from unittest.mock import Mock
import sqlite3
import pytest
import sync
import ml_window
import score_push_checkpoint


def test_checkpoint_remote_loss_overrides_acknowledged_input():
    conn = sqlite3.connect(':memory:')
    entry = {'entry_type': 'email', 'entry_id': 7, 'content': 'unchanged'}
    pending, revisions = score_push_checkpoint.pending_batch(conn, 1, [entry], {})
    score_push_checkpoint.acknowledge(conn, 1, pending, revisions)
    assert score_push_checkpoint.pending_batch(conn, 1, [entry], {})[0] == []
    assert score_push_checkpoint.pending_batch(conn, 1, [entry], {}, force_keys=[('email', 7)])[0] == [entry]
    assert score_push_checkpoint.pending_batch(conn, 2, [entry], {})[0] == [entry]


def test_repair_target_validation_and_dedup(monkeypatch):
    request = Mock()
    request.return_value.json.return_value = {'schema_version': 1, 'model_version': 3,
        'keys': [{'entry_type': 'email', 'entry_id': 7}] * 2}
    monkeypatch.setattr(sync, '_request', request)
    profile = {'seismo_url': 'https://example.test/eu', 'api_key': 'test'}
    assert sync.pull_score_repair_candidates(3, profile) == [('email', 7)]
    assert request.call_args.kwargs['seismo_target'] == profile
    assert request.call_args.kwargs['timeout'] == 10.0


@pytest.mark.parametrize('data', [None, {}, {'schema_version': True, 'model_version': 3, 'keys': []},
    {'schema_version': 1, 'model_version': 4, 'keys': []},
    {'schema_version': 1, 'model_version': 3, 'keys': [{'entry_type': 'email', 'entry_id': True}]},
    {'schema_version': 1, 'model_version': 3, 'keys': [{'entry_type': 'unknown', 'entry_id': 1}]},
    {'schema_version': 1, 'model_version': 3, 'keys': [{'entry_type': 'email', 'entry_id': 1}] * 51}])
def test_repair_rejects_invalid_response(data, monkeypatch):
    request = Mock()
    request.return_value.json.return_value = data
    monkeypatch.setattr(sync, '_request', request)
    with pytest.raises(ValueError):
        sync.pull_score_repair_candidates(3)



def test_failed_repair_does_not_suppress_next_attempt(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.sqlite'
    monkeypatch.setattr(ml_window.db, 'get_db', lambda: sqlite3.connect(path))
    monkeypatch.setattr(ml_window, '_score_input_context', lambda *_: {})
    monkeypatch.setattr(ml_window, '_model_meta_for_push', lambda *_: {})
    entry = {'entry_type': 'email', 'entry_id': 7, 'content': 'same'}
    score = Mock(return_value=[entry])
    push = Mock(return_value={'success': True, 'errors': 0})
    monkeypatch.setattr(ml_window.pipeline, 'score_entries', score)
    monkeypatch.setattr(sync, 'push_scores', push)
    run = lambda force=(): ml_window._score_and_push_changed_entries(1, {}, {'version': 3}, [entry], force_keys=force)
    run()
    assert run()['pending'] == 0
    push.return_value = {'success': False}
    with pytest.raises(RuntimeError):
        run([('email', 7)])
    push.return_value = {'success': True, 'errors': 0}
    assert run([('email', 7)])['acknowledged'] == 1
    assert run()['pending'] == 0


def test_repair_text_lookup_uses_only_indexed_key_probes():
    import score_text_scan
    conn = Mock()
    conn.execute.return_value.fetchall.return_value = []
    assert score_text_scan.by_keys(conn, [('email', 7), ('feed_item', 8)]) == []
    sql, params = conn.execute.call_args.args
    assert ' OR ' in sql and '(entry_type=? AND entry_id=?)' in sql
    assert params == ['email', 7, 'feed_item', 8]
    with pytest.raises(ValueError):
        score_text_scan.by_keys(conn, [('email', 7)] * 51)


def test_sweep_decoder_wires_cursor_limit_and_target(monkeypatch):
    request = Mock()
    request.return_value.json.return_value = {'schema_version': 1, 'model_version': 3,
        'keys': [{'entry_type':'email','entry_id':7}], 'sweep_cursor':'opaque-next',
        'sweep_examined':200,'sweep_completed':False}
    monkeypatch.setattr(sync, '_request', request)
    profile = {'seismo_url':'https://example.test/eu','api_key':'test'}
    page = sync.pull_score_repair_page(3, profile, cursor='opaque-prior', limit=40)
    assert page['keys'] == [('email',7)]
    assert request.call_args.args[1]['sweep_cursor'] == 'opaque-prior'
    assert request.call_args.args[1]['limit'] == 40
    assert request.call_args.kwargs['seismo_target'] == profile
    request.return_value.json.return_value['sweep_examined'] = 801
    with pytest.raises(ValueError):
        sync.pull_score_repair_page(3, profile)


def test_exact_id_detail_reports_only_returned_source_keys(monkeypatch):
    request = Mock()
    request.return_value.json.return_value = {'entries':[{'entry_type':'email','entry_id':7}], 'missing_ids':[8]}
    monkeypatch.setattr(sync, '_request', request)
    monkeypatch.setattr(sync.db, 'upsert_entries', Mock())
    monkeypatch.setattr(sync.db, 'log_sync', Mock())
    assert sync.pull_entries_by_ids('email',[7,8],return_keys=True) == [('email',7)]
    request.return_value.json.return_value['entries'][0]['entry_id'] = 999
    with pytest.raises(ValueError):
        sync.pull_entries_by_ids('email',[7,8],return_keys=True)
