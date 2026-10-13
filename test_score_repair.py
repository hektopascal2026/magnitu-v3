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


def test_repair_refetches_and_forces_exact_remote_keys(monkeypatch):
    keys = [('email', 7), ('feed_item', 8)]
    rows = [{'entry_type': t, 'entry_id': i, 'content': 'current'} for t, i in keys]
    monkeypatch.setattr(sync, 'pull_score_repair_candidates', Mock(return_value=keys))
    pull = Mock(return_value=1)
    monkeypatch.setattr(sync, 'pull_entries_by_ids', pull)
    conn = Mock()
    monkeypatch.setattr(ml_window.db, 'get_db', Mock(return_value=conn))
    monkeypatch.setattr(ml_window.score_text_scan, 'by_keys', Mock(return_value=rows))
    score = Mock(return_value={'acknowledged': 2})
    monkeypatch.setattr(ml_window, '_score_and_push_changed_entries', score)
    ml_window._repair_remote_scores(1, {}, {'version': 3})
    assert pull.call_count == 2
    assert score.call_args.args[-1] == rows
    assert score.call_args.kwargs['force_keys'] == keys
    score.return_value = {'acknowledged': 1}
    with pytest.raises(RuntimeError, match='partially scored'):
        ml_window._repair_remote_scores(1, {}, {'version': 3})
    pull.return_value = 0
    with pytest.raises(RuntimeError, match='refetch incomplete'):
        ml_window._repair_remote_scores(1, {}, {'version': 3})


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


def test_repair_respects_refetch_admission_deadline(monkeypatch):
    monkeypatch.setattr(sync, 'pull_score_repair_candidates', lambda *_, **__: [('email', 7)])
    pull = Mock()
    monkeypatch.setattr(sync, 'pull_entries_by_ids', pull)
    with pytest.raises(RuntimeError, match='deadline'):
        ml_window._repair_remote_scores(1, {}, {'version': 3}, deadline=0)
    pull.assert_not_called()
