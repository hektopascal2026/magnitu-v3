import sqlite3
from unittest.mock import Mock
import pytest
import ml_window


def test_worker_skips_unchanged_and_retries_failed_push(tmp_path, monkeypatch):
    database = tmp_path / 'worker.sqlite'
    monkeypatch.setattr(ml_window.db, 'get_db', lambda: sqlite3.connect(str(database)))
    monkeypatch.setattr(ml_window.db, 'get_effective_config', lambda _: {})
    monkeypatch.setattr(ml_window, 'get_config', lambda: {})
    monkeypatch.setattr(ml_window.pipeline, 'load_calibration', lambda _: None)
    monkeypatch.setattr(ml_window, '_model_meta_for_push', lambda *_: {})
    score = Mock(side_effect=lambda entries, **_: entries)
    push = Mock(return_value={'success': True, 'errors': 0})
    monkeypatch.setattr(ml_window.pipeline, 'score_entries', score)
    monkeypatch.setattr(ml_window.sync, 'push_scores', push)
    entry = {'entry_type': 'feed_item', 'entry_id': 1, 'content': 'body'}
    model = {'version': 1}
    profile = {'seismo_url': 'https://example.test'}
    run = lambda: ml_window._score_and_push_changed_entries(1, profile, model, [entry])
    run()
    run()
    assert score.call_count == push.call_count == 1
    entry['content'] = 'changed body'
    push.return_value = {'success': True, 'errors': 1}
    with pytest.raises(RuntimeError):
        run()
    push.return_value = {'success': True, 'errors': 0}
    run()
    run()
    assert score.call_count == push.call_count == 3


def test_each_http_batch_must_acknowledge_without_errors(monkeypatch):
    import sync
    request = Mock()
    request.return_value.json.return_value = {'success': True, 'errors': 1}
    monkeypatch.setattr(sync, '_request', request)
    with pytest.raises(RuntimeError):
        sync._post_json_batched_with_size('magnitu_scores', [{'x': 1}, {'x': 2}],
                                        lambda rows: {'scores': rows}, None, 1)
    assert request.call_count == 1


@pytest.mark.parametrize('entries', [[], [{'entry_type': 'feed_item', 'entry_id': 1}]])
def test_score_only_window_reuses_one_recent_text_snapshot(entries, monkeypatch):
    from unittest.mock import MagicMock
    database = MagicMock()
    database.slugify.side_effect = lambda slug: slug
    database.derive_profile_identity_from_push_url.side_effect = lambda url: (url, url)
    database.get_profile_by_slug.side_effect = lambda slug: {'id': 1 if slug == 'a' else 2}
    database.get_active_model.return_value = {'version': 1, 'trained_at': None}
    database.get_recent_entries.return_value = entries
    monkeypatch.setattr(ml_window, 'db', database)
    monkeypatch.setattr(ml_window, 'sync', MagicMock())
    monkeypatch.setattr(ml_window, 'pipeline', MagicMock())
    monkeypatch.setattr(ml_window, 'get_config', lambda: {'seismo_url': 'mother', 'api_key': 'test'})
    monkeypatch.setattr(ml_window, '_load_worker_secrets_file', lambda: None)
    monkeypatch.setattr(ml_window, 'enforce_embedding_store_cap', lambda: None)
    monkeypatch.setattr(ml_window, '_embed_pending_until_done', lambda **_: None)
    monkeypatch.setattr(ml_window, '_label_counts', lambda *_: {})
    score = Mock()
    monkeypatch.setattr(ml_window, '_score_and_push_changed_entries', score)
    monkeypatch.setenv('MAGNITU_VAULT_PASSWORD', 'test-vault')
    monkeypatch.setenv('MAGNITU_ML_SCORE_ONLY', '1')
    monkeypatch.setenv('SEISMO_DESKS_JSON', '[{"seismo_url":"a","api_key":"test"},{"seismo_url":"b","api_key":"test"}]')
    assert ml_window.main() == 0
    database.get_recent_entries.assert_called_once_with(days=14, include_embedding=False)
    assert score.call_count == 2
    assert all(call.args[-1] is entries for call in score.call_args_list)
