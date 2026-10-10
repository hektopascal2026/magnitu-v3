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
