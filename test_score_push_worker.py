import sqlite3
from unittest.mock import Mock
import pytest
import ml_window


def worker_connection(path):
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE TABLE IF NOT EXISTS profiles (id INTEGER PRIMARY KEY, seismo_url TEXT, api_key TEXT, display_name TEXT)')
    conn.commit()
    return conn


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
    profile = {'seismo_url': 'https://example.test', 'api_key': 'test'}
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
@pytest.mark.parametrize('repair_failure', [False, True])
def test_score_only_window_reads_each_profiles_own_text_page(entries, repair_failure, monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    database = MagicMock()
    database.get_db.side_effect = lambda: worker_connection(tmp_path / 'worker.sqlite')
    database.slugify.side_effect = lambda slug: slug
    database.derive_profile_identity_from_push_url.side_effect = lambda url: (url, url)
    database.get_profile_by_slug.side_effect = lambda slug: {'id': 1 if slug == 'a' else 2}
    database.get_active_model.return_value = {'version': 1, 'trained_at': None}
    text_page = Mock(side_effect=lambda profile_id: list(entries))
    monkeypatch.setattr(ml_window, '_fresh_score_text_page', lambda _: ([], 0))
    monkeypatch.setattr(ml_window, '_recent_score_text_page', text_page)
    monkeypatch.setattr(ml_window, 'db', database)
    monkeypatch.setattr(ml_window, 'sync', MagicMock())
    ml_window.sync.pull_score_repair_candidates.return_value = []
    if repair_failure:
        ml_window.sync.pull_score_repair_candidates.side_effect = RuntimeError('repair unavailable')
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
    assert ml_window.main() == int(repair_failure)
    assert ml_window.sync.pull_score_repair_candidates.call_count == 2
    assert [call.args[0] for call in text_page.call_args_list] == [1, 2]
    database.get_recent_entries.assert_not_called()
    assert score.call_count == 2
    assert all(call.args[-1] == entries for call in score.call_args_list)
    assert score.call_args_list[0].args[-1] is not score.call_args_list[1].args[-1]


def test_score_only_main_records_admission_before_interrupted_desk_work(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    class InterruptedWindow(BaseException):
        pass

    path = tmp_path / 'worker.sqlite'
    database = MagicMock()
    database.get_db.side_effect = lambda: worker_connection(path)
    database.slugify.side_effect = lambda slug: slug
    database.derive_profile_identity_from_push_url.side_effect = lambda url: (url, url)
    database.get_profile_by_slug.side_effect = lambda slug: {'id': {'a': 1, 'b': 2, 'c': 3}[slug]}
    database.get_active_model.return_value = {'version': 1, 'trained_at': None}
    monkeypatch.setattr(ml_window, '_fresh_score_text_page', lambda _: ([], 0))
    monkeypatch.setattr(ml_window, '_recent_score_text_page', lambda _: [{'entry_type': 'feed_item', 'entry_id': 1}])
    monkeypatch.setattr(ml_window, 'db', database)
    monkeypatch.setattr(ml_window, 'sync', MagicMock())
    ml_window.sync.pull_score_repair_candidates.return_value = []
    monkeypatch.setattr(ml_window, 'pipeline', MagicMock())
    monkeypatch.setattr(ml_window, 'get_config', lambda: {'seismo_url': 'mother', 'api_key': 'test'})
    monkeypatch.setattr(ml_window, '_load_worker_secrets_file', lambda: None)
    monkeypatch.setattr(ml_window, 'enforce_embedding_store_cap', lambda: None)
    monkeypatch.setattr(ml_window, '_embed_pending_until_done', lambda **_: None)
    monkeypatch.setattr(ml_window, '_label_counts', lambda *_: {})
    monkeypatch.setenv('MAGNITU_VAULT_PASSWORD', 'test-vault')
    monkeypatch.setenv('MAGNITU_ML_SCORE_ONLY', '1')
    monkeypatch.setenv('SEISMO_DESKS_JSON', '[{"slug":"c","seismo_url":"c","api_key":"test"},{"slug":"a","seismo_url":"a","api_key":"test"},{"slug":"b","seismo_url":"b","api_key":"test"}]')
    attempted = []
    def interrupt(profile_id, *_):
        with sqlite3.connect(str(path)) as conn:
            persisted_slug = conn.execute('SELECT last_slug FROM worker_score_desk_cursor').fetchone()[0]
        assert {'a': 1, 'b': 2, 'c': 3}[persisted_slug] == profile_id
        attempted.append(profile_id)
        raise InterruptedWindow()
    monkeypatch.setattr(ml_window, '_score_and_push_changed_entries', interrupt)
    for _ in range(4):
        with pytest.raises(InterruptedWindow):
            ml_window.main()
    assert attempted == [1, 2, 3, 1]


def test_actual_main_scores_more_than_one_fresh_page_for_each_desk(tmp_path, monkeypatch):
    import score_text_scan
    from unittest.mock import MagicMock
    database_path = tmp_path / 'worker.sqlite'
    def connection():
        c = sqlite3.connect(str(database_path)); c.row_factory = sqlite3.Row
        return c
    c = connection()
    c.execute('CREATE TABLE profiles (id INTEGER PRIMARY KEY,seismo_url TEXT,api_key TEXT,display_name TEXT)')
    fields = ['id INTEGER PRIMARY KEY', 'entry_id INTEGER'] + [f + " TEXT DEFAULT ''" for f in score_text_scan.TEXT_FIELDS if f not in ('id','entry_id')]
    c.execute('CREATE TABLE entries (' + ','.join(fields) + ')')
    c.executemany("INSERT INTO entries(id,entry_type,entry_id,published_date) VALUES(?,'feed_item',?,datetime('now','-8 days'))", [(i,i) for i in range(1,552)])
    c.executemany("INSERT INTO entries(id,entry_type,entry_id,published_date) VALUES(?,'feed_item',?,datetime('now'))", [(i,i) for i in range(552,1302)])
    c.commit(); c.close()
    database = MagicMock()
    database.get_db.side_effect = connection
    database.get_effective_config.return_value = {}
    database.slugify.side_effect = lambda s: s
    database.derive_profile_identity_from_push_url.side_effect = lambda s:(s,s)
    database.get_profile_by_slug.side_effect = lambda s:{'id':1 if s=='a' else 2}
    database.get_active_model.return_value = {'version':1,'trained_at':None}
    monkeypatch.setattr(ml_window,'db',database)
    monkeypatch.setattr(ml_window,'get_config',lambda:{'seismo_url':'mother','api_key':'test'})
    monkeypatch.setattr(ml_window,'_load_worker_secrets_file',lambda:None)
    monkeypatch.setattr(ml_window,'enforce_embedding_store_cap',lambda:None)
    monkeypatch.setattr(ml_window,'_embed_pending_until_done',lambda **_:None)
    monkeypatch.setattr(ml_window,'_label_counts',lambda *_:{})
    monkeypatch.setattr(ml_window,'_model_meta_for_push',lambda *_:{})
    monkeypatch.setattr(ml_window.pipeline,'load_calibration',lambda _:None)
    monkeypatch.setattr(ml_window.pipeline,'score_entries',lambda rows,**_:rows)
    pushed={1:set(),2:set()}
    def push(rows,profile,**_):
        pushed[profile['id']].update(r['entry_id'] for r in rows)
        return {'success':True,'errors':0}
    monkeypatch.setattr(ml_window.sync,'push_scores',push)
    monkeypatch.setattr(ml_window.sync,'pull_score_repair_candidates',lambda *_,**__:[])
    monkeypatch.setattr(ml_window.sync,'pull_all_entry_types',lambda **_:None)
    monkeypatch.setattr(ml_window.sync,'entry_store_watermarks',lambda:{})
    monkeypatch.setattr(ml_window.sync,'post_ml_window_report',lambda _:None)
    monkeypatch.setenv('MAGNITU_VAULT_PASSWORD','test')
    monkeypatch.setenv('MAGNITU_ML_SCORE_ONLY','1')
    monkeypatch.setenv('SEISMO_DESKS_JSON','[{"slug":"a","seismo_url":"a","api_key":"test"},{"slug":"b","seismo_url":"b","api_key":"test"}]')
    assert ml_window.main()==0
    assert all(set(range(552,1302)) <= ids for ids in pushed.values())
    assert all(set(range(1,201)) <= ids for ids in pushed.values())
    assert ml_window.main()==0  # Retry/unchanged input does not double-deliver.
    c=connection()
    assert c.execute('SELECT COUNT(*) FROM worker_score_push_state').fetchone()[0]==2300
    c.close()
