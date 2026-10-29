import sqlite3
from unittest.mock import Mock
import pytest
import db
import ml_window
import score_repair_scan as scan
import sync


def connection(path):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    return c


def setup_entries(c):
    c.execute('CREATE TABLE entries(id INTEGER PRIMARY KEY, entry_type TEXT, entry_id INTEGER, published_date TEXT, content TEXT, embedding BLOB)')
    c.execute('CREATE UNIQUE INDEX entry_key ON entries(entry_type,entry_id)')
    c.executemany("INSERT INTO entries VALUES(?,'email',?,?,?,NULL)",
                  [(i, i, '2020-01-%02d' % i, 'complete old body ' * 100) for i in range(1, 10)])
    c.executemany("INSERT INTO entries VALUES(?,'email',?,?,?,NULL)",
                  [(i, i, '2026-10-%02d' % (i - 90), 'fresh') for i in range(100, 110)])
    c.commit()


def test_priority_is_round_robin_across_profiles_deduplicated_and_missing_only(tmp_path):
    c = connection(tmp_path / 'state.sqlite')
    setup_entries(c)
    scan.admit(c, 1, 3, 'one', [('email', 1), ('email', 2), ('email', 3)])
    scan.admit(c, 2, 7, 'two', [('email', 1), ('email', 4), ('email', 5)])
    scan.admit(c, 3, 9, 'three', [('email', 6), ('email', 7), ('email', 999)])
    c.execute("UPDATE entries SET embedding=X'01' WHERE entry_id=2"); c.commit()
    before = [dict(r) for r in c.execute('SELECT * FROM worker_score_repair_scan')]
    assert scan.embedding_priority_keys(c, limit=5) == [('email', 1), ('email', 4), ('email', 6), ('email', 3), ('email', 5)]
    all_keys = scan.embedding_priority_keys(c)
    assert len(all_keys) == len(set(all_keys)) == 6
    assert ('email', 2) not in all_keys and ('email', 999) not in all_keys
    assert before == [dict(r) for r in c.execute('SELECT * FROM worker_score_repair_scan')]
    c.close()


def test_no_existing_queue_is_a_read_only_empty_priority():
    c = sqlite3.connect(':memory:')
    assert scan.embedding_priority_keys(c) == []
    assert c.execute("SELECT name FROM sqlite_master WHERE name LIKE 'worker_score_repair%'").fetchall() == []
    c.close()


def test_invalid_pending_state_is_not_silently_skipped():
    c = sqlite3.connect(':memory:')
    scan.load(c, 1, 3)
    c.execute("INSERT INTO worker_score_repair_scan VALUES(1,3,'cursor','[[\"email\",true]]')")
    c.commit()
    with pytest.raises(ValueError, match='Invalid durable repair key'):
        scan.embedding_priority_keys(c)
    c.close()


def test_bounded_priority_precedes_newest_rows_without_truncation(tmp_path, monkeypatch):
    path = tmp_path / 'entries.sqlite'
    c = connection(path); setup_entries(c)
    c.execute("UPDATE entries SET embedding=X'01' WHERE entry_id=2"); c.commit(); c.close()
    monkeypatch.setattr(db, 'get_db', lambda: connection(path))
    rows = db.get_entries_without_embeddings(limit=4, priority_keys=[('email', 3), ('email', 2), ('email', 1), ('email', 3)])
    assert [r['entry_id'] for r in rows] == [3, 1, 109, 108]
    assert rows[0]['content'] == 'complete old body ' * 100
    assert all(r['embedding'] is None for r in rows)
    assert [r['entry_id'] for r in db.get_entries_without_embeddings(limit=3)] == [109, 108, 107]


@pytest.mark.parametrize('limit,keys', [(201, [('email', 1)]), (10, [('email', 1)] * 101), (10, [('email', True)])])
def test_priority_payload_bounds_are_enforced(tmp_path, monkeypatch, limit, keys):
    path = tmp_path / 'entries.sqlite'
    c = connection(path); setup_entries(c); c.close()
    monkeypatch.setattr(db, 'get_db', lambda: connection(path))
    with pytest.raises(ValueError):
        db.get_entries_without_embeddings(limit=limit, priority_keys=keys)


def test_shared_loop_forwards_priority_inside_original_total_budget(monkeypatch):
    compute = Mock(return_value=200)
    monkeypatch.setattr(ml_window.sync, '_compute_pending_embeddings', compute)
    monkeypatch.setattr(ml_window.time, 'monotonic', lambda: 0)
    keys = [('email', 1)]
    ml_window._embed_pending_until_done(max_entries=200, deadline=150, priority_keys=keys)
    compute.assert_called_once_with(limit=200, deadline=150, priority_keys=keys)


def test_compute_forwards_priority_and_preserves_incremental_stores(monkeypatch):
    from test_embedding_budget import fake_embeddings
    _, pending, store, embed, release = fake_embeddings(monkeypatch, 12)
    keys = [('email', 1)]
    assert sync._compute_pending_embeddings(limit=200, deadline=150, priority_keys=keys) == 12
    pending.assert_called_once_with(limit=200, priority_keys=keys)
    assert [len(call.args[0]) for call in embed.call_args_list] == [10, 2]
    assert store.call_count == 2
    release.assert_called_once_with()
