import sqlite3

import pytest

import db
import score_text_scan
from score_push_checkpoint import pending_batch, acknowledge


@pytest.fixture
def conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 'worker.sqlite'))
    conn.row_factory = sqlite3.Row
    columns = ['id INTEGER PRIMARY KEY AUTOINCREMENT', 'entry_id INTEGER']
    columns.extend(field + " TEXT DEFAULT ''" for field in score_text_scan.TEXT_FIELDS
                   if field not in ('id', 'entry_id'))
    conn.execute('CREATE TABLE entries (' + ','.join(columns) + ')')
    yield conn
    conn.close()


def add(conn, start, stop, content='body'):
    conn.executemany("INSERT INTO entries (id,entry_type,entry_id,content,published_date,fetched_at) "
                     "VALUES (?, 'feed_item', ?, ?, datetime('now'), datetime('now'))",
                     [(i, i, content) for i in range(start, stop)])
    conn.commit()


def delivered(conn, profile, page):
    batch, revisions = pending_batch(conn, profile, page, {'model': 1})
    assert len(batch) <= 200
    acknowledge(conn, profile, batch, revisions)
    return batch


def test_three_independent_desks_each_cover_more_than_two_pages(conn):
    add(conn, 1, 552)
    seen = {profile: set() for profile in (1, 2, 3)}
    # Only one desk survives each simulated interrupted window. A global page
    # cursor coupled to this three-desk rotation would permanently miss rows.
    for profile in [1, 2, 3] * 3:
        page = score_text_scan.next_page(conn, profile)
        assert len(page) <= 200
        seen[profile].update(row['entry_id'] for row in delivered(conn, profile, page))
    assert all(ids == set(range(1, 552)) for ids in seen.values())


def test_unacknowledged_interrupted_page_returns_after_finite_sweep(conn):
    add(conn, 1, 552)
    failed = score_text_scan.next_page(conn, 1)
    assert failed[0]['entry_id'] == 1
    # No delivery checkpoint: interruption after text admission.
    delivered(conn, 1, score_text_scan.next_page(conn, 1))
    delivered(conn, 1, score_text_scan.next_page(conn, 1))
    retry = score_text_scan.next_page(conn, 1)
    assert [row['entry_id'] for row in delivered(conn, 1, retry)] == list(range(1, 201))


def test_process_restart_resumes_each_profiles_frozen_sweep(conn):
    add(conn, 1, 552)
    path = conn.execute('PRAGMA database_list').fetchone()[2]
    score_text_scan.next_page(conn, 1)
    add(conn, 552, 562)
    reopened = sqlite3.connect(path)
    reopened.row_factory = sqlite3.Row
    try:
        assert score_text_scan.next_page(reopened, 1)[0]['id'] == 201
        assert reopened.execute('SELECT fence_id FROM worker_score_text_scan WHERE profile_id=1').fetchone()[0] == 551
        assert score_text_scan.next_page(reopened, 2)[0]['id'] == 1
        assert reopened.execute('SELECT fence_id FROM worker_score_text_scan WHERE profile_id=2').fetchone()[0] == 561
    finally:
        reopened.close()


def test_deleted_cursor_and_suffix_do_not_skip_survivors(conn):
    add(conn, 1, 552)
    first = score_text_scan.next_page(conn, 1)
    assert first[-1]['id'] == 200
    conn.execute('DELETE FROM entries WHERE id BETWEEN 180 AND 250')
    conn.commit()
    second = score_text_scan.next_page(conn, 1)
    assert [row['id'] for row in second] == list(range(251, 451))
    conn.execute('DELETE FROM entries WHERE id > 450')
    conn.commit()
    assert score_text_scan.next_page(conn, 1)[0]['id'] == 1


def test_arrivals_do_not_extend_fence_and_are_seen_next_sweep(conn):
    add(conn, 1, 552)
    seen = set()
    for tick in range(3):
        page = score_text_scan.next_page(conn, 1)
        seen.update(row['id'] for row in page)
        add(conn, 552 + tick * 10, 562 + tick * 10)
    assert seen == set(range(1, 552))
    assert conn.execute('SELECT fence_id FROM worker_score_text_scan').fetchone()[0] == 551
    next_seen = set()
    for _ in range(3):
        next_seen.update(row['id'] for row in score_text_scan.next_page(conn, 1))
    assert set(range(552, 582)) <= next_seen


def test_revision_behind_cursor_is_reconsidered_next_sweep(conn):
    add(conn, 1, 552)
    delivered(conn, 1, score_text_scan.next_page(conn, 1))
    conn.execute("UPDATE entries SET content='hydrated body' WHERE id=1")
    conn.commit()
    delivered(conn, 1, score_text_scan.next_page(conn, 1))
    delivered(conn, 1, score_text_scan.next_page(conn, 1))
    changed = delivered(conn, 1, score_text_scan.next_page(conn, 1))
    assert [(row['id'], row['content']) for row in changed] == [(1, 'hydrated body')]


def test_body_budget_preserves_oversized_document_without_truncating(conn):
    add(conn, 1, 5, content='x' * 1000)
    page = score_text_scan.next_page(conn, 1, max_bytes=2500)
    assert len(page) == 2
    oversized = score_text_scan.next_page(conn, 1, max_bytes=256)
    assert len(oversized) == 1
    assert oversized[0]['content'] == 'x' * 1000
    assert score_text_scan.next_page(conn, 1, max_bytes=256)[0]['id'] == 4


def test_horizon_matches_recent_entries_with_missing_date_fallback(conn):
    add(conn, 1, 5)
    conn.execute("UPDATE entries SET published_date=datetime('now','-15 days') WHERE id=1")
    conn.execute("UPDATE entries SET published_date=NULL WHERE id=2")
    conn.execute("UPDATE entries SET published_date='' WHERE id=3")
    conn.commit()
    assert [row['id'] for row in score_text_scan.next_page(conn, 1, days=14)] == [2, 3, 4]


def test_empty_catalog_and_deleted_profile_cursor_start_safely(conn):
    assert score_text_scan.next_page(conn, 1) == []
    add(conn, 1, 3)
    assert [row['id'] for row in score_text_scan.next_page(conn, 1)] == [1, 2]
    assert [row['id'] for row in score_text_scan.next_page(conn, 2)] == [1, 2]


def test_metadata_and_body_queries_have_fixed_limits_and_no_embedding_blobs(conn):
    add(conn, 1, 1001)
    statements = []
    conn.set_trace_callback(statements.append)
    assert len(score_text_scan.next_page(conn, 1, limit=1000)) == 200
    metadata = [sql for sql in statements if sql.startswith('SELECT id, length(')]
    bodies = [sql for sql in statements if sql.startswith('SELECT ' + db._ENTRY_TEXT_COLUMNS)]
    assert len(metadata) == len(bodies) == 1
    assert metadata[0].endswith('LIMIT 200')
    assert 'embedding' not in bodies[0]
    assert bodies[0].split(' IN (')[1].split(')')[0].count(',') == 199


def test_fresh_pass_bypasses_old_fence_without_moving_recovery_cursor(conn):
    add(conn, 1, 552)
    conn.execute("UPDATE entries SET published_date=datetime('now','-8 days'),fetched_at=datetime('now','-8 days')")
    conn.commit()
    score_text_scan.next_page(conn, 1)
    state = tuple(conn.execute('SELECT last_id,fence_id FROM worker_score_text_scan').fetchone())
    add(conn, 552, 1302)
    seen = set()
    cursor = None
    for _ in range(5):
        page, cursor = score_text_scan.fresh_page(conn, cursor)
        seen.update(r['id'] for r in page)
    assert seen == set(range(552, 1302))
    assert tuple(conn.execute('SELECT last_id,fence_id FROM worker_score_text_scan').fetchone()) == state
    assert score_text_scan.next_page(conn, 1)[0]['id'] == 201


def test_new_window_revisits_failed_and_revised_fresh_rows(conn):
    add(conn, 1, 402)
    first, cursor = score_text_scan.fresh_page(conn)
    delivered(conn, 1, first)
    conn.execute("UPDATE entries SET content='revised' WHERE id=401")
    conn.commit()
    again, _ = score_text_scan.fresh_page(conn)
    assert [r['id'] for r in delivered(conn, 1, again)] == [401]
    # A delivery failure creates no acknowledgement, so a new window retries it.
    add(conn, 402, 403)
    failed, _ = score_text_scan.fresh_page(conn)
    pending_batch(conn, 1, failed, {'model': 1})
    retry, _ = score_text_scan.fresh_page(conn)
    assert [r['id'] for r in delivered(conn, 1, retry)] == [402]


def test_fresh_pages_use_full_bodies_byte_bound_and_finite_head(conn):
    add(conn, 1, 6, content='ü' * 1000)
    first, cursor = score_text_scan.fresh_page(conn, max_bytes=4500)
    assert [r['id'] for r in first] == [5, 4]
    add(conn, 6, 8)
    second, cursor = score_text_scan.fresh_page(conn, cursor, max_bytes=256)
    assert len(second) == 1 and second[0]['id'] == 3
    assert second[0]['content'] == 'ü' * 1000
    assert [r['id'] for r in score_text_scan.fresh_page(conn)[0]][:2] == [7, 6]


def test_recent_refetch_of_old_content_enters_fresh_lane_and_is_not_double_delivered(conn):
    add(conn, 1, 3)
    conn.execute("UPDATE entries SET published_date=datetime('now','-10 days'),fetched_at=datetime('now','-10 days')")
    conn.commit()
    assert score_text_scan.fresh_page(conn)[0] == []
    conn.execute("UPDATE entries SET fetched_at=datetime('now') WHERE id=1")
    conn.commit()
    page, _ = score_text_scan.fresh_page(conn)
    assert [r['id'] for r in delivered(conn, 1, page)] == [1]
    conn.execute("UPDATE entries SET fetched_at=datetime('now','+1 second') WHERE id=1")
    conn.commit()
    assert delivered(conn, 1, score_text_scan.fresh_page(conn)[0]) == []


def test_recent_refetch_does_not_override_configured_score_horizon(conn):
    add(conn, 1, 3)
    conn.execute("UPDATE entries SET published_date=datetime('now','-10 days')")
    conn.commit()
    assert len(score_text_scan.fresh_page(conn, lookback_days=14)[0]) == 2
    assert score_text_scan.fresh_page(conn, lookback_days=7)[0] == []
