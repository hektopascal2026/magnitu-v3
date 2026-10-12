import sqlite3
import score_push_checkpoint
from score_push_checkpoint import pending_batch, acknowledge


def entry(n=1, content='old'):
    return {'entry_type': 'feed_item', 'entry_id': n, 'content': content, 'fetched_at': 'today'}


def test_noop_after_success_but_content_model_and_desk_changes_retry():
    conn = sqlite3.connect(':memory:')
    batch, rev = pending_batch(conn, 1, [entry()], 'model1')
    assert len(batch) == 1
    acknowledge(conn, 1, batch, rev)
    assert pending_batch(conn, 1, [dict(entry(), fetched_at='tomorrow')], 'model1')[0] == []
    assert pending_batch(conn, 1, [entry(content='hydrated')], 'model1')[0]
    assert pending_batch(conn, 1, [entry()], 'model2')[0]
    assert pending_batch(conn, 2, [entry()], 'model1')[0]


def test_failed_push_and_missing_embeddings_do_not_advance():
    conn = sqlite3.connect(':memory:')
    entries = [entry(1), entry(2)]
    batch, rev = pending_batch(conn, 1, entries, 'model1')
    # No acknowledgement: crash/failure retries both.
    assert pending_batch(conn, 1, entries, 'model1')[0] == entries
    acknowledge(conn, 1, [batch[0]], rev)
    assert pending_batch(conn, 1, entries, 'model1')[0] == [entry(2)]


def test_bounded_batches_drain_without_repeating_successful_prefix():
    conn = sqlite3.connect(':memory:')
    entries = [entry(i) for i in range(501)]
    seen = []
    for _ in range(3):
        batch, rev = pending_batch(conn, 1, entries, 'model1')
        assert len(batch) <= 200
        seen.extend(e['entry_id'] for e in batch)
        acknowledge(conn, 1, batch, rev)
    assert seen == list(range(501))
    assert pending_batch(conn, 1, entries, 'model1')[0] == []


def test_failed_prefix_does_not_starve_later_rows_and_wraps():
    conn = sqlite3.connect(':memory:')
    entries = [entry(i) for i in range(5)]
    first, _ = pending_batch(conn, 1, entries, 'model1', limit=2)
    assert [e['entry_id'] for e in first] == [0, 1]
    second, rev = pending_batch(conn, 1, entries, 'model1', limit=2)
    assert [e['entry_id'] for e in second] == [2, 3]
    acknowledge(conn, 1, second, rev)
    third, _ = pending_batch(conn, 1, entries, 'model1', limit=2)
    assert [e['entry_id'] for e in third] == [4, 0]
    # Independent profile cursor starts at its own head.
    assert pending_batch(conn, 2, entries, 'model1', limit=2)[0] == first


class RecordingConnection:
    def __init__(self, conn):
        self.conn = conn
        self.lookups = []

    def execute(self, sql, params=()):
        cursor = self.conn.execute(sql, params)
        if sql.startswith('SELECT entry_type,entry_id,fingerprint'):
            rows = list(cursor)
            self.lookups.append((sql, params, rows))
            return rows
        return cursor

    def commit(self):
        self.conn.commit()


def test_checkpoint_reads_only_bounded_current_keys_not_profile_history():
    conn = sqlite3.connect(':memory:')
    entries = [entry(i) for i in range(251)]
    for _ in range(2):
        batch, rev = pending_batch(conn, 1, entries, {'model': 1})
        acknowledge(conn, 1, batch, rev)
    conn.executemany('INSERT INTO worker_score_push_state VALUES (1,?,?,?)',
                     [('feed_item', i, 'historical') for i in range(10000, 15000)])
    conn.commit()
    recorded = RecordingConnection(conn)
    assert pending_batch(recorded, 1, entries, {'model': 1})[0] == []
    assert len(recorded.lookups) == 3
    assert sum(len(rows) for _, _, rows in recorded.lookups) == 251
    for sql, params, rows in recorded.lookups:
        assert len(params) <= 300
        assert len(rows) <= 100
        assert all(row[1] < 251 for row in rows)
        # Each OR arm probes the complete indexed primary key.
        plan = conn.execute('EXPLAIN QUERY PLAN ' + sql, params).fetchall()
        assert all('SCAN worker_score_push_state' not in row[3] for row in plan)


def test_changed_output_stays_capped_and_stops_checkpoint_reads_early():
    conn = sqlite3.connect(':memory:')
    recorded = RecordingConnection(conn)
    batch, _ = pending_batch(recorded, 1, [entry(i) for i in range(1000)], {}, limit=1000)
    assert len(batch) == 200
    assert len(recorded.lookups) == 2


def test_context_hashed_once_and_config_changes_invalidate(monkeypatch):
    conn = sqlite3.connect(':memory:')
    context = {'model': {'version': 1}, 'config': {'apply_prior': False}}
    original = score_push_checkpoint.fingerprint
    calls = []
    def record(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(score_push_checkpoint, 'fingerprint', record)
    batch, rev = pending_batch(conn, 1, [entry(i) for i in range(5)], context)
    assert sum(value is context for value in calls) == 1
    assert all(value[1] == original(context) for value in calls[1:])
    acknowledge(conn, 1, batch, rev)
    assert pending_batch(conn, 1, batch, context)[0] == []
    changed_config = {'model': {'version': 1}, 'config': {'apply_prior': True}}
    assert len(pending_batch(conn, 1, batch, changed_config)[0]) == 5
    changed_model = {'model': {'version': 2}, 'config': {'apply_prior': False}}
    assert len(pending_batch(conn, 1, batch, changed_model)[0]) == 5


def test_old_fingerprint_replays_once_after_successful_acknowledgement():
    conn = sqlite3.connect(':memory:')
    item = entry()
    context = {'model': 1}
    pending_batch(conn, 1, [item], context)
    text = {k: v for k, v in item.items() if k not in ('id', 'fetched_at', 'embedding')}
    old = score_push_checkpoint.fingerprint([context, text])
    conn.execute('INSERT INTO worker_score_push_state VALUES (1,?,?,?)',
                 ('feed_item', 1, old))
    conn.commit()
    batch, rev = pending_batch(conn, 1, [item], context)
    assert batch == [item]
    assert rev[('feed_item', 1)] != old
    acknowledge(conn, 1, batch, rev)
    assert pending_batch(conn, 1, [item], context)[0] == []


def test_retry_cursor_rotates_across_lookup_page_boundary():
    conn = sqlite3.connect(':memory:')
    entries = [entry(i) for i in range(250)]
    first, _ = pending_batch(conn, 1, entries, {}, limit=101)
    assert first == entries[:101]
    second, rev = pending_batch(conn, 1, entries, {}, limit=101)
    assert second == entries[101:202]
    acknowledge(conn, 1, second, rev)
    third, _ = pending_batch(conn, 1, entries, {}, limit=101)
    assert third == entries[202:] + entries[:53]
