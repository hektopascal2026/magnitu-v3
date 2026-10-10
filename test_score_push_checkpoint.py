import sqlite3
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
