"""Revision-aware worker bookkeeping. No change to the Seismo wire contract."""
import hashlib
import json

CHECKPOINT_LOOKUP_PAGE = 100
MAX_PENDING_BATCH = 200


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()


def _known_for_page(conn, profile_id, entries):
    # Explicit full-primary-key probes avoid loading a profile's historic state.
    # 100 keys use 300 parameters, below SQLite's legacy 999-variable floor.
    predicates = []
    params = []
    for entry in entries:
        predicates.append('(profile_id=? AND entry_type=? AND entry_id=?)')
        params.extend((profile_id, entry['entry_type'], entry['entry_id']))
    rows = conn.execute(
        'SELECT entry_type,entry_id,fingerprint FROM worker_score_push_state WHERE '
        + ' OR '.join(predicates), params,
    )
    return {(row[0], row[1]): row[2] for row in rows}


def pending_batch(conn, profile_id, entries, context, limit=200):
    """Select a bounded batch; fetched_at changes alone do not invalidate text."""
    conn.execute("""CREATE TABLE IF NOT EXISTS worker_score_push_state (
        profile_id INTEGER NOT NULL, entry_type TEXT NOT NULL, entry_id INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        PRIMARY KEY (profile_id, entry_type, entry_id))""")
    conn.commit()
    conn.execute("""CREATE TABLE IF NOT EXISTS worker_score_push_cursor (
        profile_id INTEGER PRIMARY KEY, entry_type TEXT NOT NULL, entry_id INTEGER NOT NULL)""")
    cursor = conn.execute('SELECT entry_type,entry_id FROM worker_score_push_cursor WHERE profile_id=?',
                          (profile_id,)).fetchone()
    start = 0
    if cursor:
        for i, entry in enumerate(entries):
            if (entry['entry_type'], entry['entry_id']) == tuple(cursor):
                start = i + 1
                break
    limit = max(1, min(limit, MAX_PENDING_BATCH))
    context_revision = fingerprint(context)
    pending = []
    revisions = {}
    for offset in range(0, len(entries), CHECKPOINT_LOOKUP_PAGE):
        page = [entries[(start + index) % len(entries)]
                for index in range(offset, min(offset + CHECKPOINT_LOOKUP_PAGE, len(entries)))]
        known = _known_for_page(conn, profile_id, page)
        for entry in page:
            key = (entry['entry_type'], entry['entry_id'])
            text = {k: v for k, v in entry.items() if k not in ('id', 'fetched_at', 'embedding')}
            revision = fingerprint(['score-input-v2', context_revision, text])
            if known.get(key) != revision:
                pending.append(entry)
                revisions[key] = revision
                if len(pending) >= limit:
                    break
        if len(pending) >= limit:
            break
    # Advance the attempt cursor independently of successful delivery. Failed or
    # unscorable rows remain pending, but cannot monopolize every bounded batch.
    if pending:
        last = pending[-1]
        conn.execute("""INSERT INTO worker_score_push_cursor VALUES (?,?,?)
            ON CONFLICT(profile_id) DO UPDATE SET entry_type=excluded.entry_type,entry_id=excluded.entry_id""",
                     (profile_id, last['entry_type'], last['entry_id']))
    conn.commit()
    return pending, revisions


def acknowledge(conn, profile_id, scores, revisions):
    """Only scored, acknowledged rows advance. Failed/omitted rows stay pending."""
    rows = []
    for score in scores:
        key = (score['entry_type'], score['entry_id'])
        if key in revisions:
            rows.append((profile_id, *key, revisions[key]))
    with conn:
        conn.executemany('''INSERT INTO worker_score_push_state VALUES (?,?,?,?)
            ON CONFLICT(profile_id,entry_type,entry_id)
            DO UPDATE SET fingerprint=excluded.fingerprint''', rows)
