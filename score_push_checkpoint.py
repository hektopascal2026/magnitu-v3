"""Revision-aware worker bookkeeping. No change to the Seismo wire contract."""
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()


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
    known = {(r[0], r[1]): r[2] for r in conn.execute(
        'SELECT entry_type,entry_id,fingerprint FROM worker_score_push_state WHERE profile_id=?',
        (profile_id,))}
    pending = []
    revisions = {}
    for index in range(len(entries)):
        entry = entries[(start + index) % len(entries)]
        key = (entry['entry_type'], entry['entry_id'])
        text = {k: v for k, v in entry.items() if k not in ('id', 'fetched_at', 'embedding')}
        revision = fingerprint([context, text])
        if known.get(key) != revision:
            pending.append(entry)
            revisions[key] = revision
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
