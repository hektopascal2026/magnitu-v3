"""Durable bounded repair admission, independent per profile/model."""
import json
import time

MAX_PENDING = 200
RETRY_BATCH = 10


def load(conn, profile_id, model_version):
    conn.execute('''CREATE TABLE IF NOT EXISTS worker_score_repair_scan (
        profile_id INTEGER PRIMARY KEY, model_version INTEGER NOT NULL,
        cursor TEXT NOT NULL, pending TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS worker_score_repair_retry (
        profile_id INTEGER NOT NULL, model_version INTEGER NOT NULL,
        entry_type TEXT NOT NULL, entry_id INTEGER NOT NULL,
        first_seen INTEGER NOT NULL, last_attempt INTEGER NOT NULL,
        attempts INTEGER NOT NULL, cause TEXT NOT NULL,
        PRIMARY KEY(profile_id,entry_type,entry_id))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS worker_score_repair_progress (
        profile_id INTEGER PRIMARY KEY, model_version INTEGER NOT NULL,
        cursor TEXT NOT NULL, last_progress INTEGER NOT NULL)''')
    conn.commit()
    row = conn.execute('SELECT model_version,cursor,pending FROM worker_score_repair_scan WHERE profile_id=?',
                       (profile_id,)).fetchone()
    if row is None or row[0] != model_version:
        return {'cursor': '', 'pending': []}
    keys = json.loads(row[2])
    if (not isinstance(keys, list) or len(keys) > MAX_PENDING or not isinstance(row[1], str)
            or len(row[1]) > 512):
        raise ValueError('Invalid durable repair state')
    out = []
    for key in keys:
        if (not isinstance(key, list) or len(key) != 2 or key[0] not in
                ('feed_item', 'email', 'lex_item', 'calendar_event')
                or type(key[1]) is not int or key[1] < 1):
            raise ValueError('Invalid durable repair key')
        pair = tuple(key)
        if pair not in out:
            out.append(pair)
    return {'cursor': row[1], 'pending': out}


def _save(conn, profile_id, model_version, state, attempted=(), now=None):
    now = int(time.time()) if now is None else now
    with conn:
        conn.execute('''INSERT INTO worker_score_repair_scan VALUES (?,?,?,?)
            ON CONFLICT(profile_id) DO UPDATE SET model_version=excluded.model_version,
            cursor=excluded.cursor,pending=excluded.pending''',
                     (profile_id, model_version, state['cursor'], json.dumps(state['pending'])))
        # Side tables share the cursor/queue transaction and never grow beyond pending keys.
        conn.execute('DELETE FROM worker_score_repair_retry WHERE profile_id=? AND model_version<>?',
                     (profile_id, model_version))
        kept = set(state['pending'])
        for entry_type, entry_id in conn.execute(
                'SELECT entry_type,entry_id FROM worker_score_repair_retry WHERE profile_id=?', (profile_id,)).fetchall():
            if (entry_type, entry_id) not in kept:
                conn.execute('DELETE FROM worker_score_repair_retry WHERE profile_id=? AND entry_type=? AND entry_id=?',
                             (profile_id, entry_type, entry_id))
        for entry_type, entry_id in kept:
            conn.execute('''INSERT OR IGNORE INTO worker_score_repair_retry VALUES (?,?,?,?,?,0,0,'unknown_legacy')''',
                         (profile_id, model_version, entry_type, entry_id, now))
        for entry_type, entry_id in attempted:
            conn.execute('''UPDATE worker_score_repair_retry SET last_attempt=?,attempts=attempts+1,cause='interrupted'
                WHERE profile_id=? AND model_version=? AND entry_type=? AND entry_id=?''',
                         (now, profile_id, model_version, entry_type, entry_id))
        conn.execute('''INSERT INTO worker_score_repair_progress VALUES(?,?,?,?)
            ON CONFLICT(profile_id) DO UPDATE SET model_version=excluded.model_version,
            cursor=excluded.cursor,last_progress=CASE
                WHEN worker_score_repair_progress.model_version<>excluded.model_version
                  OR worker_score_repair_progress.cursor<>excluded.cursor THEN excluded.last_progress
                ELSE worker_score_repair_progress.last_progress END''',
                     (profile_id, model_version, state['cursor'], now))


def admit(conn, profile_id, model_version, cursor, new_keys, now=None):
    """Persist cursor and returned keys atomically BEFORE any network/scoring work."""
    normalized = []
    for key in new_keys:
        if (not isinstance(key, (list, tuple)) or len(key) != 2 or key[0] not in
                ('feed_item', 'email', 'lex_item', 'calendar_event')
                or type(key[1]) is not int or key[1] < 1):
            raise ValueError('Invalid repair admission key')
        normalized.append(tuple(key))
    new_keys = normalized
    state = load(conn, profile_id, model_version)
    retry = state['pending'][:RETRY_BATCH]
    batch = list(retry)
    pending = state['pending'][len(retry):]
    for key in new_keys:
        if key not in batch:
            batch.append(key)
        if key not in state['pending']:
            pending.append(key)
    pending.extend(retry)  # Rotate attempts so poison rows cannot monopolize retries.
    if len(batch) > 50 or len(pending) > MAX_PENDING or not isinstance(cursor, str) or len(cursor) > 512:
        raise ValueError('Repair admission exceeds durable bounds')
    _save(conn, profile_id, model_version, {'cursor': cursor, 'pending': pending}, batch, now)
    return batch


def complete(conn, profile_id, model_version, keys):
    state = load(conn, profile_id, model_version)
    done = set(keys)
    state['pending'] = [key for key in state['pending'] if key not in done]
    _save(conn, profile_id, model_version, state)


CAUSES = {'interrupted', 'refetch_failure', 'local_source_failure', 'text_budget',
          'embedding_budget', 'deadline_exhausted', 'scoring_delivery_failure', 'scoring_omission'}


def outcome(conn, profile_id, model_version, keys, cause):
    """Record only bounded classified causes, never source text or credentials."""
    if cause not in CAUSES:
        raise ValueError('Invalid repair cause')
    with conn:
        conn.executemany('''UPDATE worker_score_repair_retry SET cause=?
            WHERE profile_id=? AND model_version=? AND entry_type=? AND entry_id=?''',
                         [(cause, profile_id, model_version, t, i) for t, i in keys])


def health(conn, profile_id, model_version, now=None):
    """Bounded queue health; legacy age starts when telemetry first observes it."""
    now = int(time.time()) if now is None else now
    state = load(conn, profile_id, model_version)
    rows = conn.execute('''SELECT first_seen,last_attempt,attempts,cause FROM worker_score_repair_retry
        WHERE profile_id=? AND model_version=?''', (profile_id, model_version)).fetchall()
    progress = conn.execute('SELECT last_progress FROM worker_score_repair_progress WHERE profile_id=? AND model_version=?',
                            (profile_id, model_version)).fetchone()
    causes = {}
    for row in rows:
        causes[row[3]] = causes.get(row[3], 0) + 1
    return {'pending': len(state['pending']),
            'oldest_seconds': max([max(0, now-r[0]) for r in rows], default=0),
            'max_attempts': max([r[2] for r in rows], default=0),
            'cursor_stale_seconds': max(0, now-progress[0]) if progress else 0,
            'causes': causes}
