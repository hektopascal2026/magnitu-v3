"""Durable bounded repair admission, independent per profile/model."""
import json

MAX_PENDING = 200
RETRY_BATCH = 10


def load(conn, profile_id, model_version):
    conn.execute('''CREATE TABLE IF NOT EXISTS worker_score_repair_scan (
        profile_id INTEGER PRIMARY KEY, model_version INTEGER NOT NULL,
        cursor TEXT NOT NULL, pending TEXT NOT NULL)''')
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


def _save(conn, profile_id, model_version, state):
    with conn:
        conn.execute('''INSERT INTO worker_score_repair_scan VALUES (?,?,?,?)
            ON CONFLICT(profile_id) DO UPDATE SET model_version=excluded.model_version,
            cursor=excluded.cursor,pending=excluded.pending''',
                     (profile_id, model_version, state['cursor'], json.dumps(state['pending'])))


def admit(conn, profile_id, model_version, cursor, new_keys):
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
    _save(conn, profile_id, model_version, {'cursor': cursor, 'pending': pending})
    return batch


def complete(conn, profile_id, model_version, keys):
    state = load(conn, profile_id, model_version)
    done = set(keys)
    state['pending'] = [key for key in state['pending'] if key not in done]
    _save(conn, profile_id, model_version, state)
