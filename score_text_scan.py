"""Independent, resumable recent-text sweeps for each scoring profile."""

import db

MAX_PAGE_ENTRIES = 200
PAGE_TEXT_BYTES = 8 * 1024 * 1024
TEXT_FIELDS = db._ENTRY_TEXT_COLUMNS.split(', ')
TEXT_BYTES_SQL = ' + '.join(
    "length(CAST(COALESCE({}, '') AS BLOB))".format(field) for field in TEXT_FIELDS
)


def next_page(conn, profile_id, days=14, *, limit=MAX_PAGE_ENTRIES, max_bytes=PAGE_TEXT_BYTES):
    """Admit complete rows; one oversized row is processed alone, without truncation.

    The attempt cursor commits before scoring. Failed/unscorable rows have no
    delivery checkpoint and are revisited on the next finite fenced sweep.
    New ids cannot extend a sweep indefinitely; revisions behind the cursor
    are also reconsidered next sweep. Cursors are never shared between desks.
    """
    conn.execute('''CREATE TABLE IF NOT EXISTS worker_score_text_scan (
        profile_id INTEGER PRIMARY KEY, last_id INTEGER NOT NULL, fence_id INTEGER NOT NULL)''')
    conn.commit()
    limit = max(1, min(limit, MAX_PAGE_ENTRIES))
    max_bytes = max(1, max_bytes)
    # Keep admission/text in one snapshot; reserve the cursor write up front
    # to avoid a read-to-write upgrade failure if another writer commits.
    conn.execute('BEGIN IMMEDIATE')
    with conn:
        state = conn.execute(
            'SELECT last_id,fence_id FROM worker_score_text_scan WHERE profile_id=?',
            (profile_id,),
        ).fetchone()
        last_id, fence_id = tuple(state) if state else (0, 0)
        for _ in range(2):
            if fence_id == 0:
                fence_id = conn.execute('SELECT COALESCE(MAX(id),0) FROM entries').fetchone()[0]
            candidates = conn.execute(
                'SELECT id, ' + TEXT_BYTES_SQL + ' AS text_bytes FROM entries '
                "WHERE id>? AND id<=? AND COALESCE(NULLIF(published_date, ''), fetched_at) >= date('now', ?) "
                'ORDER BY id LIMIT ?',
                (last_id, fence_id, '-{} days'.format(days), limit),
            ).fetchall()
            if candidates:
                break
            # Exhausted/removed suffix: start another sweep, including revisions
            # and arrivals that could not extend the previous frozen fence.
            last_id, fence_id = 0, 0
        ids = []
        total_bytes = 0
        for row in candidates:
            if ids and total_bytes + row[1] > max_bytes:
                break
            ids.append(row[0])
            total_bytes += row[1]
        rows = []
        if ids:
            placeholders = ','.join('?' for _ in ids)
            rows = conn.execute(
                'SELECT ' + db._ENTRY_TEXT_COLUMNS + ' FROM entries WHERE id IN (' + placeholders + ') ORDER BY id',
                ids,
            ).fetchall()
            last_id = ids[-1]
        conn.execute('''INSERT INTO worker_score_text_scan VALUES (?,?,?)
            ON CONFLICT(profile_id) DO UPDATE SET last_id=excluded.last_id,fence_id=excluded.fence_id''',
                     (profile_id, last_id, fence_id))
    return [dict(row) for row in rows]


def fresh_page(conn, before_id=None, *, days=2, lookback_days=14, limit=MAX_PAGE_ENTRIES, max_bytes=PAGE_TEXT_BYTES):
    """Newest-first, bounded current-news pass, independent of recovery cursors.

    Recent re-fetches qualify even when their original scoring clock is old.
    Each window starts at the current head; subsequent pages descend within
    that finite snapshot. New arrivals wait only until the next window, not a
    historic sweep. Checkpoints filter unchanged rows and retain failed work.
    """
    limit = max(1, min(limit, MAX_PAGE_ENTRIES))
    max_bytes = max(1, max_bytes)
    conn.execute('BEGIN')
    with conn:
        if before_id is None:
            before_id = conn.execute('SELECT COALESCE(MAX(id),0)+1 FROM entries').fetchone()[0]
        candidates = conn.execute(
            'SELECT id, ' + TEXT_BYTES_SQL + ' AS text_bytes FROM entries '
            "WHERE id<? AND COALESCE(NULLIF(published_date, ''), fetched_at) >= date('now', ?) "
            "AND (COALESCE(NULLIF(published_date, ''), fetched_at) >= datetime('now', ?) "
            "OR fetched_at >= datetime('now', ?)) "
            'ORDER BY id DESC LIMIT ?',
            (before_id, '-{} days'.format(max(1, lookback_days)),
             '-{} days'.format(max(1, min(lookback_days, days))),
             '-{} days'.format(max(1, min(lookback_days, days))), limit),
        ).fetchall()
        ids = []
        total_bytes = 0
        for row in candidates:
            if ids and total_bytes + row[1] > max_bytes:
                break
            ids.append(row[0])
            total_bytes += row[1]
        if not ids:
            return [], before_id
        placeholders = ','.join('?' for _ in ids)
        rows = conn.execute(
            'SELECT ' + db._ENTRY_TEXT_COLUMNS + ' FROM entries WHERE id IN (' + placeholders + ') ORDER BY id DESC',
            ids,
        ).fetchall()
    return [dict(row) for row in rows], ids[-1]


def by_keys(conn, keys):
    """At most 50 exact indexed probes; never scan the shared text corpus."""
    if len(keys) > 50:
        raise ValueError("Too many repair keys")
    if not keys:
        return []
    predicates = ' OR '.join('(entry_type=? AND entry_id=?)' for _ in keys)
    params = [value for key in keys for value in key]
    rows = conn.execute('SELECT ' + db._ENTRY_TEXT_COLUMNS + ' FROM entries WHERE ' + predicates,
                        params).fetchall()
    return [dict(row) for row in rows]


def embedding_ready_keys(conn, keys):
    """Metadata-only point probes for the bounded repair batch."""
    if len(keys) > 50:
        raise ValueError('Too many repair keys')
    if not keys:
        return set()
    predicates = ' OR '.join('(entry_type=? AND entry_id=?)' for _ in keys)
    rows = conn.execute('SELECT entry_type,entry_id FROM entries WHERE embedding IS NOT NULL AND ('
                        + predicates + ')', [value for key in keys for value in key]).fetchall()
    return {(row[0], row[1]) for row in rows}


def admit_repair_text_keys(conn, keys, max_bytes=PAGE_TEXT_BYTES):
    """Inspect byte sizes before loading bodies; one oversized row goes alone."""
    if len(keys) > 50:
        raise ValueError('Too many repair keys')
    if not keys:
        return [], 0
    predicates = ' OR '.join('(entry_type=? AND entry_id=?)' for _ in keys)
    rows = conn.execute('SELECT entry_type,entry_id,' + TEXT_BYTES_SQL + ' AS text_bytes FROM entries WHERE '
                        + predicates, [value for key in keys for value in key]).fetchall()
    sizes = {(row[0], row[1]): row[2] for row in rows}
    if len(sizes) != len(keys):
        raise RuntimeError('Repair local source metadata incomplete')
    admitted = []
    total = 0
    for key in keys:
        if admitted and total + sizes[key] > max_bytes:
            break
        admitted.append(key)
        total += sizes[key]
    return admitted, len(keys) - len(admitted)
