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
