"""Persistent attempt order for budget-limited score-only ML windows."""


def _ensure_state(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS worker_score_desk_cursor (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        last_slug TEXT NOT NULL)""")
    conn.commit()


def ordered_desks(conn, prepared):
    """Resume after the last admitted desk, including when it was removed."""
    if not prepared:
        return []
    _ensure_state(conn)
    ordered = sorted(prepared, key=lambda item: item['slug'])
    row = conn.execute(
        'SELECT last_slug FROM worker_score_desk_cursor WHERE singleton = 1'
    ).fetchone()
    if row is None:
        return ordered
    start = next((i for i, item in enumerate(ordered) if item['slug'] > row[0]), 0)
    return ordered[start:] + ordered[:start]


def record_attempt(conn, slug):
    """Commit before expensive work: interruption must not repeat the same head."""
    _ensure_state(conn)
    with conn:
        conn.execute("""INSERT INTO worker_score_desk_cursor VALUES (1, ?)
            ON CONFLICT(singleton) DO UPDATE SET last_slug = excluded.last_slug""", (slug,))
