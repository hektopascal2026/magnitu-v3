import sqlite3

import desk_admission


def desks(*slugs):
    return [{'slug': slug} for slug in slugs]


def test_interrupted_windows_rotate_persisted_admission(tmp_path):
    path = tmp_path / 'worker.sqlite'
    admitted = []
    for _ in range(7):
        # Reopen each window, simulating process exit immediately after admission.
        with sqlite3.connect(path) as conn:
            first = desk_admission.ordered_desks(conn, desks('c', 'a', 'b'))[0]
            desk_admission.record_attempt(conn, first['slug'])
            admitted.append(first['slug'])
    assert admitted == ['a', 'b', 'c', 'a', 'b', 'c', 'a']


def test_removed_cursor_desk_resumes_at_successor_and_wraps():
    with sqlite3.connect(':memory:') as conn:
        desk_admission.record_attempt(conn, 'b')
        assert desk_admission.ordered_desks(conn, desks('a', 'c', 'd')) == desks('c', 'd', 'a')
        desk_admission.record_attempt(conn, 'z')
        assert desk_admission.ordered_desks(conn, desks('c', 'a')) == desks('a', 'c')
        assert desk_admission.ordered_desks(conn, []) == []


def test_cursor_tracks_each_attempt_rather_than_preselected_window_head():
    with sqlite3.connect(':memory:') as conn:
        prepared = desks('a', 'b', 'c')
        for item in desk_admission.ordered_desks(conn, prepared)[:2]:
            desk_admission.record_attempt(conn, item['slug'])
        assert desk_admission.ordered_desks(conn, prepared) == desks('c', 'a', 'b')
