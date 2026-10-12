import json
import sqlite3
from types import SimpleNamespace

import pytest

import score_restore_reset as resetter
from score_push_checkpoint import acknowledge, pending_batch

URL = 'https://example.test/desk/index.php'


@pytest.fixture
def worker(tmp_path, monkeypatch):
    path = tmp_path / 'magnitu.db'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE profiles(id INTEGER PRIMARY KEY,slug TEXT UNIQUE,seismo_url TEXT,api_key TEXT);
            CREATE TABLE models(id INTEGER,profile_id INTEGER,model_path TEXT);
            CREATE TABLE entries(id INTEGER,title TEXT);
            CREATE TABLE worker_score_desk_cursor(singleton INTEGER,last_slug TEXT);
            CREATE TABLE worker_score_push_state(profile_id INTEGER NOT NULL,entry_type TEXT NOT NULL,
                entry_id INTEGER NOT NULL,fingerprint TEXT NOT NULL,PRIMARY KEY(profile_id,entry_type,entry_id));
            CREATE TABLE worker_score_push_cursor(profile_id INTEGER PRIMARY KEY,entry_type TEXT NOT NULL,entry_id INTEGER NOT NULL);
            CREATE TABLE worker_score_text_scan(profile_id INTEGER PRIMARY KEY,last_id INTEGER NOT NULL,fence_id INTEGER NOT NULL);
            INSERT INTO models VALUES(1,1,'owned-model');
            INSERT INTO entries VALUES(1,'shared source');
            INSERT INTO worker_score_desk_cursor VALUES(1,'desk');
        ''')
        for pid, slug in [(1, 'desk'), (2, 'other')]:
            conn.execute('INSERT INTO profiles VALUES(?,?,?,?)', (pid, slug, URL, 'secret'))
            conn.execute('INSERT INTO worker_score_push_state VALUES(?,?,?,?)', (pid, 'feed_item', 1, 'old'))
            conn.execute('INSERT INTO worker_score_push_cursor VALUES(?,?,?)', (pid, 'feed_item', 1))
            conn.execute('INSERT INTO worker_score_text_scan VALUES(?,?,?)', (pid, 1, 1))
    monkeypatch.setattr(resetter, 'require_stopped', lambda: None)
    return path


def snapshot(path):
    with sqlite3.connect(path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: conn.execute('SELECT * FROM ' + t).fetchall() for t in tables}


def test_preview_is_read_only_and_creates_no_files(worker):
    before = worker.read_bytes()
    files = set(worker.parent.iterdir())
    result = resetter.reset(worker, 'desk', URL)
    assert result['applied'] is False
    assert set(result['rows'].values()) == {1}
    assert worker.read_bytes() == before
    assert set(worker.parent.iterdir()) == files
    assert 'secret' not in json.dumps(result)


def test_atomic_scoped_reset_preserves_ownership_and_other_profiles(worker):
    before = snapshot(worker)
    result = resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    assert result['applied'] is True
    after = snapshot(worker)
    for table in before:
        if table in resetter.TABLE_COLUMNS:
            assert after[table] == [row for row in before[table] if row[0] != 1]
        else:
            assert after[table] == before[table]
    assert set(resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')['rows'].values()) == {0}


@pytest.mark.parametrize('slug,url,confirm', [('missing', URL, 'missing'), ('desk', 'https://wrong.test', 'desk'), ('desk', URL, None)])
def test_wrong_target_profile_or_confirmation_cannot_write(worker, slug, url, confirm):
    before = snapshot(worker)
    with pytest.raises(ValueError):
        resetter.reset(worker, slug, url, apply=True, confirm=confirm)
    assert snapshot(worker) == before


def test_upgrade_missing_tables_not_created(worker):
    with sqlite3.connect(worker) as conn:
        for table in resetter.TABLE_COLUMNS:
            conn.execute('DROP TABLE ' + table)
    before = snapshot(worker)
    assert set(resetter.reset(worker, 'desk', URL)['rows'].values()) == {None}
    resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    assert snapshot(worker) == before


def test_unknown_schema_refuses_entire_reset(worker):
    with sqlite3.connect(worker) as conn:
        conn.execute('ALTER TABLE worker_score_text_scan ADD COLUMN future TEXT')
    before = snapshot(worker)
    with pytest.raises(ValueError, match='schema'):
        resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    assert snapshot(worker) == before


def test_worker_restarted_after_initial_check_refuses_write(worker, monkeypatch):
    before = snapshot(worker)
    calls = []
    def stopped():
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('worker restarted')
    monkeypatch.setattr(resetter, 'require_stopped', stopped)
    with pytest.raises(ValueError, match='restarted'):
        resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    assert snapshot(worker) == before


def test_global_destination_read_without_importing_config(worker):
    with sqlite3.connect(worker) as conn:
        conn.execute("UPDATE profiles SET seismo_url='',api_key='' WHERE id=1")
    cfg = worker.with_name('magnitu_config.json')
    cfg.write_text(json.dumps({'seismo_url': URL, 'api_key': 'global-secret'}))
    assert resetter.reset(worker, 'desk', URL)['destination'] == URL
    with sqlite3.connect(worker) as conn:
        conn.execute("UPDATE profiles SET api_key='half' WHERE id=1")
    with pytest.raises(ValueError, match='incomplete'):
        resetter.reset(worker, 'desk', URL)


@pytest.mark.parametrize('unit', resetter.UNITS)
def test_each_active_timer_or_service_blocks_apply(monkeypatch, unit):
    # Call the real checker, not the fixture's stub.
    def show(argv, **kwargs):
        state = 'active' if argv[2] == unit else 'inactive'
        return SimpleNamespace(stdout=f'LoadState=loaded\nActiveState={state}\nSubState=dead\nMainPID=0\n')
    monkeypatch.setattr(resetter.subprocess, 'run', show)
    with pytest.raises(ValueError, match=unit):
        resetter.require_stopped()


@pytest.mark.parametrize('state', ['LoadState=not-found', 'ActiveState=failed', 'MainPID=12', 'SubState=running', ''])
def test_systemd_unknown_failed_or_live_pid_fails_closed(monkeypatch, state):
    stdout = 'LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\n' + state
    if not state:
        stdout = ''
    monkeypatch.setattr(resetter.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=stdout))
    with pytest.raises(ValueError):
        resetter.require_stopped()


def test_reset_replays_through_normal_bounded_checkpoint_with_failed_retry(worker):
    resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    entries = [{'entry_type': 'feed_item', 'entry_id': i, 'title': str(i)} for i in range(450)]
    with sqlite3.connect(worker) as conn:
        pending, revisions = pending_batch(conn, 1, entries, {'model': 1})
        assert len(pending) == 200
        acknowledge(conn, 1, pending[:199], revisions)  # Last row fails delivery.
        seen = set(e['entry_id'] for e in pending[:199])
        for _ in range(4):
            pending, revisions = pending_batch(conn, 1, entries, {'model': 1})
            assert len(pending) <= 200
            seen.update(e['entry_id'] for e in pending)
            acknowledge(conn, 1, pending, revisions)
        assert seen == set(range(450))
        assert pending_batch(conn, 1, entries, {'model': 1})[0] == []


def test_missing_database_preview_never_creates_it(tmp_path):
    path = tmp_path / 'absent.db'
    with pytest.raises(sqlite3.OperationalError):
        resetter.reset(path, 'desk', URL)
    assert not path.exists()


def test_mid_delete_failure_rolls_back_all_scoped_tables(worker):
    with sqlite3.connect(worker) as conn:
        conn.execute("CREATE TRIGGER reject_reset BEFORE DELETE ON worker_score_push_cursor BEGIN SELECT RAISE(ABORT,'blocked'); END")
    before = snapshot(worker)
    with pytest.raises(sqlite3.IntegrityError, match='blocked'):
        resetter.reset(worker, 'desk', URL, apply=True, confirm='desk')
    assert snapshot(worker) == before


def test_all_four_stopped_units_are_checked(monkeypatch):
    calls = []
    def show(argv, **kwargs):
        calls.append(argv[2])
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\n')
    monkeypatch.setattr(resetter.subprocess, 'run', show)
    resetter.require_stopped()
    assert tuple(calls) == resetter.UNITS


def test_missing_service_pid_is_not_assumed_idle(monkeypatch):
    monkeypatch.setattr(resetter.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout='LoadState=loaded\nActiveState=inactive\nSubState=dead\n'))
    with pytest.raises(ValueError, match='service'):
        resetter.require_stopped()
