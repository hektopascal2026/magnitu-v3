"""Explicit operator reconciliation after restoring a Seismo destination.

Uses only the standard library: preview never imports configuration/database
initializers, creates tables, or opens the database for writing.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess

UNITS = (
    'seismo-magnitu-ml.timer', 'seismo-magnitu-ml.service',
    'seismo-magnitu-ml-score.timer', 'seismo-magnitu-ml-score.service',
)
TABLE_COLUMNS = {
    'worker_score_push_state': ('profile_id', 'entry_type', 'entry_id', 'fingerprint'),
    'worker_score_push_cursor': ('profile_id', 'entry_type', 'entry_id'),
    'worker_score_text_scan': ('profile_id', 'last_id', 'fence_id'),
}


def require_stopped():
    """Fail closed for missing units, inaccessible systemd, or running workers."""
    for unit in UNITS:
        result = subprocess.run(
            ['systemctl', 'show', unit, '--property=LoadState,ActiveState,SubState,MainPID'],
            capture_output=True, text=True, check=True, timeout=10,
        )
        state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        # Timers have no MainPID property; services must explicitly report zero.
        pid = state.get('MainPID', '0' if unit.endswith('.timer') else None)
        if (state.get('LoadState') != 'loaded' or state.get('ActiveState') != 'inactive'
                or state.get('SubState') != 'dead' or pid != '0'):
            raise ValueError('Stop both ML timers and services before applying: ' + unit)


def _inspect(conn, slug, expected_url, config_path):
    row = conn.execute('SELECT id,slug,seismo_url,api_key FROM profiles WHERE slug=?', (slug,)).fetchone()
    if row is None:
        raise ValueError('Profile does not exist: ' + slug)
    profile_id, _, url, key = row
    url, key = (url or '').strip(), (key or '').strip()
    if bool(url) != bool(key):
        raise ValueError('Profile has incomplete destination credentials')
    if not url:
        cfg = json.loads(Path(config_path).read_text())
        url = cfg.get('seismo_url', '')
        if not url or not cfg.get('api_key'):
            raise ValueError('Global destination URL and API key must be configured')
    if url != expected_url:
        raise ValueError('Expected destination does not match the profile destination')
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts = {}
    for table, columns in TABLE_COLUMNS.items():
        if table not in tables:
            counts[table] = None  # Older worker installations have no state yet.
            continue
        schema = conn.execute('PRAGMA table_info(' + table + ')').fetchall()
        expected_pk = (1, 2, 3, 0) if table == 'worker_score_push_state' else (1,) + (0,) * (len(columns) - 1)
        if tuple(row[1] for row in schema) != columns or tuple(row[5] for row in schema) != expected_pk:
            raise ValueError('Unexpected worker table schema: ' + table)
        counts[table] = conn.execute('SELECT COUNT(*) FROM ' + table + ' WHERE profile_id=?',
                                     (profile_id,)).fetchone()[0]
    return {'profile_id': profile_id, 'slug': slug, 'destination': url, 'rows': counts}


def reset(db_path, slug, expected_url, *, config_path=None, apply=False, confirm=None):
    """Preview or atomically clear only this profile's score delivery/scan state."""
    path = Path(db_path).resolve()
    config_path = config_path or path.with_name('magnitu_config.json')
    if apply:
        if confirm != slug:
            raise ValueError('--confirm-profile must exactly match --profile for apply')
        require_stopped()
    uri = path.as_uri() + ('?mode=rw' if apply else '?mode=ro')
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as conn:
        try:
            conn.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            report = _inspect(conn, slug, expected_url, config_path)
            if apply:
                require_stopped()  # Recheck after acquiring the SQLite writer lock.
                for table, count in report['rows'].items():
                    if count is not None:
                        conn.execute('DELETE FROM ' + table + ' WHERE profile_id=?', (report['profile_id'],))
                conn.commit()
            else:
                conn.rollback()
        finally:
            conn.close()
    report['applied'] = apply
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='Existing worker magnitu.db')
    parser.add_argument('--profile', required=True, help='Exact existing profile slug')
    parser.add_argument('--expected-url', required=True, help='Exact destination URL, including index.php if configured')
    parser.add_argument('--config', help='Config JSON for profiles using the global destination')
    parser.add_argument('--apply', action='store_true', help='Delete scoped bookkeeping after systemd stopped checks')
    parser.add_argument('--confirm-profile', help='Repeat the profile slug to confirm apply')
    args = parser.parse_args(argv)
    try:
        report = reset(args.db, args.profile, args.expected_url, config_path=args.config,
                       apply=args.apply, confirm=args.confirm_profile)
    except (ValueError, OSError, sqlite3.Error, subprocess.SubprocessError) as exc:
        parser.exit(1, 'Reset refused: ' + str(exc) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
