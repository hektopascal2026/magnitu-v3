"""Leg / legislation source grouping tests."""
import os
import sys
import tempfile
from pathlib import Path

_test_dir = tempfile.mkdtemp(prefix="magnitu_test_entry_sources_")
os.environ["MAGNITU_TEST"] = "1"

import config
config.DB_PATH = Path(_test_dir) / "test.db"
config.MODELS_DIR = Path(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = Path(_test_dir) / "test_config.json"
config.save_config(dict(config.DEFAULTS))

import db
db.DB_PATH = config.DB_PATH
db.init_db()

from magnitu.entry_sources import entry_matches_source_filter, is_leg_source_type

PASS = FAIL = 0


def ok():
    global PASS
    PASS += 1
    print("  OK")


def fail(msg):
    global FAIL
    FAIL += 1
    print("  FAIL:", msg)


def test_is_leg_source_type():
    assert is_leg_source_type("leg_parliament_ch")
    assert is_leg_source_type("parl_press")
    assert not is_leg_source_type("rss")
    ok()


def test_lex_and_email_filters():
    cal = {"entry_type": "calendar_event", "entry_id": 1, "source_type": "leg_parliament_ch"}
    lex = {"entry_type": "lex_item", "entry_id": 2, "source_type": "lex_ch"}
    press = {"entry_type": "feed_item", "entry_id": 3, "source_type": "parl_press"}
    rss = {"entry_type": "feed_item", "entry_id": 4, "source_type": "rss"}
    mail = {"entry_type": "email", "entry_id": 5, "source_type": "email"}
    assert entry_matches_source_filter(cal, "lex")
    assert entry_matches_source_filter(lex, "lex")
    assert not entry_matches_source_filter(press, "lex")
    assert not entry_matches_source_filter(rss, "lex")
    assert entry_matches_source_filter(mail, "email")
    assert not entry_matches_source_filter(rss, "email")
    assert not entry_matches_source_filter(cal, "email")
    ok()


def test_db_source_filters():
    db.upsert_entries([
        {
            "entry_type": "calendar_event", "entry_id": 10,
            "title": "Leg cal", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Parl", "source_category": "", "source_type": "leg_parliament_ch",
        },
        {
            "entry_type": "lex_item", "entry_id": 11,
            "title": "Lex", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Fedlex", "source_category": "", "source_type": "lex_ch",
        },
        {
            "entry_type": "feed_item", "entry_id": 12,
            "title": "Press", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Parl", "source_category": "", "source_type": "parl_press",
        },
        {
            "entry_type": "email", "entry_id": 13,
            "title": "Inbox", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Mail", "source_category": "", "source_type": "email",
        },
    ])
    lex_rows = db.get_unlabeled_entries(limit=50, source_filter="lex", profile_id=1)
    lex_types = {(r["entry_type"], r["entry_id"]) for r in lex_rows}
    if lex_types != {("calendar_event", 10), ("lex_item", 11)}:
        fail("lex filter wrong: %s" % lex_types)
        return

    email_rows = db.get_unlabeled_entries(limit=50, source_filter="email", profile_id=1)
    email_types = {(r["entry_type"], r["entry_id"]) for r in email_rows}
    if email_types != {("email", 13)}:
        fail("email filter wrong: %s" % email_types)
        return
    ok()


if __name__ == "__main__":
    print("=== entry_sources ===")
    test_is_leg_source_type()
    test_lex_and_email_filters()
    test_db_source_filters()
    print("Results: %d passed, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
