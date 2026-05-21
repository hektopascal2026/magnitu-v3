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


def test_lex_filter_includes_calendar_and_parl_press():
    cal = {"entry_type": "calendar_event", "entry_id": 1, "source_type": "leg_parliament_ch"}
    press = {"entry_type": "feed_item", "entry_id": 2, "source_type": "parl_press"}
    rss = {"entry_type": "feed_item", "entry_id": 3, "source_type": "rss"}
    assert entry_matches_source_filter(cal, "lex")
    assert entry_matches_source_filter(press, "lex")
    assert not entry_matches_source_filter(rss, "lex")
    assert entry_matches_source_filter(rss, "news")
    assert not entry_matches_source_filter(press, "news")
    ok()


def test_db_lex_unlabeled_includes_leg_rows():
    db.upsert_entries([
        {
            "entry_type": "calendar_event", "entry_id": 10,
            "title": "Leg cal", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Parl", "source_category": "", "source_type": "leg_parliament_ch",
        },
        {
            "entry_type": "feed_item", "entry_id": 11,
            "title": "Press", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "Parl", "source_category": "", "source_type": "parl_press",
        },
        {
            "entry_type": "feed_item", "entry_id": 12,
            "title": "RSS", "description": "", "content": "",
            "link": "", "author": "", "published_date": "2026-05-01",
            "source_name": "News", "source_category": "", "source_type": "rss",
        },
    ])
    lex_rows = db.get_unlabeled_entries(limit=50, source_filter="lex", profile_id=1)
    types = {(r["entry_type"], r["entry_id"]) for r in lex_rows}
    if (("calendar_event", 10) not in types) or (("feed_item", 11) not in types):
        fail("lex filter missing leg rows: %s" % types)
        return
    if ("feed_item", 12) in types:
        fail("lex filter should exclude rss feed_item")
        return
    ok()


if __name__ == "__main__":
    print("=== entry_sources ===")
    test_is_leg_source_type()
    test_lex_filter_includes_calendar_and_parl_press()
    test_db_lex_unlabeled_includes_leg_rows()
    print("Results: %d passed, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
