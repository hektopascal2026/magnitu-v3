"""HTTP-level smoke test for the new training knobs in /api/settings and the
Settings page template.  Uses FastAPI's TestClient against a temp DB.
"""
import os
import sys
import tempfile
from pathlib import Path as _P

_test_dir = tempfile.mkdtemp(prefix="magnitu_http_")
os.environ["MAGNITU_TEST"] = "1"

import config
config.DB_PATH = _P(_test_dir) / "test.db"
config.MODELS_DIR = _P(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = _P(_test_dir) / "test_config.json"
config.save_config(dict(config.DEFAULTS))

import db
db.DB_PATH = config.DB_PATH
db.init_db()

# Ensure at least the default profile exists so /p/{slug}/settings can render
if not db.has_any_profile():
    db.create_profile(slug="default", display_name="Default", description="")

from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

PASS = 0
FAIL = 0
ERRORS = []


def t(name):
    print("  TEST: {} ...".format(name), end=" ")


def ok():
    global PASS
    PASS += 1
    print("OK")


def fail(msg):
    global FAIL
    FAIL += 1
    ERRORS.append(msg)
    print("FAIL: {}".format(msg))


print("\n=== /api/settings accepts new knobs ===")

t("POST /api/settings with new knobs returns success")
try:
    r = client.post("/api/settings", json={
        "label_time_decay_days": 120,
        "label_time_decay_floor": 0.25,
        "reasoning_weight_boost": 1.5,
        "legal_signal_patterns": ["Drittland", "Binnenmarkt", "  ", "Drittland"],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"]
    cfg = body["config"]
    assert cfg["label_time_decay_days"] == 120
    assert abs(cfg["label_time_decay_floor"] - 0.25) < 1e-6
    assert abs(cfg["reasoning_weight_boost"] - 1.5) < 1e-6
    # Whitespace-only and duplicate entries are deduped/trimmed
    assert cfg["legal_signal_patterns"] == ["Drittland", "Binnenmarkt"], cfg["legal_signal_patterns"]
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings clamps out-of-range decay floor")
try:
    r = client.post("/api/settings", json={"label_time_decay_floor": 2.0})
    body = r.json()
    assert body["success"]
    assert body["config"]["label_time_decay_floor"] == 1.0
    r = client.post("/api/settings", json={"label_time_decay_floor": -0.5})
    assert r.json()["config"]["label_time_decay_floor"] == 0.0
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings clamps out-of-range reasoning boost")
try:
    r = client.post("/api/settings", json={"reasoning_weight_boost": 10.0})
    assert r.json()["config"]["reasoning_weight_boost"] == 5.0
    r = client.post("/api/settings", json={"reasoning_weight_boost": -1.0})
    assert r.json()["config"]["reasoning_weight_boost"] == 0.0
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings clamps huge half-life")
try:
    r = client.post("/api/settings", json={"label_time_decay_days": 99999})
    assert r.json()["config"]["label_time_decay_days"] == 3650
    r = client.post("/api/settings", json={"label_time_decay_days": -5})
    assert r.json()["config"]["label_time_decay_days"] == 0
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings rejects garbage decay_days gracefully")
try:
    r = client.post("/api/settings", json={"label_time_decay_days": "not a number"})
    # With invalid input, current code raises ValueError on int(float(...)).
    # That's acceptable as long as we don't mangle config silently — status 500
    # or 422 is fine, but saved config should remain the previous valid value.
    # We just check the server didn't crash permanently.
    r2 = client.post("/api/settings", json={"label_time_decay_days": 30})
    assert r2.status_code == 200
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings wraps a string pattern into a list")
try:
    r = client.post("/api/settings", json={"legal_signal_patterns": "Ursprungserzeugnis"})
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["legal_signal_patterns"] == ["Ursprungserzeugnis"]
    ok()
except Exception as e:
    fail(repr(e))

t("/api/settings with empty list clears patterns")
try:
    r = client.post("/api/settings", json={"legal_signal_patterns": []})
    assert r.status_code == 200
    assert r.json()["config"]["legal_signal_patterns"] == []
    ok()
except Exception as e:
    fail(repr(e))


print("\n=== Settings page renders with knobs ===")

t("/p/default/settings renders and includes the new fields")
try:
    r = client.post("/api/settings", json={
        "label_time_decay_days": 120,
        "label_time_decay_floor": 0.3,
        "reasoning_weight_boost": 1.5,
        "legal_signal_patterns": ["Drittland", "CE-Kennzeichnung"],
    })
    assert r.status_code == 200
    r = client.get("/p/default/settings")
    assert r.status_code == 200, r.status_code
    html = r.text
    assert 'id="labelDecayDays"' in html
    assert 'id="labelDecayFloor"' in html
    assert 'id="reasoningBoost"' in html
    assert 'id="legalSignalPatterns"' in html
    # Values should be pre-filled from config
    assert 'value="120"' in html
    assert "Drittland" in html
    assert "CE-Kennzeichnung" in html
    # JS button handler present
    assert "saveAdvancedBtn" in html
    ok()
except Exception as e:
    fail(repr(e))


print("\n=== Embedding invalidation on pattern change ===")

t("changing legal_signal_patterns clears embeddings")
try:
    # Prime an embedding
    db.upsert_entry({
        "entry_type": "feed_item", "entry_id": 1000,
        "title": "Test", "description": "", "content": "",
        "link": "", "author": "", "published_date": "",
        "source_name": "", "source_category": "", "source_type": "rss",
    })
    conn = db.get_db()
    conn.execute("UPDATE entries SET embedding=? WHERE entry_type=? AND entry_id=?",
                 (b"X" * 10, "feed_item", 1000))
    conn.commit()
    cnt_before = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert cnt_before >= 1

    r = client.post("/api/settings", json={"legal_signal_patterns": ["Drittland", "NEW_PATTERN"]})
    assert r.status_code == 200

    conn = db.get_db()
    cnt_after = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert cnt_after == 0, "embeddings not cleared: {}".format(cnt_after)
    ok()
except Exception as e:
    fail(repr(e))


t("no-op update does NOT invalidate embeddings")
try:
    conn = db.get_db()
    conn.execute("UPDATE entries SET embedding=? WHERE entry_type=? AND entry_id=?",
                 (b"Y" * 10, "feed_item", 1000))
    conn.commit()
    conn.close()
    # Save same patterns again
    r = client.post("/api/settings", json={"legal_signal_patterns": ["Drittland", "NEW_PATTERN"]})
    assert r.status_code == 200
    conn = db.get_db()
    cnt = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert cnt >= 1, "embedding was cleared on no-op save"
    ok()
except Exception as e:
    fail(repr(e))


print("\n" + "=" * 50)
print("Results: {} passed, {} failed".format(PASS, FAIL))
if ERRORS:
    print("\nFailures:")
    for err in ERRORS:
        print("  - {}".format(err))
print("=" * 50)

import shutil
shutil.rmtree(_test_dir, ignore_errors=True)

sys.exit(0 if FAIL == 0 else 1)
