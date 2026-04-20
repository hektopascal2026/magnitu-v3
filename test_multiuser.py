"""
Multi-user sync tests for Magnitu 2.

Simulates two users (Alice and Bob) with separate local databases but
syncing through the same label store.  Tests conflict resolution,
reasoning propagation, and incremental push.
"""
import os
import sys
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ── Setup: two separate temp databases ──
_test_dir = tempfile.mkdtemp(prefix="magnitu_multiuser_")
_alice_dir = os.path.join(_test_dir, "alice")
_bob_dir = os.path.join(_test_dir, "bob")
os.makedirs(_alice_dir)
os.makedirs(_bob_dir)

# We'll import db twice (by patching DB_PATH) to simulate two users.
# Since Python caches modules, we'll manually swap DB_PATH between calls.

import config
config.DB_PATH = Path(_alice_dir) / "alice.db"
config.MODELS_DIR = Path(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = Path(_test_dir) / "test_config.json"
test_config = dict(config.DEFAULTS)
test_config["min_labels_to_train"] = 4
test_config["model_architecture"] = "tfidf"
config.save_config(test_config)

import db

PASS = 0
FAIL = 0
ERRORS = []

# Shared label store (simulates what Seismo holds)
_seismo_labels = []


def test(name):
    print("  TEST: {}".format(name), end=" ... ")
    return name


def ok():
    global PASS
    PASS += 1
    print("OK")


def fail(msg):
    global FAIL
    FAIL += 1
    ERRORS.append(msg)
    print("FAIL: {}".format(msg))


def switch_to_alice():
    db.DB_PATH = Path(_alice_dir) / "alice.db"
    db.init_db()


def switch_to_bob():
    db.DB_PATH = Path(_bob_dir) / "bob.db"
    db.init_db()


# Shared test entries
test_entries = [
    {"entry_type": "feed_item", "entry_id": 1, "title": "Corruption investigation",
     "description": "Scandal", "content": "Fraud in contracts",
     "link": "", "author": "", "published_date": "2024-01-01",
     "source_name": "News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 2, "title": "Weather forecast",
     "description": "Rain", "content": "Rain this weekend",
     "link": "", "author": "", "published_date": "2024-01-02",
     "source_name": "Weather", "source_category": "weather", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 3, "title": "Sports results",
     "description": "Match", "content": "Team won the championship",
     "link": "", "author": "", "published_date": "2024-01-03",
     "source_name": "Sports", "source_category": "sports", "source_type": "rss"},
]


def simulate_seismo_push(user_labels):
    """Simulate pushing labels to Seismo (last-write-wins per entry)."""
    global _seismo_labels
    existing_map = {
        (l["entry_type"], l["entry_id"]): l for l in _seismo_labels
    }
    for lbl in user_labels:
        key = (lbl["entry_type"], lbl["entry_id"])
        existing = existing_map.get(key)
        if not existing or lbl["labeled_at"] >= existing.get("labeled_at", ""):
            existing_map[key] = lbl
    _seismo_labels = list(existing_map.values())


def simulate_seismo_pull():
    """Simulate pulling labels from Seismo."""
    return list(_seismo_labels)


def do_push_labels():
    """Push current user's labels to the simulated Seismo."""
    all_labels = db.get_all_labels_raw()
    payload_labels = [
        {
            "entry_type": lbl["entry_type"],
            "entry_id": lbl["entry_id"],
            "label": lbl["label"],
            "reasoning": lbl.get("reasoning", ""),
            "labeled_at": lbl.get("updated_at") or lbl.get("created_at", ""),
        }
        for lbl in all_labels
    ]
    simulate_seismo_push(payload_labels)
    return len(payload_labels)


def do_pull_labels():
    """Pull labels from simulated Seismo into current user's DB."""
    labels = simulate_seismo_pull()

    imported = 0
    updated = 0
    conflicts = 0

    for lbl in labels:
        entry_type = lbl.get("entry_type", "")
        entry_id = int(lbl.get("entry_id", 0))
        label = lbl.get("label", "")
        reasoning = lbl.get("reasoning", "")
        remote_time = lbl.get("labeled_at", "")
        if not entry_type or not entry_id or not label:
            continue

        conn = db.get_db()
        row = conn.execute(
            "SELECT label, reasoning, updated_at FROM labels WHERE entry_type = ? AND entry_id = ?",
            (entry_type, entry_id)
        ).fetchone()
        conn.close()

        if row is None:
            db.set_label(entry_type, entry_id, label, reasoning=reasoning)
            imported += 1
        else:
            local_time = row["updated_at"] or ""
            if row["label"] != label:
                conflicts += 1
            if remote_time > local_time:
                db.set_label(entry_type, entry_id, label, reasoning=reasoning)
                updated += 1

    return {"imported": imported, "updated": updated, "conflicts": conflicts}


# ═══════════════════════════════════════════
#  Setup both users
# ═══════════════════════════════════════════
print("\n=== Setup ===")

t = test("Initialize Alice's DB and entries")
try:
    switch_to_alice()
    db.upsert_entries(test_entries)
    assert db.get_entry_count() == 3
    ok()
except Exception as e:
    fail(str(e))

t = test("Initialize Bob's DB and entries")
try:
    switch_to_bob()
    db.upsert_entries(test_entries)
    assert db.get_entry_count() == 3
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Scenario 1: Non-overlapping labels
# ═══════════════════════════════════════════
print("\n=== Scenario 1: Non-overlapping labels ===")

t = test("Alice labels entry 1, Bob labels entry 2")
try:
    switch_to_alice()
    db.set_label("feed_item", 1, "investigation_lead", reasoning="corruption story")
    do_push_labels()

    switch_to_bob()
    db.set_label("feed_item", 2, "noise")
    do_push_labels()
    ok()
except Exception as e:
    fail(str(e))

t = test("Alice pulls — gets Bob's label for entry 2")
try:
    switch_to_alice()
    result = do_pull_labels()
    assert result["imported"] == 1, "Expected 1 imported, got {}".format(result)
    label = db.get_label("feed_item", 2)
    assert label == "noise", "Expected 'noise', got '{}'".format(label)
    ok()
except Exception as e:
    fail(str(e))

t = test("Bob pulls — gets Alice's label for entry 1 with reasoning")
try:
    switch_to_bob()
    result = do_pull_labels()
    assert result["imported"] == 1
    data = db.get_label_with_reasoning("feed_item", 1)
    assert data["label"] == "investigation_lead"
    assert data["reasoning"] == "corruption story"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Scenario 2: Conflicting labels — newer wins
# ═══════════════════════════════════════════
print("\n=== Scenario 2: Conflicting labels ===")

t = test("Alice and Bob label entry 3 differently")
try:
    import time

    switch_to_alice()
    db.set_label("feed_item", 3, "important", reasoning="alice thinks important")
    do_push_labels()

    time.sleep(1.1)  # Ensure Bob's timestamp is later

    switch_to_bob()
    db.set_label("feed_item", 3, "background", reasoning="bob thinks background")
    do_push_labels()
    ok()
except Exception as e:
    fail(str(e))

t = test("Alice pulls — Bob's newer label wins")
try:
    switch_to_alice()
    result = do_pull_labels()
    assert result["conflicts"] >= 1, "Expected conflict, got {}".format(result)
    assert result["updated"] >= 1, "Expected update, got {}".format(result)
    data = db.get_label_with_reasoning("feed_item", 3)
    assert data["label"] == "background", \
        "Expected 'background' (Bob's newer label), got '{}'".format(data["label"])
    assert data["reasoning"] == "bob thinks background", \
        "Expected Bob's reasoning, got '{}'".format(data["reasoning"])
    ok()
except Exception as e:
    fail(str(e))

t = test("Bob pulls — no change (Bob's own label is already newest)")
try:
    switch_to_bob()
    result = do_pull_labels()
    assert result["updated"] == 0, "Bob should not be updated, got {}".format(result)
    data = db.get_label_with_reasoning("feed_item", 3)
    assert data["label"] == "background"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Scenario 3: Convergence after conflict
# ═══════════════════════════════════════════
print("\n=== Scenario 3: Both users converge ===")

t = test("After sync, both users have identical labels")
try:
    switch_to_alice()
    do_push_labels()

    switch_to_bob()
    do_pull_labels()

    switch_to_alice()
    do_pull_labels()

    # Now check both have the same labels
    switch_to_alice()
    alice_labels = sorted(
        [(l["entry_id"], l["label"]) for l in db.get_all_labels_raw()],
        key=lambda x: x[0]
    )

    switch_to_bob()
    bob_labels = sorted(
        [(l["entry_id"], l["label"]) for l in db.get_all_labels_raw()],
        key=lambda x: x[0]
    )

    assert alice_labels == bob_labels, \
        "Labels should converge.\nAlice: {}\nBob:   {}".format(alice_labels, bob_labels)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Scenario 4: Reasoning survives sync
# ═══════════════════════════════════════════
print("\n=== Scenario 4: Reasoning propagation ===")

t = test("Alice updates reasoning, Bob gets it after sync")
try:
    import time

    switch_to_alice()
    time.sleep(1.1)
    db.set_label("feed_item", 1, "investigation_lead", reasoning="links X to Y through shell companies")
    do_push_labels()

    switch_to_bob()
    result = do_pull_labels()
    data = db.get_label_with_reasoning("feed_item", 1)
    assert data["reasoning"] == "links X to Y through shell companies", \
        "Expected Alice's updated reasoning, got '{}'".format(data["reasoning"])
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════
print("\n" + "=" * 50)
print("Results: {} passed, {} failed".format(PASS, FAIL))
if ERRORS:
    print("\nFailures:")
    for err in ERRORS:
        print("  - {}".format(err))
print("=" * 50)

# Cleanup
shutil.rmtree(_test_dir, ignore_errors=True)

sys.exit(0 if FAIL == 0 else 1)
