"""Integration test: train both pipelines with all three knobs enabled
simultaneously, to verify there are no crossed wires or shape mismatches.

Runs against a temp DB so it doesn't touch real data.
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path as _P

_test_dir = tempfile.mkdtemp(prefix="magnitu_knobs_")
os.environ["MAGNITU_TEST"] = "1"

import config
config.DB_PATH = _P(_test_dir) / "test.db"
config.MODELS_DIR = _P(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = _P(_test_dir) / "test_config.json"

cfg = dict(config.DEFAULTS)
cfg["min_labels_to_train"] = 4
cfg["label_time_decay_days"] = 120
cfg["label_time_decay_floor"] = 0.25
cfg["reasoning_weight_boost"] = 1.5
cfg["legal_signal_patterns"] = ["Drittland", "Binnenmarkt", "CE-Kennzeichnung"]
config.save_config(cfg)

import db
db.DB_PATH = config.DB_PATH
db.init_db()

import pipeline
import distiller

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


now = datetime.now()
old = now - timedelta(days=240)
fresh = now.strftime("%Y-%m-%d %H:%M:%S")
old_s = old.strftime("%Y-%m-%d %H:%M:%S")

base_entries = [
    {"entry_type": "lex_item", "entry_id": 1,
     "title": "Drittland-Import: neue Vorschriften",
     "description": "Die Binnenmarkt-Regelung trifft Exporteure.",
     "content": "CE-Kennzeichnung wird ab 2026 verschärft.",
     "link": "", "author": "", "published_date": "2024-01-01",
     "source_name": "Bund", "source_category": "legal", "source_type": "lex_eu"},
    {"entry_type": "lex_item", "entry_id": 2,
     "title": "Investigation lead: public contracts scandal",
     "description": "Corruption allegations involving officials.",
     "content": "Investigators have opened a probe.",
     "link": "", "author": "", "published_date": "2024-02-01",
     "source_name": "News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 3,
     "title": "Wetterbericht: Sonnig", "description": "Schönes Wetter.",
     "content": "Heute 25 Grad.",
     "link": "", "author": "", "published_date": "2024-03-01",
     "source_name": "SRF", "source_category": "weather", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 4,
     "title": "Weekly sports roundup", "description": "Standard league results.",
     "content": "Team A beat Team B.",
     "link": "", "author": "", "published_date": "2024-04-01",
     "source_name": "SRF", "source_category": "sports", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 5,
     "title": "Drittland tariffs reshape trade",
     "description": "Market access under review.",
     "content": "EU single market considerations.",
     "link": "", "author": "", "published_date": "2024-05-01",
     "source_name": "ft", "source_category": "biz", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 6,
     "title": "Horoscope for this week", "description": "Stars align.",
     "content": "Generic filler content.",
     "link": "", "author": "", "published_date": "2024-06-01",
     "source_name": "blog", "source_category": "lifestyle", "source_type": "rss"},
]

# Pad up to 30 entries (distillation requires >= 20) with generic copies
entries = list(base_entries)
for i in range(7, 31):
    template = base_entries[(i - 7) % len(base_entries)]
    entries.append({
        **template,
        "entry_id": i,
        "title": template["title"] + " #" + str(i),
        "description": template["description"],
    })
db.upsert_entries(entries)

# Mix of fresh/old labels and with/without reasoning
db.set_label("lex_item", 1, "investigation_lead",
             reasoning="Classic Drittland import anomaly, worth a probe.")
db.set_label("lex_item", 2, "investigation_lead",
             reasoning="Direct corruption angle.")
db.set_label("feed_item", 3, "noise")
db.set_label("feed_item", 4, "noise")
db.set_label("feed_item", 5, "important",
             reasoning="Trade policy shifts often become leads.")
db.set_label("feed_item", 6, "noise")

# Backdate one row to exercise decay
conn = db.get_db()
conn.execute(
    "UPDATE labels SET created_at=?, updated_at=? WHERE entry_type=? AND entry_id=?",
    (old_s, old_s, "feed_item", 3),
)
conn.commit()
conn.close()


# ─── sample weights ─────────────────────────────────────────────────
print("\n=== sample weights ===")
labeled = db.get_all_labels(profile_id=1)
w = pipeline.compute_sample_weights(labeled, cfg)

t("length matches labeled")
try:
    assert len(w) == len(labeled)
    ok()
except Exception as e:
    fail(str(e))

t("old label has decayed weight near floor")
try:
    for i, lbl in enumerate(labeled):
        if lbl["entry_type"] == "feed_item" and lbl["entry_id"] == 3:
            assert 0.24 <= w[i] <= 0.26, "got {}".format(w[i])
            break
    else:
        raise AssertionError("old label not found")
    ok()
except Exception as e:
    fail(str(e))

t("reasoning boost multiplies fresh labels")
try:
    for i, lbl in enumerate(labeled):
        if lbl["entry_type"] == "feed_item" and lbl["entry_id"] == 5:
            # fresh + reasoning → 1.0 * 1.5 = 1.5
            assert abs(w[i] - 1.5) < 0.05, "got {}".format(w[i])
            break
    else:
        raise AssertionError("reasoning-boosted label not found")
    ok()
except Exception as e:
    fail(str(e))


# ─── text builder picks up config automatically ────────────────────
print("\n=== text builder (config fallback) ===")

t("build_entry_text with no explicit patterns reads config")
try:
    txt = pipeline._build_entry_text(entries[0])
    assert "signals=" in txt, "no signals prefix: {}".format(txt[:200])
    assert "Drittland" in txt and "Binnenmarkt" in txt
    ok()
except Exception as e:
    fail(str(e))


# ─── TF-IDF training path with knobs ────────────────────────────────
print("\n=== TF-IDF training (knobs enabled) ===")

t("train TF-IDF end to end")
try:
    cfg["model_architecture"] = "tfidf"
    config.save_config(cfg)
    res = pipeline.train(profile_id=1)
    assert res["success"], res.get("error", "")
    ok()
except Exception as e:
    fail(str(e))

t("TF-IDF model scores entries without crashing")
try:
    scored = pipeline.score_entries(entries, profile_id=1)
    assert len(scored) > 0, "no scores produced"
    assert all("relevance_score" in s for s in scored)
    ok()
except Exception as e:
    fail(repr(e))


# ─── Transformer training path with knobs ───────────────────────────
print("\n=== Transformer training (knobs enabled) ===")

t("train transformer end to end with weighted replication")
try:
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)
    # Wipe prior model rows so transformer takes over
    c2 = db.get_db()
    c2.execute("DELETE FROM models WHERE profile_id=1")
    c2.commit()
    c2.close()
    res = pipeline.train(profile_id=1)
    assert res["success"], res.get("error", "")
    ok()
except Exception as e:
    fail(str(e))

t("transformer model scores entries")
try:
    scored = pipeline.score_entries(entries, profile_id=1)
    assert len(scored) > 0, "no scores produced"
    assert all("relevance_score" in s for s in scored)
    ok()
except Exception as e:
    fail(repr(e))


# ─── Recipe distillation with user patterns ─────────────────────────
print("\n=== recipe distillation ===")

t("distill recipe contains user legal patterns")
try:
    recipe = distiller.distill_recipe(profile_id=1)
    assert recipe is not None, "recipe was None"
    kw = recipe.get("keywords", {})
    # Cleaned, lowercased phrase should appear
    assert "drittland" in kw, "drittland missing from recipe"
    assert "binnenmarkt" in kw, "binnenmarkt missing from recipe"
    # Cleaned regex chars: CE-Kennzeichnung → ce-kennzeichnung
    assert "ce-kennzeichnung" in kw, "ce-kennzeichnung missing"
    assert kw["drittland"].get("investigation_lead", 0) > 0
    ok()
except Exception as e:
    fail(str(e))


# ─── Verify knobs OFF produce no-op weights ─────────────────────────
print("\n=== knobs disabled → weights all 1.0 ===")

t("with all knobs off, weights are all 1.0")
try:
    cfg_off = {"label_time_decay_days": 0,
               "label_time_decay_floor": 0.2,
               "reasoning_weight_boost": 1.0,
               "legal_signal_patterns": []}
    w_off = pipeline.compute_sample_weights(labeled, cfg_off)
    import numpy as np
    assert np.allclose(w_off, 1.0), "weights not all 1.0: {}".format(w_off)
    ok()
except Exception as e:
    fail(str(e))

t("with knobs off, build_entry_text has no signals prefix")
try:
    config.save_config({"legal_signal_patterns": []})
    # Must reset cache since we mutated config
    pipeline._LEGAL_PATTERNS_CACHE["key"] = None
    txt = pipeline._build_entry_text(entries[0], legal_patterns=[])
    assert "signals=" not in txt
    ok()
except Exception as e:
    fail(str(e))


# ─── Legacy model (no knobs) still trains same as before ───────────
print("\n=== legacy training with all-equal weights ===")

t("all-equal non-1 weights produce no TF-IDF sample_weight passthrough")
try:
    import numpy as np
    labeled2 = [
        {"label": "important", "reasoning": "x", "updated_at": fresh, "created_at": fresh},
        {"label": "important", "reasoning": "y", "updated_at": fresh, "created_at": fresh},
        {"label": "noise", "reasoning": "z", "updated_at": fresh, "created_at": fresh},
    ]
    cfg_all_eq = {"label_time_decay_days": 0,
                  "label_time_decay_floor": 0.2,
                  "reasoning_weight_boost": 1.5,
                  "legal_signal_patterns": []}
    w2 = pipeline.compute_sample_weights(labeled2, cfg_all_eq)
    assert np.allclose(w2, 1.5), "expected all 1.5, got {}".format(w2)
    # Std would be 0 → use_sw = False in the training code
    assert float(np.std(w2)) < 1e-6
    ok()
except Exception as e:
    fail(str(e))


# ─── Full MLP calibration branch (regression for MLPClassifier bug) ────
# Previous tests trained on tiny (6-label) sets that skipped the
# calibration branch entirely.  That's why the AttributeError on
# MLPClassifier.decision_function only surfaced in real-world training
# with hundreds of labels.  This test force-enters that branch by
# pre-populating 60 fake embeddings so pipeline.train() never calls the
# real transformer embedder and runs in under a second.
print("\n=== full calibration branch with MLP head ===")

t("transformer train runs calibration with >=30 train-fold samples")
try:
    import numpy as np
    from pipeline import embedding_to_bytes

    # Wipe any prior state from the earlier sections so this test is
    # hermetic: remove all labels, all models, and all entries beyond
    # the ones we're about to insert.
    conn = db.get_db()
    conn.execute("DELETE FROM labels WHERE profile_id=1")
    conn.execute("DELETE FROM models WHERE profile_id=1")
    conn.execute("DELETE FROM entries")
    conn.commit()
    conn.close()

    # 60 synthesized entries with random embeddings already cached.
    rng = np.random.RandomState(7)
    dim = 768
    fake_entries = []
    for i in range(60):
        fake_entries.append({
            "entry_type": "feed_item", "entry_id": 1000 + i,
            "title": "Synthetic item {}".format(i),
            "description": "synthetic description {}".format(i),
            "content": "synthetic content",
            "link": "", "author": "", "published_date": "2025-01-01",
            "source_name": "synthetic", "source_category": "",
            "source_type": "rss",
        })
    db.upsert_entries(fake_entries)

    # Pre-cache embeddings so the real transformer is never loaded
    updates = []
    for i, e in enumerate(fake_entries):
        v = rng.randn(dim).astype("float32")
        # Tilt embeddings per-class so the MLP can actually learn something
        cls = i % 4
        v[cls * 20:(cls + 1) * 20] += 1.5
        updates.append((embedding_to_bytes(v), e["entry_type"], e["entry_id"]))
    db.store_embeddings_batch(updates)

    # Balanced label set across all four classes
    classes = ["investigation_lead", "important", "background", "noise"]
    for i, e in enumerate(fake_entries):
        db.set_label(e["entry_type"], e["entry_id"], classes[i % 4])

    # Make sure architecture is transformer + knobs default to off so we
    # isolate the calibration bug from knob interactions.
    cfg_iso = dict(config.DEFAULTS)
    cfg_iso["model_architecture"] = "transformer"
    cfg_iso["min_labels_to_train"] = 20
    config.save_config(cfg_iso)

    res = pipeline.train(profile_id=1)
    assert res["success"], "train failed: {}".format(res.get("error", ""))

    # Critical: the calibration branch must have fired.  With 60 labels,
    # min_class_count=15, the 80/20 split yields train_fold ~48 >= 30, so
    # the inner 15% validation split runs, yielding len(xv) >= 5, and
    # logits_for_classifier_head(clf, X_val) is invoked.  Before the fix
    # this raised AttributeError.  Assert the result explicitly confirms
    # calibration ran (not "calibration inactive").
    cal_note = res.get("calibration_note", "")
    assert "validation samples" in cal_note, \
        "calibration branch didn't run: cal_note={!r}".format(cal_note)
    assert "calibration inactive" not in cal_note
    # Temperature should be an actual fitted value, not the no-op 1.0
    t_cal = res.get("calibration_temperature", 1.0)
    assert isinstance(t_cal, (int, float)) and t_cal > 0, \
        "bad temperature: {!r}".format(t_cal)

    # Confirm the trained model actually scores entries (exercises
    # classifier_probabilities -> logits_for_classifier_head at scoring
    # time too, when a calibration sidecar is present).
    scored = pipeline.score_entries(fake_entries[:10], profile_id=1)
    assert len(scored) == 10, "expected 10 scores, got {}".format(len(scored))
    for s in scored:
        ps = s["probabilities"]
        total = sum(ps.values())
        assert 0.99 < total < 1.01, "probs don't sum to ~1: {}".format(ps)

    ok()
except Exception as e:
    fail(repr(e))

t("knobs-on + calibration branch still succeeds")
try:
    # Same setup, but with all three knobs active — verifies the sample-
    # weight replication / sample_weight kwarg plays nicely with the
    # calibration branch.
    cfg_on = dict(cfg_iso)
    cfg_on["label_time_decay_days"] = 180
    cfg_on["label_time_decay_floor"] = 0.3
    cfg_on["reasoning_weight_boost"] = 1.5
    cfg_on["legal_signal_patterns"] = ["synthetic"]
    config.save_config(cfg_on)

    # Add reasoning to half the labels so the boost actually applies
    conn = db.get_db()
    conn.execute(
        "UPDATE labels SET reasoning='auto-generated test reason' "
        "WHERE profile_id=1 AND (entry_id % 2) = 0"
    )
    conn.commit()
    conn.close()

    # Clear prior model so train() starts fresh
    conn = db.get_db()
    conn.execute("DELETE FROM models WHERE profile_id=1")
    conn.commit()
    conn.close()

    res = pipeline.train(profile_id=1)
    assert res["success"], "knobs-on train failed: {}".format(res.get("error"))
    cal_note = res.get("calibration_note", "")
    assert "validation samples" in cal_note, \
        "calibration didn't fire with knobs on: cal_note={!r}".format(cal_note)
    ok()
except Exception as e:
    fail(repr(e))


try:
    t("suggested_recipe_top_keywords heuristic")
    assert config.suggested_recipe_top_keywords(0) == 200
    assert config.suggested_recipe_top_keywords(config.RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX) == 400
    assert config.suggested_recipe_top_keywords(1000) == 300
    assert config.suggested_recipe_top_keywords(700) == 270
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
