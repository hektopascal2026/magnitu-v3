"""
Integration tests for Magnitu 2.

Tests the critical paths:
1. Config: transformer settings load correctly
2. DB: schema migration, reasoning, embedding storage
3. Pipeline (TF-IDF): train, score, feature importance
4. Pipeline (Transformer): embed, train, score, knowledge distillation
5. Distiller: recipe from TF-IDF, recipe from transformer via distillation
6. Explainer: architecture-aware explanations
7. Model manager: export/import with architecture metadata
8. Embedding invalidation on model change
9. Reasoning boost in recipe
10. FastAPI endpoints: label with reasoning, train, stats
"""
import os
import sys
import json
import shutil
import tempfile
import numpy as np

# ── Setup: use a temp DB so we don't pollute the real one ──
_test_dir = tempfile.mkdtemp(prefix="magnitu_test_")
os.environ["MAGNITU_TEST"] = "1"

from pathlib import Path as _P

import config
config.DB_PATH = _P(_test_dir) / "test.db"
config.MODELS_DIR = _P(_test_dir) / "models"
config.MODELS_DIR.mkdir(exist_ok=True)
config.CONFIG_PATH = _P(_test_dir) / "test_config.json"

# Save a test config
test_config = dict(config.DEFAULTS)
test_config["min_labels_to_train"] = 4
config.save_config(test_config)

import db
db.DB_PATH = config.DB_PATH

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator-ish context for tests."""
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


# ═══════════════════════════════════════════
#  1. Config
# ═══════════════════════════════════════════
print("\n=== 1. Config ===")

t = test("Version is 2.x")
try:
    assert config.VERSION.startswith("2."), "Expected version 2.x, got " + config.VERSION
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer defaults present")
try:
    cfg = config.get_config()
    assert "model_architecture" in cfg
    assert "transformer_model_name" in cfg
    assert "embedding_dim" in cfg
    assert cfg["model_architecture"] == "transformer"
    assert cfg["embedding_dim"] == 768
    assert "discovery_lead_blend" in cfg
    assert cfg.get("discovery_lead_blend", 0) == 0.0
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  2. Database schema
# ═══════════════════════════════════════════
print("\n=== 2. Database Schema ===")

db.init_db()

t = test("Entries table has embedding column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(entries)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "embedding" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("Labels table has reasoning column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(labels)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "reasoning" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("Models table has architecture column")
try:
    conn = db.get_db()
    cursor = conn.execute("PRAGMA table_info(models)")
    cols = {row[1] for row in cursor.fetchall()}
    conn.close()
    assert "architecture" in cols
    ok()
except Exception as e:
    fail(str(e))

t = test("set_label with reasoning")
try:
    # Need an entry first
    db.upsert_entry({
        "entry_type": "feed_item", "entry_id": 100,
        "title": "Test", "description": "", "content": "",
        "link": "", "author": "", "published_date": "",
        "source_name": "", "source_category": "", "source_type": "rss",
    })
    db.set_label("feed_item", 100, "investigation_lead", reasoning="test reason")
    result = db.get_label_with_reasoning("feed_item", 100)
    assert result is not None
    assert result["label"] == "investigation_lead"
    assert result["reasoning"] == "test reason"
    db.remove_label("feed_item", 100)
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding store and retrieve")
try:
    fake_emb = np.random.randn(768).astype(np.float32)
    fake_bytes = fake_emb.tobytes()
    db.store_embedding("feed_item", 100, fake_bytes)
    assert db.get_embedding_count() >= 1
    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=100"
    ).fetchone()
    conn.close()
    retrieved = np.frombuffer(row["embedding"], dtype=np.float32)
    assert np.allclose(fake_emb, retrieved)
    ok()
except Exception as e:
    fail(str(e))

t = test("invalidate_all_embeddings")
try:
    db.invalidate_all_embeddings()
    assert db.get_embedding_count() == 0
    ok()
except Exception as e:
    fail(str(e))

t = test("get_all_reasoning_texts")
try:
    db.set_label("feed_item", 100, "investigation_lead", reasoning="contracts fraud")
    texts = db.get_all_reasoning_texts()
    assert len(texts) >= 1
    assert any(t["reasoning"] == "contracts fraud" for t in texts)
    db.remove_label("feed_item", 100)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  3. Pipeline - Setup test data
# ═══════════════════════════════════════════
print("\n=== 3. Pipeline Setup ===")

test_entries = [
    {"entry_type": "feed_item", "entry_id": 1, "title": "Investigation reveals corruption in government contracts",
     "description": "Major scandal uncovered", "content": "Deep investigation into public procurement fraud and bribery",
     "link": "", "author": "", "published_date": "2024-01-01",
     "source_name": "Investigative News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 2, "title": "New policy on data protection announced",
     "description": "Important regulation change", "content": "Government announces stricter rules for data handling",
     "link": "", "author": "", "published_date": "2024-01-02",
     "source_name": "Policy Daily", "source_category": "policy", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 3, "title": "Quarterly earnings report released",
     "description": "Company results", "content": "The company reported strong earnings above analyst expectations",
     "link": "", "author": "", "published_date": "2024-01-03",
     "source_name": "Business Wire", "source_category": "business", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 4, "title": "Weekend weather forecast",
     "description": "Rain expected", "content": "Forecast shows rain and mild temperatures for the coming weekend",
     "link": "", "author": "", "published_date": "2024-01-04",
     "source_name": "Weather Channel", "source_category": "weather", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 5, "title": "Secret documents reveal systematic cover-up",
     "description": "Leaked classified files", "content": "Documents obtained show years of systematic fraud in defense contracts",
     "link": "", "author": "", "published_date": "2024-01-05",
     "source_name": "Investigative News", "source_category": "politics", "source_type": "rss"},
    {"entry_type": "feed_item", "entry_id": 6, "title": "Local sports team wins championship",
     "description": "Match recap and highlights", "content": "The team won in overtime with a spectacular goal",
     "link": "", "author": "", "published_date": "2024-01-06",
     "source_name": "Sports Daily", "source_category": "sports", "source_type": "rss"},
]

t = test("Insert test entries")
try:
    db.upsert_entries(test_entries)
    assert db.get_entry_count() >= 6
    ok()
except Exception as e:
    fail(str(e))

t = test("Label test entries with reasoning")
try:
    db.set_label("feed_item", 1, "investigation_lead", reasoning="corruption in public contracts")
    db.set_label("feed_item", 2, "important")
    db.set_label("feed_item", 3, "background")
    db.set_label("feed_item", 4, "noise")
    assert db.get_label_count() >= 4
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  4. TF-IDF Pipeline
# ═══════════════════════════════════════════
print("\n=== 4. TF-IDF Pipeline ===")

import pipeline

t = test("TF-IDF train")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "tfidf"
    config.save_config(cfg)

    result = pipeline.train()
    assert result["success"], result.get("error", "")
    assert result["architecture"] == "tfidf"
    assert result["version"] >= 1
    assert result["accuracy"] >= 0
    assert pipeline.calibration_sidecar_path(result["model_path"]).exists(), \
        "TF-IDF training should write a calibration sidecar JSON"
    ok()
except Exception as e:
    fail(str(e))

t = test("TF-IDF score")
try:
    scores = pipeline.score_entries(test_entries)
    assert len(scores) == len(test_entries)
    for s in scores:
        assert "relevance_score" in s
        assert "predicted_label" in s
        assert "probabilities" in s
        assert 0 <= s["relevance_score"] <= 1
    ok()
except Exception as e:
    fail(str(e))

t = test("TF-IDF feature importance")
try:
    importance = pipeline.get_feature_importance()
    assert len(importance) > 0
    for cls, pairs in importance.items():
        assert len(pairs) > 0
        assert all(len(p) == 2 for p in pairs)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  5. Transformer Pipeline
# ═══════════════════════════════════════════
print("\n=== 5. Transformer Pipeline ===")

t = test("Embedding byte roundtrip")
try:
    emb = np.random.randn(768).astype(np.float32)
    b = pipeline.embedding_to_bytes(emb)
    emb2 = pipeline.bytes_to_embedding(b, 768)
    assert np.allclose(emb, emb2), "Roundtrip mismatch"
    ok()
except Exception as e:
    fail(str(e))

t = test("Compute embeddings")
try:
    texts = ["Investigation reveals corruption", "Weather forecast for weekend"]
    embeddings = pipeline.compute_embeddings(texts)
    assert embeddings.shape == (2, 768), "Expected (2, 768), got {}".format(embeddings.shape)
    assert not np.allclose(embeddings[0], embeddings[1]), "Different texts should have different embeddings"
    ok()
except Exception as e:
    fail(str(e))

t = test("embed_entries helper")
try:
    emb_bytes = pipeline.embed_entries(test_entries[:2])
    assert len(emb_bytes) == 2
    assert all(len(b) == 768 * 4 for b in emb_bytes)
    ok()
except Exception as e:
    fail(str(e))

t = test("_build_entry_text includes structured source prefix")
try:
    txt = pipeline._build_entry_text({
        "title": "Hello",
        "description": "",
        "content": "World",
        "source_type": "lex_ch",
        "source_name": "SRG",
        "source_category": "News",
    })
    assert "source_type=lex_ch" in txt
    assert "source=SRG" in txt
    assert "category=News" in txt
    assert "Hello" in txt
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer train")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)

    # Clear old models so this gets a clean version
    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"], result.get("error", "")
    assert result["architecture"] == "transformer"
    assert result["feature_count"] == 768
    assert "calibration_temperature" in result
    assert pipeline.calibration_sidecar_path(result["model_path"]).exists(), \
        "Transformer training should write a calibration sidecar JSON"
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer score")
try:
    scores = pipeline.score_entries(test_entries)
    assert len(scores) == len(test_entries)
    for s in scores:
        assert "relevance_score" in s
        assert "predicted_label" in s
        assert 0 <= s["relevance_score"] <= 1

    # The investigation entries should score higher than sports/weather
    score_map = {s["entry_id"]: s["relevance_score"] for s in scores}
    # Entry 1 (investigation) should beat entry 4 (weather)
    assert score_map[1] > score_map[4], \
        "Investigation ({:.2f}) should score > weather ({:.2f})".format(score_map[1], score_map[4])
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer feature importance returns empty (expected)")
try:
    importance = pipeline.get_feature_importance()
    assert importance == {}, "Transformer models should not have direct feature importance"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embeddings cached in DB after scoring")
try:
    count = db.get_embedding_count()
    assert count >= len(test_entries), \
        "Expected >= {} embeddings, got {}".format(len(test_entries), count)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  6. Distiller
# ═══════════════════════════════════════════
print("\n=== 6. Distiller ===")

import distiller

t = test("TF-IDF recipe distillation")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "tfidf"
    config.save_config(cfg)

    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"]

    recipe = distiller.distill_recipe()
    assert recipe is not None, "Recipe should not be None for TF-IDF"
    assert "keywords" in recipe
    assert "classes" in recipe
    assert recipe["classes"] == ["investigation_lead", "important", "background", "noise"]
    assert "class_weights" in recipe
    assert recipe["class_weights"] == [1.0, 0.66, 0.33, 0.0]
    ok()
except Exception as e:
    fail(str(e))

t = test("Reasoning boost in recipe")
try:
    db.set_label("feed_item", 1, "investigation_lead", reasoning="procurement fraud bribery")
    recipe = distiller.distill_recipe()
    assert recipe is not None
    # Check that reasoning keywords got incorporated
    kw = recipe.get("keywords", {})
    has_reasoning_term = any(
        term in kw for term in ["procurement", "fraud", "bribery"]
    )
    assert has_reasoning_term, \
        "Reasoning terms should appear in recipe keywords. Got: {}".format(
            [k for k in list(kw.keys())[:20]]
        )
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  7. Explainer
# ═══════════════════════════════════════════
print("\n=== 7. Explainer ===")

import explainer

t = test("TF-IDF explanation")
try:
    # Should have a TF-IDF model active from test 6
    exp = explainer.explain_entry(test_entries[0])
    assert exp is not None
    assert "prediction" in exp
    assert "confidence" in exp
    assert "probabilities" in exp
    assert "top_features" in exp
    assert 0 <= exp["confidence"] <= 1
    ok()
except Exception as e:
    fail(str(e))

t = test("Transformer explanation")
try:
    cfg = config.get_config()
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)

    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"]

    exp = explainer.explain_entry(test_entries[0])
    assert exp is not None
    assert "prediction" in exp
    assert "confidence" in exp
    assert "probabilities" in exp
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  8. Model Manager
# ═══════════════════════════════════════════
print("\n=== 8. Model Manager ===")

import model_manager

t = test("Create profile")
try:
    if not model_manager.has_profile():
        model_manager.create_profile("test-model", "Test model for integration tests")
    profile = model_manager.get_profile()
    assert profile is not None
    assert profile["model_name"] == "test-model"
    ok()
except Exception as e:
    fail(str(e))

t = test("Export contains architecture in manifest")
try:
    export_path = model_manager.export_model()
    assert os.path.exists(export_path)

    import zipfile
    with zipfile.ZipFile(export_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "architecture" in manifest
    assert "magnitu_version" in manifest
    assert manifest["magnitu_version"].startswith("2.")
    ok()
except Exception as e:
    fail(str(e))

t = test("Export contains reasoning in labels")
try:
    with zipfile.ZipFile(export_path) as zf:
        labels = json.loads(zf.read("labels.json"))
    has_reasoning = any(lbl.get("reasoning") for lbl in labels)
    assert has_reasoning, "Exported labels should include reasoning text"
    ok()
except Exception as e:
    fail(str(e))

t = test("Export contains calibration.json when model trained")
try:
    with zipfile.ZipFile(export_path) as zf:
        names = zf.namelist()
    assert "calibration.json" in names, "Expected calibration.json in .magnitu export"
    with zipfile.ZipFile(export_path) as zf:
        cal = json.loads(zf.read("calibration.json"))
    assert cal.get("method") == "temperature"
    assert "temperature" in cal
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  9. Embedding Invalidation
# ═══════════════════════════════════════════
print("\n=== 9. Embedding Invalidation ===")

t = test("Invalidate clears all embeddings")
try:
    before = db.get_embedding_count()
    assert before > 0, "Should have embeddings before invalidation"
    db.invalidate_all_embeddings()
    after = db.get_embedding_count()
    assert after == 0, "Should have 0 embeddings after invalidation, got {}".format(after)
    ok()
except Exception as e:
    fail(str(e))

t = test("invalidate_embedder_cache resets singleton")
try:
    pipeline.invalidate_embedder_cache()
    assert pipeline._embedder is None
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  10. Stale Embedding Invalidation on Content Update
# ═══════════════════════════════════════════
print("\n=== 10. Stale Embedding on Content Update ===")

t = test("Embedding preserved when content unchanged")
try:
    entry = {
        "entry_type": "feed_item", "entry_id": 900,
        "title": "Original title", "description": "Original desc", "content": "Original content",
        "link": "", "author": "", "published_date": "2024-06-01",
        "source_name": "Test", "source_category": "", "source_type": "rss",
    }
    db.upsert_entry(entry)
    fake_emb = np.random.randn(768).astype(np.float32)
    db.store_embedding("feed_item", 900, fake_emb.tobytes())

    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=900"
    ).fetchone()
    conn.close()
    assert row["embedding"] is not None, "Embedding should exist before re-upsert"

    # Re-upsert with identical content
    db.upsert_entry(entry)

    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=900"
    ).fetchone()
    conn.close()
    assert row["embedding"] is not None, "Embedding should be preserved when content unchanged"
    retrieved = np.frombuffer(row["embedding"], dtype=np.float32)
    assert np.allclose(fake_emb, retrieved), "Embedding bytes should be identical"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding invalidated when title changes")
try:
    updated = dict(entry)
    updated["title"] = "Changed title"
    db.upsert_entry(updated)

    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=900"
    ).fetchone()
    conn.close()
    assert row["embedding"] is None, "Embedding should be NULL after title change"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding invalidated when source metadata changes")
try:
    e = {
        "entry_type": "feed_item",
        "entry_id": 902,
        "title": "Stable title",
        "description": "",
        "content": "Body",
        "link": "",
        "author": "",
        "published_date": "2024-06-01",
        "source_name": "OldSource",
        "source_category": "cat1",
        "source_type": "rss",
    }
    db.upsert_entry(e)
    db.store_embedding("feed_item", 902, np.random.randn(768).astype(np.float32).tobytes())
    e2 = dict(e)
    e2["source_name"] = "NewSource"
    db.upsert_entry(e2)
    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=902"
    ).fetchone()
    conn.close()
    assert row["embedding"] is None, "Embedding should clear when source_name changes"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding invalidated when content changes (batch upsert)")
try:
    entry2 = {
        "entry_type": "feed_item", "entry_id": 901,
        "title": "Batch test", "description": "Desc", "content": "Short content",
        "link": "", "author": "", "published_date": "2024-06-01",
        "source_name": "Test", "source_category": "", "source_type": "rss",
    }
    db.upsert_entries([entry2])
    db.store_embedding("feed_item", 901, np.random.randn(768).astype(np.float32).tobytes())

    # Re-upsert with changed content
    entry2_updated = dict(entry2)
    entry2_updated["content"] = "Much longer updated content with more details about the topic"
    db.upsert_entries([entry2_updated])

    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=901"
    ).fetchone()
    conn.close()
    assert row["embedding"] is None, "Embedding should be NULL after content change in batch upsert"
    ok()
except Exception as e:
    fail(str(e))

t = test("Embedding preserved in batch when content unchanged")
try:
    entry3 = {
        "entry_type": "feed_item", "entry_id": 902,
        "title": "Stable entry", "description": "Same", "content": "Same content",
        "link": "", "author": "", "published_date": "2024-06-01",
        "source_name": "Test", "source_category": "", "source_type": "rss",
    }
    db.upsert_entries([entry3])
    emb_bytes = np.random.randn(768).astype(np.float32).tobytes()
    db.store_embedding("feed_item", 902, emb_bytes)

    # Re-upsert with same content
    db.upsert_entries([entry3])

    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type='feed_item' AND entry_id=902"
    ).fetchone()
    conn.close()
    assert row["embedding"] is not None, "Embedding should be preserved when batch content unchanged"
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  11. Recipe Text-Length Normalization
# ═══════════════════════════════════════════
print("\n=== 11. Recipe Text-Length Normalization ===")

t = test("Recipe normalization reduces score variance across text lengths")
try:
    # Create entries with wildly different text lengths but similar topics
    short_entry = {
        "entry_type": "feed_item", "entry_id": 800,
        "title": "Toxic baby food recall", "description": "Contamination found",
        "content": "",
        "link": "", "author": "", "published_date": "2024-06-01",
        "source_name": "News A", "source_category": "health", "source_type": "rss",
    }
    long_entry = {
        "entry_type": "feed_item", "entry_id": 801,
        "title": "Toxic baby food recall investigation",
        "description": "Major contamination found in baby food products",
        "content": (
            "The government agency announced today that baby food products from "
            "multiple manufacturers have been found contaminated with toxic substances. "
            "The investigation reveals systematic quality control failures in the production "
            "chain. Officials have issued an immediate recall of all affected products. "
            "Consumers are urged to check their pantries and return any contaminated items. "
            "The toxic contamination was first detected during routine testing at a federal "
            "laboratory. Preliminary results show that the contamination levels exceed safe "
            "thresholds by a significant margin. Health authorities are monitoring the "
            "situation closely and have set up a hotline for concerned parents."
        ),
        "link": "", "author": "", "published_date": "2024-06-01",
        "source_name": "News B", "source_category": "health", "source_type": "rss",
    }
    db.upsert_entries([short_entry, long_entry])

    # Label enough for training
    db.set_label("feed_item", 5, "investigation_lead")
    db.set_label("feed_item", 6, "noise")

    # Ensure we have a transformer model active
    cfg = config.get_config()
    cfg["model_architecture"] = "transformer"
    config.save_config(cfg)

    conn = db.get_db()
    conn.execute("DELETE FROM models")
    conn.commit()
    conn.close()

    result = pipeline.train()
    assert result["success"], result.get("error", "")

    recipe = distiller.distill_recipe()
    if recipe:
        # Evaluate recipe scores for both entries
        kw = recipe.get("keywords", {})
        sw = recipe.get("source_weights", {})
        classes = recipe["classes"]
        class_wts = recipe["class_weights"]

        def recipe_score(entry_dict):
            text = "{} {} {}".format(
                entry_dict.get("title", ""),
                entry_dict.get("description", ""),
                entry_dict.get("content", ""),
            ).lower()
            tokens = text.split()
            bigrams = ["{} {}".format(tokens[j], tokens[j + 1])
                       for j in range(len(tokens) - 1)]
            all_tokens = tokens + bigrams
            cs = {c: 0.0 for c in classes}
            for tok in all_tokens:
                if tok in kw:
                    for cls2, wt2 in kw[tok].items():
                        if cls2 in cs:
                            cs[cls2] += wt2
            src = entry_dict.get("source_type", "")
            if src in sw:
                for cls2, wt2 in sw[src].items():
                    if cls2 in cs:
                        cs[cls2] += wt2
            max_s = max(cs.values()) if cs else 0
            es = {c: np.exp(s - max_s) for c, s in cs.items()}
            es_sum = sum(es.values())
            probs = {c: es[c] / es_sum if es_sum > 0 else 0.25 for c in classes}
            return sum(probs.get(c, 0) * class_wts[i] for i, c in enumerate(classes))

        short_score = recipe_score(short_entry)
        long_score = recipe_score(long_entry)
        diff = abs(short_score - long_score)

        # After normalization, the difference between same-topic entries of
        # different lengths should be < 0.8 (before normalization it could
        # be 0.95+ i.e. 5 vs 100 on seismo's display)
        assert diff < 0.8, \
            "Same-topic entries should have score difference < 0.8, got {:.3f} " \
            "(short={:.3f}, long={:.3f})".format(diff, short_score, long_score)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  11a. Regression: MLPClassifier has no decision_function
# ═══════════════════════════════════════════
print("\n=== 11a. Calibration logits: MLP has no decision_function ===")

t = test("logits_for_classifier_head works on MLP pipeline (no decision_function)")
try:
    from sklearn.pipeline import Pipeline as _Pipe
    from sklearn.preprocessing import StandardScaler as _SS, LabelEncoder as _LE
    from sklearn.neural_network import MLPClassifier as _MLP
    from pipeline import (
        logits_for_classifier_head, _LabelDecodingClassifier,
        _softmax_rows, CLASSES,
    )
    import numpy as _np

    _rng = _np.random.RandomState(0)
    _X = _rng.randn(60, 16)
    _y = _rng.choice(CLASSES, size=60)
    _le = _LE(); _le.fit(CLASSES)
    _y_enc = _le.transform(_y)

    _inner = _Pipe([
        ("scaler", _SS()),
        ("mlp", _MLP(hidden_layer_sizes=(8,), max_iter=80, random_state=0)),
    ])
    _inner.fit(_X, _y_enc)
    _clf = _LabelDecodingClassifier(_inner, _le)

    # Previously raised AttributeError: 'MLPClassifier' has no decision_function.
    _logits = logits_for_classifier_head(_clf, _X[:10])
    assert _logits.shape == (10, 4)
    assert _np.all(_np.isfinite(_logits))

    # Softmax of log(probs) must round-trip to predict_proba (temperature = 1).
    _probs_recon = _softmax_rows(_logits / 1.0)
    _probs_direct = _clf.predict_proba(_X[:10])
    assert _np.max(_np.abs(_probs_recon - _probs_direct)) < 1e-10
    ok()
except Exception as e:
    fail(repr(e))


# ═══════════════════════════════════════════
#  11b. Advanced training knobs (decay, reasoning boost, legal signals)
# ═══════════════════════════════════════════
print("\n=== 11b. Advanced training knobs ===")

t = test("Config defaults include new training knobs")
try:
    cfg = config.get_config()
    for k in ("label_time_decay_days", "label_time_decay_floor",
              "reasoning_weight_boost", "legal_signal_patterns"):
        assert k in cfg, "missing default: " + k
    assert cfg["label_time_decay_days"] == 0
    assert cfg["reasoning_weight_boost"] == 1.0
    assert cfg["legal_signal_patterns"] == []
    ok()
except Exception as e:
    fail(str(e))

t = test("compute_sample_weights defaults to ones")
try:
    from datetime import datetime, timedelta
    now = datetime.now()
    labeled = [
        {"label": "important", "reasoning": "", "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"), "created_at": None},
        {"label": "noise", "reasoning": "because spam", "updated_at": None, "created_at": None},
    ]
    w = pipeline.compute_sample_weights(labeled, config={})
    assert len(w) == 2
    assert abs(w[0] - 1.0) < 1e-6 and abs(w[1] - 1.0) < 1e-6
    ok()
except Exception as e:
    fail(str(e))

t = test("compute_sample_weights applies time decay and floor")
try:
    from datetime import datetime, timedelta
    now = datetime.now()
    old_ts = (now - timedelta(days=240)).strftime("%Y-%m-%d %H:%M:%S")
    fresh_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    labeled = [
        {"label": "important", "reasoning": "", "updated_at": fresh_ts, "created_at": None},
        {"label": "important", "reasoning": "", "updated_at": old_ts, "created_at": None},
    ]
    cfg = {"label_time_decay_days": 120, "label_time_decay_floor": 0.25,
           "reasoning_weight_boost": 1.0}
    w = pipeline.compute_sample_weights(labeled, config=cfg)
    assert abs(w[0] - 1.0) < 1e-3, "fresh label should be ~1.0, got {}".format(w[0])
    # 240/120 = 2 half-lives → 0.25 which is right at the floor
    assert 0.24 <= w[1] <= 0.26, "old label should be near floor 0.25, got {}".format(w[1])
    ok()
except Exception as e:
    fail(str(e))

t = test("compute_sample_weights applies reasoning boost")
try:
    labeled = [
        {"label": "important", "reasoning": "", "updated_at": None, "created_at": None},
        {"label": "important", "reasoning": "clear investigative angle", "updated_at": None, "created_at": None},
    ]
    cfg = {"label_time_decay_days": 0, "reasoning_weight_boost": 1.5}
    w = pipeline.compute_sample_weights(labeled, config=cfg)
    assert abs(w[0] - 1.0) < 1e-6
    assert abs(w[1] - 1.5) < 1e-6
    ok()
except Exception as e:
    fail(str(e))

t = test("_build_entry_text injects legal-signal prefix on match")
try:
    entry = {
        "title": "Drittland-Klausel sorgt für Diskussionen",
        "description": "Die Binnenmarkt-Regelung ändert sich.",
        "content": "",
        "source_type": "rss", "source_name": "Test", "source_category": "",
    }
    patterns = ["Drittland", "Binnenmarkt", "CE-Kennzeichnung"]
    text = pipeline._build_entry_text(entry, legal_patterns=patterns)
    assert "signals=" in text, "prefix missing: {}".format(text[:120])
    assert "Drittland" in text and "Binnenmarkt" in text
    # A non-matching phrase should not appear in the signals tag
    # (title still contains other words, but the signals= chunk is bounded)
    sig_line = [ln for ln in text.split("\n") if "signals=" in ln][0]
    assert "CE-Kennzeichnung" not in sig_line
    ok()
except Exception as e:
    fail(str(e))

t = test("_build_entry_text is unchanged when no patterns configured")
try:
    entry = {
        "title": "Some random news",
        "description": "Nothing legal here.",
        "content": "",
        "source_type": "rss", "source_name": "Test", "source_category": "",
    }
    text_no_pat = pipeline._build_entry_text(entry, legal_patterns=[])
    assert "signals=" not in text_no_pat
    ok()
except Exception as e:
    fail(str(e))

t = test("distiller._boost_legal_templates injects user patterns")
try:
    old_patterns = config.get_config().get("legal_signal_patterns", [])
    cfg = config.get_config()
    cfg["legal_signal_patterns"] = ["Drittland", "Ursprungserzeugnis"]
    config.save_config(cfg)
    from distiller import _boost_legal_templates
    kw = _boost_legal_templates({})
    assert "drittland" in kw, "user pattern missing from recipe keywords"
    assert kw["drittland"].get("investigation_lead", 0) >= 0.35
    assert "ursprungserzeugnis" in kw
    cfg["legal_signal_patterns"] = old_patterns
    config.save_config(cfg)
    ok()
except Exception as e:
    fail(str(e))


# ═══════════════════════════════════════════
#  12. FastAPI App
# ═══════════════════════════════════════════
print("\n=== 12. FastAPI App ===")

t = test("App creates with correct version")
try:
    from main import app
    assert app.version == config.VERSION
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
