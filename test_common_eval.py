"""Common-eval: score a stored model on the current labeled holdout."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

import config
import db
import pipeline


CLASSES = pipeline.CLASSES


def _patch_data_dir(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "magnitu.db"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cfg_path = tmp_path / "magnitu_config.json"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(db, "DB_PATH", db_path)
    monkeypatch.setattr(pipeline, "MODELS_DIR", models_dir)


def _save_cfg(**overrides) -> dict:
    cfg = dict(config.DEFAULTS)
    cfg.update(overrides)
    config.save_config(cfg)
    return cfg


def _seed_labeled_entries(n: int = 24) -> None:
    entries = []
    for i in range(1, n + 1):
        cls = CLASSES[(i - 1) % 4]
        entries.append(
            {
                "entry_type": "feed_item",
                "entry_id": i,
                "title": "{} story number {} unique tokens xxx{}".format(cls, i, i),
                "description": "description for {} item {}".format(cls, i),
                "content": "body text {} with extra words for vectorizer {}".format(
                    cls, i
                ),
                "link": "",
                "author": "",
                "published_date": "2024-01-{:02d}".format((i % 28) + 1),
                "source_name": "src{}".format(i % 3),
                "source_category": "news",
                "source_type": "rss",
            }
        )
    db.upsert_entries(entries)
    for i, entry in enumerate(entries, start=1):
        db.set_label(
            entry["entry_type"],
            entry["entry_id"],
            CLASSES[(i - 1) % 4],
        )


def _fit_transformer_head(X: np.ndarray, y):
    le = LabelEncoder()
    le.fit(CLASSES)
    pipe = pipeline.build_transformer_head_pipeline()
    y_enc = le.transform(y)
    pipe.fit(X, y_enc)
    return pipeline._LabelDecodingClassifier(pipe, le)


def test_sidecar_v1_apply_prior_is_noop():
    X, y_idx = make_classification(
        n_samples=80,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    y = [CLASSES[i] for i in y_idx]
    clf = _fit_transformer_head(X, y)
    cal = {
        "method": "temperature",
        "temperature": 2.0,
        "class_names": list(clf.classes_),
    }
    assert "prior_fit" not in cal
    p_on, _ = pipeline.classifier_probabilities(
        clf, X, "", cal=cal, apply_prior=True
    )
    p_off, _ = pipeline.classifier_probabilities(
        clf, X, "", cal=cal, apply_prior=False
    )
    np.testing.assert_allclose(p_on, p_off)


def test_path_parity_transformer_embeddings_and_tfidf_text():
    rng = np.random.RandomState(1)
    X_emb = rng.randn(40, 16)
    y = [CLASSES[i % 4] for i in range(40)]
    clf_t = _fit_transformer_head(X_emb, y)

    m_t = pipeline.evaluate_fitted_model(clf_t, X_emb, y, model_path="")
    assert m_t.get("success") is True
    probs_t, cn_t = pipeline.classifier_probabilities(clf_t, X_emb, "")
    rank_t = pipeline._ranking_metrics(probs_t, cn_t, y)
    assert m_t["precision_at_30"] == round(rank_t["precision_at_30"], 4)
    assert m_t["lead_recall_at_30"] == round(rank_t["lead_recall_at_30"], 4)

    rows = [
        {
            "text": "{} document {}".format(y[i], i),
            "source_type": "rss",
            "text_length": 20,
        }
        for i in range(40)
    ]
    df = pd.DataFrame(rows)
    pipe = pipeline.build_tfidf_pipeline()
    pipe.fit(df, y)
    m_f = pipeline.evaluate_fitted_model(pipe, df, y, model_path="")
    assert m_f.get("success") is True
    probs_f, cn_f = pipeline.classifier_probabilities(pipe, df, "")
    rank_f = pipeline._ranking_metrics(probs_f, cn_f, y)
    assert m_f["precision_at_30"] == round(rank_f["precision_at_30"], 4)


def test_evaluate_fitted_model_roundtrips_joblib(tmp_path):
    X, y_idx = make_classification(
        n_samples=60,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=2,
    )
    y = [CLASSES[i] for i in y_idx]
    clf = _fit_transformer_head(X, y)
    cal = {
        "method": "temperature",
        "temperature": 1.4,
        "class_names": list(clf.classes_),
    }
    path = tmp_path / "head.joblib"
    joblib.dump(clf, path)
    pipeline.write_calibration_sidecar(str(path), cal)
    loaded = joblib.load(path)
    live = pipeline.evaluate_fitted_model(clf, X, y, model_path=str(path), cal=cal)
    disk = pipeline.evaluate_fitted_model(
        loaded, X, y, model_path=str(path)
    )
    assert live.get("success") is True
    assert disk.get("success") is True
    for key in ("precision_at_30", "f1_score", "accuracy", "lead_recall_at_30"):
        assert live[key] == disk[key]


def test_evaluate_stored_model_missing_artifact():
    out = pipeline.evaluate_stored_model(
        {"model_path": "/no/such/model.joblib", "architecture": "tfidf"},
        profile_id=1,
    )
    assert out.get("success") is not True
    assert "missing" in (out.get("error") or "")


def test_evaluate_stored_model_matches_tfidf_train_holdout(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_cfg(min_labels_to_train=8, model_architecture="tfidf")
    db.init_db()
    _seed_labeled_entries(24)

    loads = []
    real_load = pipeline.joblib.load

    def _spy(path):
        loads.append(str(path))
        return real_load(path)

    monkeypatch.setattr(pipeline.joblib, "load", _spy)

    res = pipeline.train(profile_id=1, activate=False)
    assert res.get("success") is True, res.get("error")
    rows = db.get_all_models(1)
    assert rows
    rematch = pipeline.evaluate_stored_model(rows[0], profile_id=1)
    assert rematch.get("success") is True, rematch.get("error")
    assert any(str(rows[0]["model_path"]) in p for p in loads)
    for key in ("precision_at_30", "f1_score", "accuracy", "lead_recall_at_30"):
        assert rematch[key] == pytest.approx(res[key], abs=1e-4)


def test_evaluate_stored_model_matches_transformer_train_holdout(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_cfg(min_labels_to_train=8, model_architecture="transformer", embedding_dim=32)
    db.init_db()
    _seed_labeled_entries(24)
    rng = np.random.RandomState(3)
    updates = []
    for i in range(1, 25):
        vec = rng.randn(32).astype(np.float32)
        updates.append((pipeline.embedding_to_bytes(vec), "feed_item", i))
    db.store_embeddings_batch(updates)

    res = pipeline.train(profile_id=1, activate=False)
    assert res.get("success") is True, res.get("error")
    rows = db.get_all_models(1)
    rematch = pipeline.evaluate_stored_model(rows[0], profile_id=1)
    assert rematch.get("success") is True, rematch.get("error")
    for key in ("precision_at_30", "f1_score", "accuracy", "lead_recall_at_30"):
        assert rematch[key] == pytest.approx(res[key], abs=1e-4)


def _load_replay_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "replay_promote_gate",
        Path(__file__).resolve().parent / "scripts" / "replay_promote_gate.py",
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    return replay


def _restore_db_paths(prev_db, prev_cfg):
    db.DB_PATH = prev_db
    config.DB_PATH = prev_cfg


def test_replay_skips_missing_joblib(tmp_path, capsys):
    replay = _load_replay_module()
    prev_db, prev_cfg = db.DB_PATH, config.DB_PATH
    db_path = tmp_path / "hist.db"
    conn = __import__("sqlite3").connect(str(db_path))
    conn.execute(
        "CREATE TABLE profiles (id INTEGER PRIMARY KEY, slug TEXT, display_name TEXT)"
    )
    conn.execute(
        "CREATE TABLE models ("
        "profile_id INTEGER, version INTEGER, precision_at_30 REAL, "
        "f1_score REAL, lead_recall_at_30 REAL, model_path TEXT, architecture TEXT)"
    )
    conn.execute("INSERT INTO profiles VALUES (1, 'eu', 'EU')")
    conn.execute(
        "INSERT INTO models VALUES (1, 1, 0.2, 0.4, 0.5, '/missing-a.joblib', 'tfidf')"
    )
    conn.execute(
        "INSERT INTO models VALUES (1, 2, 0.3, 0.41, 0.5, '/missing-b.joblib', 'tfidf')"
    )
    conn.commit()
    conn.close()

    try:
        replay.replay_common_eval(db_path)
    finally:
        _restore_db_paths(prev_db, prev_cfg)
    out = capsys.readouterr().out
    assert "old artifact missing" in out
    assert "stored_vs_rematch_diff=0" in out


def test_replay_common_eval_rematches_both_artifacts(tmp_path, capsys, monkeypatch):
    replay = _load_replay_module()
    prev_db, prev_cfg = db.DB_PATH, config.DB_PATH
    old_p = tmp_path / "old.joblib"
    new_p = tmp_path / "new.joblib"
    old_p.write_bytes(b"stub")
    new_p.write_bytes(b"stub")
    db_path = tmp_path / "hist.db"
    conn = __import__("sqlite3").connect(str(db_path))
    conn.execute(
        "CREATE TABLE profiles (id INTEGER PRIMARY KEY, slug TEXT, display_name TEXT)"
    )
    conn.execute(
        "CREATE TABLE models ("
        "profile_id INTEGER, version INTEGER, precision_at_30 REAL, "
        "f1_score REAL, lead_recall_at_30 REAL, model_path TEXT, architecture TEXT)"
    )
    conn.execute("INSERT INTO profiles VALUES (1, 'eu', 'EU')")
    conn.execute(
        "INSERT INTO models VALUES (1, 1, 0.167, 0.415, 0.8, ?, 'tfidf')",
        (str(old_p),),
    )
    conn.execute(
        "INSERT INTO models VALUES (1, 2, 0.200, 0.382, 0.8, ?, 'tfidf')",
        (str(new_p),),
    )
    conn.commit()
    conn.close()

    calls = []

    def _fake_eval(info, profile_id=1, apply_prior=True):
        ver = int(info["version"])
        calls.append(ver)
        if ver == 1:
            return {
                "success": True,
                "precision_at_30": 0.167,
                "f1_score": 0.415,
                "lead_recall_at_30": 0.8,
            }
        return {
            "success": True,
            "precision_at_30": 0.300,
            "f1_score": 0.335,
            "lead_recall_at_30": 0.8,
        }

    monkeypatch.setattr(pipeline, "evaluate_stored_model", _fake_eval)
    try:
        replay.replay_common_eval(db_path)
    finally:
        _restore_db_paths(prev_db, prev_cfg)
    out = capsys.readouterr().out
    assert calls == [1, 2]
    assert "DIFF" in out
    assert "stored_vs_rematch_diff=1" in out
