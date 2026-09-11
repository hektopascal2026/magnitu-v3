"""Regression tests for scoring-plan bugs and UTIL@30 gate."""
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import pipeline
import ml_window


def test_expand_absent_class_gets_neg_inf_logit_not_zero():
    """Bug 1: missing class must not get logit 0.0 (phantom ~25% mass)."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 8)
    y = np.array(["noise"] * 20 + ["important"] * 20)
    le = LabelEncoder()
    le.fit(pipeline.CLASSES)
    pipe = pipeline.build_transformer_head_pipeline()
    pipe.fit(X, le.transform(y))
    clf = pipeline._LabelDecodingClassifier(pipe, le)
    cn = list(clf.classes_)
    assert "investigation_lead" in cn
    logits = clf.decision_function(X)
    assert logits.shape == (40, 4)
    il = cn.index("investigation_lead")
    # Absent class must be driven to ~0 after softmax.
    cal = {"method": "temperature", "temperature": 1.0, "class_names": cn}
    pr, _ = pipeline.classifier_probabilities(clf, X, "", cal=cal)
    assert float(pr[:, il].max()) < 1e-6
    # predict_proba also zero on absent class
    pp = clf.predict_proba(X)
    assert float(pp[:, il].max()) < 1e-12


def test_binary_fold_decision_function_keeps_sample_axis():
    """Bug 2: binary decision_function must not collapse (n,) → (1, n)."""
    rng = np.random.RandomState(1)
    X = rng.randn(40, 8)
    y = np.array(["noise"] * 20 + ["important"] * 20)
    le = LabelEncoder()
    le.fit(pipeline.CLASSES)
    pipe = pipeline.build_transformer_head_pipeline()
    pipe.fit(X, le.transform(y))
    clf = pipeline._LabelDecodingClassifier(pipe, le)
    logits = clf.decision_function(X)
    assert logits.shape == (40, 4)
    cal = {"method": "temperature", "temperature": 1.0, "class_names": list(clf.classes_)}
    pr, cn = pipeline.classifier_probabilities(clf, X, "", cal=cal)
    assert pr.shape == (40, 4)
    # Temperature fit on OOF-shaped binary logits must not raise.
    ol, oy = pipeline._collect_oof_logits(X, list(y), None, le, 3)
    assert len(oy) == 40
    T = pipeline._fit_temperature_scalar(np.asarray(ol), np.array(oy), list(le.classes_))
    assert 0.25 <= T <= 12.0


def test_binary_pipeline_scoring_keeps_one_row_per_sample():
    """A 2-class model over a batch must yield (n_samples, 2), not (1, n).

    ``decision_function`` is 1-D for binary estimators; softmaxing it directly
    collapsed a whole batch into a single bogus row.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(7)
    n = 50  # deliberately > 16, the old ambiguity threshold
    X = rng.randn(n, 6)
    y = np.array(["background"] * 25 + ["noise"] * 25)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, y)

    logits = pipeline.logits_for_classifier_head(pipe, X)
    assert logits.shape == (n, 2)

    cal = {
        "method": "temperature",
        "temperature": 1.5,
        "class_names": list(pipe.classes_),
    }
    probs, cn = pipeline.classifier_probabilities(pipe, X, "", cal=cal)
    assert probs.shape == (n, 2)
    assert cn == list(pipe.classes_)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Softmax of the split margin must agree with the binary sigmoid at T=1.
    cal1 = dict(cal, temperature=1.0)
    p1, _ = pipeline.classifier_probabilities(pipe, X, "", cal=cal1)
    assert np.allclose(p1, pipe.predict_proba(X), atol=1e-8)


def test_softmax_rejects_1d_input():
    with pytest.raises(ValueError):
        pipeline._softmax_rows(np.arange(50, dtype=float))


def test_temperature_fit_refuses_misaligned_class_width():
    """Never guess a column→class alignment; fall back to T=1.0."""
    logits = np.random.RandomState(0).randn(20, 2)
    y = np.array(["noise"] * 20)
    T = pipeline._fit_temperature_scalar(logits, y, ["investigation_lead", "important", "background", "noise"])
    assert T == 1.0


def test_util_at_30_prefers_leads_over_important_padding():
    """UTIL@30 is graded: packing top-30 with important scores lower than leads."""
    cn = pipeline.CLASSES
    # Two items: one lead (high score), one important (slightly lower).
    # Expand to 4 rows so ranking is defined.
    probs = np.array([
        [0.9, 0.05, 0.03, 0.02],   # lead
        [0.1, 0.8, 0.05, 0.05],    # important
        [0.02, 0.03, 0.05, 0.9],   # noise
        [0.02, 0.03, 0.9, 0.05],   # background
    ], dtype=float)
    y_lead_heavy = ["investigation_lead", "noise", "noise", "noise"]
    y_imp_heavy = ["important", "noise", "noise", "noise"]
    # Force ranking by making first row highest composite in both cases via probs.
    r1 = pipeline._ranking_metrics(probs, cn, y_lead_heavy, k=2)
    r2 = pipeline._ranking_metrics(probs, cn, y_imp_heavy, k=2)
    assert r1["util_at_30"] > r2["util_at_30"]


def test_bootstrap_util_rejects_clearly_worse():
    # Large holdout: old ranks all leads first; new ranks them last.
    tw = np.array([1.0] * 10 + [0.0] * 50)
    old_c = np.linspace(1.0, 0.0, 60)
    new_c = np.linspace(0.0, 1.0, 60)
    boot = pipeline.bootstrap_util_delta(old_c, new_c, tw, k=10, n_boot=300)
    assert boot["reject_new_worse"] is True
    assert boot["ci_hi"] < 0.0


def test_bootstrap_util_ties_identical_rankings():
    tw = np.array([1.0, 0.8, 0.2, 0.0] * 5)
    c = np.linspace(1.0, 0.0, len(tw))
    boot = pipeline.bootstrap_util_delta(c, c.copy(), tw, k=5, n_boot=100)
    assert boot["tie"] is True
    assert boot["reject_new_worse"] is False


def test_evaluate_recent_gate_util_reject():
    old = {
        "success": True, "n_recent": 40, "n_leads": 5,
        "util_at_30": 0.70, "precision_at_30": 0.8, "lead_recall_at_30": 0.8,
    }
    new = {
        "success": True, "n_recent": 40, "n_leads": 5,
        "util_at_30": 0.40, "precision_at_30": 0.8, "lead_recall_at_30": 0.8,
    }
    assert ml_window.evaluate_recent_gate(old, new) is False


def test_gate_refuses_blind_promote_when_incumbent_eval_fails():
    """A broken holdout must not silently disable the gate."""
    new = {
        "success": True, "n_recent": 40, "n_leads": 5,
        "util_at_30": 0.9, "precision_at_30": 0.9, "lead_recall_at_30": 0.9,
    }
    failed = {"success": False, "error": "artifact missing"}
    assert ml_window.evaluate_recent_gate(failed, new, has_incumbent=True) is False
    assert ml_window.evaluate_recent_gate(None, new, has_incumbent=True) is False
    # Genuine cold start still promotes.
    assert ml_window.evaluate_recent_gate(None, new, has_incumbent=False) is True


def test_gate_rejects_when_candidate_cannot_be_evaluated():
    old = {
        "success": True, "n_recent": 40, "n_leads": 5, "util_at_30": 0.5,
    }
    assert ml_window.evaluate_recent_gate(
        old, {"success": False, "error": "boom"}, has_incumbent=True
    ) is False


def test_evaluate_recent_gate_util_promote_or_tie():
    old = {
        "success": True, "n_recent": 40, "n_leads": 5,
        "util_at_30": 0.50, "precision_at_30": 0.5, "lead_recall_at_30": 0.5,
    }
    new = {
        "success": True, "n_recent": 40, "n_leads": 5,
        "util_at_30": 0.55, "precision_at_30": 0.5, "lead_recall_at_30": 0.5,
    }
    assert ml_window.evaluate_recent_gate(old, new) is True


def test_gate_4a_twin_models_tie_then_promote():
    """Test 4a: identical rankings → bootstrap tie; gate still promotes."""
    rng = np.random.RandomState(0)
    n = 60
    tw = np.array([1.0] * 8 + [0.8] * 8 + [0.2] * 20 + [0.0] * (n - 36))
    composites = rng.rand(n)
    arm = {
        "success": True,
        "n_recent": n,
        "n_leads": 8,
        "util_at_30": 0.5,
        "precision_at_30": 0.5,
        "lead_recall_at_30": 0.5,
        "_composites": composites,
        "_true_weights": tw,
    }
    twin = dict(arm)
    twin["_composites"] = composites.copy()
    boot = pipeline.bootstrap_util_delta(composites, composites.copy(), tw, k=30)
    assert boot["tie"] is True
    assert boot["reject_new_worse"] is False
    assert ml_window.evaluate_recent_gate(arm, twin, has_incumbent=True) is True


def test_gate_4b_degraded_challenger_rejects():
    """Test 4b: challenger ranks leads last → CI entirely below 0 → reject."""
    tw = np.array([1.0] * 10 + [0.0] * 50)
    old_c = np.linspace(1.0, 0.0, 60)
    new_c = np.linspace(0.0, 1.0, 60)
    old = {
        "success": True,
        "n_recent": 60,
        "n_leads": 10,
        "util_at_30": 0.9,
        "precision_at_30": 0.9,
        "lead_recall_at_30": 0.9,
        "_composites": old_c,
        "_true_weights": tw,
    }
    new = {
        "success": True,
        "n_recent": 60,
        "n_leads": 10,
        "util_at_30": 0.1,
        "precision_at_30": 0.1,
        "lead_recall_at_30": 0.1,
        "_composites": new_c,
        "_true_weights": tw,
    }
    boot = pipeline.bootstrap_util_delta(old_c, new_c, tw, k=10, n_boot=300)
    assert boot["reject_new_worse"] is True
    assert ml_window.evaluate_recent_gate(old, new, has_incumbent=True) is False


def test_bug3_apply_prior_false_ignores_sidecar_prior_fit():
    """Bug 3: with apply_prior=False, prior_fit in the sidecar must not move probs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(3)
    X = rng.randn(60, 6)
    y = np.array(
        ["investigation_lead"] * 15
        + ["important"] * 15
        + ["background"] * 15
        + ["noise"] * 15
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=300, multi_class="multinomial")),
        ]
    )
    pipe.fit(X, y)
    cn = list(pipe.classes_)
    prior_fit = {
        "target_priors": {c: 0.25 for c in cn},
        "empirical_priors": {c: 0.25 for c in cn},
        "prior_log_offsets": {c: (2.0 if c == "investigation_lead" else -2.0) for c in cn},
        "base_rate_composite": 0.5,
    }
    cal_with = {
        "method": "temperature",
        "temperature": 1.5,
        "class_names": cn,
        "prior_fit": prior_fit,
    }
    cal_without = {
        "method": "temperature",
        "temperature": 1.5,
        "class_names": cn,
    }
    p_with, _ = pipeline.classifier_probabilities(
        pipe, X, "", cal=cal_with, apply_prior=False
    )
    p_without, _ = pipeline.classifier_probabilities(
        pipe, X, "", cal=cal_without, apply_prior=False
    )
    assert np.allclose(p_with, p_without, atol=1e-10)
    # Sanity: turning prior on *does* move mass when offsets are large.
    p_on, _ = pipeline.classifier_probabilities(
        pipe, X, "", cal=cal_with, apply_prior=True
    )
    assert not np.allclose(p_on, p_without, atol=1e-3)


def test_bug4_separable_temperature_hits_grid_endpoint():
    """Bug 4: separable logits land on a grid endpoint (callers set clamped)."""
    logits = np.tile(np.array([40.0, -40.0, -40.0, -40.0]), (40, 1))
    y = np.array(["investigation_lead"] * 40)
    T = pipeline._fit_temperature_scalar(logits, y, list(pipeline.CLASSES))
    clamped = float(T) <= 0.25 + 1e-9 or float(T) >= 12.0 - 1e-9
    assert clamped, "expected endpoint T, got {}".format(T)


def _seed_labels(dbmod, n):
    conn = dbmod.get_db()
    for i in range(1, n + 1):
        conn.execute(
            "INSERT OR IGNORE INTO entries (entry_type, entry_id, title) "
            "VALUES ('feed_item', ?, ?)", (i, "t%d" % i),
        )
    conn.commit()
    conn.close()
    for i in range(1, n + 1):
        dbmod.set_label("feed_item", i, "important" if i % 2 else "noise")
    conn = dbmod.get_db()
    conn.execute("UPDATE labels SET created_at = datetime('now','-30 days')")
    conn.commit()
    conn.close()


@pytest.mark.parametrize("total", [25, 60, 110, 400])
def test_eval_reserve_never_starves_training(tmp_path, monkeypatch, total):
    """The withheld reserve must always leave min_labels_to_train rows behind."""
    monkeypatch.setenv("MAGNITU_DATA_DIR", str(tmp_path))
    import importlib
    import config as config_mod
    importlib.reload(config_mod)
    dbmod = importlib.reload(importlib.import_module("db"))
    dbmod.init_db()
    _seed_labels(dbmod, total)

    dbmod.roll_eval_reserve(1, min_train_labels=20)
    reserved = dbmod.get_eval_reserve_keys(1)
    trainable = total - len(reserved)
    assert trainable >= 20, "reserve starved training: {} left".format(trainable)
    assert len(reserved) <= 100

    # Idempotent: a second roll must not grow past the cap.
    dbmod.roll_eval_reserve(1, min_train_labels=20)
    assert len(dbmod.get_eval_reserve_keys(1)) == len(reserved)

    importlib.reload(config_mod)
    importlib.reload(dbmod)


def test_calreport_writes_buckets_ece_and_deciles(tmp_path):
    # Re-export from test_calreport via running the real functions.
    import math, json
    names = ["investigation_lead", "important", "background", "noise"]
    rows = (
        [[2.0, 1.0, -1.0, -2.0]] * 6
        + [[-2.0, -1.0, 0.5, 2.0]] * 6
    )
    out = []
    for r in rows:
        m = max(r)
        e = [math.exp(x - m) for x in r]
        s = sum(e)
        out.append([x / s for x in e])
    probs = np.array(out)
    y = ["investigation_lead"] * 3 + ["important"] * 3 + ["noise"] * 6
    model = tmp_path / "model_pX_vY.joblib"
    model.write_text("x")
    pipeline._write_calibration_report(probs, names, y, str(model))
    path = pipeline.calibration_report_path(str(model))
    assert path.exists()
    rep = json.loads(path.read_text())
    assert rep["holdout_n"] == 12
    assert rep["ece"] < 0.10
    assert len(rep["composite_deciles"]) == 5
