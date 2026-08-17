"""Property tests for score v2 prior correction (engineering notes §5 T1–T6)."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pipeline


CLASSES = pipeline.CLASSES


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _example_offsets():
    pi_hat = np.array([0.25, 0.25, 0.25, 0.25])
    pi = np.array([0.04, 0.16, 0.45, 0.35])
    return np.log(pi) - np.log(pi_hat), pi, pi_hat


def test_t2_likelihood_ratio_invariant():
    rng = np.random.RandomState(0)
    for _ in range(20):
        raw = rng.dirichlet(np.ones(4))
        p = np.clip(raw, 1e-9, 1.0)
        p = p / p.sum()
        pi = rng.dirichlet(np.ones(4) * 2)
        pi_hat = rng.dirichlet(np.ones(4) * 2)
        z = np.log(p)
        p_prime = _softmax(z + np.log(pi) - np.log(pi_hat))[0]
        ratio = (p_prime / pi) / (p / pi_hat)
        assert np.allclose(ratio, ratio[0], atol=1e-9)


def test_t3_correction_to_self_is_identity():
    y = np.array(CLASSES * 12)
    pf = pipeline.build_prior_fit(y, None, CLASSES)
    for c in CLASSES:
        assert abs(pf["prior_log_offsets"][c]) < 1e-12


def test_t6_worked_example_a_balanced_mirage():
    offsets, pi, pi_hat = _example_offsets()
    p = np.array([0.45, 0.30, 0.15, 0.10])
    p_corr = _softmax(np.log(p) + offsets)[0]
    expected = np.array([0.107, 0.285, 0.401, 0.208])
    assert np.allclose(p_corr, expected, atol=5e-4)
    w = np.array([1.0, 0.8, 0.2, 0.0])
    assert abs(float(p.dot(w)) - 0.72) < 1e-12
    assert abs(float(p_corr.dot(w)) - 0.41) < 0.005


def test_t6_worked_example_b_anchor_compression():
    offsets, _, _ = _example_offsets()
    p = np.array([0.02, 0.96, 0.01, 0.01])
    p_corr = _softmax(np.log(p) + offsets)[0]
    expected = np.array([0.005, 0.946, 0.028, 0.022])
    assert np.allclose(p_corr, expected, atol=5e-3)
    w = np.array([1.0, 0.8, 0.2, 0.0])
    assert abs(float(p.dot(w)) - 0.79) < 0.005
    assert abs(float(p_corr.dot(w)) - 0.77) < 0.01


def test_t6_worked_example_c_cant_tell_base_rate():
    offsets, pi, _ = _example_offsets()
    p = np.array([0.25, 0.25, 0.25, 0.25])
    p_corr = _softmax(np.log(p) + offsets)[0]
    assert np.allclose(p_corr, pi, atol=1e-9)
    w = np.array([1.0, 0.8, 0.2, 0.0])
    assert abs(float(p.dot(w)) - 0.50) < 1e-12
    assert abs(float(pi.dot(w)) - 0.258) < 0.005


def test_absent_class_floor_finite_offsets():
    y = np.array(
        ["investigation_lead"] * 10 + ["important"] * 10 + ["background"] * 10
    )
    pf = pipeline.build_prior_fit(y, None, CLASSES)
    assert pf["target_priors"]["noise"] > 0.0
    assert np.isfinite(pf["prior_log_offsets"]["noise"])


def test_t4_sidecar_v1_matches_temperature_only():
    X, y_idx = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=1,
    )
    names = np.array(CLASSES)
    y = names[y_idx]
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X, y)
    class_names = clf.classes_.tolist()
    cal_v1 = {
        "version": 1,
        "method": "temperature",
        "temperature": 1.37,
        "class_names": class_names,
    }
    probs, cn = pipeline.classifier_probabilities(clf, X, "", cal=cal_v1)
    logits = pipeline.logits_for_classifier_head(clf, X)
    manual = pipeline._softmax_rows(np.asarray(logits, dtype=np.float64) / 1.37)
    assert cn == class_names
    assert np.allclose(probs, manual, atol=1e-12)


def test_t5_ranking_metrics_follow_corrected_path():
    X, y_idx = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=2,
    )
    names = np.array(CLASSES)
    y = names[y_idx]
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X, y)
    class_names = clf.classes_.tolist()
    cal_v1 = {
        "version": 1,
        "method": "temperature",
        "temperature": 1.0,
        "class_names": class_names,
    }
    extreme = {c: 8.0 if c == "investigation_lead" else -8.0 for c in class_names}
    cal_v2 = dict(cal_v1)
    cal_v2["version"] = 2
    cal_v2["prior_fit"] = {"prior_log_offsets": extreme}
    p1, cn = pipeline.classifier_probabilities(clf, X, "", cal=cal_v1)
    p2, _ = pipeline.classifier_probabilities(clf, X, "", cal=cal_v2)
    r1 = pipeline._ranking_metrics(p1, cn, y)
    r2 = pipeline._ranking_metrics(p2, cn, y)
    assert not np.allclose(p1, p2, atol=1e-3)
    assert r1["precision_at_30"] != r2["precision_at_30"] or r1["lead_recall_at_30"] != r2["lead_recall_at_30"]


def test_t1_balanced_fit_correction_recovers_unweighted():
    X, y = make_blobs(
        n_samples=600, centers=3, n_features=2, cluster_std=0.45, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    clf_a = LogisticRegression(
        class_weight=None, C=1e6, max_iter=8000, random_state=42, solver="lbfgs"
    )
    clf_a.fit(X_tr, y_tr)
    sw = np.ones(len(y_tr), dtype=np.float64)
    sw[y_tr == 2] = 2.0
    clf_b = LogisticRegression(
        class_weight=None, C=1e6, max_iter=8000, random_state=42, solver="lbfgs"
    )
    clf_b.fit(X_tr, y_tr, sample_weight=sw)

    n = float(len(y_tr))
    names = list(clf_a.classes_)
    y_tr_arr = np.asarray(y_tr)
    pi = np.array([np.sum(y_tr_arr == c) / n for c in names], dtype=np.float64)
    w_c = np.array([float(np.sum(sw[y_tr_arr == c])) for c in names], dtype=np.float64)
    pi_hat = w_c / w_c.sum()
    offsets = np.log(np.clip(pi, 1e-12, 1.0)) - np.log(np.clip(pi_hat, 1e-12, 1.0))

    logits_b = clf_b.decision_function(X_te)
    p_b_corr = _softmax(logits_b + offsets)
    p_a = clf_a.predict_proba(X_te)
    assert float(np.max(np.abs(p_a - p_b_corr))) < 0.02
    assert p_a.shape == (len(y_te), 3)
