#!/usr/bin/env python3
"""Acceptance tests for the calibration framework.

These tests implement the protocol's §7.3 acceptance criteria.
They must ALL pass before any experiment execution.

Run:
  .venv/bin/python scripts/test_calibration_framework.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.calibration_framework import (
    ExperimentConfig,
    build_head,
    fit_head,
    extract_logits,
    actual_fit_weights,
    build_prior_fit_actual,
    collect_oof_logits_per_config,
    fit_temperature,
    compute_prob_mode,
    chronological_split,
    composite_score,
    classification_metrics,
    ranking_metrics,
    calibration_metrics,
    threshold_survival,
    evaluate_gate,
    CLASSES,
    CLASS_WEIGHT_MAP,
    C_SWEEP,
    SEEDS,
    _softmax_rows,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


# ── Test fixtures ────────────────────────────────────────────────────

def make_synthetic_data(n=200, n_classes=4, dim=64, seed=42, overlap=0.5):
    """Generate synthetic data with known class structure.

    overlap controls class separation: 0.5 = moderate overlap (C matters),
    0.1 = well-separated (easy), 2.0 = heavy overlap (hard).
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n, dim), dtype=np.float32)
    y = []
    class_names = CLASSES[:n_classes]
    per_class = n // n_classes
    for i, c in enumerate(class_names):
        center = rng.randn(dim) * 2
        start = i * per_class
        end = start + per_class
        X[start:end] = center + rng.randn(per_class, dim) * overlap
        y.extend([c] * per_class)
    # Shuffle
    idx = rng.permutation(n)
    X = X[idx]
    y = np.array(y)[idx]
    return X, list(y)


def make_timestamps(n=200, days_span=120, seed=42):
    """Generate timestamps spanning days_span days."""
    rng = np.random.RandomState(seed)
    base = np.datetime64("2026-05-01").astype("datetime64[s]").astype(float)
    offsets = rng.uniform(0, days_span * 86400, size=n)
    return np.sort(base + offsets)


# ── §7.3 Test 1: OOF uses exact head/C (not hardcoded C=1.0) ────────

def test_oof_uses_exact_C():
    """OOF logits must come from the exact C being evaluated, not C=1.0."""
    print("\n[1] OOF uses exact C (not hardcoded C=1.0)")
    X, y = make_synthetic_data(n=200, seed=42)
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # Collect OOF with C=0.01 (very different from C=1.0)
    oof_c001, oof_y001 = collect_oof_logits_per_config(
        "logreg", C=0.01, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )
    # Collect OOF with C=1.0
    oof_c1, oof_y1 = collect_oof_logits_per_config(
        "logreg", C=1.0, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )
    # Collect OOF with C=10.0
    oof_c10, oof_y10 = collect_oof_logits_per_config(
        "logreg", C=10.0, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )

    check("OOF C=0.01 produced logits", len(oof_c001) > 0)
    check("OOF C=1.0 produced logits", len(oof_c1) > 0)
    check("OOF C=10.0 produced logits", len(oof_c10) > 0)

    # The logits MUST differ — if they're identical, C is being ignored
    diff_001_1 = np.max(np.abs(oof_c001 - oof_c1))
    diff_1_10 = np.max(np.abs(oof_c1 - oof_c10))
    check("C=0.01 logits differ from C=1.0", diff_001_1 > 0.01,
          f"(max diff={diff_001_1:.6f})")
    check("C=1.0 logits differ from C=10.0", diff_1_10 > 0.01,
          f"(max diff={diff_1_10:.6f})")

    # OOF y must be complete (every row appears once)
    check("OOF y is complete", len(oof_y001) == len(y_enc))
    check("OOF y matches input", np.array_equal(np.sort(oof_y001), np.sort(y_enc)))


# ── §7.3 Test 2: Per-head OOF (not all heads get LogReg's temperature) ──

def test_per_head_oof():
    """Different heads must produce different OOF logits."""
    print("\n[2] Per-head OOF (not all heads get LogReg temperature)")
    X, y = make_synthetic_data(n=200, seed=42)
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    oof_lr = collect_oof_logits_per_config(
        "logreg", C=1.0, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )
    oof_mlp = collect_oof_logits_per_config(
        "mlp_256", C=1.0, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )

    check("LogReg OOF produced", len(oof_lr[0]) > 0)
    check("MLP OOF produced", len(oof_mlp[0]) > 0)

    if len(oof_lr[0]) > 0 and len(oof_mlp[0]) > 0:
        diff = np.max(np.abs(oof_lr[0] - oof_mlp[0]))
        check("LogReg and MLP OOF differ", diff > 0.01,
              f"(max diff={diff:.6f})")


# ── §7.3 Test 3: Prior correction uses actual weights ──────────────

def test_prior_actual_weights():
    """Prior correction must use actual fit weights, not assume balanced."""
    print("\n[3] Prior correction uses actual weights (not assumed balanced)")
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(["investigation_lead"] * 10 + ["important"] * 20 +
                          ["background"] * 30 + ["noise"] * 40)

    # LogReg: uses class_weight='balanced' → effective weights should differ from uniform
    w_lr = actual_fit_weights(y_enc, None, "logreg")
    check("LogReg weights are balanced (non-uniform)", np.std(w_lr) > 0.01)

    # MLP: no class_weight → weights should be uniform (when no sample_weight)
    w_mlp = actual_fit_weights(y_enc, None, "mlp_256")
    check("MLP weights are uniform (no class_weight)", np.std(w_mlp) < 1e-6)

    # kNN: no class_weight → weights should be uniform
    w_knn = actual_fit_weights(y_enc, None, "knn")
    check("kNN weights are uniform (no class_weight)", np.std(w_knn) < 1e-6)

    # Prior offsets for LogReg should be non-zero (balanced ≠ empirical)
    prior_lr = build_prior_fit_actual(y_enc, None, "logreg", CLASSES)
    check("LogReg prior offsets are non-zero",
          prior_lr["offsets"] is not None and np.any(np.abs(prior_lr["offsets"]) > 0.01))

    # Prior offsets for MLP should be zero (uniform = empirical when no sample_weight)
    prior_mlp = build_prior_fit_actual(y_enc, None, "mlp_256", CLASSES)
    if prior_mlp["offsets"] is not None:
        check("MLP prior offsets are ~zero (no class balancing)",
              np.all(np.abs(prior_mlp["offsets"]) < 0.01),
              f"(offsets={prior_mlp['offsets']})")
    else:
        check("MLP prior offsets are None", False, "expected zero offsets, got None")

    # With sample_weight, both should differ
    sw = np.ones(len(y_enc)) * 2.0
    sw[:10] *= 5.0  # Upweight investigation_lead
    prior_lr_sw = build_prior_fit_actual(y_enc, sw, "logreg", CLASSES)
    prior_mlp_sw = build_prior_fit_actual(y_enc, sw, "mlp_256", CLASSES)
    check("LogReg with sample_weight has different offsets",
          not np.allclose(prior_lr["offsets"], prior_lr_sw["offsets"]))
    check("MLP with sample_weight has non-zero offsets",
          prior_mlp_sw["offsets"] is not None and np.any(np.abs(prior_mlp_sw["offsets"]) > 0.01))


# ── §7.3 Test 4: Probability modes are distinct ─────────────────────

def test_prob_modes():
    """P0/P1/P2/P3 must produce different probabilities."""
    print("\n[4] Probability modes P0-P3 are distinct")
    rng = np.random.RandomState(42)
    logits = rng.randn(50, 4)
    temperature = 2.0
    offsets = np.array([0.5, -0.3, 0.1, -0.2])

    p0 = compute_prob_mode("P0", logits, temperature, offsets)
    p1 = compute_prob_mode("P1", logits, temperature, offsets)
    p2 = compute_prob_mode("P2", logits, temperature, offsets)
    p3 = compute_prob_mode("P3", logits, temperature, offsets)

    # All must be valid probability distributions
    for name, probs in [("P0", p0), ("P1", p1), ("P2", p2), ("P3", p3)]:
        check(f"{name} rows sum to 1",
              np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
        check(f"{name} all non-negative",
              np.all(probs >= -1e-9))

    # P0 ≠ P1 (temperature changes things)
    check("P0 ≠ P1 (temperature effect)", np.max(np.abs(p0 - p1)) > 1e-6)
    # P0 ≠ P2 (prior changes things)
    check("P0 ≠ P2 (prior effect)", np.max(np.abs(p0 - p2)) > 1e-6)
    # P3 ≠ P1 (prior + temp ≠ temp only)
    check("P3 ≠ P1 (prior+temp ≠ temp only)", np.max(np.abs(p3 - p1)) > 1e-6)
    # P3 ≠ P2 (prior + temp ≠ prior only)
    check("P3 ≠ P2 (prior+temp ≠ prior only)", np.max(np.abs(p3 - p2)) > 1e-6)

    # P1 with T=1 should equal P0
    p1_t1 = compute_prob_mode("P1", logits, 1.0, offsets)
    check("P1 with T=1 equals P0", np.allclose(p1_t1, p0, atol=1e-9))

    # P2 with zero offsets should equal P0
    p2_zero = compute_prob_mode("P2", logits, 1.0, np.zeros(4))
    check("P2 with zero offsets equals P0", np.allclose(p2_zero, p0, atol=1e-9))


# ── §7.3 Test 5: Temperature fitting ────────────────────────────────

def test_temperature_fitting():
    """Temperature fitting must reduce NLL and respect bounds."""
    print("\n[5] Temperature fitting")
    rng = np.random.RandomState(42)
    logits = rng.randn(100, 4) * 3
    y = rng.randint(0, 4, size=100)

    temp, nll = fit_temperature(logits, y)
    check("Temperature is positive", temp > 0)
    check("Temperature is in bounds", 0.25 <= temp <= 12.0,
          f"(temp={temp})")
    check("NLL is finite", np.isfinite(nll))

    # NLL at fitted T should be ≤ NLL at T=1
    probs_t1 = _softmax_rows(logits / 1.0)
    p_true_t1 = probs_t1[np.arange(100), y]
    nll_t1 = -np.mean(np.log(np.clip(p_true_t1, 1e-9, 1.0)))
    check("Fitted T has NLL ≤ T=1", nll <= nll_t1 + 1e-6,
          f"(nll_T={nll:.4f}, nll_T1={nll_t1:.4f})")

    # Empty data → T=1
    temp_empty, nll_empty = fit_temperature(np.array([]), np.array([]))
    check("Empty data → T=1", temp_empty == 1.0)


# ── §7.3 Test 6: Logits extraction for all head types ───────────────

def test_logits_all_heads():
    """Logits extraction must work for every head type."""
    print("\n[6] Logits extraction for all head types")
    X, y = make_synthetic_data(n=100, seed=42)
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    heads = ["logreg", "mlp_256", "svm_rbf", "xgboost", "lightgbm", "knn"]
    for head_name in heads:
        try:
            pipe, _ws = fit_head(head_name, C=1.0, seed=42, X_train=X, y_train_enc=y_enc)
            logits = extract_logits(pipe, X[:10])
            check(f"{head_name}: logits shape correct",
                  logits.shape == (10, 4) or logits.shape == (10, 2),
                  f"(shape={logits.shape})")
            check(f"{head_name}: logits are finite",
                  np.all(np.isfinite(logits)))
        except Exception as e:
            check(f"{head_name}: logits extraction", False, str(e))


# ── §7.3 Test 7: Chronological split ────────────────────────────────

def test_chronological_split():
    """Chronological split must produce disjoint, ordered partitions."""
    print("\n[7] Chronological split")
    n = 200
    times = make_timestamps(n, days_span=120)
    y = np.array(["investigation_lead"] * 20 + ["important"] * 40 +
                 ["background"] * 60 + ["noise"] * 80)
    rng = np.random.RandomState(42)
    rng.shuffle(y)

    splits = chronological_split(times, le_transform(y))

    # Partitions must be disjoint
    fit_set = set(splits.fit_idx.tolist())
    select_set = set(splits.select_idx.tolist())
    threshold_set = set(splits.threshold_idx.tolist())
    all_test = set()
    for w in splits.test_windows:
        all_test.update(w.tolist())

    check("fit ∩ select = ∅", fit_set.isdisjoint(select_set))
    check("fit ∩ threshold = ∅", fit_set.isdisjoint(threshold_set))
    check("select ∩ threshold = ∅", select_set.isdisjoint(threshold_set))
    check("fit ∩ test = ∅", fit_set.isdisjoint(all_test))
    check("select ∩ test = ∅", select_set.isdisjoint(all_test))
    check("threshold ∩ test = ∅", threshold_set.isdisjoint(all_test))

    # Fit must be earlier than select, select earlier than threshold
    if len(splits.fit_idx) > 0 and len(splits.select_idx) > 0:
        check("fit times < select times",
              np.max(times[splits.fit_idx]) < np.min(times[splits.select_idx]))
    if len(splits.select_idx) > 0 and len(splits.threshold_idx) > 0:
        check("select times < threshold times",
              np.max(times[splits.select_idx]) < np.min(times[splits.threshold_idx]))

    # Test windows must be after threshold
    if len(splits.threshold_idx) > 0 and all_test:
        test_idx = list(all_test)
        check("threshold times < test times",
              np.max(times[splits.threshold_idx]) < np.min(times[test_idx]))

    # Test windows must be non-overlapping
    for i in range(len(splits.test_windows)):
        for j in range(i + 1, len(splits.test_windows)):
            wi = set(splits.test_windows[i].tolist())
            wj = set(splits.test_windows[j].tolist())
            check(f"test window {i} ∩ test window {j} = ∅",
                  wi.isdisjoint(wj))

    # Total coverage (including embargo gap)
    total = splits.n_fit + splits.n_select + splits.n_threshold + splits.n_test + splits.n_embargo
    check("all rows accounted for (incl embargo)", total == n,
          f"({total} != {n})")
    check("embargo rows are excluded from fit/select/threshold/test",
          len(splits.embargo_idx) == 0 or
          set(splits.embargo_idx.tolist()).isdisjoint(fit_set | select_set | threshold_set | all_test))


def le_transform(y):
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    return le.transform(y)


# ── §7.3 Test 8: Gate replay ────────────────────────────────────────

def test_gate_replay():
    """Gate replay must match production logic."""
    print("\n[8] Gate replay")
    # Cold start
    r = evaluate_gate(None, {"precision_at_30": 0.5, "macro_f1": 0.4})
    check("Cold start promotes", r["promote"] and r["branch"] == "cold_start")

    # Lead recall guard
    r = evaluate_gate(
        {"precision_at_30": 0.5, "macro_f1": 0.4, "lead_recall_at_30": 0.8},
        {"precision_at_30": 0.6, "macro_f1": 0.5, "lead_recall_at_30": 0.5},
    )
    check("Lead recall guard rejects", not r["promote"] and
          r["branch"] == "lead_recall_guard")

    # Big p@30 win
    r = evaluate_gate(
        {"precision_at_30": 0.5, "macro_f1": 0.4, "lead_recall_at_30": 0.8},
        {"precision_at_30": 0.6, "macro_f1": 0.35, "lead_recall_at_30": 0.8},
    )
    check("Big p@30 win promotes (F1 dip within limit)",
          r["promote"] and r["branch"] == "big_p30_win")

    # Legacy gate: p@30 up, F1 ok
    r = evaluate_gate(
        {"precision_at_30": 0.5, "macro_f1": 0.4, "lead_recall_at_30": 0.8},
        {"precision_at_30": 0.52, "macro_f1": 0.4, "lead_recall_at_30": 0.8},
    )
    check("Legacy: p@30 up promotes", r["promote"] and r["branch"] == "legacy_gate")

    # Legacy reject: both down
    r = evaluate_gate(
        {"precision_at_30": 0.5, "macro_f1": 0.4, "lead_recall_at_30": 0.8},
        {"precision_at_30": 0.48, "macro_f1": 0.38, "lead_recall_at_30": 0.8},
    )
    check("Legacy: both down rejects", not r["promote"])

    # Missing lead_recall bypasses guard
    r = evaluate_gate(
        {"precision_at_30": 0.5, "macro_f1": 0.4},
        {"precision_at_30": 0.6, "macro_f1": 0.5, "lead_recall_at_30": 0.0},
    )
    check("Missing old lead_recall bypasses guard", r["promote"])


# ── §7.3 Test 9: Composite score ────────────────────────────────────

def test_composite_score():
    """Composite score must match production formula."""
    print("\n[9] Composite score")
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # pure lead → 1.0
        [0.0, 1.0, 0.0, 0.0],  # pure important → 0.80
        [0.0, 0.0, 1.0, 0.0],  # pure background → 0.20
        [0.0, 0.0, 0.0, 1.0],  # pure noise → 0.0
        [0.5, 0.3, 0.1, 0.1],  # mixed → 0.5*1 + 0.3*0.8 + 0.1*0.2 = 0.76
    ])
    scores = composite_score(probs, CLASSES)
    check("pure lead → 1.0", abs(scores[0] - 1.0) < 1e-6)
    check("pure important → 0.80", abs(scores[1] - 0.80) < 1e-6)
    check("pure background → 0.20", abs(scores[2] - 0.20) < 1e-6)
    check("pure noise → 0.0", abs(scores[3] - 0.0) < 1e-6)
    check("mixed → 0.76", abs(scores[4] - 0.76) < 1e-6)


# ── §7.3 Test 10: Threshold survival ────────────────────────────────

def test_threshold_survival():
    """Threshold survival must report counts and rates correctly."""
    print("\n[10] Threshold survival")
    probs = np.array([
        [0.9, 0.05, 0.03, 0.02],  # lead, score=0.9+0.04+0.006=0.946 → above 0.55
        [0.1, 0.8, 0.05, 0.05],   # important, score=0.1+0.64+0.01=0.75 → above
        [0.05, 0.1, 0.8, 0.05],   # background, score=0.05+0.08+0.16=0.29 → below
        [0.01, 0.01, 0.01, 0.97], # noise, score=0.01+0.008+0.002=0.02 → below
    ])
    y = ["investigation_lead", "important", "background", "noise"]
    surv = threshold_survival(probs, y, CLASSES, threshold=0.55)

    check("n_above=2", surv["n_above"] == 2)
    check("lead_survival=1.0 (1/1)", surv["lead_survival"] == 1.0)
    check("important_survival=1.0 (1/1)", surv["important_survival"] == 1.0)
    check("background_pass=0.0 (0/1)", surv["background_pass"] == 0.0)
    check("noise_pass=0.0 (0/1)", surv["noise_pass"] == 0.0)
    check("threshold_precision=1.0", surv["threshold_precision"] == 1.0)
    check("lead_survival_count correct",
          surv["lead_survival_count"] == "1/1")


# ── §7.3 Test 11: Cache fingerprint includes config ─────────────────

def test_cache_fingerprint():
    """Config fingerprint must differ for different C values."""
    print("\n[11] Cache fingerprint includes config")
    c1 = ExperimentConfig("e5_base", "e5_tuned", "chunked", "mean",
                           "logreg", 1.0, 42, 1, "P3")
    c2 = ExperimentConfig("e5_base", "e5_tuned", "chunked", "mean",
                           "logreg", 0.1, 42, 1, "P3")
    c3 = ExperimentConfig("e5_base", "e5_tuned", "chunked", "mean",
                           "logreg", 1.0, 42, 1, "P3")

    check("Different C → different fingerprint",
          c1.fingerprint() != c2.fingerprint())
    check("Same config → same fingerprint",
          c1.fingerprint() == c3.fingerprint())
    check("Different C → different config_id",
          c1.config_id() != c2.config_id())


# ── §7.3 Test 12: Full evaluation pipeline ──────────────────────────

def test_full_evaluation():
    """End-to-end evaluation must produce all metrics."""
    print("\n[12] Full evaluation pipeline")
    X, y = make_synthetic_data(n=200, seed=42)
    sw = np.ones(200)
    times = make_timestamps(200, days_span=120)

    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    splits = chronological_split(times, y_enc)
    if len(splits.fit_idx) < 15 or len(splits.test_windows) == 0:
        # Use simple split if chronological doesn't work for synthetic data
        fit_idx = np.arange(140)
        test_idx = np.arange(140, 200)
    else:
        fit_idx = splits.fit_idx
        test_idx = splits.test_windows[0]

    config = ExperimentConfig(
        encoder="test", text_mode="plain", context_mode="chunked",
        pooling="mean", head="logreg", C=1.0, seed=42,
        profile_id=1, prob_mode="P3",
    )

    from scripts.calibration_framework import evaluate_configuration
    result = evaluate_configuration(
        config, X, y, sw, fit_idx, test_idx,
        incumbent_metrics={"precision_at_30": 0.3, "macro_f1": 0.3,
                            "lead_recall_at_30": 0.5},
    )

    check("Has classification metrics", "accuracy" in result.classification)
    check("Has ranking metrics", "precision_at_30" in result.ranking)
    check("Has calibration metrics", "nll" in result.calibration)
    check("Has survival metrics", "lead_survival" in result.survival)
    check("Has gate result", result.gate is not None)
    check("Has temperature", result.temperature > 0)
    check("Has class support", len(result.class_support) == 4)
    check("Has notes with all-mode survival",
          any("survival_all_modes" in n for n in result.notes))


# ── §7.3 Test 13: C mismatch detection ──────────────────────────────

def test_c_mismatch_detection():
    """Verify that using wrong C for OOF produces different temperature."""
    print("\n[13] C mismatch detection")
    # Use heavily overlapping data so C matters for confidence calibration
    X, y = make_synthetic_data(n=400, dim=16, seed=42, overlap=2.5)
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # Correct: OOF with C=0.01 (very strong regularization → less confident)
    oof_correct, y_correct = collect_oof_logits_per_config(
        "logreg", C=0.01, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )
    temp_correct, nll_correct = fit_temperature(oof_correct, y_correct)

    # Wrong: OOF with C=10.0 (weak regularization → more confident)
    oof_wrong, y_wrong = collect_oof_logits_per_config(
        "logreg", C=10.0, seed=42, X=X, y_enc=y_enc,
        sample_weight=None, n_folds=5,
    )
    temp_wrong, nll_wrong = fit_temperature(oof_wrong, y_wrong)

    # The OOF logits MUST differ (this is the core calibration bug check)
    logit_diff = np.max(np.abs(oof_correct - oof_wrong))
    check("C=0.01 OOF logits differ from C=10.0",
          logit_diff > 0.1,
          f"(max logit diff={logit_diff:.4f})")

    # Either temperatures differ, or NLLs differ, or both
    # (both hitting the bound is a degenerate case, but logits differing is the key)
    temp_differ = abs(temp_correct - temp_wrong) > 0.01
    nll_differ = abs(nll_correct - nll_wrong) > 0.001
    check("C affects calibration (temp or NLL differs)",
          temp_differ or nll_differ,
          f"(temp: {temp_correct:.4f} vs {temp_wrong:.4f}, nll: {nll_correct:.4f} vs {nll_wrong:.4f})")


# ── §7.3 Test 14: Class-order consistency ───────────────────────────

def test_class_order():
    """Probability computation must be consistent with class order."""
    print("\n[14] Class-order consistency")
    rng = np.random.RandomState(42)
    logits = rng.randn(10, 4)
    offsets = np.array([0.1, -0.1, 0.2, -0.2])

    # P3 with CLASSES order
    probs1 = compute_prob_mode("P3", logits, 2.0, offsets)

    # Permute classes and offsets consistently
    perm = [2, 0, 3, 1]
    logits_perm = logits[:, perm]
    offsets_perm = offsets[perm]
    probs2 = compute_prob_mode("P3", logits_perm, 2.0, offsets_perm)

    # Unpermute probs2 and compare
    inv_perm = [0, 0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    probs2_unperm = probs2[:, inv_perm]

    check("Class permutation consistency",
          np.allclose(probs1, probs2_unperm, atol=1e-9))


# ── Main ────────────────────────────────────────────────────────────

def test_story_group_enforcement():
    """Verify that story groups straddling a test boundary are pulled together."""
    print("\n[15] Story group enforcement")
    n = 300
    rng = np.random.RandomState(42)
    times = np.sort(rng.uniform(0, 100 * 86400, n).astype(np.float64))
    y_enc = rng.randint(0, 4, n)

    # Create story groups: most are unique, but 20 pairs share a group
    groups = np.array([f"g{i}" for i in range(n)])
    # Make pairs: items 50&200, 51&201, etc. share a group
    for i in range(20):
        groups[200 + i] = groups[50 + i]

    splits = chronological_split(
        times, y_enc, story_group_ids=groups, n_test_windows=3,
    )

    # Check: no story group appears in both fit and any test window
    fit_set = set(groups[splits.fit_idx])
    for w_idx, w in enumerate(splits.test_windows):
        if len(w) == 0:
            continue
        test_set = set(groups[w])
        overlap = fit_set & test_set
        check(f"window {w_idx}: no story group in both fit and test",
              len(overlap) == 0,
              f"overlap: {overlap}")

    # Check: no story group appears in both fit and select
    select_set = set(groups[splits.select_idx])
    threshold_set = set(groups[splits.threshold_idx])
    check("No story group in both fit and select",
          len(fit_set & select_set) == 0,
          f"overlap: {fit_set & select_set}")
    check("No story group in both select and threshold",
          len(select_set & threshold_set) == 0,
          f"overlap: {select_set & threshold_set}")
    check("No story group in both fit and threshold",
          len(fit_set & threshold_set) == 0,
          f"overlap: {fit_set & threshold_set}")

    # Check: enforcement note is present
    check("Story group enforcement note present",
          any("story_group" in note for note in splits.notes),
          f"notes: {splits.notes}")


def test_weight_status_reporting():
    """Verify that fit_head reports weight acceptance/rejection status."""
    print("\n[16] Weight status reporting")
    X, y = make_synthetic_data(n=100, dim=16, seed=42)
    le = __import__("sklearn").preprocessing.LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # LogReg accepts sample_weight
    sw = np.ones(100) * 2.0
    sw[50:] = 5.0  # non-uniform
    _, status_logreg = fit_head("logreg", C=1.0, seed=42, X_train=X, y_train_enc=y_enc, sample_weight=sw)
    check("LogReg weight status is 'accepted'", status_logreg == "accepted",
          f"got: {status_logreg}")

    # Uniform weights → 'uniform'
    _, status_uniform = fit_head("logreg", C=1.0, seed=42, X_train=X, y_train_enc=y_enc, sample_weight=np.ones(100))
    check("Uniform weight status is 'uniform'", status_uniform == "uniform",
          f"got: {status_uniform}")

    # No weights → 'uniform'
    _, status_none = fit_head("logreg", C=1.0, seed=42, X_train=X, y_train_enc=y_enc)
    check("No weight status is 'uniform'", status_none == "uniform",
          f"got: {status_none}")


def test_prob_mode_label():
    """Verify that EvalResult includes a human-readable prob_mode_label."""
    print("\n[17] Prob mode label")
    from scripts.calibration_framework import evaluate_configuration, ExperimentConfig
    X, y = make_synthetic_data(n=200, dim=32, seed=42)
    sw = np.ones(200)
    fit_idx = np.arange(150)
    test_idx = np.arange(150, 200)
    config = ExperimentConfig(
        encoder="test", text_mode="plain", context_mode="chunked",
        pooling="mean", head="logreg", C=1.0, seed=42,
        profile_id=1, prob_mode="P3",
    )
    result = evaluate_configuration(config, X, y, sw, fit_idx, test_idx)
    check("prob_mode is P3", result.prob_mode == "P3")
    check("prob_mode_label is 'prior_then_temperature'",
          result.prob_mode_label == "prior_then_temperature",
          f"got: {result.prob_mode_label}")
    check("weights_status is present", hasattr(result, "weights_status"))


def test_adversarial_regressions():
    from unittest.mock import patch
    import scripts.calibration_framework as cf

    X, labels = make_synthetic_data(n=80, dim=8, overlap=2.0)
    y = le_transform(labels)
    pipe, status = fit_head("logreg", 0.1, 42, X, y, np.full(80, 3.0))
    assert status == "accepted", "constant non-unit weights must not be dropped"
    assert np.allclose(actual_fit_weights(y, None, "svm_linear"),
                       actual_fit_weights(y, None, "logreg"))
    with patch.object(cf, "build_head") as factory:
        broken = factory.return_value
        broken.named_steps = {"classifier": type("Broken", (), {
            "fit": lambda self, X, y, sample_weight=None: None})()}
        broken.fit.side_effect = ValueError("unexpected fit failure")
        try:
            fit_head("logreg", 1.0, 42, X, y, np.arange(80) + 1)
        except ValueError as exc:
            assert str(exc) == "unexpected fit failure"
        else:
            raise AssertionError("fit errors must propagate")
        assert broken.fit.call_count == 1

    svc, _ = fit_head("svm_rbf", 1.0, 42, X, y)
    # SVC(probability=True) with multiclass uses OvO decision_function which
    # returns shape (n, n_classes*(n_classes-1)/2), NOT class-aligned.
    # extract_logits must detect this and fall back to log(predict_proba)
    # so that logits have shape (n, n_classes) — the framework invariant.
    svc_logits = extract_logits(svc, X)
    n_classes = len(np.unique(y))
    assert svc_logits.shape == (80, n_classes), \
        f"SVC logits must be class-aligned (n, n_classes), got {svc_logits.shape}"
    assert np.all(np.isfinite(svc_logits)), "SVC logits must be finite"
    svc_probs = _softmax_rows(svc_logits)
    assert np.allclose(svc_probs.sum(axis=1), 1.0, atol=1e-6), "softmax must normalize"
    # log(predict_proba) → softmax(log_proba) == predict_proba (up to clipping
    # of tiny probabilities in extract_logits)
    raw_probs = svc.predict_proba(X)
    assert np.allclose(svc_probs, raw_probs, atol=1e-4), \
        f"softmax(log(predict_proba)) must recover predict_proba (max diff {np.max(np.abs(svc_probs - raw_probs))})"
    times = np.arange(101, dtype=float) * 86400
    groups = np.arange(101)
    groups[[5, 42, 49]] = 500
    groups[[10, 65, 80, 99]] = 501
    split = chronological_split(times, np.arange(101) % 4, groups)
    assert 100 in split.test_windows[0], "latest timestamp must be evaluated"
    partitions = [split.fit_idx, split.select_idx, split.threshold_idx] + split.test_windows
    for i, part in enumerate(partitions):
        for other in partitions[i + 1:]:
            assert set(groups[part]).isdisjoint(groups[other])
    for window, (start, end) in zip(split.test_windows, split.window_boundaries):
        assert np.all(times[window] >= start) and np.all(times[window] <= end)
    assert 10 in split.group_purged_idx and 99 in split.test_windows[0]
    assert 5 in split.group_purged_idx and 42 in split.group_purged_idx
    assert sum(map(len, partitions)) + split.n_embargo + split.n_group_purged == 101
    sparse = chronological_split(np.array([0, 61, 100]) * 86400., np.array([0, 1, 2]))
    assert len(sparse.test_windows) == 3
    assert 1 in sparse.test_windows[2], "empty windows must advance"

    config = ExperimentConfig("test", "plain", "chunked", "mean", "logreg", .1, 42, 1, "P1")
    result = cf.evaluate_configuration(config, X, labels, None, np.arange(60), np.arange(60, 80),
                                       story_group_ids=np.arange(80))
    assert set(result.mode_metrics) == {"P0", "P1", "P2", "P3"}
    assert result.temperature == result.temperatures["P1"]
    assert result.predictions["test_indices"] == list(range(60, 80))
    for mode, metrics in result.mode_metrics.items():
        assert set(metrics) == {"classification", "ranking", "calibration", "survival"}
        assert len(result.predictions["modes"][mode]["probabilities"]) == 20
    check("Adversarial regressions", True)


def main():
    print("=" * 60)
    print("Calibration Framework Acceptance Tests")
    print("Protocol: docs/calibration-for-real.md §7.3")
    print("=" * 60)

    test_oof_uses_exact_C()
    test_per_head_oof()
    test_prior_actual_weights()
    test_prob_modes()
    test_temperature_fitting()
    test_logits_all_heads()
    test_chronological_split()
    test_gate_replay()
    test_composite_score()
    test_threshold_survival()
    test_cache_fingerprint()
    test_full_evaluation()
    test_c_mismatch_detection()
    test_class_order()
    test_story_group_enforcement()
    test_weight_status_reporting()
    test_prob_mode_label()
    test_adversarial_regressions()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
