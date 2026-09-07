#!/usr/bin/env python3
"""D3 chronological validation: E5-base vs E5-large-instruct.

Compares the current production recipe (e5_base, C=0.01, P1) against the
best challenger from the C sweep (e5_large_instruct, C=0.005, P2) using
chronological splits — the most production-realistic evaluation.

Also tests a few nearby configs to confirm the challenger is robust, not
a single-point artifact.

Uses the same evaluation methodology as lab_e5_sweep.py:
  - Always fit temperature on prior-adjusted OOF logits (matches production)
  - P1 = no prior at scoring time (apply_prior=False)
  - P2 = prior at scoring time (apply_prior=True)
  - C-aware OOF collection (not config-bound)

Chronological split from calibration_framework.py:
  - 3 test windows of 14 days each (most recent data)
  - 48h embargo between fit and test
  - Story group enforcement (no title-leakage across partitions)

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_d3.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_d3.py --profile 3
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import db
from pipeline import (
    CLASSES,
    CLASS_WEIGHT_MAP,
    compute_sample_weights,
    build_prior_fit,
    _prior_offset_vector,
    _add_logit_offsets,
    _fit_temperature_scalar,
    _ranking_metrics,
    _softmax_rows,
    logits_for_classifier_head,
    _transformer_fit_kwargs,
    _oof_fold_count,
    _min_class_count_in_labels,
)
from config import get_config
from scripts.calibration_framework import chronological_split

# ── Config ───────────────────────────────────────────────────────────

ENCODERS = {
    "e5_base": {
        "cache_path": "encoder_comparison_cache/e5_base_p{p}.npy",
        "dim": 768,
    },
    "e5_large_instruct": {
        "cache_path": "encoder_comparison_cache/e5_large_instruct_p{p}.npy",
        "dim": 1024,
    },
}

# Configs to test: (label, encoder, C, apply_prior)
CONFIGS = [
    # Production baseline
    ("prod_e5_base_c01_P1",     "e5_base",            0.01,  False),
    # Best challenger from sweep
    ("chall_e5_large_c005_P2",  "e5_large_instruct",  0.005, True),
    # Nearby configs for robustness check
    ("chall_e5_large_c01_P2",   "e5_large_instruct",  0.01,  True),
    ("chall_e5_large_c002_P2",  "e5_large_instruct",  0.02,  True),
    ("chall_e5_large_c005_P1",  "e5_large_instruct",  0.005, False),
    # E5-base with P2 for ablation (does prior help base too?)
    ("ablat_e5_base_c01_P2",    "e5_base",            0.01,  True),
    ("ablat_e5_base_c005_P2",   "e5_base",            0.005, True),
]

SEED = 42


# ── Data loading ─────────────────────────────────────────────────────

def load_profile_data(profile_id: int) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Load labeled entries with timestamps and story groups for chronological split.

    Uses the SAME loading method as lab_encoder_comparison.py and lab_e5_sweep.py
    (db.get_all_labels + db.get_all_entries joined in Python) so that cached
    embeddings align row-for-row with the returned labeled list.
    """
    labels = db.get_all_labels(profile_id=profile_id)
    all_entries = db.get_all_entries(include_embedding=False)
    entry_map = {}
    for e in all_entries:
        key = db.entry_key_from_mapping(e)
        entry_map[key] = e

    labeled = []
    for lbl in labels:
        key = db.entry_key_from_mapping(lbl)
        entry = entry_map.get(key)
        if entry:
            merged = {**entry, **lbl}
            labeled.append(merged)

    if not labeled:
        return [], np.array([]), np.array([])

    # Timestamps (fetched_at = ingest time = earliest prediction possible)
    times = []
    for r in labeled:
        ts_str = r.get("fetched_at")
        try:
            ts = np.datetime64(ts_str).astype("datetime64[s]").astype(float)
        except Exception:
            ts = 0.0
        times.append(ts)
    times = np.array(times)

    # Story groups: normalized title hash to prevent leakage
    story_groups = []
    for r in labeled:
        title = (r.get("title") or "").strip().lower()
        title = " ".join(title.split())
        story_groups.append(hashlib.md5(title.encode("utf-8")).hexdigest())
    story_groups = np.array(story_groups)

    return labeled, times, story_groups


def load_embeddings(encoder_key: str, profile_id: int, labeled: List[dict]) -> np.ndarray:
    """Load cached embeddings and verify row count."""
    cfg = ENCODERS[encoder_key]
    p = BASE_DIR / "lab_data" / cfg["cache_path"].format(p=profile_id)
    if not p.exists():
        raise FileNotFoundError(f"Cached embeddings not found: {p}")
    X = np.load(p)
    if X.shape[0] != len(labeled):
        raise ValueError(
            f"Cached embeddings ({X.shape[0]} rows) != labeled entries ({len(labeled)}). "
            f"Cache is stale — re-run lab_encoder_comparison.py."
        )
    return X


# ── OOF collection (C-aware, same as lab_e5_sweep.py) ────────────────

def collect_oof_logits_with_c(
    X: np.ndarray,
    y_enc: np.ndarray,
    sample_weight,
    n_folds: int,
    c_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF logits using a pipeline built with the given C value."""
    from sklearn.model_selection import StratifiedKFold

    n_samples = len(y_enc)
    sw_arr = None
    if sample_weight is not None and len(sample_weight) == n_samples:
        sw_arr = np.asarray(sample_weight, dtype=np.float64)

    oof_logits = [None] * n_samples
    oof_y = [None] * n_samples

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    try:
        split_iter = skf.split(X, y_enc)
    except ValueError:
        return np.array([]), np.array([])

    for train_idx, val_idx in split_iter:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=c_value, class_weight="balanced",
                max_iter=1000, solver="lbfgs",
                random_state=42,
            )),
        ])
        fit_kw = _transformer_fit_kwargs(
            sw_arr[train_idx] if sw_arr is not None else None
        )
        pipe.fit(X[train_idx], y_enc[train_idx], **fit_kw)
        logits_val = logits_for_classifier_head(pipe, X[val_idx])
        for j, vi in enumerate(val_idx):
            oof_logits[vi] = logits_val[j]
            oof_y[vi] = y_enc[vi]

    valid = [(l, y) for l, y in zip(oof_logits, oof_y) if l is not None]
    if not valid:
        return np.array([]), np.array([])
    return np.vstack([l for l, _ in valid]), np.array([y for _, y in valid])


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_on_window(
    X_all: np.ndarray,
    y_all: List[str],
    sw_all: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    c_value: float,
    apply_prior: bool,
) -> dict:
    """Train on fit_idx, evaluate on test_idx with given C and prior mode.

    Mirrors production _train_transformer + classifier_probabilities:
      1. Fit LogReg with swept C on fit partition
      2. Collect OOF logits with the SAME C
      3. ALWAYS apply prior offsets to OOF logits before fitting temperature
      4. At scoring: apply prior offsets only if apply_prior=True
      5. Temperature-scale and softmax
    """
    y_all_arr = np.asarray(y_all)
    y_fit = list(y_all_arr[fit_idx])
    y_test = list(y_all_arr[test_idx])

    sw_fit = None
    if sw_all is not None and len(sw_all) > 0:
        sw_fit = np.asarray(sw_all, dtype=np.float64)[fit_idx]

    X_fit = X_all[fit_idx]
    X_test = X_all[test_idx]

    # Always use ALL 4 classes (matches production). LabelEncoder sorts
    # alphabetically: ['background', 'important', 'investigation_lead', 'noise']
    le = LabelEncoder()
    le.fit(CLASSES)
    class_names = le.classes_.tolist()
    y_fit_enc = le.transform(y_fit)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
            random_state=42,
        )),
    ])
    fit_kwargs = _transformer_fit_kwargs(sw_fit)
    clf.fit(X_fit, y_fit_enc, **fit_kwargs)

    # OOF temperature calibration
    config = get_config()
    min_class = _min_class_count_in_labels(y_fit)
    n_folds = _oof_fold_count(len(y_fit), min_class)
    prior_fit = build_prior_fit(y_fit, sw_fit, class_names, config)
    temperature = 1.0

    if n_folds >= 2:
        try:
            oof_logits, oof_y_enc = collect_oof_logits_with_c(
                X_fit, y_fit_enc, sw_fit, n_folds, c_value,
            )
            if len(oof_y_enc) >= 3:
                # Production ALWAYS applies prior offsets to OOF before fitting T
                off = _prior_offset_vector({"prior_fit": prior_fit}, class_names)
                if off is not None:
                    oof_logits = _add_logit_offsets(oof_logits, off)
                temperature = _fit_temperature_scalar(
                    oof_logits, le.inverse_transform(oof_y_enc), class_names,
                )
        except Exception:
            temperature = 1.0

    # Get test logits
    logits = logits_for_classifier_head(clf, X_test)
    logits = np.asarray(logits, dtype=np.float64)

    # Apply prior at scoring time only if apply_prior=True (P2)
    if apply_prior:
        offsets = _prior_offset_vector({"prior_fit": prior_fit}, class_names)
        if offsets is not None:
            logits = _add_logit_offsets(logits, offsets)

    # Temperature scale
    probs = _softmax_rows(logits / max(temperature, 1e-3))

    # Predictions
    y_pred_enc = np.argmax(probs, axis=1)
    y_pred = [class_names[i] for i in y_pred_enc]

    # Metrics
    f1 = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    rank = _ranking_metrics(probs, class_names, y_test, k=30)

    return {
        "n_train": len(fit_idx),
        "n_test": len(test_idx),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "calibration_temperature": round(temperature, 4),
        "test_class_distribution": dict(pd.Series(y_test).value_counts().to_dict()),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="D3 chronological validation: E5-base vs E5-large")
    parser.add_argument("--profile", type=int, nargs="+", default=[2, 3, 4],
                        help="Profile IDs (default: 2=digital, 3=sicherheit, 4=eu)")
    parser.add_argument("--windows", type=int, default=1,
                        help="Number of chronological test windows (default 1)")
    parser.add_argument("--window-days", type=int, default=14,
                        help="Test window size in days (default 14)")
    args = parser.parse_args()

    all_results = []

    for pid in args.profile:
        print(f"\n{'='*70}")
        print(f"Profile {pid}")
        print(f"{'='*70}")

        labeled, times, story_groups = load_profile_data(pid)
        if len(labeled) < 20:
            print(f"  Skipping: only {len(labeled)} labeled entries")
            continue

        y = [l["label"] for l in labeled]
        print(f"  {len(labeled)} labeled entries, classes: {dict(pd.Series(y).value_counts().to_dict())}")
        print(f"  Time range: {np.datetime64(int(times.min()), 's')} to {np.datetime64(int(times.max()), 's')}")

        le = LabelEncoder()
        le.fit(CLASSES)
        y_enc = le.transform(y)

        # Chronological split
        splits = chronological_split(
            times, y_enc,
            story_group_ids=story_groups,
            n_test_windows=args.windows,
            test_window_days=args.window_days,
        )

        if not splits.test_windows or len(splits.test_windows[0]) == 0:
            print(f"  SKIP: no valid test windows")
            continue

        print(f"  Chronological split: fit={splits.n_fit} select={splits.n_select} "
              f"threshold={splits.n_threshold} test={splits.n_test} "
              f"embargo={splits.n_embargo} windows={len(splits.test_windows)}")
        if splits.notes:
            print(f"  Notes: {'; '.join(splits.notes)}")

        # For D3 validation of pre-selected configs, train on ALL pre-window
        # data (fit+select+threshold). We're not doing model selection, so
        # the select partition is not needed for config comparison, and we're
        # not calibrating thresholds (using ranking metrics, not threshold
        # metrics). This matches production, which trains on all available
        # labeled data.
        train_idx = np.concatenate([
            splits.fit_idx, splits.select_idx, splits.threshold_idx
        ])
        # Sort by original index to keep a stable order
        train_idx = np.sort(train_idx)
        print(f"  Training on all pre-window data: {len(train_idx)} entries "
              f"(fit={splits.n_fit} + select={splits.n_select} + threshold={splits.n_threshold})")

        # Sample weights (computed on ALL labeled data, then indexed)
        sw = compute_sample_weights(labeled)

        for config_label, enc_key, c_val, ap_prior in CONFIGS:
            print(f"\n  --- {config_label} ({enc_key}, C={c_val}, {'P2' if ap_prior else 'P1'}) ---")

            try:
                X = load_embeddings(enc_key, pid, labeled)
            except (FileNotFoundError, ValueError) as e:
                print(f"    SKIP: {e}")
                continue

            window_results = []
            for w_idx, test_idx in enumerate(splits.test_windows):
                if len(test_idx) == 0:
                    continue

                m = evaluate_on_window(
                    X, y, sw, train_idx, test_idx,
                    c_val, ap_prior,
                )
                m["window"] = w_idx
                m["config_label"] = config_label
                m["encoder"] = enc_key
                m["C"] = c_val
                m["apply_prior"] = ap_prior
                m["profile"] = pid
                window_results.append(m)
                all_results.append(m)

                print(f"    w{w_idx} (n_test={m['n_test']}): "
                      f"F1={m['f1_score']:<7} p@30={m['precision_at_30']:<7} "
                      f"lr@30={m['lead_recall_at_30']:<7} AUC={m['ranking_auc']:<7} "
                      f"T={m['calibration_temperature']}")

            # Aggregate across windows
            if window_results:
                avg_f1 = np.mean([r["f1_score"] for r in window_results])
                avg_p30 = np.mean([r["precision_at_30"] for r in window_results])
                avg_lr30 = np.mean([r["lead_recall_at_30"] for r in window_results])
                avg_auc = np.mean([r["ranking_auc"] for r in window_results])
                print(f"    AVG across {len(window_results)} windows: "
                      f"F1={avg_f1:.4f} p@30={avg_p30:.4f} lr@30={avg_lr30:.4f} AUC={avg_auc:.4f}")

    # Save
    out_path = BASE_DIR / "lab_data" / "e5_d3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary: aggregate per config per profile
    print(f"\n{'='*90}")
    print("D3 SUMMARY: avg across chronological test windows")
    print(f"{'='*90}")

    df = pd.DataFrame(all_results)
    if df.empty:
        print("  No results.")
        return

    summary = df.groupby(["profile", "config_label", "encoder", "C", "apply_prior"]).agg({
        "f1_score": "mean",
        "precision_at_30": "mean",
        "lead_recall_at_30": "mean",
        "ranking_auc": "mean",
        "calibration_temperature": "mean",
        "n_test": "sum",
        "window": "count",
    }).reset_index()

    for pid in sorted(summary["profile"].unique()):
        sub = summary[summary["profile"] == pid].sort_values("precision_at_30", ascending=False)
        print(f"\n  Profile {pid} (n_windows={sub['window'].iloc[0]}, n_test_total={sub['n_test'].iloc[0]}):")
        print(f"  {'Config':<28} {'Encoder':<22} {'C':>6} {'Mode':>4} {'F1':>7} {'p@30':>7} {'lr@30':>7} {'AUC':>7} {'T':>7}")
        for _, r in sub.iterrows():
            mode = "P2" if r["apply_prior"] else "P1"
            print(f"  {r['config_label']:<28} {r['encoder']:<22} {r['C']:>6} {mode:>4} "
                  f"{r['f1_score']:>7.4f} {r['precision_at_30']:>7.4f} {r['lead_recall_at_30']:>7.4f} "
                  f"{r['ranking_auc']:>7.4f} {r['calibration_temperature']:>7.4f}")

    # Cross-desk shared recipe comparison
    print(f"\n{'='*90}")
    print("CROSS-DESK SHARED RECIPE COMPARISON")
    print(f"{'='*90}")

    cross = df.groupby(["config_label", "encoder", "C", "apply_prior"]).agg({
        "f1_score": "mean",
        "precision_at_30": "mean",
        "lead_recall_at_30": "mean",
        "ranking_auc": "mean",
    }).reset_index()

    cross = cross.sort_values("precision_at_30", ascending=False)
    print(f"\n  {'Config':<28} {'Encoder':<22} {'C':>6} {'Mode':>4} {'avg_F1':>8} {'avg_p@30':>8} {'avg_lr@30':>8} {'avg_AUC':>8}")
    for _, r in cross.iterrows():
        mode = "P2" if r["apply_prior"] else "P1"
        print(f"  {r['config_label']:<28} {r['encoder']:<22} {r['C']:>6} {mode:>4} "
              f"{r['f1_score']:>8.4f} {r['precision_at_30']:>8.4f} {r['lead_recall_at_30']:>8.4f} "
              f"{r['ranking_auc']:>8.4f}")

    # Head-to-head: production vs challenger
    print(f"\n{'='*90}")
    print("HEAD-TO-HEAD: production vs best challenger (per desk)")
    print(f"{'='*90}")

    for pid in sorted(df["profile"].unique()):
        prod = df[(df["profile"] == pid) & (df["config_label"] == "prod_e5_base_c01_P1")]
        chall = df[(df["profile"] == pid) & (df["config_label"] == "chall_e5_large_c005_P2")]
        if prod.empty or chall.empty:
            continue

        print(f"\n  Profile {pid}:")
        for metric in ["f1_score", "precision_at_30", "lead_recall_at_30", "ranking_auc"]:
            p_val = prod[metric].mean()
            c_val = chall[metric].mean()
            delta = c_val - p_val
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"    {metric:<22} prod={p_val:.4f}  chall={c_val:.4f}  delta={delta:+.4f} {arrow}")

    print()


if __name__ == "__main__":
    main()
