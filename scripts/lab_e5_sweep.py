#!/usr/bin/env python3
"""E5-base vs E5-large-instruct: C sweep × probability mode × prior correction.

Sweeps both encoders across regularization and calibration levers so we can
decide whether E5-large-instruct justifies the migration cost or whether
tuning E5-base further closes the gap.

Levers tested:
  Encoders:  e5_base (768d), e5_large_instruct (1024d)
  C values:  [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
  Prob mode: P1 (temperature only), P2 (temperature + prior offset)
  Prior:     apply_prior True/False (only meaningful for P2)

Uses cached embeddings from encoder_comparison_cache.
Same seed=42, same StratifiedShuffleSplit, same sample weights as the
encoder comparison script.

Focuses on desks 2 (Digital), 3 (Sicherheit), 4 (EU) — not mothership.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_sweep.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_sweep.py --profile 3
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_sweep.py --top 10
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
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

C_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEED = 42

# (label, apply_prior) — P1 = temperature only, P2 = temperature + prior
PROB_MODES = [
    ("P1", False),   # temperature only (current production)
    ("P2", True),    # temperature + prior offset
]


# ── Data loading ─────────────────────────────────────────────────────

def load_labeled_entries(profile_id: int) -> List[dict]:
    """Load labeled entries for a profile, joined with entry text."""
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
    return labeled


def load_embeddings(encoder_key: str, profile_id: int) -> np.ndarray:
    cfg = ENCODERS[encoder_key]
    p = BASE_DIR / "lab_data" / cfg["cache_path"].format(p=profile_id)
    if not p.exists():
        raise FileNotFoundError(f"Cached embeddings not found: {p}")
    return np.load(p)


# ── Evaluation ───────────────────────────────────────────────────────

def _collect_oof_logits_with_c(
    X: np.ndarray,
    y_list: List[str],
    sample_weight,
    label_encoder,
    n_folds: int,
    c_value: float,
) -> Tuple[np.ndarray, List[str]]:
    """Out-of-fold logits using a pipeline built with the given C value.

    This mirrors pipeline._collect_oof_logits but uses c_value instead of
    reading classifier_c from the global config.  Production's OOF function
    always uses build_transformer_head_pipeline() which reads config — so
    sweeping C without this function would fit temperature on logits from
    the wrong regularization strength.
    """
    from sklearn.model_selection import StratifiedKFold

    y_enc = label_encoder.transform(y_list)
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
        return np.array([]), []

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
            oof_y[vi] = y_list[vi]

    valid = [(l, y) for l, y in zip(oof_logits, oof_y) if l is not None]
    if not valid:
        return np.array([]), []
    return np.vstack([l for l, _ in valid]), [y for _, y in valid]


def evaluate_combo(
    X: np.ndarray,
    labeled: List[dict],
    c_value: float,
    apply_prior: bool,
    seed: int = SEED,
) -> dict:
    """Train LogReg with given C, evaluate with given probability mode.

    Mirrors production _train_transformer + classifier_probabilities:
      1. Fit LogReg with swept C on train split
      2. Collect OOF logits with the SAME C (not config C)
      3. ALWAYS apply prior offsets to OOF logits before fitting temperature
         (production does this unconditionally — apply_prior only controls
         scoring-time prior application, not temperature fitting)
      4. At scoring time: apply prior offsets only if apply_prior=True
      5. Temperature-scale and softmax
    """
    from sklearn.preprocessing import LabelEncoder

    y = [l["label"] for l in labeled]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    labeled_train = [labeled[i] for i in train_idx]

    sw = compute_sample_weights(labeled_train)

    # Build pipeline with the specific C value
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
            random_state=42,
        )),
    ])
    fit_kwargs = _transformer_fit_kwargs(sw)
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    clf.fit(X_train, y_train_enc, **fit_kwargs)

    class_names = le.classes_.tolist()

    # OOF temperature calibration (mirrors production _train_transformer)
    config = get_config()
    min_class = _min_class_count_in_labels(y_train)
    n_folds = _oof_fold_count(len(y_train), min_class)
    prior_fit = build_prior_fit(y_train, sw, class_names, config)
    temperature = 1.0

    if n_folds >= 2:
        try:
            # Use our C-aware OOF collector, not the config-bound one
            oof_logits, oof_y = _collect_oof_logits_with_c(
                X_train, y_train, sw, le, n_folds, c_value,
            )
            if len(oof_y) >= 3:
                # Production ALWAYS applies prior offsets to OOF logits
                # before fitting temperature, regardless of apply_prior.
                # apply_prior only controls scoring-time application.
                off = _prior_offset_vector({"prior_fit": prior_fit}, class_names)
                if off is not None:
                    oof_logits = _add_logit_offsets(oof_logits, off)
                temperature = _fit_temperature_scalar(
                    oof_logits, np.array(oof_y), class_names,
                )
        except Exception:
            temperature = 1.0

    # Get logits for test set (logits_for_classifier_head handles scaler)
    logits = logits_for_classifier_head(clf, X_test)
    logits = np.asarray(logits, dtype=np.float64)

    # Apply prior offset at scoring time only if apply_prior=True (P2)
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
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "calibration_temperature": round(temperature, 4),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E5-base vs E5-large sweep")
    parser.add_argument("--profile", type=int, nargs="+", default=[2, 3, 4],
                        help="Profile IDs (default: 2=digital, 3=sicherheit, 4=eu)")
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()),
                        help="Encoder keys to test")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only top N configs per desk (0 = show all)")
    args = parser.parse_args()

    results = []

    for pid in args.profile:
        print(f"\n{'='*70}")
        print(f"Profile {pid}")
        print(f"{'='*70}")

        labeled = load_labeled_entries(pid)
        if len(labeled) < 20:
            print(f"  Skipping: only {len(labeled)} labeled entries")
            continue

        y = [l["label"] for l in labeled]
        print(f"  {len(labeled)} labeled entries, classes: {dict(pd.Series(y).value_counts().to_dict())}")

        for enc_key in args.encoders:
            if enc_key not in ENCODERS:
                print(f"  Unknown encoder: {enc_key}")
                continue

            try:
                X = load_embeddings(enc_key, pid)
            except FileNotFoundError as e:
                print(f"  {enc_key}: {e}")
                continue

            # Verify embedding row count matches labeled entry count
            if X.shape[0] != len(labeled):
                print(f"  WARNING: {enc_key} has {X.shape[0]} cached embeddings "
                      f"but {len(labeled)} labeled entries — skipping (stale cache)")
                continue

            print(f"\n  --- {enc_key} ({X.shape[1]}d) ---")

            for c_val in C_VALUES:
                for prob_label, ap_prior in PROB_MODES:
                    m = evaluate_combo(X, labeled, c_val, ap_prior)
                    row = {
                        "profile": pid,
                        "encoder": enc_key,
                        "C": c_val,
                        "prob_mode": prob_label,
                        "apply_prior": ap_prior,
                        **m,
                    }
                    results.append(row)
                    print(f"    C={c_val:<6} {prob_label}  F1={m['f1_score']:<7} "
                          f"p@30={m['precision_at_30']:<7} lr@30={m['lead_recall_at_30']:<7} "
                          f"AUC={m['ranking_auc']:<7} T={m['calibration_temperature']}")

    # Save
    out_path = BASE_DIR / "lab_data" / "e5_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary: best configs per desk
    print(f"\n{'='*90}")
    print("BEST CONFIGS PER DESK (by p@30, then lr@30, then F1)")
    print(f"{'='*90}")

    df = pd.DataFrame(results)
    for pid in sorted(df["profile"].unique()):
        sub = df[df["profile"] == pid].copy()
        # Composite sort: p@30 first, then lr@30, then F1
        sub = sub.sort_values(
            ["precision_at_30", "lead_recall_at_30", "f1_score"],
            ascending=False,
        )
        print(f"\n  Profile {pid} — top {args.top or len(sub)}:")
        print(f"  {'Encoder':<22} {'C':>6} {'Mode':<5} {'F1':>7} {'p@30':>7} {'lr@30':>7} {'AUC':>7} {'T':>7}")
        for _, r in sub.head(args.top or len(sub)).iterrows():
            print(f"  {r['encoder']:<22} {r['C']:>6} {r['prob_mode']:<5} "
                  f"{r['f1_score']:>7} {r['precision_at_30']:>7} {r['lead_recall_at_30']:>7} "
                  f"{r['ranking_auc']:>7} {r['calibration_temperature']:>7}")

    # Cross-desk comparison: best shared recipe
    print(f"\n{'='*90}")
    print("BEST SHARED RECIPE (same C + prob_mode across all desks)")
    print(f"{'='*90}")

    # For each (encoder, C, prob_mode), compute average p@30 and lr@30 across desks
    shared = []
    for enc_key in args.encoders:
        for c_val in C_VALUES:
            for prob_label, ap_prior in PROB_MODES:
                rows = df[
                    (df["encoder"] == enc_key) &
                    (df["C"] == c_val) &
                    (df["prob_mode"] == prob_label)
                ]
                if len(rows) < len(args.profile):
                    continue
                avg_p30 = rows["precision_at_30"].mean()
                avg_lr30 = rows["lead_recall_at_30"].mean()
                avg_f1 = rows["f1_score"].mean()
                avg_auc = rows["ranking_auc"].mean()
                min_p30 = rows["precision_at_30"].min()
                min_lr30 = rows["lead_recall_at_30"].min()
                shared.append({
                    "encoder": enc_key,
                    "C": c_val,
                    "prob_mode": prob_label,
                    "avg_p30": round(avg_p30, 4),
                    "avg_lr30": round(avg_lr30, 4),
                    "avg_f1": round(avg_f1, 4),
                    "avg_auc": round(avg_auc, 4),
                    "min_p30": round(min_p30, 4),
                    "min_lr30": round(min_lr30, 4),
                })

    shared_df = pd.DataFrame(shared)
    # Sort by average p@30, then average lr@30
    shared_df = shared_df.sort_values(["avg_p30", "avg_lr30", "avg_f1"], ascending=False)

    print(f"\n  {'Encoder':<22} {'C':>6} {'Mode':<5} {'avg_p30':>8} {'avg_lr30':>8} "
          f"{'avg_f1':>8} {'avg_auc':>8} {'min_p30':>8} {'min_lr30':>8}")
    for _, r in shared_df.head(15).iterrows():
        print(f"  {r['encoder']:<22} {r['C']:>6} {r['prob_mode']:<5} "
              f"{r['avg_p30']:>8} {r['avg_lr30']:>8} {r['avg_f1']:>8} "
              f"{r['avg_auc']:>8} {r['min_p30']:>8} {r['min_lr30']:>8}")

    # Current production reference
    print(f"\n  Current production: e5_base, C=0.01, P1")
    prod = df[(df["encoder"] == "e5_base") & (df["C"] == 0.01) & (df["prob_mode"] == "P1")]
    if not prod.empty:
        for _, r in prod.iterrows():
            print(f"    p{int(r['profile'])}: F1={r['f1_score']} p@30={r['precision_at_30']} "
                  f"lr@30={r['lead_recall_at_30']} AUC={r['ranking_auc']}")

    print()


if __name__ == "__main__":
    main()
