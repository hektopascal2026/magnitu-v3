#!/usr/bin/env python3
"""Re-run e5-large-instruct with mean_norm (L2 normalization) vs e5-base mean_norm.

The original encoder comparison tested e5-large with raw mean pooling (l2_final=False).
Production e5-base now uses mean_norm (L2-normalized mean pooling). This script re-runs
both encoders with mean_norm using the cached embeddings, so the comparison is fair:
  e5-base mean_norm  vs  e5-large mean_norm

Also tests a C sweep [0.005, 0.01, 0.02, 0.05] and P1 (temperature-only) calibration,
matching the production configuration.

Uses cached embeddings from lab_data/encoder_comparison_cache/ — no re-encoding needed.
Same seed=42, same StratifiedShuffleSplit, same sample weights as the original.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_large_norm_rerun.py
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import db
from pipeline import (
    CLASSES,
    CLASS_WEIGHT_MAP,
    compute_sample_weights,
    _fit_temperature_scalar,
    _collect_oof_logits,
    _ranking_metrics,
    _oof_fold_count,
)
from config import get_config

SEED = 42
C_VALUES = [0.005, 0.01, 0.02, 0.05]
PROFILES = {2: "digital", 3: "sicherheit", 4: "eu"}


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def load_cached(model_key: str, profile_id: int) -> np.ndarray:
    p = Path("lab_data/encoder_comparison_cache") / f"{model_key}_p{profile_id}.npy"
    return np.load(p)


def load_labeled(profile_id: int) -> List[dict]:
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


def evaluate(X: np.ndarray, labeled: List[dict], C: float, seed: int = 42) -> dict:
    y = [l["label"] for l in labeled]
    n = len(y)
    min_class = int(pd.Series(y).value_counts().min())
    n_folds = min(5, max(2, min_class // 2)) if min_class >= 4 else 2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    labeled_train = [labeled[i] for i in train_idx]

    sw = compute_sample_weights(labeled_train)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
            multi_class="multinomial",
        )),
    ])
    clf.fit(X_train, y_train, clf__sample_weight=sw)

    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    class_names = clf.named_steps["clf"].classes_

    f1 = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    rank = _ranking_metrics(probs, class_names, y_test, k=30)

    # OOF temperature (P1)
    try:
        oof_logits = _collect_oof_logits(
            X_train, y_train,
            n_folds=n_folds, seed=seed,
            C=C, sample_weight=sw,
        )
        if oof_logits is not None:
            temperature = _fit_temperature_scalar(oof_logits, y_train, class_names)
        else:
            temperature = 1.0
    except Exception:
        temperature = 1.0

    # Per-class F1
    report = classification_report(
        y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

    # Survival at production threshold 0.55
    w = np.array([CLASS_WEIGHT_MAP.get(cn, 0.0) for cn in class_names], dtype=float)
    composite = probs.dot(w)
    threshold = 0.55
    y_te_arr = np.array(y_test)
    leads = y_te_arr == "investigation_lead"
    importants = y_te_arr == "important"
    above = composite >= threshold
    lead_surv = float(above[leads].sum()) / max(int(leads.sum()), 1) if leads.sum() > 0 else 0.0
    imp_surv = float(above[importants].sum()) / max(int(importants.sum()), 1) if importants.sum() > 0 else 0.0
    relevant = leads | importants
    n_above = int(above.sum())
    prec_at_t = float(above[relevant].sum()) / n_above if n_above > 0 else 0.0

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "C": C,
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "calibration_temperature": round(temperature, 4),
        "lead_survival_055": round(lead_surv, 4),
        "important_survival_055": round(imp_surv, 4),
        "threshold_precision_055": round(prec_at_t, 4),
        "n_above_055": n_above,
        "class_distribution": dict(pd.Series(y_test).value_counts().to_dict()),
        "per_class": {
            cls: {
                "precision": round(report.get(cls, {}).get("precision", 0), 4),
                "recall": round(report.get(cls, {}).get("recall", 0), 4),
                "f1": round(report.get(cls, {}).get("f1-score", 0), 4),
                "support": int(report.get(cls, {}).get("support", 0)),
            }
            for cls in CLASSES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--c-values", type=float, nargs="+", default=C_VALUES)
    args = parser.parse_args()

    config = get_config()

    results = {}
    for pid in args.profiles:
        desk = PROFILES.get(pid, f"p{pid}")
        print(f"\n{'='*70}")
        print(f"Profile {pid} ({desk})")
        print(f"{'='*70}")

        labeled = load_labeled(pid)
        if len(labeled) < 20:
            print(f"  Skipping: only {len(labeled)} labeled entries")
            continue
        print(f"  {len(labeled)} labeled entries")

        for model_key in ["e5_base", "e5_large_instruct"]:
            try:
                X_raw = load_cached(model_key, pid)
            except FileNotFoundError:
                print(f"  {model_key}: no cached embeddings, skipping")
                continue

            if X_raw.shape[0] != len(labeled):
                print(f"  {model_key}: cached shape {X_raw.shape[0]} != {len(labeled)} labels, skipping")
                continue

            # Apply mean_norm (L2 normalization) — this is the fix
            X_norm = l2_normalize(X_raw)

            print(f"\n  {model_key} (mean_norm, dim={X_norm.shape[1]}):")
            print(f"    {'C':>8} {'F1':>7} {'Acc':>7} {'p@30':>7} {'LR@30':>7} {'AUC':>7} {'LeadS':>7} {'ImpS':>7} {'ThrP':>7} {'Temp':>7}")
            print("    " + "-" * 80)

            model_results = []
            for C in args.c_values:
                m = evaluate(X_norm, labeled, C=C, seed=SEED)
                model_results.append(m)
                print(f"    {C:>8.3f} {m['f1_score']:>7.4f} {m['accuracy']:>7.4f} {m['precision_at_30']:>7.4f} {m['lead_recall_at_30']:>7.4f} {m['ranking_auc']:>7.4f} {m['lead_survival_055']:>7.4f} {m['important_survival_055']:>7.4f} {m['threshold_precision_055']:>7.4f} {m['calibration_temperature']:>7.4f}")

            # Also test raw (no L2) at C=0.01 for comparison
            m_raw = evaluate(X_raw, labeled, C=0.01, seed=SEED)
            print(f"    {'raw':>8} {m_raw['f1_score']:>7.4f} {m_raw['accuracy']:>7.4f} {m_raw['precision_at_30']:>7.4f} {m_raw['lead_recall_at_30']:>7.4f} {m_raw['ranking_auc']:>7.4f} {m_raw['lead_survival_055']:>7.4f} {m_raw['important_survival_055']:>7.4f} {m_raw['threshold_precision_055']:>7.4f} {m_raw['calibration_temperature']:>7.4f}  (C=0.01, no L2)")

            # Best C by F1
            best = max(model_results, key=lambda r: r["f1_score"])
            print(f"    Best C={best['C']} (F1={best['f1_score']}, p@30={best['precision_at_30']}, LR@30={best['lead_recall_at_30']})")

            # Per-class for best
            print(f"    Per-class (best C={best['C']}):")
            for cls in CLASSES:
                pc = best["per_class"].get(cls, {})
                print(f"      {cls:20s} P={pc.get('precision',0):.4f} R={pc.get('recall',0):.4f} F1={pc.get('f1',0):.4f} n={pc.get('support',0)}")

            results.setdefault(desk, {})[model_key] = {
                "mean_norm": model_results,
                "raw_c001": m_raw,
            }

    # Save
    out_path = Path("lab_e5_large_norm_rerun_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: e5-base mean_norm vs e5-large mean_norm (best C per desk)")
    print(f"{'='*70}")
    print(f"  {'Desk':>12} {'Model':>20} {'C':>8} {'F1':>7} {'p@30':>7} {'LR@30':>7} {'AUC':>7} {'LeadS':>7} {'ImpS':>7}")
    print("  " + "-" * 80)
    for desk in sorted(results.keys()):
        for model_key in ["e5_base", "e5_large_instruct"]:
            if model_key not in results[desk]:
                continue
            best = max(results[desk][model_key]["mean_norm"], key=lambda r: r["f1_score"])
            print(f"  {desk:>12} {model_key:>20} {best['C']:>8.3f} {best['f1_score']:>7.4f} {best['precision_at_30']:>7.4f} {best['lead_recall_at_30']:>7.4f} {best['ranking_auc']:>7.4f} {best['lead_survival_055']:>7.4f} {best['important_survival_055']:>7.4f}")


if __name__ == "__main__":
    main()
