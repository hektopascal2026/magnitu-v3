#!/usr/bin/env python3
"""Phase 0 head ablation: LogReg vs MLP vs XGBoost on cached e5 vectors.

Answers the audit's open question #4: "Is e5-base actually good, or just less
bad? The bottleneck may be the LogReg head itself."

Uses the **exact same** pipeline as _train_transformer():
  - Same stable entry_key hash holdout split
  - Same sample weights (time decay, reasoning boost, synthetic downweight)
  - Same OOF temperature calibration
  - Same prior correction (empirical label distribution)
  - Same ranking metrics (p@30, lead_recall@30, ranking_auc, macro-F1)
  - Same promote gate (evaluate_model_update)

No writes to the DB, no model files saved, no pushes to Seismo.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_head_ablation.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_head_ablation.py --profile 4
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_head_ablation.py --all
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Ensure we can import magnitu-v3 modules
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import db
import pipeline as pipe
from pipeline import (
    CLASSES,
    CLASS_WEIGHT_MAP,
    _stable_split_bucket,
    _holdout_test_fraction,
    _stable_train_test_split,
    compute_sample_weights,
    build_prior_fit,
    _prior_offset_vector,
    _add_logit_offsets,
    _fit_temperature_scalar,
    _collect_oof_logits,
    _oof_fold_count,
    _min_class_count_in_labels,
    _holdout_classification_metrics,
    _ranking_metrics,
    _softmax_rows,
    logits_for_classifier_head,
    bytes_to_embedding,
    _LabelDecodingClassifier,
    build_transformer_head_pipeline,
)
from ml_window import evaluate_model_update


# ── Head definitions ────────────────────────────────────────────────

def logreg_head() -> Pipeline:
    """Incumbent: StandardScaler + balanced LogReg C=1.0 (identical to pipeline)."""
    return build_transformer_head_pipeline()


def mlp_head() -> Pipeline:
    """MLP: StandardScaler + single hidden layer (256), L2 reg, early stopping."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=0.01,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def xgboost_head() -> Pipeline:
    """XGBoost: gradient-boosted trees on scaled embeddings."""
    from xgboost import XGBClassifier
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )),
    ])


def logreg_c01_head() -> Pipeline:
    """LogReg with stronger regularization (C=0.1)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        )),
    ])


def logreg_c10_head() -> Pipeline:
    """LogReg with weaker regularization (C=10)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=10.0, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        )),
    ])


def svm_rbf_head() -> Pipeline:
    """SVM with RBF kernel — classic for small-data nonlinear classification."""
    from sklearn.svm import SVC
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])


def svm_linear_head() -> Pipeline:
    """Linear SVM — simpler linear baseline, may be more stable than LogReg."""
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42, dual="auto")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", CalibratedClassifierCV(svc, cv=3)),
    ])


def mlp_128_head() -> Pipeline:
    """MLP with smaller hidden layer (128) — fewer params, less overfit risk."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=0.01,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def mlp_512_head() -> Pipeline:
    """MLP with larger hidden layer (512) + stronger L2 — more capacity, more reg."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="relu",
            alpha=0.05,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def mlp_2layer_head() -> Pipeline:
    """MLP with two hidden layers (256, 128) — deeper, may capture interactions."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=0.02,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def lightgbm_head() -> Pipeline:
    """LightGBM — typically better regularized than XGBoost on small data."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None  # will be filtered out
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])


def ridge_head() -> Pipeline:
    """Ridge classifier — simplest linear baseline, very low overfit risk."""
    from sklearn.linear_model import RidgeClassifier
    # RidgeClassifier has no predict_proba — wrap with CalibratedClassifierCV
    from sklearn.calibration import CalibratedClassifierCV
    rc = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", CalibratedClassifierCV(rc, cv=3)),
    ])


def knn_head() -> Pipeline:
    """k-NN in embedding space — distance-based, may exploit e5 geometry directly."""
    from sklearn.neighbors import KNeighborsClassifier
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="cosine",
        )),
    ])


# Build HEADS dict, filtering out heads that require unavailable libs
def _build_heads():
    candidates = [
        ("logreg", ("LogReg C=1.0 (incumbent)", logreg_head)),
        ("logreg_c01", ("LogReg C=0.1", logreg_c01_head)),
        ("logreg_c10", ("LogReg C=10", logreg_c10_head)),
        ("svm_rbf", ("SVM RBF", svm_rbf_head)),
        ("svm_linear", ("LinearSVC", svm_linear_head)),
        ("mlp_128", ("MLP-128", mlp_128_head)),
        ("mlp_256", ("MLP-256", mlp_head)),
        ("mlp_512", ("MLP-512", mlp_512_head)),
        ("mlp_2layer", ("MLP-256x128", mlp_2layer_head)),
        ("xgboost", ("XGBoost", xgboost_head)),
        ("lightgbm", ("LightGBM", lightgbm_head)),
        ("ridge", ("Ridge", ridge_head)),
        ("knn", ("kNN-15-cos", knn_head)),
    ]
    heads = {}
    for key, (label, fn) in candidates:
        try:
            p = fn()
            if p is not None:
                heads[key] = (label, fn)
        except ImportError:
            pass
    return heads


HEADS = _build_heads()


# ── Per-head OOF calibration ────────────────────────────────────────

def _collect_oof_logits_per_head(
    head_name: str,
    X: np.ndarray,
    y_list: List[str],
    sample_weight,
    label_encoder,
    n_folds: int,
) -> Tuple[np.ndarray, List[str]]:
    """Out-of-fold logits using the ACTUAL head being evaluated, not LogReg.

    This fixes the calibration inequality flagged in the granite encoder audit:
    the original _collect_oof_logits uses build_transformer_head_pipeline()
    (LogReg) internally, so all heads got LogReg's temperature. Each head should
    get its own temperature fitted on its own OOF logits.
    """
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

    head_fn = HEADS[head_name][1]
    step_names = list(head_fn().named_steps.keys())
    final_step = step_names[-1]

    try:
        for train_idx, val_idx in split_iter:
            pipe_head = head_fn()

            # Route sample weights to the right step
            if head_name.startswith("mlp"):
                pipe_head.fit(X[train_idx], y_enc[train_idx])
            elif final_step == "classifier" and sw_arr is not None:
                pipe_head.fit(X[train_idx], y_enc[train_idx],
                              **{f"{final_step}__sample_weight": sw_arr[train_idx]})
            else:
                pipe_head.fit(X[train_idx], y_enc[train_idx])

            logits_val = logits_for_classifier_head(pipe_head, X[val_idx])
            for j, vi in enumerate(val_idx):
                oof_logits[vi] = logits_val[j]
                oof_y[vi] = y_list[vi]
    except (ValueError, TypeError):
        return np.array([]), []

    valid_idx = [i for i in range(n_samples) if oof_logits[i] is not None]
    if not valid_idx:
        return np.array([]), []
    return (
        np.array([oof_logits[i] for i in valid_idx]),
        [oof_y[i] for i in valid_idx],
    )


# ── Training + eval (mirrors _train_transformer exactly) ────────────

def train_and_eval_head(
    head_name: str,
    X_train: np.ndarray,
    y_train: List[str],
    X_test: np.ndarray,
    y_test: List[str],
    sw_train: np.ndarray,
    config: dict,
) -> dict:
    """Train one head with the same calibration + prior path as the pipeline."""

    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    class_names_fit = le.classes_.tolist()

    # OOF temperature calibration (same as pipeline)
    train_min_class = _min_class_count_in_labels(y_train)
    n_folds = _oof_fold_count(len(y_train), train_min_class)
    prior_fit = build_prior_fit(y_train, sw_train, class_names_fit, config)

    oof_samples = 0
    if n_folds >= 2:
        # Per-head OOF calibration (fixes audit issue: each head gets its own
        # temperature fitted on its own OOF logits, not LogReg's)
        oof_logits, oof_y = _collect_oof_logits_per_head(
            head_name, X_train, y_train, sw_train, le, n_folds,
        )
        oof_samples = len(oof_y)
        if oof_samples >= 3:
            off = _prior_offset_vector({"prior_fit": prior_fit}, class_names_fit)
            if off is not None:
                oof_logits = _add_logit_offsets(oof_logits, off)
            temperature = _fit_temperature_scalar(
                oof_logits, np.array(oof_y), class_names_fit
            )
        else:
            temperature = 1.0
    else:
        temperature = 1.0

    cal_dict = {
        "version": 2,
        "method": "temperature",
        "calibration_fit": "oof" if oof_samples >= 3 else "none",
        "oof_folds": n_folds if oof_samples >= 3 else 0,
        "oof_samples": oof_samples,
        "temperature": temperature,
        "class_names": class_names_fit,
        "prior_fit": prior_fit,
    }

    # Build + fit head
    head_fn = HEADS[head_name][1]
    clf_pipeline = head_fn()

    # Determine the final step name for sample_weight routing
    step_names = list(clf_pipeline.named_steps.keys())
    final_step = step_names[-1]  # "classifier", "mlp", etc.

    # Heads that accept sample_weight in fit():
    sw_capable = {"classifier"}  # LogReg, XGB, LGBM, SVC all use "classifier"
    # MLP and kNN don't accept sample_weight
    # CalibratedClassifierCV (ridge, svm_linear) doesn't accept sample_weight
    sw_ignored = {"mlp"}  # MLP steps

    has_sw = (sw_train is not None and len(sw_train) > 0
              and float(np.std(sw_train)) > 1e-6)

    if final_step in sw_ignored or final_step == "mlp":
        # MLP / kNN: no sample_weight, rely on prior correction
        clf_pipeline.fit(X_train, y_train_enc)
    elif has_sw:
        try:
            clf_pipeline.fit(X_train, y_train_enc,
                             **{f"{final_step}__sample_weight": sw_train})
        except (TypeError, ValueError):
            # CalibratedClassifierCV or other wrapper that rejects sample_weight
            clf_pipeline.fit(X_train, y_train_enc)
    else:
        clf_pipeline.fit(X_train, y_train_enc)

    clf = _LabelDecodingClassifier(clf_pipeline, le)

    # Evaluate with calibration + prior (same as classifier_probabilities)
    probs_test, cn = pipe.classifier_probabilities(clf, X_test, "", cal=cal_dict)

    # Classification metrics
    y_pred_idx = np.argmax(probs_test, axis=1)
    y_pred = np.array([cn[i] for i in y_pred_idx])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Ranking metrics (p@30, lead_recall@30, AUC)
    rank = _ranking_metrics(probs_test, cn, y_test)

    # Survival metrics at production alert threshold (0.55)
    w = np.array([CLASS_WEIGHT_MAP.get(cn[i], 0.0) for i in range(len(cn))], dtype=float)
    composite = probs_test.dot(w)
    threshold = 0.55
    y_te_arr = np.array(y_test)
    leads = y_te_arr == "investigation_lead"
    importants = y_te_arr == "important"
    above = composite >= threshold
    lead_survival = float(above[leads].sum()) / max(int(leads.sum()), 1)
    important_survival = float(above[importants].sum()) / max(int(importants.sum()), 1)
    relevant = leads | importants
    thresh_precision = float(relevant[above].sum()) / max(int(above.sum()), 1)
    n_above = int(above.sum())

    return {
        "head": head_name,
        "head_label": HEADS[head_name][0],
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_note": rank["ranking_note"],
        "temperature": round(temperature, 4),
        "oof_samples": oof_samples,
        "n_folds": n_folds,
        "survival_threshold": threshold,
        "lead_survival": round(lead_survival, 4),
        "important_survival": round(important_survival, 4),
        "threshold_precision": round(thresh_precision, 4),
        "n_above_threshold": n_above,
        "n_leads_holdout": int(leads.sum()),
        "n_importants_holdout": int(importants.sum()),
    }


def run_ablation(profile_id: int, config: dict) -> dict:
    """Run head ablation for one profile."""

    labeled = db.get_all_labels(profile_id)
    if len(labeled) < 20:
        return {"profile_id": profile_id, "error": f"only {len(labeled)} labels"}

    # Load embeddings (same as _train_transformer)
    embedding_dim = config.get("embedding_dim", 768)
    conn = db.get_db()
    emb_map = {}
    rows = conn.execute(
        "SELECT entry_type, entry_id, embedding FROM entries WHERE embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        emb_map[db.entry_key_from_mapping(row)] = row["embedding"]
    conn.close()

    X_list, y_list, lbl_list = [], [], []
    skipped = 0
    for lbl in labeled:
        key = db.entry_key_from_mapping(lbl)
        emb_bytes = emb_map.get(key)
        if not emb_bytes:
            skipped += 1
            continue
        X_list.append(bytes_to_embedding(emb_bytes, embedding_dim))
        y_list.append(lbl["label"])
        lbl_list.append(lbl)

    if len(X_list) < 20:
        return {"profile_id": profile_id, "error": f"only {len(X_list)} labeled embeddings ({skipped} skipped)"}

    X = np.array(X_list)
    y = y_list
    sw_all = compute_sample_weights(lbl_list, config)

    # Same stable split
    import pandas as pd
    min_class_count = int(pd.Series(y).value_counts().min())

    if min_class_count < 2:
        return {"profile_id": profile_id, "error": f"min class count < 2 ({min_class_count})"}

    test_size = _holdout_test_fraction(min_class_count, len(y))
    X_train, X_test, y_train, y_test, sw_train, _sw_test = _stable_train_test_split(
        X, y, sw_all, lbl_list, test_size=test_size
    )

    # Run all heads
    results = []
    incumbent_metrics = None
    for head_name in HEADS:
        print(f"  Training {HEADS[head_name][0]}...", flush=True)
        res = train_and_eval_head(head_name, X_train, y_train, X_test, y_test, sw_train, config)
        results.append(res)
        if head_name == "logreg":
            incumbent_metrics = res

    # Promote gate: does any challenger beat LogReg?
    gate_results = []
    for res in results:
        if res["head"] == "logreg":
            continue
        promoted = evaluate_model_update(incumbent_metrics, res)
        gate_results.append({
            "challenger": res["head_label"],
            "promote_gate": promoted,
            "p30_delta": round(res["precision_at_30"] - incumbent_metrics["precision_at_30"], 4),
            "f1_delta": round(res["f1_score"] - incumbent_metrics["f1_score"], 4),
            "lead_recall_delta": round(
                res["lead_recall_at_30"] - incumbent_metrics["lead_recall_at_30"], 4
            ),
        })

    # Class distribution
    label_dist = {k: int(v) for k, v in pd.Series(y).value_counts().items()}

    return {
        "profile_id": profile_id,
        "profile_name": db.get_profile_by_id(profile_id).get("display_name", str(profile_id)),
        "total_labels": len(labeled),
        "labeled_with_embeddings": len(X_list),
        "skipped_no_embedding": skipped,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "test_fraction": round(test_size, 4),
        "label_distribution": label_dist,
        "embedding_dim": embedding_dim,
        "results": results,
        "promote_gate": gate_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 0 head ablation on cached e5 vectors")
    parser.add_argument("--profile", type=int, default=None, help="Profile ID (default: all main profiles)")
    parser.add_argument("--all", action="store_true", help="Run all profiles")
    parser.add_argument("--json", type=str, default=None, help="Write results to JSON file")
    args = parser.parse_args()

    # Determine profiles to run
    if args.profile is not None:
        profiles = [args.profile]
    elif args.all:
        profiles = [1, 2, 3, 4]
    else:
        # Default: the 3 desks from the encoder audit
        profiles = [1, 3, 4]

    all_results = []
    for pid in profiles:
        profile = db.get_profile_by_id(pid)
        name = profile.get("display_name", str(pid)) if profile else str(pid)
        print(f"\n{'='*60}")
        print(f"Profile {pid}: {name}")
        print(f"{'='*60}")

        config = db.get_effective_config(pid)
        result = run_ablation(pid, config)
        all_results.append(result)

        if "error" in result:
            print(f"  SKIP: {result['error']}")
            continue

        # Print results table
        print(f"\n  Labels: {result['total_labels']} (train={result['train_size']}, test={result['test_size']})")
        print(f"  Distribution: {result['label_distribution']}")
        print()
        print(f"  {'Head':<20s} {'Acc':>6s} {'F1':>6s} {'p@30':>6s} {'LR@30':>6s} {'AUC':>6s} {'Temp':>6s} "
              f"{'leadS':>6s} {'impS':>6s} {'n>=':>4s} {'tPrec':>6s}")
        print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*6}")
        for r in result["results"]:
            print(f"  {r['head_label']:<20s} {r['accuracy']:>6.4f} {r['f1_score']:>6.4f} "
                  f"{r['precision_at_30']:>6.4f} {r['lead_recall_at_30']:>6.4f} "
                  f"{r['ranking_auc']:>6.4f} {r['temperature']:>6.3f} "
                  f"{r['lead_survival']:>6.2f} {r['important_survival']:>6.2f} "
                  f"{r['n_above_threshold']:>4d} {r['threshold_precision']:>6.2f}")

        print(f"\n  Promote gate (vs LogReg incumbent):")
        for g in result["promote_gate"]:
            verdict = "PROMOTE" if g["promote_gate"] else "reject"
            print(f"    {g['challenger']:<20s} {verdict:>8s}  "
                  f"Δp@30={g['p30_delta']:+.4f}  ΔF1={g['f1_delta']:+.4f}  "
                  f"ΔLR@30={g['lead_recall_delta']:+.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"  Profile {r['profile_id']}: {r['error']}")
            continue
        logreg = next(x for x in r["results"] if x["head"] == "logreg")
        best = max(r["results"], key=lambda x: x["precision_at_30"])
        print(f"  Profile {r['profile_id']} ({r['profile_name']}): "
              f"best p@30 = {best['head_label']} ({best['precision_at_30']:.4f}), "
              f"LogReg p@30 = {logreg['precision_at_30']:.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
