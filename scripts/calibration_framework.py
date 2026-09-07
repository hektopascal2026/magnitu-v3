#!/usr/bin/env python3
"""Calibration framework for the Magnitu evaluation protocol.

Implements the protocol in docs/calibration-for-real.md:
  - Per-configuration OOF calibration (exact head/C/weights, not hardcoded C=1.0)
  - Chronological splits with prediction_time and three disjoint partitions
  - Four probability modes (P0-P3) with correct prior/temperature ordering
  - Actual-weight prior correction (not assumed balanced for all heads)
  - Full metrics: classification, ranking, calibration, threshold survival
  - Promote gate replay with uncertainty

Python 3.9 compatible. No production writes, no Seismo sync.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

# ── Constants from production pipeline.py ────────────────────────────

CLASSES = ["investigation_lead", "important", "background", "noise"]
CLASS_WEIGHT_MAP = {
    "investigation_lead": 1.0,
    "important": 0.80,
    "background": 0.20,
    "noise": 0.0,
}
_RANKING_RELEVANT = {"investigation_lead", "important"}
ALERT_THRESHOLD = 0.55

# Gate constants from ml_window.py
PROMOTE_MARGIN = 0.01
PROMOTE_RANKING_SLACK = 0.05
PROMOTE_BIG_P30_WIN = 0.05
F1_HARD_DROP_LIMIT = 0.10
LEAD_RECALL_SLACK = 0.10

# Temperature search grid (matches pipeline.py _fit_temperature_scalar)
TEMP_GRID = np.geomspace(0.25, 12.0, num=40)
TEMP_BOUNDS = (0.25, 12.0)

# C sweep grid (matches lab_encoder_c_sweep.py)
C_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# Seeds for stochastic training
SEEDS = [13, 29, 42, 71, 101]


# ═══════════════════════════════════════════════════════════════════
#  1. Configuration registry
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Complete specification of one experiment cell."""
    encoder: str               # "e5_base", "granite", "e5_large_instruct"
    text_mode: str             # "e5_tuned", "plain", "plain_context", "plain_full", etc.
    context_mode: str          # "chunked", "single_pass_512", "single_pass_2048", etc.
    pooling: str               # "mean", "cls", "cls_norm", "mean_norm"
    head: str                  # "logreg", "mlp_256", "svm_rbf", etc.
    C: float                   # regularization (1.0 for non-C heads)
    seed: int                  # training seed
    profile_id: int            # desk/profile
    prob_mode: str = "P3"      # P0/P1/P2/P3

    def config_id(self) -> str:
        """Stable unique ID for this configuration."""
        parts = [
            self.encoder, self.text_mode, self.context_mode,
            self.pooling, self.head, f"C{self.C}", f"s{self.seed}",
            f"p{self.profile_id}", self.prob_mode,
        ]
        return "_".join(parts)

    def fingerprint(self) -> str:
        """SHA-256 fingerprint for cache keys."""
        h = hashlib.sha256()
        h.update(self.config_id().encode("utf-8"))
        return h.hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════
#  2. Head registry — each head instantiated with exact parameters
# ═══════════════════════════════════════════════════════════════════

def build_head(head_name: str, C: float = 1.0, seed: int = 42) -> Pipeline:
    """Build a head pipeline with the EXACT parameters being evaluated.

    This is the core fix for the C=1.0 calibration bug: the OOF collection
    must instantiate the same C value, not a hardcoded C=1.0.
    """
    if head_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C, class_weight="balanced", max_iter=1000,
                solver="lbfgs", random_state=seed,
            )),
        ])
    if head_name == "svm_rbf":
        from sklearn.svm import SVC
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=C, kernel="rbf", class_weight="balanced",
                probability=True, random_state=seed,
            )),
        ])
    if head_name == "svm_linear":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        svc = LinearSVC(C=C, class_weight="balanced", max_iter=5000,
                        random_state=seed, dual=False)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(svc, cv=3)),
        ])
    if head_name == "mlp_128":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128,), activation="relu", alpha=0.01,
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed,
            )),
        ])
    if head_name == "mlp_256":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256,), activation="relu", alpha=0.01,
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed,
            )),
        ])
    if head_name == "mlp_512":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(512,), activation="relu", alpha=0.05,
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed,
            )),
        ])
    if head_name == "mlp_2layer":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128), activation="relu", alpha=0.02,
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=seed,
            )),
        ])
    if head_name == "xgboost":
        from xgboost import XGBClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=seed, eval_metric="mlogloss",
                use_label_encoder=False,
            )),
        ])
    if head_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                class_weight="balanced", random_state=seed, verbose=-1,
            )),
        ])
    if head_name == "ridge":
        from sklearn.linear_model import RidgeClassifier
        from sklearn.calibration import CalibratedClassifierCV
        rc = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=seed)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(rc, cv=3)),
        ])
    if head_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=15, weights="distance", metric="cosine",
            )),
        ])
    raise ValueError(f"Unknown head: {head_name}")


# Heads that accept sample_weight via classifier__sample_weight
_WEIGHTABLE_HEADS = {"logreg", "svm_rbf"}
# Heads that accept sample_weight via mlp__sample_weight
_MLP_HEADS = {"mlp_128", "mlp_256", "mlp_512", "mlp_2layer"}
# Heads where C is not a parameter
_C_INSENSITIVE = {"mlp_128", "mlp_256", "mlp_512", "mlp_2layer",
                  "xgboost", "lightgbm", "ridge", "knn"}


def fit_head(
    head_name: str,
    C: float,
    seed: int,
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, str]:
    """Fit exact parameters; non-unit unsupported weights raise before fitting.

    Returns (pipeline, "accepted" or "uniform"). Only None or exactly unit
    weights are omitted. Runtime estimator errors are never retried.
    """
    X_train, y_train_enc = _validate_xy(X_train, y_train_enc)
    sw = _validate_weights(sample_weight, len(y_train_enc))
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    pipe = build_head(head_name, C=C, seed=seed)
    final_step = list(pipe.named_steps)[-1]
    estimator = pipe.named_steps[final_step]
    supports_weights = "sample_weight" in inspect.signature(estimator.fit).parameters
    use_weights = sw is not None and not np.all(sw == 1.0)
    if use_weights and not supports_weights:
        # CalibratedClassifierCV or wrapper rejects sample_weight
        raise ValueError(f"unsupported sample_weight for {head_name}; no unweighted retry")
    weight_status = "accepted" if use_weights else "uniform"
    kwargs = {f"{final_step}__sample_weight": sw} if use_weights else {}
    pipe.fit(X_train, y_train_enc, **kwargs)
    pipe.fit_metadata_ = {
        "weight_status": weight_status,
        "supports_sample_weight": supports_weights,
        "sample_weights": sw.tolist() if sw is not None else None,
        "scaler_weighting": "unweighted, fitted on training rows only",
        "probability_adapter": "clipped_log_predict_proba",
        "internal_probability_calibration": head_name in {"svm_rbf", "svm_linear", "ridge"},
        "early_stopping": head_name in _MLP_HEADS,
        "prior_correction_supported": head_name not in _MLP_HEADS | {"svm_rbf", "svm_linear", "ridge"},
    }
    return pipe, weight_status


def _validate_weights(weights, n):
    if weights is None:
        return None
    sw = np.asarray(weights, dtype=np.float64)
    if sw.shape != (n,) or not np.all(np.isfinite(sw)) or np.any(sw <= 0):
        raise ValueError("sample_weight must be aligned, finite and strictly positive")
    return sw


def _validate_xy(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X/y must be nonempty aligned 2D/1D arrays")
    if not np.all(np.isfinite(X)) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("X must be finite and labels integer encoded")
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))) or len(np.unique(y)) < 2:
        raise ValueError("training labels must cover contiguous classes starting at zero")
    return X, y


def _validate_groups(groups, n):
    if groups is None:
        return None
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError("story_group_ids must be globally aligned")
    for group in groups.tolist():
        if group is None or group != group or str(group).strip() == "":
            raise ValueError("story_group_ids must not contain missing values")
        hash(group)
    return groups


def _validate_indices(indices, n, name):
    idx = np.asarray(indices)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(idx) == 0:
        raise ValueError(f"{name} must be nonempty integer indices")
    if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} contains out-of-range or duplicate indices")
    return idx


def _auditable_prior(y, sw, head_name, class_names):
    if head_name in _MLP_HEADS | {"svm_rbf", "svm_linear", "ridge"}:
        raise ValueError(f"unsupported prior correction for {head_name}: internal calibration/early-stopping sampling is not audited")
    return build_prior_fit_actual(y, sw, head_name, class_names)


# ═══════════════════════════════════════════════════════════════════
#  3. Logits extraction — works for all head types
# ═══════════════════════════════════════════════════════════════════

def extract_logits(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Extract class-aligned logits from any head pipeline.

    For LogReg/LinearSVC: use decision_function (true pre-softmax logits,
    shape (n, n_classes) via OvR).
    For SVC(kernel="rbf", probability=True): decision_function returns OvO
    shape (n, n_classes*(n_classes-1)/2) which is NOT class-aligned, so we
    fall back to log(predict_proba) (Platt-scaled, but class-aligned).
    For MLP/XGBoost/LightGBM/kNN: use log(predict_proba) (logits up to
    a per-row constant, which softmax normalizes away).
    For CalibratedClassifierCV: use log(predict_proba) directly.
    """
    steps = pipe.named_steps
    if "scaler" in steps:
        X_scaled = steps["scaler"].transform(X)
    else:
        X_scaled = X

    final_name = list(steps.keys())[-1]
    step = steps[final_name]

    # Determine expected number of classes from predict_proba if available
    n_classes = None
    if hasattr(step, "predict_proba"):
        n_classes = step.predict_proba(X_scaled[:1]).shape[1]

    # Try decision_function first (true logits for linear models)
    if hasattr(step, "decision_function"):
        try:
            logits = step.decision_function(X_scaled)
            logits = np.asarray(logits, dtype=np.float64)
            # Binary case: (n,) → (n, 2) via stacking
            if logits.ndim == 1:
                logits = np.column_stack([-logits, logits])
            # Verify class alignment: columns must match n_classes.
            # SVC OvO decision_function returns n_classes*(n_classes-1)/2
            # columns, which is NOT class-aligned — fall through to log(proba).
            if n_classes is not None and logits.shape[1] != n_classes:
                pass  # fall through to log(predict_proba)
            elif n_classes is not None and hasattr(step, "predict_proba"):
                # Check if softmax(decision_function) matches predict_proba.
                # For LogReg they match (predict_proba IS softmax(decision_function)).
                # For SVC(probability=True) they don't (Platt scaling is applied
                # separately), so decision_function is not a true pre-softmax logit.
                n_check = min(5, logits.shape[0])
                df_softmax = _softmax_rows(logits[:n_check])
                proba_check = np.asarray(
                    step.predict_proba(X_scaled[:n_check]), dtype=np.float64
                )
                if np.allclose(df_softmax, proba_check, atol=1e-4):
                    return logits  # true logits
                # else: fall through to log(predict_proba)
            else:
                return logits  # no predict_proba to compare, assume true logits
        except (AttributeError, NotImplementedError):
            pass

    # Fall back to log(predict_proba)
    if hasattr(step, "predict_proba"):
        probs = step.predict_proba(X_scaled)
        probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
        return np.log(probs)

    raise ValueError(f"Cannot extract logits from {type(step)}")


# ═══════════════════════════════════════════════════════════════════
#  4. Prior correction — actual-weight, not assumed balanced
# ═══════════════════════════════════════════════════════════════════

def actual_fit_weights(
    y_enc: np.ndarray,
    sample_weight: Optional[np.ndarray],
    head_name: str,
) -> np.ndarray:
    """Compute the actual per-row weights used by the head during fit.

    For heads with class_weight='balanced' (logreg, svm_rbf, ridge):
      effective_weight = balanced_class_weight * sample_weight

    For heads WITHOUT class_weight (mlp, xgboost, lightgbm, knn):
      effective_weight = sample_weight (if accepted) or uniform

    This is the core fix for the prior-weight assumption bug: we cannot
    assume balanced class_weight for heads that don't use it.
    """
    y_arr = np.asarray(y_enc)
    n = len(y_arr)
    if n == 0:
        return np.array([])

    # Heads that use class_weight='balanced'
    if head_name in {"logreg", "svm_rbf", "svm_linear", "ridge"}:
        balanced = np.asarray(
            compute_sample_weight("balanced", y_arr), dtype=np.float64
        )
    elif head_name == "lightgbm":
        # LightGBM has class_weight='balanced' in our config
        balanced = np.asarray(
            compute_sample_weight("balanced", y_arr), dtype=np.float64
        )
    else:
        # MLP, XGBoost, kNN: no class_weight balancing
        balanced = np.ones(n, dtype=np.float64)

    if sample_weight is not None and len(sample_weight) == n:
        sw = np.asarray(sample_weight, dtype=np.float64)
        return balanced * sw
    return balanced


def build_prior_fit_actual(
    y_enc: np.ndarray,
    sample_weight: Optional[np.ndarray],
    head_name: str,
    class_names: List[str],
) -> Dict[str, Any]:
    """Build prior correction using ACTUAL fit weights for the head.

    Key fix: uses actual_fit_weights() which checks whether the head
    actually uses class_weight='balanced', rather than always assuming it.
    """
    y_arr = np.asarray(y_enc)
    n = int(y_arr.shape[0])
    if n == 0:
        return {"offsets": None, "effective_priors": None, "target_priors": None}

    floor = 0.5 / float(max(n, 1))

    # Target priors: unweighted class proportions (labeled empirical distribution)
    counts = {}
    for i, c in enumerate(class_names):
        counts[c] = int(np.sum(y_arr == i))
    target_raw = {c: counts[c] / float(max(n, 1)) for c in class_names}
    target = _renormalize_priors(target_raw, class_names, floor)

    # Effective priors: from ACTUAL fit weights (not assumed balanced)
    weights = actual_fit_weights(y_arr, sample_weight, head_name)
    w_c = {}
    for i, c in enumerate(class_names):
        w_c[c] = float(np.sum(weights[y_arr == i]))
    w_tot = float(sum(w_c.values()))
    if w_tot > 0:
        effective_raw = {c: w_c[c] / w_tot for c in class_names}
    else:
        effective_raw = {c: 1.0 / len(class_names) for c in class_names}
    effective = _renormalize_priors(effective_raw, class_names, floor)

    offsets = np.array([
        float(np.log(target[c]) - np.log(effective[c]))
        for c in class_names
    ], dtype=np.float64)

    return {
        "offsets": offsets,
        "effective_priors": effective,
        "target_priors": target,
        "base_rate": float(sum(
            target[c] * CLASS_WEIGHT_MAP.get(c, 0.0) for c in class_names
        )),
    }


def _renormalize_priors(
    raw: Dict[str, float], class_names: List[str], floor: float
) -> Dict[str, float]:
    out = {c: max(float(raw.get(c, 0.0)), floor) for c in class_names}
    z = float(sum(out.values()))
    if z <= 0.0:
        u = 1.0 / float(max(len(class_names), 1))
        return {c: u for c in class_names}
    return {c: out[c] / z for c in class_names}


# ═══════════════════════════════════════════════════════════════════
#  5. Per-configuration OOF calibration
# ═══════════════════════════════════════════════════════════════════

def collect_oof_logits_per_config(
    head_name: str,
    C: float,
    seed: int,
    X: np.ndarray,
    y_enc: np.ndarray,
    sample_weight: Optional[np.ndarray],
    n_folds: int,
    story_group_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF logits using the EXACT head/C/seed being evaluated.

    This is the core fix for the calibration bug: _collect_oof_logits in
    pipeline.py hardcodes build_transformer_head_pipeline() (C=1.0 LogReg).
    This function instantiates the exact configuration.

    When story_group_ids is provided, uses StratifiedGroupKFold to ensure
    duplicate/syndicated stories don't leak across OOF train/validation folds.

    Returns (oof_logits, oof_y_enc) where each row appears exactly once
    as OOF validation.
    """
    n = len(y_enc)
    if n == 0 or n_folds < 2:
        return np.array([]), np.array([])

    oof_logits = [None] * n
    oof_y = [None] * n

    if story_group_ids is not None and len(story_group_ids) == n:
        from sklearn.model_selection import StratifiedGroupKFold
        sgf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        try:
            splits = sgf.split(X, y_enc, groups=story_group_ids)
        except ValueError:
            return np.array([]), np.array([])
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        try:
            splits = skf.split(X, y_enc)
        except ValueError:
            return np.array([]), np.array([])

    for train_idx, val_idx in splits:
        sw_fold = None
        if sample_weight is not None and len(sample_weight) == n:
            sw_fold = np.asarray(sample_weight, dtype=np.float64)[train_idx]

        pipe, _sw_status = fit_head(
            head_name, C, seed,
            X[train_idx], y_enc[train_idx], sw_fold,
        )
        logits_val = extract_logits(pipe, X[val_idx])
        for j, vi in enumerate(val_idx):
            oof_logits[vi] = logits_val[j]
            oof_y[vi] = y_enc[vi]

    valid = [i for i in range(n) if oof_logits[i] is not None]
    if not valid:
        return np.array([]), np.array([])
    return (
        np.array([oof_logits[i] for i in valid]),
        np.array([oof_y[i] for i in valid]),
    )


def fit_temperature(
    logits: np.ndarray,
    y_enc: np.ndarray,
) -> Tuple[float, float]:
    """Fit scalar temperature by OOF NLL minimization.

    Returns (temperature, nll_at_optimum).
    Falls back to (1.0, inf) when data is degenerate.
    """
    if logits is None or len(logits) == 0:
        return 1.0, float("inf")
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    y = np.asarray(y_enc, dtype=np.int64)

    best_t, best_nll = 1.0, float("inf")
    for t in TEMP_GRID:
        probs = _softmax_rows(logits / float(t))
        p_true = probs[np.arange(len(y)), y]
        nll = -float(np.mean(np.log(np.clip(p_true, 1e-9, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, best_nll


# ═══════════════════════════════════════════════════════════════════
#  6. Probability modes P0-P3
# ═══════════════════════════════════════════════════════════════════

def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_prob_mode(
    mode: str,
    raw_logits: np.ndarray,
    temperature: float,
    prior_offsets: Optional[np.ndarray],
) -> np.ndarray:
    """Compute probabilities for one of the four modes.

    P0: No temperature, no prior offset (raw predict_proba or softmax(logits))
    P1: Temperature only: softmax(logits / T)
    P2: Prior adjustment only: softmax(logits + offsets)
    P3: Prior then temperature: softmax((logits + offsets) / T)

    The order for P3 matches pipeline.py classifier_probabilities():
    offsets applied first, then temperature division.
    """
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    if mode == "P0":
        return _softmax_rows(logits)
    elif mode == "P1":
        t = max(float(temperature), 1e-3)
        return _softmax_rows(logits / t)
    elif mode == "P2":
        if prior_offsets is not None:
            off = np.asarray(prior_offsets, dtype=np.float64)
            if logits.shape[1] == off.shape[0]:
                logits = logits + off.reshape(1, -1)
        return _softmax_rows(logits)
    elif mode == "P3":
        if prior_offsets is not None:
            off = np.asarray(prior_offsets, dtype=np.float64)
            if logits.shape[1] == off.shape[0]:
                logits = logits + off.reshape(1, -1)
        t = max(float(temperature), 1e-3)
        return _softmax_rows(logits / t)
    else:
        raise ValueError(f"Unknown probability mode: {mode}")


# ═══════════════════════════════════════════════════════════════════
#  7. Chronological splitting
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChronologicalSplits:
    """Three disjoint partitions + test windows + embargo gap."""
    fit_idx: np.ndarray
    select_idx: np.ndarray
    threshold_idx: np.ndarray
    test_windows: List[np.ndarray]  # three 14-day windows
    embargo_idx: np.ndarray         # rows in embargo gaps (intentionally excluded)
    prediction_times: np.ndarray
    window_boundaries: List[Tuple[float, float]]
    n_total: int
    n_fit: int
    n_select: int
    n_threshold: int
    n_test: int
    n_embargo: int
    group_purged_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_group_purged: int = 0
    notes: List[str] = field(default_factory=list)


def _enforce_story_groups(
    test_windows: List[np.ndarray],
    pre_idx: np.ndarray,
    embargo_idx: np.ndarray,
    story_group_ids: np.ndarray,
    times: np.ndarray,
    window_boundaries: List[Tuple[float, float]],
    embargo_seconds: float,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Enforce story-group integrity across split boundaries.

    For any group with members in multiple partitions, keep members in the
    LATEST partition (by chronological order) and purge members from all
    earlier partitions. Purged rows go into group_purged_idx and are excluded
    from all partitions. This prevents syndicated/duplicate stories from
    leaking label information across train/select/threshold/test boundaries.

    Partition rank (higher = later chronologically):
      0=fit, 1=select, 2=threshold, 3=test_w_last, ..., n=test_w_0

    Returns (updated_test_windows, updated_pre_idx, updated_embargo_idx,
             group_purged_idx, notes).
    """
    notes: List[str] = []
    n = len(times)
    groups = np.asarray(story_group_ids)

    # Assign each row a partition rank: -1=embargo/unassigned, 0=fit, 1=select,
    # 2=threshold, 3+n_test_windows-1-w=test_w_w (higher rank = more recent)
    n_tw = len(test_windows)
    rank = np.full(n, -1, dtype=int)
    for r in pre_idx:
        rank[r] = 0  # will be refined after pre-window split
    for w_idx, w in enumerate(test_windows):
        # More recent test windows get higher rank
        w_rank = 3 + (n_tw - 1 - w_idx)
        for r in w:
            rank[r] = w_rank

    # Build group → members map
    unique_groups = np.unique(groups)
    purged_mask = np.zeros(n, dtype=bool)

    for g in unique_groups:
        members = np.where(groups == g)[0]
        if len(members) < 2:
            continue
        member_ranks = rank[members]
        # Only consider members that are in a partition (rank >= 0)
        in_partition = member_ranks[member_ranks >= 0]
        if len(in_partition) < 2:
            continue
        unique_ranks = np.unique(in_partition)
        if len(unique_ranks) <= 1:
            continue  # All in same partition — no straddle
        # Keep members in the HIGHEST rank (latest partition), purge from lower
        max_rank = int(np.max(in_partition))
        for m in members:
            if rank[m] >= 0 and rank[m] < max_rank:
                purged_mask[m] = True
                rank[m] = -1  # remove from partition

    n_purged = int(np.sum(purged_mask))
    if n_purged > 0:
        notes.append(
            f"story_group_enforcement: purged {n_purged} rows from earlier "
            f"partitions to prevent cross-split leakage"
        )

    group_purged_idx = np.where(purged_mask)[0]

    # Rebuild partitions excluding purged rows
    new_test_windows: List[np.ndarray] = []
    for w_idx in range(n_tw):
        w_rows = test_windows[w_idx]
        w_rows = w_rows[~purged_mask[w_rows]]
        if len(w_rows) > 0:
            w_rows = w_rows[np.argsort(times[w_rows])]
        new_test_windows.append(w_rows)

    new_pre_idx = pre_idx[~purged_mask[pre_idx]] if len(pre_idx) > 0 else pre_idx
    new_embargo_idx = embargo_idx[~purged_mask[embargo_idx]] if len(embargo_idx) > 0 else embargo_idx

    return new_test_windows, new_pre_idx, new_embargo_idx, group_purged_idx, notes


def chronological_split(
    prediction_times: np.ndarray,
    y_enc: np.ndarray,
    story_group_ids: Optional[np.ndarray] = None,
    test_window_days: int = 14,
    n_test_windows: int = 3,
    embargo_hours: int = 48,
    fit_fraction: float = 0.70,
    select_fraction: float = 0.15,
) -> ChronologicalSplits:
    """Split data chronologically into fit/select/threshold + test windows.

    prediction_times: UTC timestamps (as float seconds or datetime64).
    y_enc: integer-encoded labels (for stratification checks).
    story_group_ids: optional grouping to keep duplicate/syndicated stories
        in the same partition. When provided, any group that straddles a
        test-window boundary is pulled entirely into the earliest test window
        it touches, preventing train→test label leakage.

    The test windows are the most recent n_test_windows × test_window_days.
    The pre-window history is split into fit/select/threshold by chronology.
    """
    times = np.asarray(prediction_times, dtype=np.float64)
    y_arr = np.asarray(y_enc)
    n = len(times)
    if n == 0:
        return ChronologicalSplits(
            np.array([]), np.array([]), np.array([]), [],
            np.array([]), times, [], 0, 0, 0, 0, 0, 0, ["empty input"]
        )

    # Sort by prediction time
    order = np.argsort(times)
    times_sorted = times[order]

    # Define test window boundaries (from the end)
    max_time = times_sorted[-1]
    window_seconds = test_window_days * 86400
    embargo_seconds = embargo_hours * 3600

    test_windows = []
    window_boundaries = []
    window_start = max_time
    notes = []

    for w in range(n_test_windows):
        w_end = window_start
        w_start = w_end - window_seconds
        if w_start < times_sorted[0]:
            notes.append(f"test window {w} extends before data start")
            break
        # Use <= for the last (most recent) window to include the max timestamp
        if w == 0:
            w_mask = (times >= w_start) & (times <= w_end)
        else:
            w_mask = (times >= w_start) & (times < w_end)
        w_idx = np.where(w_mask)[0]
        if len(w_idx) == 0:
            notes.append(f"test window {w} is empty")
            # Still append empty window and advance so later windows can capture
            # sparse data further back in time.
            test_windows.append(np.array([], dtype=int))
            window_boundaries.append((w_start, w_end))
            window_start = w_start
            continue
        test_windows.append(w_idx)
        window_boundaries.append((w_start, w_end))
        window_start = w_start  # next window starts where this one starts

    # Pre-window history: everything before the earliest test window
    if window_boundaries:
        pre_window_end = window_boundaries[-1][0] - embargo_seconds
        pre_mask = times < pre_window_end
    else:
        pre_mask = np.ones(n, dtype=bool)
        notes.append("no test windows defined; using all data as pre-window")

    pre_idx = np.where(pre_mask)[0]

    # Embargo rows: in the gap between pre-window and earliest test window
    test_all_mask = np.zeros(n, dtype=bool)
    for w in test_windows:
        test_all_mask[w] = True
    embargo_mask = (~pre_mask) & (~test_all_mask)
    embargo_idx = np.where(embargo_mask)[0]

    # ── Story group enforcement (test vs pre-window) ───────────────
    # Purge straddling group members from earlier partitions to prevent
    # train→test leakage from duplicate/syndicated stories.
    group_purged_idx = np.array([], dtype=int)
    if story_group_ids is not None and len(story_group_ids) == n:
        test_windows, pre_idx, embargo_idx, group_purged_idx, sg_notes = _enforce_story_groups(
            test_windows, pre_idx, embargo_idx,
            np.asarray(story_group_ids), times,
            window_boundaries, embargo_seconds,
        )
        notes.extend(sg_notes)

    pre_times = times[pre_idx]
    pre_order = np.argsort(pre_times)
    pre_idx_sorted = pre_idx[pre_order]
    n_pre = len(pre_idx_sorted)

    if n_pre == 0:
        notes.append("no pre-window training data")
        return ChronologicalSplits(
            np.array([]), np.array([]), np.array([]), test_windows,
            embargo_idx, times, window_boundaries, n, 0, 0, 0,
            sum(len(w) for w in test_windows), len(embargo_idx),
            group_purged_idx, len(group_purged_idx), notes,
        )

    # Split pre-window into fit/select/threshold by chronology
    n_fit = int(n_pre * fit_fraction)
    n_select = int(n_pre * select_fraction)
    # Remainder goes to threshold
    n_threshold = n_pre - n_fit - n_select

    fit_idx = pre_idx_sorted[:n_fit]
    select_idx = pre_idx_sorted[n_fit:n_fit + n_select]
    threshold_idx = pre_idx_sorted[n_fit + n_select:]

    # ── Story group enforcement (fit/select/threshold) ─────────────
    # Purge straddling group members from earlier pre-window partitions.
    if story_group_ids is not None and len(story_group_ids) == n:
        groups_arr = np.asarray(story_group_ids)
        # Assign each pre-window row a partition rank: 0=fit, 1=select, 2=threshold
        partition_rank = np.full(n, -1, dtype=int)
        for r in fit_idx:
            partition_rank[r] = 0
        for r in select_idx:
            partition_rank[r] = 1
        for r in threshold_idx:
            partition_rank[r] = 2

        additional_purged = []
        for g in np.unique(groups_arr):
            members = np.where(groups_arr == g)[0]
            if len(members) < 2:
                continue
            member_ranks = partition_rank[members]
            # Only consider members that are in a pre-window partition
            in_pre = member_ranks[member_ranks >= 0]
            if len(in_pre) < 2:
                continue
            min_rank = int(np.min(in_pre))
            max_rank = int(np.max(in_pre))
            if min_rank == max_rank:
                continue  # All in same partition — no straddle
            # Purge members from earlier partitions (keep in latest)
            for m in members:
                r = partition_rank[m]
                if r >= 0 and r < max_rank:
                    additional_purged.append(m)
                    partition_rank[m] = -1

        if additional_purged:
            additional_purged = np.array(additional_purged, dtype=int)
            group_purged_idx = np.concatenate([group_purged_idx, additional_purged])
            notes.append(
                f"story_group_enforcement_pre: purged {len(additional_purged)} "
                f"rows from earlier pre-window partitions"
            )

        fit_idx = np.where(partition_rank == 0)[0]
        select_idx = np.where(partition_rank == 1)[0]
        threshold_idx = np.where(partition_rank == 2)[0]

    # Check class support in each partition
    for name, idx in [("fit", fit_idx), ("select", select_idx),
                       ("threshold", threshold_idx)]:
        if len(idx) > 0:
            classes_present = np.unique(y_arr[idx])
            if len(classes_present) < len(CLASSES):
                notes.append(
                    f"{name} partition has {len(classes_present)}/{len(CLASSES)} classes"
                )

    return ChronologicalSplits(
        fit_idx=fit_idx,
        select_idx=select_idx,
        threshold_idx=threshold_idx,
        test_windows=test_windows,
        embargo_idx=embargo_idx,
        prediction_times=times,
        window_boundaries=window_boundaries,
        n_total=n,
        n_fit=len(fit_idx),
        n_select=len(select_idx),
        n_threshold=len(threshold_idx),
        n_test=sum(len(w) for w in test_windows),
        n_embargo=len(embargo_idx),
        group_purged_idx=group_purged_idx,
        n_group_purged=len(group_purged_idx),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════
#  7b. Grouped development split (D2 evidence class)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GroupedDevSplit:
    """Fixed development split using StratifiedGroupKFold.

    This is D2 evidence: controlled development result, not temporal holdout.
    Duplicate/syndicated stories are kept in one partition via group keys.
    The test partition (~20%) is held out; the remaining ~80% is split into
    fit (~75% of remaining), select (~12.5%), and threshold (~12.5%) by
    a second grouped split.
    """
    fit_idx: np.ndarray
    select_idx: np.ndarray
    threshold_idx: np.ndarray
    test_idx: np.ndarray
    n_total: int
    n_fit: int
    n_select: int
    n_threshold: int
    n_test: int
    notes: List[str] = field(default_factory=list)


def grouped_dev_split(
    y_enc: np.ndarray,
    story_group_ids: np.ndarray,
    seed: int = 42,
    test_fraction: float = 0.20,
) -> GroupedDevSplit:
    """Fixed group-isolated development split.

    Uses StratifiedGroupKFold to keep story groups in one partition while
    stratifying by class. The outer split produces a test fold (~20%);
    the remaining rows are split again into fit/select/threshold.

    This is NOT a temporal holdout — it is a D2 development evaluation.
    The split is deterministic given the seed and data ordering.
    """
    y_arr = np.asarray(y_enc)
    groups = np.asarray(story_group_ids)
    n = len(y_arr)
    notes = []

    if n == 0:
        return GroupedDevSplit(
            np.array([]), np.array([]), np.array([]), np.array([]),
            0, 0, 0, 0, 0, ["empty input"]
        )

    # Outer split: test vs rest (5 folds, take fold 0 as test ~20%)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(np.zeros(n), y_arr, groups))

    # Verify no group overlap between rest and test
    rest_groups = set(groups[rest_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    overlap = rest_groups & test_groups
    if overlap:
        notes.append(f"WARNING: {len(overlap)} groups overlap rest/test")

    # Inner split: fit vs (select+threshold) on the rest
    y_rest = y_arr[rest_idx]
    g_rest = groups[rest_idx]
    n_rest = len(rest_idx)

    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
    # fold 0 → select (~12.5% of rest), fold 1 → threshold (~12.5%), rest → fit (~75%)
    fold_count = 0
    select_local = None
    threshold_local = None
    fit_local = None
    for train_local_idx, val_local_idx in inner.split(np.zeros(n_rest), y_rest, g_rest):
        if fold_count == 0:
            select_local = val_local_idx
        elif fold_count == 1:
            threshold_local = val_local_idx
        else:
            break
        fold_count += 1

    if select_local is not None and threshold_local is not None:
        # fit = rest minus select and threshold
        used = set(select_local.tolist()) | set(threshold_local.tolist())
        fit_local = np.array([i for i in range(n_rest) if i not in used])
    else:
        # Fallback: simple chronological-style split
        n_fit = int(n_rest * 0.75)
        n_select = int(n_rest * 0.125)
        fit_local = np.arange(n_fit)
        select_local = np.arange(n_fit, n_fit + n_select)
        threshold_local = np.arange(n_fit + n_select, n_rest)
        notes.append("fallback inner split (StratifiedGroupKFold failed)")

    fit_idx = rest_idx[fit_local]
    select_idx = rest_idx[select_local]
    threshold_idx = rest_idx[threshold_local]

    # Verify no group overlap across fit/select/threshold
    for name_a, idx_a in [("fit", fit_idx), ("select", select_idx), ("threshold", threshold_idx)]:
        for name_b, idx_b in [("select", select_idx), ("threshold", threshold_idx)]:
            if name_a >= name_b:
                continue
            ga = set(groups[idx_a].tolist())
            gb = set(groups[idx_b].tolist())
            ov = ga & gb
            if ov:
                notes.append(f"WARNING: {len(ov)} groups overlap {name_a}/{name_b}")

    # Check class support
    for name, idx in [("fit", fit_idx), ("test", test_idx)]:
        if len(idx) > 0:
            classes_present = np.unique(y_arr[idx])
            if len(classes_present) < len(CLASSES):
                notes.append(f"{name} has {len(classes_present)}/{len(CLASSES)} classes")

    return GroupedDevSplit(
        fit_idx=fit_idx,
        select_idx=select_idx,
        threshold_idx=threshold_idx,
        test_idx=test_idx,
        n_total=n,
        n_fit=len(fit_idx),
        n_select=len(select_idx),
        n_threshold=len(threshold_idx),
        n_test=len(test_idx),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════
#  8. Composite score and metrics
# ═══════════════════════════════════════════════════════════════════

def composite_score(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Production composite: weighted sum of class probabilities."""
    w = np.array([CLASS_WEIGHT_MAP.get(c, 0.0) for c in class_names], dtype=float)
    return probs.dot(w)


def classification_metrics(
    probs: np.ndarray, y_true: List[str], class_names: List[str]
) -> Dict[str, float]:
    """Full classification metrics with per-class breakdown."""
    y_arr = np.asarray(y_true)
    y_pred_idx = np.argmax(probs, axis=1)
    y_pred = np.array([class_names[i] for i in y_pred_idx])

    # Per-class support
    per_class = {}
    for c in class_names:
        mask = y_arr == c
        n_c = int(mask.sum())
        if n_c > 0:
            pred_c = y_pred[mask] == c
            tp = int(pred_c.sum())
            per_class[c] = {
                "precision": round(float(tp / max(int((y_pred == c).sum()), 1)), 4),
                "recall": round(float(tp / n_c), 4),
                "f1": round(
                    float(2 * tp / max(int((y_pred == c).sum()) + n_c, 1)), 4
                ),
                "support": n_c,
            }
        else:
            per_class[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    return {
        "accuracy": round(float(accuracy_score(y_arr, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_arr, y_pred, average="macro",
                                          zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_arr, y_pred,
                                                        average="macro",
                                                        zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_arr, y_pred, average="macro",
                                                  zero_division=0)), 4),
        "per_class": per_class,
    }


def ranking_metrics(
    probs: np.ndarray, y_true: List[str], class_names: List[str],
    k_values: List[int] = [10, 30, 50],
) -> Dict[str, Any]:
    """Ranking metrics: p@k, lead_recall@k, important_recall@k, AUC."""
    y_arr = np.asarray(y_true)
    n = len(y_arr)
    scores = composite_score(probs, class_names)

    rel = np.array([1 if y in _RANKING_RELEVANT else 0 for y in y_arr], dtype=int)
    n_rel = int(rel.sum())
    n_neg = n - n_rel

    out: Dict[str, Any] = {}
    out["ranking_auc"] = 0.0
    out["ranking_note"] = ""

    notes = []
    if n_rel < 2 or n_neg < 2:
        notes.append("holdout has <2 relevant or <2 irrelevant; AUC undefined")
    else:
        try:
            out["ranking_auc"] = round(float(roc_auc_score(rel, scores)), 4)
        except Exception:
            notes.append("AUC computation failed")

    order = np.argsort(-scores)
    for k in k_values:
        k_eff = min(k, n)
        top = order[:k_eff]
        out[f"precision_at_{k}"] = round(float(rel[top].mean()), 4) if k_eff > 0 else 0.0

        n_lead = int(np.sum(y_arr == "investigation_lead"))
        if n_lead > 0:
            out[f"lead_recall_at_{k}"] = round(
                float(np.sum(y_arr[top] == "investigation_lead") / n_lead), 4
            )
        else:
            out[f"lead_recall_at_{k}"] = 0.0

        n_imp = int(np.sum(y_arr == "important"))
        if n_imp > 0:
            out[f"important_recall_at_{k}"] = round(
                float(np.sum(y_arr[top] == "important") / n_imp), 4
            )
        else:
            out[f"important_recall_at_{k}"] = 0.0

    out["ranking_note"] = "; ".join(notes)
    return out


def calibration_metrics(
    probs: np.ndarray, y_true: List[str], class_names: List[str],
) -> Dict[str, Any]:
    """Probability calibration: NLL, Brier, ECE, per-class ECE, reliability bins."""
    y_arr = np.asarray(y_true)
    idx_map = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([idx_map[str(y)] for y in y_arr], dtype=np.int64)

    p_true = probs[np.arange(len(y_idx)), y_idx]
    nll = -float(np.mean(np.log(np.clip(p_true, 1e-9, 1.0))))

    # Brier: sum over classes of (prob - onehot)^2, then mean over rows
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    # ECE: 10-bin equal-width on max probability
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_idx).astype(float)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability_bins = []
    for b in range(n_bins):
        mask = (confidences >= bin_edges[b]) & (confidences < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (confidences >= bin_edges[b]) & (confidences <= bin_edges[b + 1])
        n_b = int(mask.sum())
        if n_b > 0:
            acc_b = float(correct[mask].mean())
            conf_b = float(confidences[mask].mean())
            ece += (n_b / len(y_idx)) * abs(acc_b - conf_b)
            reliability_bins.append({
                "bin": b,
                "lo": round(float(bin_edges[b]), 2),
                "hi": round(float(bin_edges[b + 1]), 2),
                "count": n_b,
                "accuracy": round(acc_b, 4),
                "confidence": round(conf_b, 4),
                "gap": round(acc_b - conf_b, 4),
            })

    # Per-class ECE: calibration of p(class) vs 1{y==class}
    per_class_ece = {}
    for ci, cname in enumerate(class_names):
        class_probs = probs[:, ci]
        class_correct = (y_idx == ci).astype(float)
        cls_ece = 0.0
        for b in range(n_bins):
            mask = (class_probs >= bin_edges[b]) & (class_probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (class_probs >= bin_edges[b]) & (class_probs <= bin_edges[b + 1])
            n_b = int(mask.sum())
            if n_b > 0:
                acc_b = float(class_correct[mask].mean())
                conf_b = float(class_probs[mask].mean())
                cls_ece += (n_b / len(y_idx)) * abs(acc_b - conf_b)
        per_class_ece[cname] = round(float(cls_ece), 4)

    return {
        "nll": round(nll, 4),
        "brier": round(brier, 4),
        "ece": round(float(ece), 4),
        "per_class_ece": per_class_ece,
        "reliability_bins": reliability_bins,
    }


def threshold_survival(
    probs: np.ndarray, y_true: List[str], class_names: List[str],
    threshold: float = ALERT_THRESHOLD,
) -> Dict[str, Any]:
    """Threshold survival metrics at a fixed threshold."""
    y_arr = np.asarray(y_true)
    scores = composite_score(probs, class_names)
    n = len(y_arr)

    n_lead = int(np.sum(y_arr == "investigation_lead"))
    n_imp = int(np.sum(y_arr == "important"))
    n_bg = int(np.sum(y_arr == "background"))
    n_noise = int(np.sum(y_arr == "noise"))

    above = scores >= threshold
    n_above = int(above.sum())

    lead_surv = int(np.sum(above & (y_arr == "investigation_lead")))
    imp_surv = int(np.sum(above & (y_arr == "important")))
    bg_pass = int(np.sum(above & (y_arr == "background")))
    noise_pass = int(np.sum(above & (y_arr == "noise")))

    # Precision among threshold-selected rows
    relevant_above = lead_surv + imp_surv
    threshold_precision = float(relevant_above / n_above) if n_above > 0 else 0.0

    return {
        "threshold": threshold,
        "n_above": n_above,
        "n_total": n,
        "lead_survival": round(float(lead_surv / n_lead), 4) if n_lead > 0 else 0.0,
        "lead_survival_count": f"{lead_surv}/{n_lead}",
        "important_survival": round(float(imp_surv / n_imp), 4) if n_imp > 0 else 0.0,
        "important_survival_count": f"{imp_surv}/{n_imp}",
        "background_pass": round(float(bg_pass / n_bg), 4) if n_bg > 0 else 0.0,
        "background_pass_count": f"{bg_pass}/{n_bg}",
        "noise_pass": round(float(noise_pass / n_noise), 4) if n_noise > 0 else 0.0,
        "noise_pass_count": f"{noise_pass}/{n_noise}",
        "threshold_precision": round(threshold_precision, 4),
        "queue_composition": {
            "investigation_lead": lead_surv,
            "important": imp_surv,
            "background": bg_pass,
            "noise": noise_pass,
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  9. Promote gate replay
# ═══════════════════════════════════════════════════════════════════

def evaluate_gate(
    old_metrics: Optional[Dict[str, float]],
    new_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Replay the production promote gate from ml_window.py.

    Returns the Boolean decision plus the branch taken and all intermediate
    values, so the replay is fully auditable.
    """
    if not old_metrics:
        return {"promote": True, "branch": "cold_start", "deltas": {}}

    old_p30 = float(old_metrics.get("precision_at_30") or 0.0)
    new_p30 = float(new_metrics.get("precision_at_30") or 0.0)
    # Accept both macro_f1 (framework naming) and f1_score (production naming).
    # Both compute f1_score(average="macro", zero_division=0).
    old_f1 = float(old_metrics.get("macro_f1") or old_metrics.get("f1_score") or 0.0)
    new_f1 = float(new_metrics.get("macro_f1") or new_metrics.get("f1_score") or 0.0)
    old_lr = old_metrics.get("lead_recall_at_30")
    new_lr = new_metrics.get("lead_recall_at_30")

    p30_gain = new_p30 - old_p30
    f1_gain = new_f1 - old_f1

    deltas = {
        "p30_gain": round(p30_gain, 4),
        "f1_gain": round(f1_gain, 4),
    }

    # Lead-recall guard
    if old_lr and new_lr is not None and float(old_lr) > 0.0:
        lr_gain = float(new_lr) - float(old_lr)
        deltas["lead_recall_gain"] = round(lr_gain, 4)
        if lr_gain < -LEAD_RECALL_SLACK:
            return {
                "promote": False,
                "branch": "lead_recall_guard",
                "deltas": deltas,
                "reason": f"lead_recall_at_30 cratered ({old_lr:.3f}→{new_lr:.3f})",
            }

    # Big p@30 win path
    if p30_gain >= PROMOTE_BIG_P30_WIN and f1_gain >= -F1_HARD_DROP_LIMIT:
        return {
            "promote": True,
            "branch": "big_p30_win",
            "deltas": deltas,
        }

    # Legacy gate
    p30_up = p30_gain >= PROMOTE_MARGIN
    f1_up = f1_gain >= PROMOTE_MARGIN
    f1_ok = f1_gain >= -PROMOTE_MARGIN
    p30_not_collapsed = p30_gain >= -PROMOTE_RANKING_SLACK

    if (p30_up and f1_ok) or (f1_up and p30_not_collapsed):
        return {
            "promote": True,
            "branch": "legacy_gate",
            "deltas": deltas,
        }

    return {
        "promote": False,
        "branch": "legacy_reject",
        "deltas": deltas,
    }


# ═══════════════════════════════════════════════════════════════════
#  10. Full evaluation pipeline for one configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """Complete results for one configuration evaluation."""
    config: Dict[str, Any]
    n_train: int
    n_test: int
    class_support: Dict[str, int]
    temperature: float
    oof_nll: float
    oof_samples: int
    calibration_method: str
    prob_mode: str
    prob_mode_label: str
    weights_status: str
    classification: Dict[str, Any]
    ranking: Dict[str, Any]
    calibration: Dict[str, Any]
    survival: Dict[str, Any]
    gate: Optional[Dict[str, Any]] = None
    # Per-mode diagnostics (blocker fix: separate temperatures for P1/P3)
    temperatures: Dict[str, float] = field(default_factory=dict)
    mode_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def evaluate_configuration(
    config: ExperimentConfig,
    X_all: np.ndarray,
    y_all: List[str],
    sample_weights: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    incumbent_metrics: Optional[Dict[str, float]] = None,
    story_group_ids: Optional[np.ndarray] = None,
) -> EvalResult:
    """Train and evaluate one configuration on a given split.

    This is the main entry point. It:
    1. Fits the head on fit_idx with exact C/seed/weights
    2. Collects OOF logits on fit_idx using the SAME head/C/seed
    3. Fits temperature on OOF logits
    4. Builds prior correction using ACTUAL fit weights
    5. Computes all four probability modes (but reports the config's mode)
    6. Computes all metrics on test_idx
    7. Replays the promote gate against incumbent_metrics
    """
    le = LabelEncoder()
    le.fit(CLASSES)
    class_names = le.classes_.tolist()

    y_all_arr = np.asarray(y_all)
    y_fit = y_all_arr[fit_idx]
    y_test = y_all_arr[test_idx]
    y_fit_enc = le.transform(y_fit)

    sw_fit = None
    if sample_weights is not None and len(sample_weights) > 0:
        sw_fit = np.asarray(sample_weights, dtype=np.float64)[fit_idx]

    X_fit = X_all[fit_idx]
    X_test = X_all[test_idx]

    n_train = len(fit_idx)
    n_test = len(test_idx)

    # Class support
    class_support = {c: int(np.sum(y_fit == c)) for c in CLASSES}

    notes = []

    # ── Fit final head FIRST ────────────────────────────────────────
    # We fit the final head before OOF/prior/temperature so we know whether
    # sample_weight was accepted. The new fit_head raises on unsupported
    # non-unit weights, so we check support first and pass None for heads
    # that can't accept weights (MLP, CalibratedClassifierCV wrappers, kNN).
    # This is explicit, not silent — the note records the omission.
    sw_for_fit = sw_fit
    if sw_for_fit is not None and not np.all(sw_for_fit == 1.0):
        _probe = build_head(config.head, C=config.C, seed=config.seed)
        _final_name = list(_probe.named_steps)[-1]
        _est = _probe.named_steps[_final_name]
        _supports = "sample_weight" in inspect.signature(_est.fit).parameters
        if not _supports:
            notes.append(
                f"weights_omitted: {config.head} does not support sample_weight; "
                f"training unweighted, prior correction uses uniform effective weights"
            )
            sw_for_fit = None

    final_pipe, weights_status = fit_head(
        config.head, config.C, config.seed,
        X_fit, y_fit_enc, sw_for_fit,
    )

    # Effective sample weights for prior correction: None if omitted
    sw_for_prior = sw_for_fit if weights_status == "accepted" else None

    # ── OOF calibration ─────────────────────────────────────────────
    min_class = min(class_support.values()) if class_support else 0
    if n_train >= 15 and min_class >= 2:
        n_folds = min(5, min_class)
        if n_train < 50:
            n_folds = min(n_folds, 3)
    else:
        n_folds = 0
        notes.append(f"insufficient class support for OOF (min_class={min_class})")

    temperature = 1.0
    temperature_raw = 1.0   # T1 for P1 (fit on raw OOF logits)
    temperature_prior = 1.0  # T3 for P3 (fit on prior-adjusted OOF logits)
    oof_nll = float("inf")
    oof_samples = 0
    cal_method = "none"
    prior_offsets = None

    # Extract story groups for fit partition (for group-disjoint OOF)
    fit_groups = None
    if story_group_ids is not None and len(story_group_ids) == len(y_all_arr):
        fit_groups = np.asarray(story_group_ids)[fit_idx]

    if n_folds >= 2:
        # OOF uses the same sw_fit; fit_head inside will reject if needed
        oof_logits, oof_y_enc = collect_oof_logits_per_config(
            config.head, config.C, config.seed,
            X_fit, y_fit_enc, sw_fit, n_folds,
            story_group_ids=fit_groups,
        )
        oof_samples = len(oof_y_enc)
        if oof_samples >= 3:
            # Build prior correction using ACTUAL effective weights.
            # If weights were rejected by the final head, pass None so
            # the prior correction reflects unweighted training.
            prior = build_prior_fit_actual(
                y_fit_enc, sw_for_prior, config.head, class_names,
            )
            prior_offsets = prior["offsets"]

            # Fit TWO temperatures (blocker fix: P1 and P3 must not share T)
            # T1 (temperature_raw): fit on RAW OOF logits → for P1
            temperature_raw, oof_nll_raw = fit_temperature(oof_logits, oof_y_enc)

            # T3 (temperature_prior): fit on prior-ADJUSTED OOF logits → for P3
            if prior_offsets is not None:
                oof_adjusted = oof_logits + prior_offsets.reshape(1, -1)
            else:
                oof_adjusted = oof_logits
            temperature_prior, oof_nll = fit_temperature(oof_adjusted, oof_y_enc)

            # The "primary" temperature is the one for the configured mode
            if config.prob_mode in ("P1",):
                temperature = temperature_raw
            elif config.prob_mode in ("P3",):
                temperature = temperature_prior
            else:
                temperature = temperature_prior  # P0/P2 don't use T
            cal_method = "oof"
        else:
            notes.append(f"OOF produced only {oof_samples} samples")

    # ── Extract test logits ─────────────────────────────────────────
    test_logits = extract_logits(final_pipe, X_test)

    # ── Compute probabilities for each mode with its own temperature ──
    # P0: raw softmax (no T, no prior)
    # P1: softmax(logits / T1)  — T1 fit on raw OOF
    # P2: softmax(logits + offsets)  — no T
    # P3: softmax((logits + offsets) / T3)  — T3 fit on prior-adjusted OOF
    all_modes = {}
    mode_temps = {"P0": 1.0, "P1": temperature_raw, "P2": 1.0, "P3": temperature_prior}
    for mode in ["P0", "P1", "P2", "P3"]:
        all_modes[mode] = compute_prob_mode(
            mode, test_logits, mode_temps[mode], prior_offsets,
        )

    # The configured mode's probabilities are the primary result
    probs_test = all_modes[config.prob_mode]

    # ── Metrics ─────────────────────────────────────────────────────
    y_test_list = list(y_test)
    cls_metrics = classification_metrics(probs_test, y_test_list, class_names)
    rank_metrics = ranking_metrics(probs_test, y_test_list, class_names)
    cal_metrics = calibration_metrics(probs_test, y_test_list, class_names)
    surv_metrics = threshold_survival(probs_test, y_test_list, class_names)

    # ── Gate replay ─────────────────────────────────────────────────
    gate_result = None
    if incumbent_metrics is not None:
        new_for_gate = {
            "precision_at_30": rank_metrics.get("precision_at_30", 0.0),
            "macro_f1": cls_metrics["macro_f1"],
            "f1_score": cls_metrics["macro_f1"],  # alias for production gate compat
            "lead_recall_at_30": rank_metrics.get("lead_recall_at_30", 0.0),
        }
        gate_result = evaluate_gate(incumbent_metrics, new_for_gate)

    # ── Per-mode metrics (diagnostic) ───────────────────────────────
    survival_all_modes = {}
    mode_metrics: Dict[str, Dict[str, Any]] = {}
    for mode in ["P0", "P1", "P2", "P3"]:
        mode_probs = all_modes[mode]
        mode_surv = threshold_survival(mode_probs, y_test_list, class_names)
        survival_all_modes[mode] = mode_surv
        mode_metrics[mode] = {
            "classification": classification_metrics(mode_probs, y_test_list, class_names),
            "ranking": ranking_metrics(mode_probs, y_test_list, class_names),
            "calibration": calibration_metrics(mode_probs, y_test_list, class_names),
            "survival": mode_surv,
        }

    _PROB_MODE_LABELS = {
        "P0": "raw_softmax",
        "P1": "temperature_only",
        "P2": "prior_only",
        "P3": "prior_then_temperature",
    }

    # Per-mode temperatures
    temperatures_out = {
        "P0": 1.0,
        "P1": round(temperature_raw, 4),
        "P2": 1.0,
        "P3": round(temperature_prior, 4),
    }

    # Predictions dict for downstream inspection
    predictions_out = {
        "test_indices": list(test_idx),
        "modes": {
            mode: {"probabilities": all_modes[mode].tolist()}
            for mode in ["P0", "P1", "P2", "P3"]
        },
    }

    return EvalResult(
        config=asdict(config),
        n_train=n_train,
        n_test=n_test,
        class_support=class_support,
        temperature=round(temperature, 4),
        oof_nll=round(oof_nll, 4) if oof_nll != float("inf") else None,
        oof_samples=oof_samples,
        calibration_method=cal_method,
        prob_mode=config.prob_mode,
        prob_mode_label=_PROB_MODE_LABELS.get(config.prob_mode, config.prob_mode),
        weights_status=weights_status,
        classification=cls_metrics,
        ranking=rank_metrics,
        calibration=cal_metrics,
        survival=surv_metrics,
        gate=gate_result,
        temperatures=temperatures_out,
        mode_metrics=mode_metrics,
        predictions=predictions_out,
        notes=notes + [
            f"survival_all_modes={json.dumps({m: {k: v for k, v in s.items() if k in ['lead_survival', 'important_survival', 'n_above', 'threshold_precision']} for m, s in survival_all_modes.items()})}",
        ],
    )
