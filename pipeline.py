"""
ML pipeline for Magnitu.

Two architectures behind the same interface:
- "tfidf":       TF-IDF + Logistic Regression (classic fallback + recipe distiller)
- "transformer": Cached multilingual E5 embeddings + LogReg head (default)

The transformer path computes mean-pooled embeddings once at sync time and
stores them in the DB.  Training and scoring use these cached embeddings with
a regularized linear classifier — so labeling stays snappy.  The TF-IDF path is
kept as a fallback and is used by the recipe distiller for knowledge
distillation.
"""
import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

from typing import Optional, List, Tuple, Dict, Callable

import db
from config import (
    DEFAULT_TRANSFORMER_MODEL,
    MODELS_DIR,
    get_config,
)
from magnitu.entry_preview import is_legal_training_entry, training_corpus_text

logger = logging.getLogger(__name__)

CLASSES = ["investigation_lead", "important", "background", "noise"]

CLASS_WEIGHT_MAP = {
    "investigation_lead": 1.0,
    "important": 0.66,
    "background": 0.33,
    "noise": 0.0,
}


_STABLE_HOLDOUT_SALT = b"magnitu-v3-stable-holdout-v1"


def _stable_split_bucket(entry_type, entry_id) -> int:
    """0..9999 bucket from entry identity; stable across retrains and label order."""
    import hashlib

    et, eid = db.entry_key(entry_type, entry_id)
    raw = ("%s\x00%s" % (et, eid)).encode("utf-8")
    digest = hashlib.sha256(_STABLE_HOLDOUT_SALT + raw).hexdigest()
    return int(digest[:8], 16) % 10000


def _holdout_test_fraction(min_class_count: int, n: int) -> float:
    """Match legacy sizing: ~10–20% test, capped by rarest class."""
    test_size = min(0.2, min_class_count / max(n, 1))
    return max(float(test_size), 0.1)


def _min_class_count_in_labels(y_list: List[str]) -> int:
    """Smallest class count in a label list (0 when empty)."""
    if not y_list:
        return 0
    counts = pd.Series(y_list).value_counts()
    return int(counts.min())


def _repair_stratified_holdout(is_test, y_arr, keys):
    """Ensure each class with >=2 rows has at least one train and one test sample."""
    is_test = np.asarray(is_test, dtype=bool).copy()
    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        if len(cls_idx) < 2:
            continue
        test_cls = cls_idx[is_test[cls_idx]]
        train_cls = cls_idx[~is_test[cls_idx]]
        if len(test_cls) == 0 and len(train_cls) > 0:
            buckets = [_stable_split_bucket(*keys[i]) for i in train_cls]
            is_test[train_cls[int(np.argmax(buckets))]] = True
        elif len(train_cls) == 0 and len(test_cls) > 0:
            buckets = [_stable_split_bucket(*keys[i]) for i in test_cls]
            is_test[test_cls[int(np.argmin(buckets))]] = False
    return is_test


def _slice_by_indices(X, y, sample_weight, train_idx, test_idx):
    """Index rows for ndarray, DataFrame, or list features."""

    def _take(arr, idx):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr[idx]
        if hasattr(arr, "iloc"):
            return arr.iloc[idx]
        return [arr[i] for i in idx]

    y_list = list(y)
    return (
        _take(X, train_idx),
        _take(X, test_idx),
        [y_list[i] for i in train_idx],
        [y_list[i] for i in test_idx],
        _take(sample_weight, train_idx),
        _take(sample_weight, test_idx),
    )


def _stable_train_test_split(X, y, sample_weight, labeled_rows, test_size: float):
    """
    Hold out a fixed fraction of entries by (entry_type, entry_id) hash.

    Unlike ``train_test_split(random_state=42)``, adding labels does not reshuffle
    which older entries sit in the test fold — training-history metrics compare fairly.
    """
    n = len(y)
    y_arr = np.asarray(y)
    keys = [db.entry_key_from_mapping(row) for row in labeled_rows]
    threshold = int(float(test_size) * 10000)
    is_test = np.array(
        [_stable_split_bucket(et, eid) < threshold for et, eid in keys],
        dtype=bool,
    )
    is_test = _repair_stratified_holdout(is_test, y_arr, keys)
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    if len(test_idx) == 0 and n >= 2:
        is_test[np.argmax([_stable_split_bucket(*keys[i]) for i in range(n)])] = True
        train_idx = np.where(~is_test)[0]
        test_idx = np.where(is_test)[0]
    return _slice_by_indices(X, y, sample_weight, train_idx, test_idx)


def _train_test_split_stratified_safe(X, y, sample_weight, test_size: float,
                                      random_state: int = 42):
    """
    Stratified train/test split that sklearn can apply without error.

    With a small labeled set and four classes, a 10%% holdout can yield fewer
    test samples than classes; ``stratify=y`` then raises ValueError.  We bump
    the test fraction to at least n_classes/n (when both folds can still hold
    every class), otherwise fall back to an unstratified split.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    n_classes = len(np.unique(y_arr))
    min_frac = float(n_classes) / float(max(n, 1))
    if n < 2 * n_classes:
        return train_test_split(
            X, y, sample_weight, test_size=test_size,
            stratify=None, random_state=random_state,
        )
    max_test_frac = 1.0 - min_frac
    ts = float(test_size)
    ts = max(ts, min_frac)
    ts = min(ts, max_test_frac)
    try:
        return train_test_split(
            X, y, sample_weight, test_size=ts,
            stratify=y, random_state=random_state,
        )
    except ValueError:
        return train_test_split(
            X, y, sample_weight, test_size=test_size,
            stratify=None, random_state=random_state,
        )


# ═══════════════════════════════════════════════════════════════════
#  Probability calibration (temperature scaling on validation logits)
# ═══════════════════════════════════════════════════════════════════


def calibration_sidecar_path(model_path: str) -> Path:
    """Path to JSON sidecar written next to the .joblib classifier."""
    p = Path(model_path)
    return p.with_name(p.stem + ".calibration.json")


def load_calibration(model_path: str) -> Optional[dict]:
    """Load calibration JSON if present; otherwise return None."""
    side = calibration_sidecar_path(model_path)
    if not side.exists():
        return None
    try:
        with open(side, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_calibration_sidecar(model_path: str, cal: dict) -> None:
    """Persist calibration next to the serialized classifier."""
    side = calibration_sidecar_path(model_path)
    with open(str(side), "w") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)


def _step_logits(step, X) -> np.ndarray:
    """
    Logits from a single classifier step.

    Prefers ``decision_function`` (true pre-softmax logits for linear models
    like LogisticRegression).  Falls back to ``log(predict_proba)`` for
    classifiers that don't expose ``decision_function`` — notably
    ``MLPClassifier``.  Mathematically this is logits up to a per-row
    constant (``log Z``), which softmax normalizes away, so temperature
    scaling ``softmax(logits / T)`` stays well-defined.
    """
    if hasattr(step, "decision_function"):
        try:
            return step.decision_function(X)
        except (AttributeError, NotImplementedError):
            pass
    probs = step.predict_proba(X)
    return np.log(np.clip(probs, 1e-12, 1.0))


def logits_for_classifier_head(clf, X) -> np.ndarray:
    """
    Raw classifier logits for temperature scaling.

    Sklearn ``Pipeline`` does not always expose ``decision_function`` even when
    the final step supports it (depends on version), so we unwrap known
    Magnitu layouts explicitly.
    """
    if hasattr(clf, "_pipeline"):
        return logits_for_classifier_head(clf._pipeline, X)
    if hasattr(clf, "named_steps"):
        steps = clf.named_steps
        if "scaler" in steps:
            scaled = steps["scaler"].transform(X)
            if "classifier" in steps:
                return _step_logits(steps["classifier"], scaled)
            if "mlp" in steps:
                return _step_logits(steps["mlp"], scaled)
        if "classifier" in steps and "features" in steps:
            Xf = steps["features"].transform(X)
            return _step_logits(steps["classifier"], Xf)
    return _step_logits(clf, X)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax. logits shape (n_samples, n_classes)."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_temperature_scalar(
    logits: np.ndarray,
    y_str: np.ndarray,
    class_names: List[str],
) -> float:
    """
    Choose T > 0 that minimizes mean NLL of true labels on softmax(logits / T).
    Falls back to 1.0 when data is degenerate.
    """
    if logits is None or len(logits) == 0:
        return 1.0
    idx_map = {c: i for i, c in enumerate(class_names)}
    try:
        y_idx = np.array([idx_map[str(yi)] for yi in y_str], dtype=np.int64)
    except KeyError:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.25, 12.0, num=40):
        probs = _softmax_rows(logits / float(t))
        p_true = probs[np.arange(len(y_idx)), y_idx]
        nll = -float(np.mean(np.log(np.clip(p_true, 1e-9, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _oof_fold_count(n_train: int, min_class_count: int) -> int:
    """Stratified fold count for OOF calibration on the training fold."""
    if n_train < 15 or min_class_count < 2:
        return 0
    folds = min(5, min_class_count)
    if n_train < 50:
        folds = min(folds, 3)
    return folds if folds >= 2 else 0


def _transformer_fit_kwargs(sample_weight) -> dict:
    """Build sklearn fit kwargs for the transformer LogReg head."""
    if sample_weight is None:
        return {}
    sw = np.asarray(sample_weight, dtype=np.float64)
    if len(sw) == 0 or float(np.std(sw)) <= 1e-6:
        return {}
    return {"classifier__sample_weight": sw}


def build_transformer_head_pipeline() -> Pipeline:
    """StandardScaler + balanced LogReg on frozen embedding vectors."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def _collect_oof_logits(
    X: np.ndarray,
    y_list: List[str],
    sample_weight,
    label_encoder,
    n_folds: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Out-of-fold logits on the training fold for stable temperature scaling.
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
        logger.warning(
            "OOF StratifiedKFold skipped: rarest class has fewer than %d train samples",
            n_folds,
        )
        return np.array([]), []
    try:
        for train_idx, val_idx in split_iter:
            pipe = build_transformer_head_pipeline()
            fit_kwargs = _transformer_fit_kwargs(
                sw_arr[train_idx] if sw_arr is not None else None
            )
            pipe.fit(X[train_idx], y_enc[train_idx], **fit_kwargs)
            logits_val = logits_for_classifier_head(pipe, X[val_idx])
            for j, vi in enumerate(val_idx):
                oof_logits[vi] = logits_val[j]
                oof_y[vi] = y_list[vi]
    except ValueError:
        logger.warning(
            "OOF fold fit failed (class too small for %d-fold split); calibration skipped",
            n_folds,
        )
        return np.array([]), []

    valid_idx = [i for i in range(n_samples) if oof_logits[i] is not None]
    if not valid_idx:
        return np.array([]), []
    return (
        np.array([oof_logits[i] for i in valid_idx]),
        [oof_y[i] for i in valid_idx],
    )


def classifier_probabilities(
    clf,
    X: np.ndarray,
    model_path: str,
    cal: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (probabilities, class_names) using temperature scaling when a
    calibration sidecar exists (or when ``cal`` is passed during training).
    """
    class_names = clf.classes_.tolist()
    if cal is None and model_path:
        cal = load_calibration(model_path)
    if cal and cal.get("method") == "temperature":
        stored = cal.get("class_names")
        if stored and list(stored) != list(class_names):
            logger.warning("Calibration class_names mismatch; ignoring calibration.")
            cal = None
    if cal and cal.get("method") == "temperature":
        t_scale = max(float(cal.get("temperature", 1.0)), 1e-3)
        try:
            logits = logits_for_classifier_head(clf, X)
        except AttributeError:
            return clf.predict_proba(X), class_names
        return _softmax_rows(logits / t_scale), class_names
    return clf.predict_proba(X), class_names


def _relevance_from_probs(probs: Dict[str, float], class_names: List[str]) -> float:
    """Weighted class composite in [0, 1] from CLASS_WEIGHT_MAP."""
    return float(
        sum(probs.get(c, 0.0) * CLASS_WEIGHT_MAP.get(c, 0.0) for c in class_names)
    )


def _discovery_adjusted_relevance(composite: float, p_lead: float,
                                   profile_id: Optional[int] = None) -> float:
    """
    Optional blend toward investigation_lead for discovery (config:
    discovery_lead_blend in [0, 0.25]).
    """
    if profile_id is not None:
        cfg = db.get_effective_config(profile_id)
    else:
        cfg = get_config()
    blend = float(cfg.get("discovery_lead_blend", 0.0) or 0.0)
    blend = max(0.0, min(0.25, blend))
    if blend <= 0.0:
        return composite
    raw = (1.0 - blend) * composite + blend * p_lead
    return float(min(1.0, max(0.0, raw)))


# ═══════════════════════════════════════════════════════════════════
#  Embedding helpers (transformer path)
# ═══════════════════════════════════════════════════════════════════

_embedder = None   # lazy-loaded singleton


def _select_device():
    """Pick the best available torch device based on config."""
    import torch

    config = get_config()
    use_gpu = config.get("use_gpu", True)

    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _hf_hub_cache_dir() -> str:
    """Directory where huggingface_hub stores downloaded model snapshots."""
    return (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )


def _log_hf_model_cache_status(model_name: str) -> None:
    """Tell the operator whether E5 will load from disk or hit the network."""
    logger.info("HuggingFace hub cache: %s", _hf_hub_cache_dir())
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        logger.warning(
            "HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set — download disabled; "
            "model must already be cached or load will fail."
        )
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        if cached:
            logger.info(
                "E5 backbone %s: config found in local cache (%s)",
                model_name, cached,
            )
        else:
            logger.info(
                "E5 backbone %s: not in local cache — will download from "
                "huggingface.co (~1.1 GB for e5-base; needs working internet)",
                model_name,
            )
    except Exception as exc:
        logger.info(
            "Could not inspect HF cache for %s (%s); proceeding with from_pretrained",
            model_name, exc,
        )


def _enable_hf_download_logging() -> None:
    """Surface huggingface_hub download activity in the server terminal."""
    for name in ("huggingface_hub", "huggingface_hub.file_download", "transformers"):
        logging.getLogger(name).setLevel(logging.INFO)


def _get_embedder():
    """Lazy-load the transformer model + tokenizer. Cached after first call."""
    global _embedder
    if _embedder is not None:
        return _embedder

    import torch
    from transformers import AutoTokenizer, AutoModel

    config = get_config()
    model_name = config.get("transformer_model_name", DEFAULT_TRANSFORMER_MODEL)

    _enable_hf_download_logging()
    _log_hf_model_cache_status(model_name)

    logger.info("Loading E5 tokenizer: %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer ready for %s", model_name)

    device = _select_device()
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(
        "Loading E5 weights: %s (device=%s, dtype=%s) ...",
        model_name, device.type, str(model_dtype).split(".")[-1],
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    logger.info("E5 model ready on %s", device.type)

    _embedder = {"tokenizer": tokenizer, "model": model, "device": device}
    return _embedder


def release_embedder():
    """Unload the transformer model to free memory after batch operations."""
    global _embedder
    if _embedder is not None:
        import gc
        device_type = _embedder["device"].type
        del _embedder
        _embedder = None
        gc.collect()
        if device_type == "mps":
            import torch
            torch.mps.empty_cache()
        elif device_type == "cuda":
            import torch
            torch.cuda.empty_cache()
        logger.info("Transformer model released from memory.")


def _model_uses_passage_prefix(model_name: str) -> bool:
    """E5-family models expect a passage: prefix for document embedding."""
    name = (model_name or "").lower()
    return "e5" in name


def _embedding_input_text(text: str, model_name: Optional[str] = None) -> str:
    """Apply model-specific input formatting at embed time (not TF-IDF/recipe)."""
    if model_name is None:
        model_name = get_config().get("transformer_model_name", DEFAULT_TRANSFORMER_MODEL)
    body = text or ""
    if _model_uses_passage_prefix(model_name) and not body.startswith("passage:"):
        return "passage: " + body
    return body


def _embedding_runtime_settings():
    """Resolve max tokens and batch size from config."""
    config = get_config()
    model_name = config.get("transformer_model_name", DEFAULT_TRANSFORMER_MODEL)
    try:
        max_length = int(config.get("embedding_max_tokens", EMBEDDING_MAX_LENGTH))
    except (TypeError, ValueError):
        max_length = EMBEDDING_MAX_LENGTH
    max_length = max(64, min(512, max_length))
    try:
        batch_size = int(config.get("embedding_batch_size", 0) or 0)
    except (TypeError, ValueError):
        batch_size = 0
    if batch_size <= 0:
        batch_size = 4 if max_length >= 512 else 8
    return model_name, max_length, batch_size


def compute_embeddings(
    texts: List[str],
    batch_size: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Compute mean-pooled embeddings for a list of texts using the transformer.
    Returns ndarray of shape (len(texts), embedding_dim).

    Mean pooling averages all non-padding token embeddings.  E5 inputs receive
    a ``passage:`` prefix at embed time only.
    """
    import torch

    model_name, max_length, default_batch = _embedding_runtime_settings()
    if batch_size is None:
        batch_size = default_batch

    embedder = _get_embedder()
    tokenizer = embedder["tokenizer"]
    model = embedder["model"]
    device = embedder["device"]

    n_texts = len(texts)
    all_embeddings = []
    for i in range(0, n_texts, batch_size):
        batch_texts = [
            _embedding_input_text(t, model_name) for t in texts[i:i + batch_size]
        ]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        token_embeddings = outputs.last_hidden_state.float()
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / counts).cpu().numpy()
        all_embeddings.append(mean_pooled)
        done = min(i + len(batch_texts), n_texts)
        if progress_cb:
            progress_cb(done, n_texts)
        elif done == n_texts or (done % max(batch_size * 10, 40) == 0):
            logger.info("Embedded %d / %d texts", done, n_texts)

    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a 1-D float32 embedding to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize bytes back to a 1-D float32 embedding."""
    emb = np.frombuffer(data, dtype=np.float32)
    if len(emb) != dim:
        logger.warning(
            "Embedding dimension mismatch: expected %d, got %d. "
            "Stale embedding? Try recomputing embeddings from Settings.",
            dim, len(emb)
        )
    return emb


def invalidate_embedder_cache():
    """Clear the cached transformer model (call when model name changes)."""
    global _embedder
    _embedder = None


CONTENT_CAP = 3000
LEGAL_CONTENT_CAP = 12000  # lex / Leg statutory body text from Seismo
EMBEDDING_MAX_LENGTH = 512
SOURCE_NAME_CAP = 120
SOURCE_CATEGORY_CAP = 80
LEGAL_SIGNAL_CAP = 8  # max distinct signals to prepend per entry
LEGAL_EXTRACTIVE_SNIPPET_CAP = 6
EXTRACTIVE_SNIPPET_CAP = 3  # max excerpts from full content around legal-signal hits

# Human-readable source_type lines for E5 embedding context (not sent to Seismo).
SOURCE_TYPE_DESCRIPTIONS = {
    "rss": "This is a news feed article.",
    "substack": "This is a newsletter article.",
    "email": "This is an email message.",
    "lex_eu": "This is an official EU legal publication.",
    "lex_ch": "This is an official Swiss legal publication.",
    "leg_eu": "This is an EU legislative document.",
    "leg_ch": "This is a Swiss legislative document.",
}

PRIORITY_TRAINING_LABELS = frozenset({"investigation_lead", "important"})
SOFT_DISTILL_MIN_PROBA = 0.02
HUMAN_DISTILL_WEIGHT = 3.0


_LEGAL_PATTERNS_CACHE = {"key": None, "compiled": []}


def _compiled_legal_patterns(patterns: Optional[List[str]] = None):
    """Return cached compiled regexes for the configured legal-signal patterns.

    Each entry is (original_phrase, compiled_regex).  Malformed regexes fall
    back to a literal search so a single bad pattern never kills training.
    """
    if patterns is None:
        patterns = get_config().get("legal_signal_patterns") or []
    key = tuple(patterns)
    if _LEGAL_PATTERNS_CACHE["key"] == key:
        return _LEGAL_PATTERNS_CACHE["compiled"]
    import re as _re
    compiled = []
    for raw in patterns:
        phrase = (raw or "").strip()
        if not phrase:
            continue
        try:
            rx = _re.compile(phrase, _re.IGNORECASE | _re.UNICODE)
        except _re.error:
            rx = _re.compile(_re.escape(phrase), _re.IGNORECASE | _re.UNICODE)
        compiled.append((phrase, rx))
    _LEGAL_PATTERNS_CACHE["key"] = key
    _LEGAL_PATTERNS_CACHE["compiled"] = compiled
    return compiled


def _detect_legal_signals(text: str, patterns: Optional[List[str]] = None) -> List[str]:
    """Return the list of configured phrases that appear in text (order preserved)."""
    if not text:
        return []
    hits = []
    seen = set()
    for phrase, rx in _compiled_legal_patterns(patterns):
        if phrase in seen:
            continue
        if rx.search(text):
            hits.append(phrase)
            seen.add(phrase)
            if len(hits) >= LEGAL_SIGNAL_CAP:
                break
    return hits


def _extractive_snippets(full_content: str, patterns: Optional[List[str]] = None,
                         window: Optional[int] = None,
                         max_snippets: int = EXTRACTIVE_SNIPPET_CAP) -> List[str]:
    """
    Pull short excerpts from uncapped content around legal-signal pattern hits so
    clauses buried after the content cap still reach the embedding.
    """
    if not full_content or not patterns:
        return []
    if window is None:
        try:
            window = int(get_config().get("embedding_extractive_window", 280) or 280)
        except (TypeError, ValueError):
            window = 280
    window = max(80, min(800, window))

    snippets = []
    seen_norm = set()
    for _phrase, rx in _compiled_legal_patterns(patterns):
        for match in rx.finditer(full_content):
            start = max(0, match.start() - window // 2)
            end = min(len(full_content), match.end() + window // 2)
            snippet = full_content[start:end].strip()
            if not snippet:
                continue
            norm = snippet.lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            snippets.append(snippet)
            if len(snippets) >= max_snippets:
                return snippets
    return snippets


def _content_cap_for_entry(entry: dict) -> int:
    """Character cap for corpus text; higher for lex / Leg entries."""
    cfg = get_config()
    if is_legal_training_entry(entry):
        try:
            cap = int(cfg.get("embedding_legal_content_cap", LEGAL_CONTENT_CAP) or LEGAL_CONTENT_CAP)
        except (TypeError, ValueError):
            cap = LEGAL_CONTENT_CAP
    else:
        try:
            cap = int(cfg.get("embedding_content_cap", CONTENT_CAP) or CONTENT_CAP)
        except (TypeError, ValueError):
            cap = CONTENT_CAP
    return max(500, min(50000, cap))


def _natural_source_context(entry: dict, signals: Optional[List[str]] = None) -> str:
    """Build a short natural-language context block for E5 (not Seismo schema)."""
    parts = []
    st = (entry.get("source_type") or "").strip()
    if st:
        parts.append(SOURCE_TYPE_DESCRIPTIONS.get(st, "Source type: {}.".format(st)))
    sn = (entry.get("source_name") or "").strip()
    if sn:
        parts.append("Published by {}.".format(sn[:SOURCE_NAME_CAP]))
    sc = (entry.get("source_category") or "").strip()
    if sc:
        parts.append("Category: {}.".format(sc[:SOURCE_CATEGORY_CAP]))
    if signals:
        parts.append("Legal signals detected: {}.".format(", ".join(signals)))
    return " ".join(parts)


def _build_entry_text(entry: dict, legal_patterns: Optional[List[str]] = None) -> str:
    """
    Build text for embedding/scoring from an entry's fields.

    Title is repeated so it dominates the embedding even for entries with long
    content.  Corpus text uses ``training_corpus_text`` (HTML stripped; lex/Leg
    boilerplate skipped) and is capped per entry type — higher for statutory
    sources so Seismo's full body text informs training.

    A natural-language context prefix (source type, publisher, legal signals)
    helps E5 interpret institutional gravity.  Legal-signal patterns are scanned
    on the full uncapped corpus; matching regions are injected as extractive
    snippets so buried clauses still influence the embedding.
    """
    title = entry.get("title", "").strip()
    full_corpus = training_corpus_text(entry)
    content_cap = _content_cap_for_entry(entry)
    corpus = full_corpus[:content_cap]

    scan_text = " ".join(part for part in [title, full_corpus] if part)
    signals = _detect_legal_signals(scan_text, legal_patterns)
    context = _natural_source_context(entry, signals)
    snippet_cap = (
        LEGAL_EXTRACTIVE_SNIPPET_CAP
        if is_legal_training_entry(entry)
        else EXTRACTIVE_SNIPPET_CAP
    )
    snippets = _extractive_snippets(
        full_corpus, legal_patterns, max_snippets=snippet_cap,
    )

    body_parts = []
    if snippets:
        body_parts.extend(snippets)
    body_parts.extend(part for part in [title, title, corpus] if part)
    body = "\n".join(body_parts) if body_parts else "(empty)"

    if context:
        return "{}\n\n{}".format(context, body)
    return body


def embed_entries(
    entries: List[dict],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[bytes]:
    """Compute embeddings for a list of entry dicts. Returns list of bytes."""
    patterns = get_config().get("legal_signal_patterns") or []
    n = len(entries)
    logger.info(
        "Preparing entry text for %d rows (lex/Leg bodies can take minutes before E5 loads)...",
        n,
    )
    texts = []
    for i, entry in enumerate(entries):
        texts.append(_build_entry_text(entry, legal_patterns=patterns))
        if n <= 50 or (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info("  entry text prep %d / %d", i + 1, n)
    logger.info("Entry text ready; loading E5 and encoding %d texts", n)
    embeddings = compute_embeddings(texts, progress_cb=progress_cb)
    return [embedding_to_bytes(emb) for emb in embeddings]


# ═══════════════════════════════════════════════════════════════════
#  Sample-weight helpers (time decay + reasoning boost)
# ═══════════════════════════════════════════════════════════════════


def _parse_label_ts(value) -> Optional[float]:
    """Parse an ISO-ish timestamp stored in the labels table into a POSIX ts."""
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    from datetime import datetime as _dt
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    try:
        return _dt.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _decay_half_life_for_label(label: str, base_half_life: float,
                                config: dict) -> float:
    """Effective half-life in days for a label class; 0 means no decay."""
    if base_half_life <= 0:
        return 0.0
    if config.get("label_time_decay_priority_exempt", True):
        if label in PRIORITY_TRAINING_LABELS:
            return 0.0
    if label == "noise":
        try:
            accel = float(config.get("label_time_decay_noise_accel", 3.0) or 3.0)
        except (TypeError, ValueError):
            accel = 3.0
        accel = max(1.0, accel)
        return base_half_life / accel
    return base_half_life


def compute_sample_weights(labeled: List[dict],
                            config: Optional[dict] = None) -> np.ndarray:
    """Per-label training weight combining time decay and reasoning boost.

    - Time decay: ``weight = 0.5 ** (age_days / half_life)`` clamped to a floor.
      Uses ``updated_at`` when present (freshly re-labeled rows stay strong)
      else ``created_at``.  Missing timestamps get weight 1.0.
      When decay is enabled, ``investigation_lead`` and ``important`` skip decay
      by default; ``noise`` decays faster (``label_time_decay_noise_accel``).
    - Reasoning boost: multiplied in for labels with a non-empty reasoning note.
    - Both default to no-op so existing models train identically.
    """
    if config is None:
        config = get_config()
    try:
        half_life = float(config.get("label_time_decay_days", 0) or 0.0)
    except (TypeError, ValueError):
        half_life = 0.0
    try:
        floor = float(config.get("label_time_decay_floor", 0.2) or 0.0)
    except (TypeError, ValueError):
        floor = 0.0
    floor = max(0.0, min(1.0, floor))
    try:
        boost = float(config.get("reasoning_weight_boost", 1.0) or 1.0)
    except (TypeError, ValueError):
        boost = 1.0
    boost = max(0.0, boost)

    import time as _time
    now = _time.time()
    weights = np.ones(len(labeled), dtype=np.float64)

    if half_life > 0:
        for i, lbl in enumerate(labeled):
            ts = _parse_label_ts(lbl.get("updated_at")) or _parse_label_ts(lbl.get("created_at"))
            if ts is None:
                continue
            label = lbl.get("label") or ""
            effective_hl = _decay_half_life_for_label(label, half_life, config)
            if effective_hl <= 0:
                continue
            age_days = max(0.0, (now - ts) / 86400.0)
            w = 0.5 ** (age_days / effective_hl)
            weights[i] = max(floor, w)

    if abs(boost - 1.0) > 1e-6:
        for i, lbl in enumerate(labeled):
            reason = (lbl.get("reasoning") or "").strip()
            if reason:
                weights[i] *= boost

    return weights


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF pipeline (fallback + recipe distiller)
# ═══════════════════════════════════════════════════════════════════

def _prepare_text(entries: List[dict]) -> pd.DataFrame:
    """Convert entries into a DataFrame with text and structured features."""
    patterns = get_config().get("legal_signal_patterns") or []
    rows = []
    for e in entries:
        text = _build_entry_text(e, legal_patterns=patterns)
        rows.append({
            "text": text,
            "source_type": e.get("source_type", "unknown"),
            "text_length": len(text),
        })
    return pd.DataFrame(rows)


def build_tfidf_pipeline() -> Pipeline:
    """Build the scikit-learn pipeline: TF-IDF + structured features -> LogReg."""
    text_transformer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95,
    )

    source_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "text"),
            ("source", source_transformer, ["source_type"]),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    return pipeline


def _relaxed_tfidf_vectorizer(min_df: int = 1) -> TfidfVectorizer:
    """TF-IDF settings for small distillation corpora."""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=min_df,
        max_df=1.0 if min_df <= 1 else 0.95,
    )


def _fit_tfidf_student(student: Pipeline, df: pd.DataFrame,
                       targets: List[dict], class_names: List[str]) -> Pipeline:
    """
    Fit a TF-IDF student from distillation targets.

    Each target is either:
    - ``{"hard": "important", "weight": 3.0}`` for human labels
    - ``{"soft": {"important": 0.6, "investigation_lead": 0.35, ...}}`` for teacher probs

    Soft targets expand into one weighted row per class (equivalent to soft CE).
    """
    expanded_rows = []
    y_train = []
    sample_weights = []

    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        target = targets[i]
        if "hard" in target:
            expanded_rows.append(row)
            y_train.append(target["hard"])
            sample_weights.append(float(target.get("weight", 1.0)))
            continue
        probs = target.get("soft") or {}
        for cls in class_names:
            p = float(probs.get(cls, 0.0) or 0.0)
            if p < SOFT_DISTILL_MIN_PROBA:
                continue
            expanded_rows.append(row)
            y_train.append(cls)
            sample_weights.append(p)

    if len(expanded_rows) < 20:
        raise ValueError("Not enough distillation rows after soft-label expansion")

    fit_df = pd.DataFrame(expanded_rows)
    sw = np.asarray(sample_weights, dtype=np.float64)

    try:
        student.fit(fit_df, y_train, classifier__sample_weight=sw)
    except ValueError:
        student.named_steps["features"].transformers[0] = (
            "text",
            _relaxed_tfidf_vectorizer(min_df=1),
            "text",
        )
        student.fit(fit_df, y_train, classifier__sample_weight=sw)
    return student


# ═══════════════════════════════════════════════════════════════════
#  Unified interface — delegates to the configured architecture
# ═══════════════════════════════════════════════════════════════════

def _get_architecture() -> str:
    """Return the current architecture from config."""
    config = get_config()
    return config.get("model_architecture", "transformer")


def train(profile_id: int = 1) -> dict:
    """Train a new model on labeled entries for the given profile."""
    arch = _get_architecture()
    if arch == "transformer":
        return _train_transformer(profile_id=profile_id)
    return _train_tfidf(profile_id=profile_id)


def get_active_model_paths(profile_id: int = 1) -> Optional[dict]:
    """Resolved paths for the active model row (model, optional recipe, optional calibration)."""
    model_info = db.get_active_model(profile_id=profile_id)
    if not model_info or not model_info.get("model_path"):
        return None
    model_path = model_info["model_path"]
    mp = Path(model_path)
    if not mp.exists():
        return None
    out = {
        "model_path": str(mp.resolve()),
        "recipe_path": "",
        "calibration_path": "",
    }
    rp = model_info.get("recipe_path") or ""
    if rp and Path(rp).exists():
        out["recipe_path"] = str(Path(rp).resolve())
    cal = calibration_sidecar_path(str(mp))
    if cal.exists():
        out["calibration_path"] = str(cal.resolve())
    return out


def load_active_model(profile_id: int = 1):
    """Load the currently active model for a profile."""
    paths = get_active_model_paths(profile_id=profile_id)
    if not paths:
        return None
    return joblib.load(paths["model_path"])


def score_entries(entries: List[dict], profile_id: int = 1) -> List[dict]:
    """Score entries using the active model for the given profile."""
    model_info = db.get_active_model(profile_id=profile_id)
    if not model_info:
        return []

    arch = model_info.get("architecture", "tfidf")
    if arch == "transformer":
        return _score_transformer(entries, model_info)
    return _score_tfidf(entries, model_info)


def get_feature_importance(profile_id: int = 1) -> dict:
    """Get feature importance. For TF-IDF models only (used by recipe distiller)."""
    model_info = db.get_active_model(profile_id=profile_id)
    if not model_info:
        return {}

    arch = model_info.get("architecture", "tfidf")

    if arch == "tfidf":
        return _get_tfidf_feature_importance(profile_id=profile_id)

    return {}


# ═══════════════════════════════════════════════════════════════════
#  Transformer training + scoring
# ═══════════════════════════════════════════════════════════════════

class _LabelDecodingClassifier:
    """Wraps a Pipeline that was trained on integer-encoded labels and
    translates predictions back to the original string labels.  Exposes
    the same interface as sklearn classifiers (predict, predict_proba,
    classes_) so scoring and explainer code works unchanged."""

    def __init__(self, pipeline, label_encoder):
        self._pipeline = pipeline
        self._le = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        encoded = self._pipeline.predict(X)
        return self._le.inverse_transform(encoded)

    def predict_proba(self, X):
        return self._pipeline.predict_proba(X)


def _train_transformer(profile_id: int = 1) -> dict:
    """Train a LogReg classifier on cached transformer embeddings for a profile."""
    config = db.get_effective_config(profile_id)
    min_labels = config.get("min_labels_to_train", 20)
    embedding_dim = config.get("embedding_dim", 768)

    labeled = db.get_all_labels(profile_id)
    if len(labeled) < min_labels:
        return {
            "success": False,
            "error": "Need at least {} labels to train. Currently have {}.".format(
                min_labels, len(labeled)
            ),
            "label_count": len(labeled),
        }

    # Batch-fetch all embeddings in one query
    conn = db.get_db()
    emb_map = {}
    rows = conn.execute(
        "SELECT entry_type, entry_id, embedding FROM entries WHERE embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        emb_map[db.entry_key_from_mapping(row)] = row["embedding"]
    conn.close()

    X_list = []
    y_list = []
    lbl_list = []
    missing_embeddings = []

    for lbl in labeled:
        key = db.entry_key_from_mapping(lbl)
        emb_bytes = emb_map.get(key)
        if emb_bytes:
            emb = bytes_to_embedding(emb_bytes, embedding_dim)
            X_list.append(emb)
            y_list.append(lbl["label"])
            lbl_list.append(lbl)
        else:
            missing_embeddings.append(lbl)

    # Compute missing embeddings on the fly
    if missing_embeddings:
        logger.info("Computing %d missing embeddings for training", len(missing_embeddings))
        emb_bytes = embed_entries(missing_embeddings)
        updates = []
        for lbl, eb in zip(missing_embeddings, emb_bytes):
            updates.append((eb, lbl["entry_type"], lbl["entry_id"]))
            emb = bytes_to_embedding(eb, embedding_dim)
            X_list.append(emb)
            y_list.append(lbl["label"])
            lbl_list.append(lbl)
        db.store_embeddings_batch(updates)

    if len(X_list) < min_labels:
        return {
            "success": False,
            "error": "Not enough entries with embeddings. Try syncing first.",
            "label_count": len(labeled),
        }

    X = np.array(X_list)
    y = y_list
    sw_all = compute_sample_weights(lbl_list, config)
    labels_series = pd.Series(y)
    label_counts = labels_series.value_counts()
    min_class_count = label_counts.min()

    if min_class_count < 2:
        X_train, X_test = X, X
        y_train, y_test = y, y
        sw_train = sw_all
        split_note = "All data used for training (some classes have <2 samples)"
    else:
        test_size = _holdout_test_fraction(min_class_count, len(y))
        X_train, X_test, y_train, y_test, sw_train, _sw_test = (
            _stable_train_test_split(X, y, sw_all, lbl_list, test_size=test_size)
        )
        te = len(y_test)
        tr = len(y_train)
        split_note = "stable entry holdout {}/{} train/test (~{}% test)".format(
            int(round(100.0 * tr / (tr + te))),
            int(round(100.0 * te / (tr + te))),
            int(round(100.0 * te / (tr + te))),
        )

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    fit_kwargs = _transformer_fit_kwargs(sw_train)

    train_min_class = _min_class_count_in_labels(y_train)
    n_folds = _oof_fold_count(len(y_train), train_min_class)
    class_names_fit = le.classes_.tolist()
    oof_samples = 0
    if n_folds >= 2:
        oof_logits, oof_y = _collect_oof_logits(
            X_train, y_train, sw_train, le, n_folds
        )
        oof_samples = len(oof_y)
        if oof_samples >= 3:
            temperature = _fit_temperature_scalar(
                oof_logits, np.array(oof_y), class_names_fit
            )
            cal_note = "temperature T={:.3f} fit on {} OOF samples ({} folds)".format(
                temperature, oof_samples, n_folds
            )
        else:
            temperature = 1.0
            cal_note = "temperature T=1.0 (OOF degenerate; calibration inactive)"
    else:
        temperature = 1.0
        cal_note = "temperature T=1.0 (too few samples for OOF; calibration inactive)"

    cal_dict = {
        "version": 1,
        "method": "temperature",
        "calibration_fit": "oof" if oof_samples >= 3 else "none",
        "oof_folds": n_folds if oof_samples >= 3 else 0,
        "oof_samples": oof_samples,
        "temperature": temperature,
        "class_names": class_names_fit,
    }

    clf_pipeline = build_transformer_head_pipeline()
    clf_pipeline.fit(X_train, y_train_enc, **fit_kwargs)

    clf = _LabelDecodingClassifier(clf_pipeline, le)

    # Evaluate on held-out test using the same probabilities as scoring
    version = db.get_next_model_version(profile_id)
    model_filename = "model_v{}.joblib".format(version)
    model_path = str(MODELS_DIR / model_filename)
    joblib.dump(clf, model_path)
    write_calibration_sidecar(model_path, cal_dict)

    probs_test, cn = classifier_probabilities(clf, X_test, "", cal=cal_dict)
    y_pred_idx = np.argmax(probs_test, axis=1)
    y_pred = np.array([cn[i] for i in y_pred_idx])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    label_dist = {k: int(v) for k, v in labels_series.value_counts().items()}

    db.save_model_record(
        version=version,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        label_count=len(labeled),
        feature_count=X.shape[1],
        model_path=model_path,
        architecture="transformer",
        profile_id=profile_id,
        label_distribution=label_dist,
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = json.loads(json.dumps(report, default=float))

    return {
        "success": True,
        "version": version,
        "architecture": "transformer",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "label_count": len(labeled),
        "label_distribution": label_dist,
        "feature_count": int(X.shape[1]),
        "model_path": model_path,
        "split_note": split_note + "; " + cal_note,
        "calibration_temperature": round(temperature, 4),
        "calibration_note": cal_note,
        "class_report": report,
    }


MAX_ONTHEFLY_EMBEDDINGS = 10


def _score_transformer(entries: List[dict], model_info: dict) -> List[dict]:
    """Score entries using cached embeddings + LogReg classifier.

    On-the-fly embedding computation is capped at MAX_ONTHEFLY_EMBEDDINGS so
    that page loads stay fast.  Entries beyond the cap are silently omitted
    from results -- callers already handle partial score lists.  The sync
    path (_compute_pending_embeddings) handles bulk embedding computation.
    """
    model_path = model_info.get("model_path")
    if not model_path or not Path(model_path).exists():
        return []

    clf = joblib.load(model_path)
    pid = int(model_info.get("profile_id") or 1)
    config = db.get_effective_config(pid)
    embedding_dim = config.get("embedding_dim", 768)

    # Batch-fetch all embeddings in one query
    conn = db.get_db()
    emb_map = {}
    rows = conn.execute(
        "SELECT entry_type, entry_id, embedding FROM entries WHERE embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        emb_map[db.entry_key_from_mapping(row)] = row["embedding"]
    conn.close()

    embeddings = []
    to_compute = []
    to_compute_indices = []

    for i, entry in enumerate(entries):
        key = db.entry_key_from_mapping(entry)
        emb_bytes = emb_map.get(key)
        if emb_bytes:
            embeddings.append((i, bytes_to_embedding(emb_bytes, embedding_dim)))
        else:
            to_compute.append(entry)
            to_compute_indices.append(i)

    # Compute missing embeddings -- capped to avoid blocking page loads.
    # When the model was just changed, hundreds of entries may lack embeddings;
    # computing them all here would hang the UI.  The sync path handles bulk
    # computation; here we only do a small batch for immediate scoring.
    if to_compute:
        if len(to_compute) > MAX_ONTHEFLY_EMBEDDINGS:
            logger.info(
                "Skipping on-the-fly embedding for %d entries (cap=%d). "
                "Run sync to compute pending embeddings.",
                len(to_compute), MAX_ONTHEFLY_EMBEDDINGS,
            )
            to_compute = to_compute[:MAX_ONTHEFLY_EMBEDDINGS]
            to_compute_indices = to_compute_indices[:MAX_ONTHEFLY_EMBEDDINGS]

        try:
            new_emb_bytes = embed_entries(to_compute)
            new_emb_arrays = [bytes_to_embedding(b, embedding_dim) for b in new_emb_bytes]
            updates = []
            for entry, eb in zip(to_compute, new_emb_bytes):
                updates.append((eb, entry["entry_type"], entry["entry_id"]))
            db.store_embeddings_batch(updates)
            for idx, emb in zip(to_compute_indices, new_emb_arrays):
                embeddings.append((idx, emb))
        except Exception as e:
            logger.warning("On-the-fly embedding failed (model loading?): %s", e)

    if not embeddings:
        return []

    # Sort by original index and build feature matrix
    embeddings.sort(key=lambda x: x[0])
    scored_indices = [idx for idx, _ in embeddings]
    X = np.array([emb for _, emb in embeddings])

    probabilities, class_names = classifier_probabilities(clf, X, model_path)

    # Build results only for entries that had embeddings
    results = []
    for j, orig_idx in enumerate(scored_indices):
        entry = entries[orig_idx]
        probs = dict(zip(class_names, probabilities[j].tolist()))
        composite = _relevance_from_probs(probs, class_names)
        p_lead = float(probs.get("investigation_lead", 0.0))
        relevance = _discovery_adjusted_relevance(composite, p_lead, profile_id=pid)
        pred_idx = int(np.argmax(probabilities[j]))
        predicted_label = class_names[pred_idx]
        results.append({
            "entry_type": entry["entry_type"],
            "entry_id": entry["entry_id"],
            "relevance_score": round(relevance, 4),
            "predicted_label": predicted_label,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        })

    return results


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF training + scoring (fallback)
# ═══════════════════════════════════════════════════════════════════

def _train_tfidf(profile_id: int = 1) -> dict:
    """Train using the original TF-IDF + LogReg pipeline for a profile."""
    config = db.get_effective_config(profile_id)
    min_labels = config.get("min_labels_to_train", 20)

    labeled = db.get_all_labels(profile_id)
    if len(labeled) < min_labels:
        return {
            "success": False,
            "error": "Need at least {} labels to train. Currently have {}.".format(
                min_labels, len(labeled)
            ),
            "label_count": len(labeled),
        }

    df = _prepare_text(labeled)
    labels = [e["label"] for e in labeled]
    sw_all = compute_sample_weights(labeled, config)

    label_counts = pd.Series(labels).value_counts()
    min_class_count = label_counts.min()

    if min_class_count < 2:
        X_train, X_test = df, df
        y_train, y_test = labels, labels
        sw_train = sw_all
        split_note = "All data used for training (some classes have <2 samples)"
    else:
        test_size = _holdout_test_fraction(min_class_count, len(labels))
        X_train, X_test, y_train, y_test, sw_train, _sw_test = (
            _stable_train_test_split(df, labels, sw_all, labeled, test_size=test_size)
        )
        te = len(y_test)
        tr = len(y_train)
        split_note = "stable entry holdout {}/{} train/test (~{}% test)".format(
            int(round(100.0 * tr / (tr + te))),
            int(round(100.0 * te / (tr + te))),
            int(round(100.0 * te / (tr + te))),
        )

    # Slice of training fold for temperature calibration (kept out of TF-IDF fit)
    X_tr, y_tr = X_train, y_train
    sw_tr = sw_train
    X_val, y_val = None, None
    train_min_class = _min_class_count_in_labels(y_train)
    if train_min_class >= 2 and len(X_train) >= 30:
        try:
            xtr, xv, ytr, yv, sw_tr_new, _sw_v = train_test_split(
                X_train, y_train, sw_train,
                test_size=0.15, stratify=y_train, random_state=43,
            )
            if len(xv) >= 5:
                X_tr, X_val = xtr, xv
                y_tr, y_val = ytr, yv
                sw_tr = sw_tr_new
        except ValueError:
            pass

    pipeline = build_tfidf_pipeline()

    use_sw = (
        sw_tr is not None
        and len(sw_tr) == len(y_tr)
        and float(np.std(np.asarray(sw_tr, dtype=np.float64))) > 1e-6
    )
    fit_kwargs = {"classifier__sample_weight": np.asarray(sw_tr, dtype=np.float64)} if use_sw else {}

    try:
        pipeline.fit(X_tr, y_tr, **fit_kwargs)
    except ValueError:
        pipeline.named_steps["features"].transformers[0] = (
            "text",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                sublinear_tf=True,
                strip_accents="unicode",
                min_df=1,
                max_df=1.0,
            ),
            "text",
        )
        pipeline.fit(X_tr, y_tr, **fit_kwargs)

    class_names_fit = pipeline.classes_.tolist()
    if X_val is not None and len(X_val) >= 3:
        logits_val = logits_for_classifier_head(pipeline, X_val)
        temperature = _fit_temperature_scalar(
            logits_val, np.array(y_val), class_names_fit
        )
        cal_note = "temperature T={:.3f} fit on {} validation samples".format(
            temperature, len(X_val)
        )
    else:
        temperature = 1.0
        cal_note = "temperature T=1.0 (no validation slice; calibration inactive)"

    cal_dict = {
        "version": 1,
        "method": "temperature",
        "temperature": temperature,
        "class_names": class_names_fit,
    }

    version = db.get_next_model_version(profile_id)
    model_filename = "model_v{}.joblib".format(version)
    model_path = str(MODELS_DIR / model_filename)
    joblib.dump(pipeline, model_path)
    write_calibration_sidecar(model_path, cal_dict)

    probs_test, cn = classifier_probabilities(pipeline, X_test, "", cal=cal_dict)
    y_pred_idx = np.argmax(probs_test, axis=1)
    y_pred = np.array([cn[i] for i in y_pred_idx])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    tfidf = pipeline.named_steps["features"].transformers_[0][1]
    feature_count = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0

    label_dist = {k: int(v) for k, v in pd.Series(labels).value_counts().items()}

    db.save_model_record(
        version=version,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        label_count=len(labeled),
        feature_count=feature_count,
        model_path=model_path,
        architecture="tfidf",
        profile_id=profile_id,
        label_distribution=label_dist,
    )

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = json.loads(json.dumps(report, default=float))

    return {
        "success": True,
        "version": version,
        "architecture": "tfidf",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "label_count": len(labeled),
        "label_distribution": label_dist,
        "feature_count": int(feature_count),
        "model_path": model_path,
        "split_note": split_note + "; " + cal_note,
        "calibration_temperature": round(temperature, 4),
        "calibration_note": cal_note,
        "class_report": report,
    }


def _score_tfidf(entries: List[dict], model_info: Optional[dict] = None) -> List[dict]:
    """Score entries using the TF-IDF + LogReg pipeline."""
    if model_info is None:
        model_info = db.get_active_model()
    if not model_info:
        return []
    pid = int(model_info.get("profile_id") or 1)
    model = load_active_model(profile_id=pid)
    if model is None:
        return []
    model_path = model_info.get("model_path") or ""

    df = _prepare_text(entries)
    probabilities, class_names = classifier_probabilities(model, df, model_path)

    results = []
    for i, entry in enumerate(entries):
        probs = dict(zip(class_names, probabilities[i].tolist()))
        composite = _relevance_from_probs(probs, class_names)
        p_lead = float(probs.get("investigation_lead", 0.0))
        relevance = _discovery_adjusted_relevance(composite, p_lead, profile_id=pid)
        pred_idx = int(np.argmax(probabilities[i]))
        predicted_label = class_names[pred_idx]
        results.append({
            "entry_type": entry["entry_type"],
            "entry_id": entry["entry_id"],
            "relevance_score": round(relevance, 4),
            "predicted_label": predicted_label,
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        })

    return results


def _get_tfidf_feature_importance(profile_id: int = 1) -> dict:
    """Get feature coefficients from the active TF-IDF model."""
    model = load_active_model(profile_id=profile_id)
    if model is None:
        return {}

    # Check if this is actually a TF-IDF pipeline (has 'features' step)
    if not hasattr(model, "named_steps"):
        return {}

    preprocessor = model.named_steps.get("features")
    if preprocessor is None:
        return {}

    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()

    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []

    all_names = tfidf_names + source_names
    classifier = model.named_steps["classifier"]
    class_names = classifier.classes_.tolist()
    coef_matrix = classifier.coef_

    result = {}
    for i, cls in enumerate(class_names):
        coefs = coef_matrix[i]
        pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        result[cls] = pairs

    return result


# ═══════════════════════════════════════════════════════════════════
#  Knowledge distillation: train a TF-IDF student from transformer scores
#  (used by the recipe distiller to produce keyword-weight recipes)
# ═══════════════════════════════════════════════════════════════════

def train_tfidf_student(profile_id: int = 1) -> Optional[Pipeline]:
    """
    Train a TF-IDF + LogReg 'student' model that learns from the transformer
    model's predictions on ALL entries (not just labeled ones).

    Uses soft teacher probabilities by default so borderline entries teach
    mixed class weights to the keyword recipe.  Human labels override with
    a higher sample weight.

    Returns the trained student pipeline, or None if not possible.
    """
    model_info = db.get_active_model(profile_id)
    if not model_info or model_info.get("architecture") != "transformer":
        return None

    config = db.get_effective_config(profile_id)
    use_soft = bool(config.get("distillation_soft_labels", True))

    all_entries = db.get_all_entries()
    if len(all_entries) < 20:
        return None

    scores = score_entries(all_entries, profile_id=profile_id)
    if not scores:
        return None

    teacher_prob_map = {
        db.entry_key_from_mapping(s): s.get("probabilities") or {}
        for s in scores
    }
    teacher_label_map = {
        db.entry_key_from_mapping(s): s["predicted_label"]
        for s in scores
    }

    human_labels = {
        db.entry_key_from_mapping(lbl): lbl["label"]
        for lbl in db.get_all_labels(profile_id)
    }

    scored_entries = []
    targets = []
    class_names = list(CLASSES)

    for entry in all_entries:
        key = db.entry_key_from_mapping(entry)
        if key in human_labels:
            scored_entries.append(entry)
            targets.append({"hard": human_labels[key], "weight": HUMAN_DISTILL_WEIGHT})
        elif use_soft and teacher_prob_map.get(key):
            scored_entries.append(entry)
            targets.append({"soft": teacher_prob_map[key]})
        elif key in teacher_label_map:
            scored_entries.append(entry)
            targets.append({"hard": teacher_label_map[key], "weight": 1.0})

    if len(scored_entries) < 20:
        return None

    df = _prepare_text(scored_entries)
    student = build_tfidf_pipeline()

    try:
        _fit_tfidf_student(student, df, targets, class_names)
    except ValueError as exc:
        logger.warning("TF-IDF student distillation failed: %s", exc)
        return None

    return student
