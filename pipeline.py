"""
ML Pipeline for Magnitu 2.

Two architectures behind the same interface:
- "tfidf":       TF-IDF + Logistic Regression (original Magnitu 1)
- "transformer": Cached XLM-RoBERTa embeddings + MLP classifier (Magnitu 2)

The transformer path computes mean-pooled embeddings once at sync time and
stores them in the DB.  Training and scoring use these cached embeddings with
a lightweight MLP classifier — so labeling stays snappy.  The TF-IDF path is
kept as a fallback and is used by the recipe distiller for knowledge
distillation.
"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

from typing import Optional, List, Tuple, Dict

import db
from config import MODELS_DIR, get_config

logger = logging.getLogger(__name__)

CLASSES = ["investigation_lead", "important", "background", "noise"]

CLASS_WEIGHT_MAP = {
    "investigation_lead": 1.0,
    "important": 0.66,
    "background": 0.33,
    "noise": 0.0,
}


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
    Magnitu layouts explicitly.  For the transformer path the head is an
    ``MLPClassifier`` which has no ``decision_function`` — we fall back to
    ``log(predict_proba)`` in that case (see ``_step_logits``).
    """
    if hasattr(clf, "_pipeline"):
        return logits_for_classifier_head(clf._pipeline, X)
    if hasattr(clf, "named_steps"):
        steps = clf.named_steps
        if "mlp" in steps and "scaler" in steps:
            scaled = steps["scaler"].transform(X)
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


def _discovery_adjusted_relevance(composite: float, p_lead: float) -> float:
    """
    Optional blend toward investigation_lead for discovery (config:
    discovery_lead_blend in [0, 0.25]).
    """
    cfg = get_config()
    blend = float(cfg.get("discovery_lead_blend", 0.0) or 0.0)
    blend = max(0.0, min(0.25, blend))
    if blend <= 0.0:
        return composite
    raw = (1.0 - blend) * composite + blend * p_lead
    return float(min(1.0, max(0.0, raw)))


# ═══════════════════════════════════════════════════════════════════
#  Embedding helpers (Magnitu 2)
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


def _get_embedder():
    """Lazy-load the transformer model + tokenizer. Cached after first call."""
    global _embedder
    if _embedder is not None:
        return _embedder

    import torch
    from transformers import AutoTokenizer, AutoModel

    config = get_config()
    model_name = config.get("transformer_model_name", "xlm-roberta-base")

    logger.info("Loading transformer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = _select_device()

    model_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    model = AutoModel.from_pretrained(
        model_name,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    logger.info("Transformer loaded on %s (%s)", device.type, str(model_dtype).split(".")[-1])

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


def compute_embeddings(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Compute mean-pooled embeddings for a list of texts using the transformer.
    Returns ndarray of shape (len(texts), embedding_dim).
    Batch size kept small (8) and max_length capped at 256 to limit memory.

    Mean pooling averages all non-padding token embeddings, producing better
    representations than [CLS] alone for models like XLM-RoBERTa that weren't
    trained with a dedicated [CLS] objective.
    """
    import torch

    embedder = _get_embedder()
    tokenizer = embedder["tokenizer"]
    model = embedder["model"]
    device = embedder["device"]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
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


CONTENT_CAP = 500
SOURCE_NAME_CAP = 120
SOURCE_CATEGORY_CAP = 80
LEGAL_SIGNAL_CAP = 8  # max distinct signals to prepend per entry


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


def _build_entry_text(entry: dict, legal_patterns: Optional[List[str]] = None) -> str:
    """
    Build text for embedding/scoring from an entry's fields.

    Title is repeated so it dominates the [CLS] embedding even for entries
    with long content.  Content is capped at CONTENT_CAP chars so a 3000-
    char government press release doesn't push the title out of the
    tokenizer's 256-token window.  This makes embeddings comparable across
    sources that provide wildly different amounts of text (e.g. a Bund
    Medienmitteilung vs. an SRF teaser).

    A short structured prefix (source type, name, category) helps the model
    separate feeds without changing the Seismo entry schema.

    When legal_signal_patterns are configured and any match the entry body,
    they are prepended as a "signals=..." tag so the transformer sees them
    as emphasized context.  Changing this list invalidates cached embeddings.
    """
    meta_parts = []
    st = (entry.get("source_type") or "").strip()
    if st:
        meta_parts.append("source_type={}".format(st))
    sn = (entry.get("source_name") or "").strip()
    if sn:
        meta_parts.append("source={}".format(sn[:SOURCE_NAME_CAP]))
    sc = (entry.get("source_category") or "").strip()
    if sc:
        meta_parts.append("category={}".format(sc[:SOURCE_CATEGORY_CAP]))

    title = entry.get("title", "").strip()
    desc = entry.get("description", "").strip()
    content = entry.get("content", "").strip()[:CONTENT_CAP]

    scan_text = " ".join(part for part in [title, desc, content] if part)
    signals = _detect_legal_signals(scan_text, legal_patterns)
    if signals:
        meta_parts.append("signals={}".format(", ".join(signals)))

    meta = "; ".join(meta_parts)

    body = "\n".join(part for part in [title, title, desc, content] if part)
    if not body:
        body = "(empty)"
    if meta:
        return "{}\n\n{}".format(meta, body)
    return body


def embed_entries(entries: List[dict]) -> List[bytes]:
    """Compute embeddings for a list of entry dicts. Returns list of bytes."""
    patterns = get_config().get("legal_signal_patterns") or []
    texts = [_build_entry_text(e, legal_patterns=patterns) for e in entries]
    embeddings = compute_embeddings(texts)
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


def compute_sample_weights(labeled: List[dict],
                            config: Optional[dict] = None) -> np.ndarray:
    """Per-label training weight combining time decay and reasoning boost.

    - Time decay: ``weight = 0.5 ** (age_days / half_life)`` clamped to a floor.
      Uses ``updated_at`` when present (freshly re-labeled rows stay strong)
      else ``created_at``.  Missing timestamps get weight 1.0.
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
            age_days = max(0.0, (now - ts) / 86400.0)
            w = 0.5 ** (age_days / half_life)
            weights[i] = max(floor, w)

    if abs(boost - 1.0) > 1e-6:
        for i, lbl in enumerate(labeled):
            reason = (lbl.get("reasoning") or "").strip()
            if reason:
                weights[i] *= boost

    return weights


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF pipeline (Magnitu 1 — kept as fallback + recipe distiller)
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
#  Transformer training + scoring (Magnitu 2)
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
    """Train an MLP classifier on cached transformer embeddings for a profile."""
    config = get_config()
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
        emb_map[(row["entry_type"], row["entry_id"])] = row["embedding"]
    conn.close()

    X_list = []
    y_list = []
    lbl_list = []
    missing_embeddings = []

    for lbl in labeled:
        key = (lbl["entry_type"], lbl["entry_id"])
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
        test_size = min(0.2, min_class_count / len(y))
        test_size = max(test_size, 0.1)
        X_train, X_test, y_train, y_test, sw_train, _sw_test = train_test_split(
            X, y, sw_all, test_size=test_size, stratify=y, random_state=42
        )
        split_note = "{}/{} train/test split".format(int((1 - test_size) * 100), int(test_size * 100))

    # Hold out part of the training fold for temperature scaling (kept out of oversampling)
    X_tr_raw, y_tr_raw = X_train, y_train
    sw_tr = sw_train
    X_val, y_val = None, None
    if min_class_count >= 2 and len(X_train) >= 30:
        try:
            xtr, xv, ytr, yv, sw_tr_new, _sw_v = train_test_split(
                X_train, y_train, sw_train,
                test_size=0.15, stratify=y_train, random_state=43,
            )
            if len(xv) >= 5:
                X_tr_raw, X_val = xtr, xv
                y_tr_raw, y_val = ytr, yv
                sw_tr = sw_tr_new
        except ValueError:
            pass

    # Fold per-sample weights into the training set via replication.  MLPClassifier
    # doesn't support sample_weight, so we express "this label matters more" by
    # letting it appear more often.  Capped at 5× to keep training tractable.
    if sw_tr is not None and len(sw_tr) == len(y_tr_raw) and len(sw_tr) > 0:
        w_arr = np.asarray(sw_tr, dtype=np.float64)
        w_min = max(float(np.min(w_arr)), 1e-6)
        reps = np.clip(np.round(w_arr / w_min).astype(int), 1, 5)
    else:
        reps = np.ones(len(y_tr_raw), dtype=int)

    if np.any(reps > 1):
        X_rep = []
        y_rep = []
        X_tr_arr = np.asarray(X_tr_raw)
        for i in range(len(y_tr_raw)):
            n = int(reps[i])
            for _ in range(n):
                X_rep.append(X_tr_arr[i])
                y_rep.append(y_tr_raw[i])
        X_tr_raw = np.array(X_rep)
        y_tr_raw = y_rep

    # Oversample minority classes so the MLP sees equal representation
    # (MLPClassifier doesn't support class_weight or sample_weight)
    from collections import Counter
    train_counts = Counter(y_tr_raw)
    max_count = max(train_counts.values())
    rng = np.random.RandomState(42)

    X_balanced = list(X_tr_raw)
    y_balanced = list(y_tr_raw)
    for cls, count in train_counts.items():
        if count < max_count:
            cls_indices = [i for i, label in enumerate(y_tr_raw) if label == cls]
            extra = rng.choice(cls_indices, size=max_count - count, replace=True)
            for idx in extra:
                X_balanced.append(X_tr_raw[idx])
                y_balanced.append(y_tr_raw[idx])
    X_train_bal = np.array(X_balanced)
    y_train_bal = y_balanced

    # Encode string labels to integers for MLP (early_stopping + string labels
    # triggers a numpy isnan bug in some sklearn versions)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train_bal)

    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])
    clf_pipeline.fit(X_train_bal, y_train_enc)

    # Wrap in a thin adapter that decodes integer predictions back to strings
    # so the rest of the codebase (scoring, explainer) works unchanged
    clf = _LabelDecodingClassifier(clf_pipeline, le)

    # Temperature scaling on validation logits (matches production scoring)
    class_names_fit = clf.classes_.tolist()
    if X_val is not None and len(X_val) >= 3:
        logits_val = logits_for_classifier_head(clf, X_val)
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

    # Evaluate on held-out test using the same probabilities as scoring
    version = db.get_next_model_version()
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
    """Score entries using cached embeddings + MLP classifier.

    On-the-fly embedding computation is capped at MAX_ONTHEFLY_EMBEDDINGS so
    that page loads stay fast.  Entries beyond the cap are silently omitted
    from results -- callers already handle partial score lists.  The sync
    path (_compute_pending_embeddings) handles bulk embedding computation.
    """
    model_path = model_info.get("model_path")
    if not model_path or not Path(model_path).exists():
        return []

    clf = joblib.load(model_path)
    config = get_config()
    embedding_dim = config.get("embedding_dim", 768)

    # Batch-fetch all embeddings in one query
    conn = db.get_db()
    emb_map = {}
    rows = conn.execute(
        "SELECT entry_type, entry_id, embedding FROM entries WHERE embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        emb_map[(row["entry_type"], row["entry_id"])] = row["embedding"]
    conn.close()

    embeddings = []
    to_compute = []
    to_compute_indices = []

    for i, entry in enumerate(entries):
        key = (entry["entry_type"], entry["entry_id"])
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
        relevance = _discovery_adjusted_relevance(composite, p_lead)
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
#  TF-IDF training + scoring (Magnitu 1 fallback)
# ═══════════════════════════════════════════════════════════════════

def _train_tfidf(profile_id: int = 1) -> dict:
    """Train using the original TF-IDF + LogReg pipeline for a profile."""
    config = get_config()
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
        test_size = min(0.2, min_class_count / len(labels))
        test_size = max(test_size, 0.1)
        X_train, X_test, y_train, y_test, sw_train, _sw_test = train_test_split(
            df, labels, sw_all, test_size=test_size, stratify=labels, random_state=42
        )
        split_note = "{}/{} train/test split".format(int((1 - test_size) * 100), int(test_size * 100))

    # Slice of training fold for temperature calibration (kept out of TF-IDF fit)
    X_tr, y_tr = X_train, y_train
    sw_tr = sw_train
    X_val, y_val = None, None
    if min_class_count >= 2 and len(X_train) >= 30:
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

    version = db.get_next_model_version()
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
    model = load_active_model(profile_id=model_info.get("profile_id", 1))
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
        relevance = _discovery_adjusted_relevance(composite, p_lead)
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

    The student captures the transformer's knowledge in a form that can be
    distilled into a keyword recipe for seismo's PHP.

    Returns the trained student pipeline, or None if not possible.
    """
    model_info = db.get_active_model(profile_id)
    if not model_info or model_info.get("architecture") != "transformer":
        return None

    # Score all entries with the transformer model for this profile
    all_entries = db.get_all_entries()
    if len(all_entries) < 20:
        return None

    scores = score_entries(all_entries, profile_id=profile_id)
    if not scores:
        return None

    # Build a lookup of transformer predictions keyed by (entry_type, entry_id)
    score_map = {
        (s["entry_type"], s["entry_id"]): s["predicted_label"]
        for s in scores
    }

    # Human labels override transformer predictions (profile-specific)
    human_labels = {
        (lbl["entry_type"], lbl["entry_id"]): lbl["label"]
        for lbl in db.get_all_labels(profile_id)
    }

    # Only include entries that have either a score or a human label
    scored_entries = []
    teacher_labels = []
    for entry in all_entries:
        key = (entry["entry_type"], entry["entry_id"])
        if key in human_labels:
            scored_entries.append(entry)
            teacher_labels.append(human_labels[key])
        elif key in score_map:
            scored_entries.append(entry)
            teacher_labels.append(score_map[key])

    if len(scored_entries) < 20:
        return None

    df = _prepare_text(scored_entries)

    # Build and train the student pipeline
    student = build_tfidf_pipeline()

    try:
        student.fit(df, teacher_labels)
    except ValueError:
        student.named_steps["features"].transformers[0] = (
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
        student.fit(df, teacher_labels)

    return student
