"""
Explainability module: per-entry explanations and global keyword rankings.

Magnitu 2: for transformer models, provides class probability breakdowns.
For TF-IDF models, provides keyword-level feature attribution.
"""
import numpy as np
import joblib
import json
import re
from pathlib import Path
from typing import Optional

import db
from config import get_config
from pipeline import (
    load_active_model,
    _prepare_text,
    get_feature_importance,
    bytes_to_embedding,
    embed_entries,
    CLASS_WEIGHT_MAP,
    classifier_probabilities,
    _relevance_from_probs,
)


def _extract_recipe_tokens(text: str) -> list:
    """Extract unigram, bigram, and trigram candidates for recipe matching."""
    tokens = re.findall(r"\b[a-zA-Z0-9\u00C0-\u024F]{2,}\b", (text or "").lower())
    grams = []
    for n in (2, 3):
        if len(tokens) < n:
            break
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i:i + n]))
    return tokens + grams


def _recipe_phrase_contributions(entry: dict, model_info: dict, probs: dict, limit: int = 8) -> list:
    """Collect matched recipe phrase contributions weighted by class probabilities."""
    recipe_path = model_info.get("recipe_path", "")
    if not recipe_path or not Path(recipe_path).exists():
        return []
    try:
        with open(recipe_path) as f:
            recipe = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    keywords = recipe.get("keywords", {})
    text = "{} {} {}".format(
        entry.get("title", ""),
        entry.get("description", ""),
        entry.get("content", ""),
    )
    all_tokens = _extract_recipe_tokens(text)

    phrase_scores = {}
    for tok in all_tokens:
        if " " not in tok:
            continue
        cls_wts = keywords.get(tok)
        if not cls_wts:
            continue
        weighted = 0.0
        for cls, p in probs.items():
            weighted += float(cls_wts.get(cls, 0.0)) * float(p)
        if abs(weighted) > 1e-6:
            phrase_scores[tok] = phrase_scores.get(tok, 0.0) + weighted

    items = sorted(phrase_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    return [
        {
            "feature": phrase,
            "weight": round(float(weight), 4),
            "direction": "positive" if weight > 0 else "negative",
        }
        for phrase, weight in items
    ]


def explain_entry(entry: dict, profile_id: int = 1) -> Optional[dict]:
    """
    Explain why a single entry received its score.
    Architecture-aware: uses coefficient analysis for TF-IDF,
    class probabilities for transformer.
    """
    model_info = db.get_active_model(profile_id)
    if not model_info:
        return None

    arch = model_info.get("architecture", "tfidf")

    if arch == "transformer":
        return _explain_transformer(entry, model_info)
    return _explain_tfidf(entry, profile_id=profile_id)


def _explain_tfidf(entry: dict, profile_id: int = 1) -> Optional[dict]:
    """Explain using TF-IDF coefficient * feature_value (original Magnitu 1 path)."""
    model = load_active_model(profile_id=profile_id)
    if model is None:
        return None

    if not hasattr(model, "named_steps"):
        return None

    df = _prepare_text([entry])
    model_info = db.get_active_model(profile_id)
    model_path = (model_info or {}).get("model_path", "") or ""
    probabilities, class_names = classifier_probabilities(model, df, model_path)
    row = probabilities[0]
    prediction = class_names[int(np.argmax(row))]

    preprocessor = model.named_steps["features"]
    features = preprocessor.transform(df)

    tfidf = preprocessor.transformers_[0][1]
    tfidf_names = tfidf.get_feature_names_out().tolist()
    try:
        source_enc = preprocessor.transformers_[1][1]
        source_names = source_enc.get_feature_names_out().tolist()
    except (IndexError, AttributeError):
        source_names = []
    all_names = tfidf_names + source_names

    classifier = model.named_steps["classifier"]
    pred_idx = class_names.index(prediction)
    coefs = classifier.coef_[pred_idx]

    if hasattr(features, "toarray"):
        feature_values = features.toarray()[0]
    else:
        feature_values = np.array(features[0]).flatten()

    contributions = coefs[:len(feature_values)] * feature_values[:len(coefs)]

    n_features = min(len(all_names), len(contributions))
    feature_contribs = []
    for j in range(n_features):
        if abs(contributions[j]) > 0.001:
            feature_contribs.append({
                "feature": all_names[j],
                "weight": round(float(contributions[j]), 4),
                "direction": "positive" if contributions[j] > 0 else "negative",
            })

    feature_contribs.sort(key=lambda x: abs(x["weight"]), reverse=True)
    top_features = feature_contribs[:8]

    probs = dict(zip(class_names, [round(float(p), 4) for p in row]))
    relevance = _relevance_from_probs(probs, class_names)

    return {
        "prediction": prediction,
        "confidence": round(float(max(row)), 4),
        "relevance_score": round(relevance, 4),
        "probabilities": probs,
        "top_features": top_features,
    }


def _explain_transformer(entry: dict, model_info: dict) -> Optional[dict]:
    """
    Explain using the transformer model.  Since we use LogReg on embeddings,
    we can report class probabilities.  Token-level attribution is kept
    simple — we report the source type contribution if relevant.
    """
    model_path = model_info.get("model_path")
    if not model_path or not Path(model_path).exists():
        return None

    clf = joblib.load(model_path)
    config = get_config()
    embedding_dim = config.get("embedding_dim", 768)

    # Get or compute embedding
    conn = db.get_db()
    row = conn.execute(
        "SELECT embedding FROM entries WHERE entry_type = ? AND entry_id = ?",
        (entry["entry_type"], entry["entry_id"])
    ).fetchone()
    conn.close()

    if row and row["embedding"]:
        emb = bytes_to_embedding(row["embedding"], embedding_dim)
    else:
        emb_bytes_list = embed_entries([entry])
        emb = bytes_to_embedding(emb_bytes_list[0], embedding_dim)

    X = emb.reshape(1, -1)
    probabilities, class_names = classifier_probabilities(clf, X, model_path)
    row = probabilities[0]
    prediction = class_names[int(np.argmax(row))]
    probs = dict(zip(class_names, [round(float(p), 4) for p in row]))
    relevance = _relevance_from_probs(probs, class_names)

    # Phrase-aware explanation: first show matched recipe phrases for the
    # predicted class (especially useful for legal text), then class probs.
    top_features = _recipe_phrase_contributions(entry, model_info, probs, limit=8)
    if len(top_features) < 4:
        for cls in class_names:
            top_features.append({
                "feature": cls.replace("_", " "),
                "weight": round(float(probs.get(cls, 0)), 4),
                "direction": "positive" if probs.get(cls, 0) > 0.25 else "neutral",
            })

    return {
        "prediction": prediction,
        "confidence": round(float(max(row)), 4),
        "relevance_score": round(relevance, 4),
        "probabilities": probs,
        "top_features": top_features,
    }


def global_keywords(class_name: Optional[str] = None, limit: int = 50,
                    profile_id: int = 1) -> dict:
    """
    Get top keywords across all classes or for a specific class.
    Returns {class: [(keyword, weight), ...]} limited to top N per class.

    For transformer models, attempts knowledge distillation to extract keywords.
    """
    model_info = db.get_active_model(profile_id)
    if not model_info:
        return {}

    arch = model_info.get("architecture", "tfidf")

    if arch == "transformer":
        # Use the student model for keyword extraction
        from pipeline import train_tfidf_student
        student = train_tfidf_student(profile_id=profile_id)
        if student is None:
            return {}

        preprocessor = student.named_steps["features"]
        tfidf = preprocessor.transformers_[0][1]
        tfidf_names = tfidf.get_feature_names_out().tolist()
        try:
            source_enc = preprocessor.transformers_[1][1]
            source_names = source_enc.get_feature_names_out().tolist()
        except (IndexError, AttributeError):
            source_names = []
        all_names = tfidf_names + source_names
        classifier = student.named_steps["classifier"]
        class_names = classifier.classes_.tolist()
        coef_matrix = classifier.coef_

        importance = {}
        for i, cls in enumerate(class_names):
            coefs = coef_matrix[i]
            pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            importance[cls] = pairs
    else:
        importance = get_feature_importance()

    if not importance:
        return {}

    if class_name and class_name in importance:
        return {class_name: importance[class_name][:limit]}

    return {cls: pairs[:limit] for cls, pairs in importance.items()}


def compare_models(version_a: int, version_b: int) -> dict:
    """
    Compare keywords between two model versions.
    Returns changes: new keywords, removed keywords, weight shifts.
    """
    import joblib as jl

    all_models = db.get_all_models()

    path_a = None
    path_b = None
    arch_a = "tfidf"
    arch_b = "tfidf"
    for m in all_models:
        if m["version"] == version_a:
            path_a = m["model_path"]
            arch_a = m.get("architecture", "tfidf")
        if m["version"] == version_b:
            path_b = m["model_path"]
            arch_b = m.get("architecture", "tfidf")

    if not path_a or not path_b:
        return {"error": "Model version not found"}

    if not Path(path_a).exists() or not Path(path_b).exists():
        return {"error": "Model file not found on disk"}

    # Only compare TF-IDF models directly
    if arch_a != "tfidf" or arch_b != "tfidf":
        return {
            "version_a": version_a,
            "version_b": version_b,
            "note": "Keyword comparison only available between TF-IDF model versions.",
            "changes": {},
        }

    pipe_a = jl.load(path_a)
    pipe_b = jl.load(path_b)

    def _top_keywords(pipe, n=30):
        tfidf = pipe.named_steps["features"].transformers_[0][1]
        names = tfidf.get_feature_names_out().tolist()
        clf_step = pipe.named_steps["classifier"]
        result = {}
        for i, cls in enumerate(clf_step.classes_):
            coefs = clf_step.coef_[i]
            pairs = sorted(zip(names[:len(coefs)], coefs), key=lambda x: abs(x[1]), reverse=True)
            result[cls] = dict(pairs[:n])
        return result

    kw_a = _top_keywords(pipe_a)
    kw_b = _top_keywords(pipe_b)

    changes = {}
    all_classes = set(list(kw_a.keys()) + list(kw_b.keys()))
    for cls in all_classes:
        a = kw_a.get(cls, {})
        b = kw_b.get(cls, {})
        new_kw = {k: v for k, v in b.items() if k not in a}
        removed_kw = {k: v for k, v in a.items() if k not in b}
        shifted = {}
        for k in set(a.keys()) & set(b.keys()):
            diff = b[k] - a[k]
            if abs(diff) > 0.05:
                shifted[k] = {
                    "old": round(a[k], 4),
                    "new": round(b[k], 4),
                    "change": round(diff, 4),
                }
        changes[cls] = {"new": new_kw, "removed": removed_kw, "shifted": shifted}

    return {
        "version_a": version_a,
        "version_b": version_b,
        "changes": changes,
    }
