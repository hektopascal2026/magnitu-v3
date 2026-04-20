"""
Recipe Distiller: converts a trained model into a lightweight keyword-weight
JSON recipe that seismo's PHP can evaluate.

Magnitu 2: when the active model is a transformer, uses knowledge distillation.
A TF-IDF 'student' model is trained on the transformer's predictions across all
entries.  The recipe is then extracted from the student's coefficients — same
format, same PHP, but informed by transformer-quality classifications.

Also incorporates user reasoning text: key phrases from reasoning annotations
are boosted in the recipe so human-stated priorities carry extra weight.
"""
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import db
from config import MODELS_DIR, get_config
from pipeline import (
    load_active_model,
    get_feature_importance,
    score_entries,
    train_tfidf_student,
    build_tfidf_pipeline,
    _prepare_text,
)

logger = logging.getLogger(__name__)

LOW_SIGNAL_STOPWORDS = {
    # German
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einem",
    "einen", "und", "oder", "aber", "in", "im", "am", "an", "auf", "aus", "von",
    "mit", "ohne", "zu", "zum", "zur", "für", "ist", "sind", "war", "waren", "wird",
    "werden", "nach", "bei", "als", "auch", "wie", "was", "wo", "wenn", "dass",
    # English
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    # French
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "dans", "sur", "pour",
    "avec", "sans", "est", "sont", "été", "ce", "cette", "ces",
    # Italian
    "il", "lo", "gli", "i", "l", "di", "del", "della", "dei", "delle", "un", "una",
    "e", "o", "in", "su", "per", "con", "senza", "è", "sono", "era", "che",
}

LEGAL_TEMPLATE_PHRASES = {
    # Cross-border market access / third-country signals
    "third country": {"investigation_lead": 0.45, "important": 0.25},
    "third countries": {"investigation_lead": 0.45, "important": 0.25},
    "pays tiers": {"investigation_lead": 0.45, "important": 0.25},
    "drittstaaten": {"investigation_lead": 0.45, "important": 0.25},
    "member states only": {"investigation_lead": 0.55, "important": 0.2},
    "eu eea": {"investigation_lead": 0.45, "important": 0.2},
    "eu ewr": {"investigation_lead": 0.45, "important": 0.2},
    "market access": {"investigation_lead": 0.35, "important": 0.25},
    "single market": {"important": 0.25, "background": 0.12},
    "internal market": {"important": 0.22, "background": 0.1},
    "equivalence decision": {"investigation_lead": 0.35, "important": 0.2},
    # Compliance and technical barriers
    "conformity assessment": {"important": 0.35, "investigation_lead": 0.18},
    "ce marking": {"important": 0.25, "background": 0.1},
    "technical regulation": {"important": 0.22, "background": 0.12},
    "compliance requirement": {"important": 0.28, "background": 0.1},
    # Frequent procedural boilerplate (negative relevance signal)
    "implementing act": {"noise": 0.35, "background": 0.12},
    "delegated regulation": {"noise": 0.28, "background": 0.12},
    "corrigendum": {"noise": 0.3},
    "annex amendment": {"noise": 0.3},
    "administrative procedure": {"noise": 0.3, "background": 0.15},
}


def _tokenize_text(text: str) -> list:
    """Tokenize text into lowercase word tokens with unicode support."""
    return re.findall(r"\b[a-zA-Z0-9\u00C0-\u024F]{2,}\b", (text or "").lower())


def _compose_ngrams(tokens: list, max_n: int = 3) -> list:
    """Create n-gram phrases (2..max_n) from tokens."""
    grams = []
    if not tokens:
        return grams
    for n in range(2, max_n + 1):
        if len(tokens) < n:
            break
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i:i + n]))
    return grams


def _extract_recipe_tokens(text: str) -> list:
    """Extract unigram, bigram, and trigram candidates for recipe matching."""
    tokens = _tokenize_text(text)
    return tokens + _compose_ngrams(tokens, max_n=3)


def _is_low_signal_unigram(token: str) -> bool:
    """Return True for unigrams that are too generic to export."""
    if " " in token:
        return False
    if len(token) < 3:
        return True
    return token in LOW_SIGNAL_STOPWORDS


def _clip(value: float, max_abs: float) -> float:
    return max(-max_abs, min(max_abs, value))


def _select_signed_features(pairs: list, top_n: int) -> list:
    """
    Select a balanced set of positive and negative features.

    This ensures recipe export contains both "raise score" and "lower score"
    evidence per class (important for filtering domains like sports/noise).
    """
    if top_n <= 0:
        return []

    positives = [(f, w) for f, w in pairs if w > 0]
    negatives = [(f, w) for f, w in pairs if w < 0]

    positives.sort(key=lambda x: x[1], reverse=True)   # strongest positive first
    negatives.sort(key=lambda x: x[1])                 # most negative first

    pos_budget = top_n // 2
    neg_budget = top_n - pos_budget

    selected = positives[:pos_budget] + negatives[:neg_budget]

    if len(selected) < top_n:
        selected_keys = {f for f, _ in selected}
        remainder = [(f, w) for f, w in pairs if f not in selected_keys]
        remainder.sort(key=lambda x: abs(x[1]), reverse=True)
        selected.extend(remainder[: max(0, top_n - len(selected))])

    return selected[:top_n]


def distill_recipe(top_n: Optional[int] = None, profile_id: int = 1):
    """
    Extract top keywords per class from the active model and package them as
    a JSON recipe for seismo.

    For TF-IDF models: extracts directly from coefficients (Magnitu 1 path).
    For transformer models: trains a TF-IDF student via knowledge distillation,
    then extracts from the student's coefficients (Magnitu 2 path).

    Returns the recipe dict, or None if no active model.
    """
    config = get_config()
    if top_n is None:
        top_n = config.get("recipe_top_keywords", 200)

    model_info = db.get_active_model(profile_id)
    if not model_info:
        return None

    arch = model_info.get("architecture", "tfidf")

    if arch == "transformer":
        importance = _distill_from_transformer(top_n, profile_id=profile_id)
    else:
        importance = get_feature_importance(profile_id=profile_id)

    if not importance:
        return None

    # Build keyword map: {keyword: {class: weight, ...}}
    keywords = {}
    source_weights = {}

    for cls, pairs in importance.items():
        signed_pairs = _select_signed_features(pairs, top_n)
        for feature, weight in signed_pairs:
            if abs(weight) < 0.01:
                continue

            if feature.startswith("source_type_") or feature.startswith("x0_"):
                source_name = feature.replace("source_type_", "").replace("x0_", "")
                if source_name not in source_weights:
                    source_weights[source_name] = {}
                source_weights[source_name][cls] = round(float(weight), 4)
            else:
                if _is_low_signal_unigram(feature):
                    continue
                if feature not in keywords:
                    keywords[feature] = {}
                keywords[feature][cls] = round(float(weight), 4)

    # Boost keywords from user reasoning annotations
    keywords = _boost_from_reasoning(keywords)
    # Add stable legal template phrases as prior signals for legislative text
    keywords = _boost_legal_templates(keywords)

    # Normalize weights so accumulated scores stay in a reasonable range for
    # softmax regardless of text length — prevents long entries from getting
    # extreme scores while short entries cluster near 0.5
    keywords, source_weights = _normalize_weights(keywords, source_weights)

    # Stabilize export so repeated tokens and source priors cannot dominate.
    keywords, source_weights = _stabilize_export_weights(keywords, source_weights)

    # Build recipe
    recipe = {
        "version": model_info["version"],
        "trained_at": model_info.get("trained_at", datetime.now().isoformat()),
        "labels_used": model_info.get("label_count", 0),
        "metrics": {
            "accuracy": round(model_info.get("accuracy", 0), 4),
            "f1_macro": round(model_info.get("f1_score", 0), 4),
        },
        "alert_threshold": config.get("alert_threshold", 0.75),
        "classes": ["investigation_lead", "important", "background", "noise"],
        "class_weights": [1.0, 0.66, 0.33, 0.0],
        "keywords": keywords,
        "source_weights": source_weights,
    }

    # Save recipe to disk
    recipe_filename = "recipe_v{}.json".format(model_info["version"])
    recipe_path = str(MODELS_DIR / recipe_filename)
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    # Update model record with recipe path
    conn = db.get_db()
    conn.execute(
        "UPDATE models SET recipe_path = ? WHERE version = ?",
        (recipe_path, model_info["version"])
    )
    conn.commit()
    conn.close()

    return recipe


def _normalize_weights(keywords: dict, source_weights: dict) -> tuple:
    """
    Scale recipe weights so that the median accumulated class_score across all
    entries lands near 1.0 before softmax.  This prevents long entries (many
    keyword matches) from producing extremely peaked softmax distributions
    while short entries get near-uniform scores.

    Without this, a 500-token government press release can score 100 while
    a 50-token news teaser about the same topic scores 5.
    """
    all_entries = db.get_all_entries()
    if not all_entries or not keywords:
        return keywords, source_weights

    classes = ["investigation_lead", "important", "background", "noise"]
    magnitudes = []

    for entry in all_entries:
        text = "{} {} {}".format(
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("content", ""),
        ).lower()
        all_tokens = _extract_recipe_tokens(text)

        class_scores = {c: 0.0 for c in classes}
        for token in all_tokens:
            if token in keywords:
                for cls, wt in keywords[token].items():
                    if cls in class_scores:
                        class_scores[cls] += wt

        src = entry.get("source_type", "")
        if src in source_weights:
            for cls, wt in source_weights[src].items():
                if cls in class_scores:
                    class_scores[cls] += wt

        mag = max(abs(v) for v in class_scores.values()) if class_scores else 0
        if mag > 0:
            magnitudes.append(mag)

    if not magnitudes:
        return keywords, source_weights

    magnitudes.sort()
    median_mag = magnitudes[len(magnitudes) // 2]

    TARGET_MAGNITUDE = 2.0
    if median_mag < 0.01:
        return keywords, source_weights

    scale = TARGET_MAGNITUDE / median_mag

    logger.info(
        "Recipe normalization: median magnitude %.2f, scaling weights by %.3f",
        median_mag, scale,
    )

    normalized_kw = {}
    for word, class_weights in keywords.items():
        normalized_kw[word] = {
            cls: round(wt * scale, 4)
            for cls, wt in class_weights.items()
        }

    normalized_sw = {}
    for src, class_weights in source_weights.items():
        normalized_sw[src] = {
            cls: round(wt * scale, 4)
            for cls, wt in class_weights.items()
        }

    return normalized_kw, normalized_sw


def _stabilize_export_weights(keywords: dict, source_weights: dict) -> tuple:
    """
    Clamp exported weights to keep Seismo recipe scoring stable:
    - lower cap for unigrams (repeat more often in text)
    - slightly higher cap for phrase patterns
    - strict cap for source priors so text remains dominant
    """
    cfg = get_config()
    max_unigram_abs = float(cfg.get("recipe_max_unigram_abs", 0.12))
    max_phrase_abs = float(cfg.get("recipe_max_phrase_abs", 0.24))
    max_source_abs = float(cfg.get("recipe_max_source_abs", 0.08))
    min_abs_keep = float(cfg.get("recipe_min_abs_keep", 0.01))

    stable_kw = {}
    for token, cls_wts in keywords.items():
        if _is_low_signal_unigram(token):
            continue
        cap = max_phrase_abs if " " in token else max_unigram_abs
        clipped = {}
        for cls, wt in cls_wts.items():
            w = round(_clip(float(wt), cap), 4)
            if abs(w) >= min_abs_keep:
                clipped[cls] = w
        if clipped:
            stable_kw[token] = clipped

    stable_sw = {}
    for src, cls_wts in source_weights.items():
        clipped = {}
        for cls, wt in cls_wts.items():
            w = round(_clip(float(wt), max_source_abs), 4)
            if abs(w) >= min_abs_keep:
                clipped[cls] = w
        if clipped:
            stable_sw[src] = clipped

    return stable_kw, stable_sw


def _distill_from_transformer(top_n: int, profile_id: int = 1) -> dict:
    """
    Knowledge distillation: train a TF-IDF student from the transformer's
    predictions, then extract feature importance from the student.
    """
    logger.info("Knowledge distillation: training TF-IDF student from transformer...")
    student = train_tfidf_student(profile_id=profile_id)
    if student is None:
        logger.warning("Could not train TF-IDF student for distillation.")
        return {}

    # Extract feature importance from the student
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

    result = {}
    for i, cls in enumerate(class_names):
        coefs = coef_matrix[i]
        pairs = list(zip(all_names[:len(coefs)], coefs.tolist()))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        result[cls] = pairs

    logger.info("Knowledge distillation complete. %d features extracted.", len(all_names))
    return result


def _boost_from_reasoning(keywords: dict) -> dict:
    """
    Extract key phrases from user reasoning annotations and boost their
    weights in the recipe.  This ensures that explicitly-stated reasons
    ('links politician X to company Y') increase the recipe's sensitivity
    to those terms.
    """
    reasoning_labels = db.get_all_reasoning_texts(profile_id=1)  # reasoning boosts are global
    if not reasoning_labels:
        return keywords

    BOOST_FACTOR = 1.5

    for rl in reasoning_labels:
        reasoning = rl.get("reasoning", "")
        label = rl.get("label", "")
        if not reasoning or not label:
            continue

        tokens = _tokenize_text(reasoning)
        phrases = _compose_ngrams(tokens, max_n=3)
        # Unigrams + phrase patterns from the reasoning itself
        for token in tokens + phrases:
            if " " not in token and len(token) < 3:
                continue
            phrase_boost = BOOST_FACTOR
            if " " in token:
                phrase_boost = 1.8
            if token in keywords:
                if label in keywords[token]:
                    keywords[token][label] = round(keywords[token][label] * phrase_boost, 4)
                else:
                    keywords[token][label] = round(0.08 * phrase_boost, 4)
            else:
                base = 0.1 if " " not in token else 0.16
                keywords[token] = {label: round(base, 4)}

    return keywords


def _boost_legal_templates(keywords: dict) -> dict:
    """Inject legal phrase priors to improve Seismo-side recipe approximation.

    Includes the built-in ``LEGAL_TEMPLATE_PHRASES`` plus any user-configured
    ``legal_signal_patterns`` from config — the latter get a single boost toward
    ``investigation_lead`` so Seismo's keyword recipe tracks what the user
    marked as legislative signal.
    """
    for phrase, cls_wts in LEGAL_TEMPLATE_PHRASES.items():
        if phrase not in keywords:
            keywords[phrase] = {}
        for cls, wt in cls_wts.items():
            prev = float(keywords[phrase].get(cls, 0.0))
            keywords[phrase][cls] = round(prev + float(wt), 4)

    try:
        user_patterns = get_config().get("legal_signal_patterns") or []
    except Exception:
        user_patterns = []
    for raw in user_patterns:
        phrase = (raw or "").strip().lower()
        if not phrase or len(phrase) < 2:
            continue
        # Strip common regex metacharacters so Seismo's keyword scorer has a
        # plain token to match (Seismo does literal lowercase containment).
        cleaned = re.sub(r"[\\^$.|?*+()\[\]{}]", "", phrase).strip()
        if not cleaned:
            continue
        if cleaned not in keywords:
            keywords[cleaned] = {}
        prev_lead = float(keywords[cleaned].get("investigation_lead", 0.0))
        prev_imp = float(keywords[cleaned].get("important", 0.0))
        keywords[cleaned]["investigation_lead"] = round(prev_lead + 0.35, 4)
        keywords[cleaned]["important"] = round(prev_imp + 0.15, 4)
    return keywords


def evaluate_recipe_quality(recipe: dict, sample_size: int = 100,
                            profile_id: int = 1) -> float:
    """
    Compare recipe-based scoring against full model scoring on a sample.
    Returns correlation score (0-1) indicating how well recipe approximates the model.
    """
    entries = db.get_all_entries()[:sample_size]
    if not entries:
        return 0.0

    full_scores = score_entries(entries, profile_id=profile_id)
    if not full_scores:
        return 0.0

    # Build lookup of model scores keyed by (entry_type, entry_id)
    model_score_map = {
        (s["entry_type"], s["entry_id"]): s["relevance_score"]
        for s in full_scores
    }

    # Compute recipe-based scores (matching Seismo's PHP logic)
    kw = recipe.get("keywords", {})
    source_weights_map = recipe.get("source_weights", {})
    classes = recipe.get("classes", ["investigation_lead", "important", "background", "noise"])
    class_wts = recipe.get("class_weights", [1.0, 0.66, 0.33, 0.0])

    paired_model = []
    paired_recipe = []

    for entry in entries:
        key = (entry["entry_type"], entry["entry_id"])
        if key not in model_score_map:
            continue

        text = "{} {} {}".format(
            entry.get("title", ""),
            entry.get("description", ""),
            entry.get("content", ""),
        ).lower()
        all_tokens = _extract_recipe_tokens(text)

        class_scores = {c: 0.0 for c in classes}
        for token in all_tokens:
            if token in kw:
                for cls, wt in kw[token].items():
                    if cls in class_scores:
                        class_scores[cls] += wt

        src = entry.get("source_type", "")
        if src in source_weights_map:
            for cls, wt in source_weights_map[src].items():
                if cls in class_scores:
                    class_scores[cls] += wt

        max_s = max(class_scores.values()) if class_scores else 0
        exp_scores = {c: np.exp(s - max_s) for c, s in class_scores.items()}
        exp_sum = sum(exp_scores.values())
        probs = {
            c: exp_scores[c] / exp_sum if exp_sum > 0 else 1 / len(classes)
            for c in classes
        }

        composite = sum(probs.get(c, 0) * class_wts[idx] for idx, c in enumerate(classes))
        paired_model.append(model_score_map[key])
        paired_recipe.append(composite)

    if len(paired_model) < 2:
        return 0.0

    correlation = float(np.corrcoef(paired_model, paired_recipe)[0, 1])
    quality = max(0.0, correlation) if not (correlation != correlation) else 0.0

    return round(quality, 4)
