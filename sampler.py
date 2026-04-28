"""
Smart sampling for active learning.
Instead of always showing newest unlabeled entries, prioritise entries that
will improve the model fastest: uncertain predictions, model/recipe conflicts,
and underrepresented source categories.
"""
import math
import json
from typing import List, Dict, Optional

import numpy as np

import db
import pipeline
import distiller


# ─── Helpers ───

def _normalize_entry_key(entry_type, entry_id):
    """Stable identity for deduping (avoids int/str entry_id mismatches)."""
    try:
        eid = int(entry_id)
    except (TypeError, ValueError):
        eid = entry_id
    return (str(entry_type or ""), eid)


def _normalize_entry_dict_key(entry: dict):
    return _normalize_entry_key(entry.get("entry_type"), entry.get("entry_id"))


def _dedupe_entries_preserve_order(entries: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for e in entries:
        k = _normalize_entry_dict_key(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy of a probability distribution. Higher = more uncertain."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def _recipe_predict(entry: dict, recipe: dict) -> Optional[str]:
    """Score a single entry with the recipe and return the predicted label."""
    keywords = recipe.get("keywords", {})
    source_weights_map = recipe.get("source_weights", {})
    classes = recipe.get("classes", ["investigation_lead", "important", "background", "noise"])

    text = f"{entry.get('title', '')} {entry.get('description', '')} {entry.get('content', '')}".lower()
    tokens = text.split()
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]
    all_tokens = tokens + bigrams + trigrams

    class_scores = {c: 0.0 for c in classes}
    for token in all_tokens:
        if token in keywords:
            for cls, wt in keywords[token].items():
                if cls in class_scores:
                    class_scores[cls] += wt

    src = entry.get("source_type", "")
    if src in source_weights_map:
        for cls, wt in source_weights_map[src].items():
            if cls in class_scores:
                class_scores[cls] += wt

    # Softmax
    max_s = max(class_scores.values()) if class_scores else 0
    exp_scores = {c: math.exp(s - max_s) for c, s in class_scores.items()}
    exp_sum = sum(exp_scores.values())
    if exp_sum == 0:
        return None
    probs = {c: exp_scores[c] / exp_sum for c in classes}
    return max(probs, key=probs.get)


# ─── Sampling strategies ───

def _get_uncertain(unlabeled: List[dict], scores: List[dict], limit: int = 10) -> List[dict]:
    """
    Uncertainty sampling: entries where the model's probability distribution
    is flattest (highest entropy). These are the entries the model is most
    confused about — labeling them teaches the model the most.
    """
    scored_with_entropy = []
    score_map = {}
    for s in scores:
        k = _normalize_entry_key(s["entry_type"], s["entry_id"])
        if k not in score_map:
            score_map[k] = s

    for entry in unlabeled:
        key = _normalize_entry_dict_key(entry)
        s = score_map.get(key)
        if not s or "probabilities" not in s:
            continue
        ent = _entropy(s["probabilities"])
        scored_with_entropy.append((entry, ent, s))

    # Highest entropy first
    scored_with_entropy.sort(key=lambda x: x[1], reverse=True)

    results = []
    for entry, ent, s in scored_with_entropy[:limit]:
        entry = dict(entry)
        entry["_sampling_reason"] = "uncertain"
        entry["_sampling_detail"] = f"{s['predicted_label']} ({max(s['probabilities'].values()):.0%})"
        results.append(entry)
    return results


def _get_conflicts(unlabeled: List[dict], scores: List[dict], limit: int = 5,
                   profile_id: int = 1) -> List[dict]:
    """
    Conflict analysis: entries where the ML model and the recipe disagree
    on the predicted label. These are high-value training samples because
    they reveal where the lightweight recipe diverges from the full model.
    """
    # Load the latest recipe
    model_info = db.get_active_model(profile_id=profile_id)
    if not model_info or not model_info.get("recipe_path"):
        return []

    try:
        with open(model_info["recipe_path"]) as f:
            recipe = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if not recipe.get("keywords"):
        return []

    score_map = {}
    for s in scores:
        k = _normalize_entry_key(s["entry_type"], s["entry_id"])
        if k not in score_map:
            score_map[k] = s
    conflicts = []

    for entry in unlabeled:
        key = _normalize_entry_dict_key(entry)
        s = score_map.get(key)
        if not s:
            continue

        model_label = s["predicted_label"]
        recipe_label = _recipe_predict(entry, recipe)

        if recipe_label and model_label != recipe_label:
            entry = dict(entry)
            entry["_sampling_reason"] = "conflict"
            entry["_sampling_detail"] = f"model: {model_label} vs recipe: {recipe_label}"
            conflicts.append(entry)

    return conflicts[:limit]


def _get_diverse(unlabeled: List[dict], limit: int = 5,
                 profile_id: int = 1) -> List[dict]:
    """
    Diversity sampling: surface entries from source categories that are
    underrepresented in the labeled training set. This broadens the model's
    horizons beyond the types of content the user naturally gravitates to.
    """
    conn = db.get_db()
    rows = conn.execute("""
        SELECT e.source_category, COUNT(*) as cnt
        FROM labels l
        JOIN entries e ON l.entry_type = e.entry_type AND l.entry_id = e.entry_id
        WHERE l.profile_id = ?
          AND (l.pending_gemini_job_id IS NULL OR TRIM(COALESCE(l.pending_gemini_job_id,''))='')
          AND e.source_category IS NOT NULL AND e.source_category != ''
        GROUP BY e.source_category
    """, (profile_id,)).fetchall()
    conn.close()
    labeled_counts = {r["source_category"]: r["cnt"] for r in rows}

    # Group unlabeled entries by source_category
    by_category = {}
    for entry in unlabeled:
        cat = entry.get("source_category", "")
        if not cat:
            continue
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(entry)

    # Score categories: lower labeled count = higher priority
    # Categories with 0 labels get highest priority
    category_scores = []
    for cat, entries in by_category.items():
        label_count = labeled_counts.get(cat, 0)
        # Score: inverse of label count (add 1 to avoid division by zero)
        score = 1.0 / (label_count + 1)
        category_scores.append((cat, score, entries))

    # Sort by score descending (least-labeled categories first)
    category_scores.sort(key=lambda x: x[1], reverse=True)

    # Pick one entry from each underrepresented category, round-robin
    results = []
    for cat, score, entries in category_scores:
        if len(results) >= limit:
            break
        entry = dict(entries[0])  # Pick the first (newest) entry from this category
        entry["_sampling_reason"] = "diverse"
        labeled_count = labeled_counts.get(cat, 0)
        entry["_sampling_detail"] = (
            f"{cat} · {labeled_count} label(s) in training from this category"
        )
        results.append(entry)

    return results


def _get_chronological(unlabeled: List[dict], limit: int = 10) -> List[dict]:
    """Newest unlabeled entries — keeps the feed fresh."""
    results = []
    for entry in unlabeled[:limit]:
        entry = dict(entry)
        entry["_sampling_reason"] = "new"
        entry["_sampling_detail"] = ""
        results.append(entry)
    return results


# ─── Main entry point ───

def get_smart_entries(limit: int = 30, entry_type: str = None,
                      profile_id: int = 1) -> List[dict]:
    """
    Get a mixed set of entries optimised for active learning for a profile.

    If a model exists, returns a mix of:
    - ~10 uncertain (model is confused)
    - ~5 conflicts (model vs recipe disagree)
    - ~5 diverse (underrepresented categories)
    - ~10 chronological (newest, for freshness)

    If no model exists, falls back to newest unlabeled (current behaviour).
    """
    unlabeled = db.get_unlabeled_entries(limit=200, entry_type=entry_type,
                                         profile_id=profile_id)
    unlabeled = _dedupe_entries_preserve_order(unlabeled)

    if not unlabeled:
        return []

    model = pipeline.load_active_model(profile_id=profile_id)
    if model is None:
        entries = unlabeled[:limit]
        for e in entries:
            e["_sampling_reason"] = "new"
            e["_sampling_detail"] = ""
        return entries

    scores = pipeline.score_entries(unlabeled, profile_id=profile_id)
    if not scores:
        entries = unlabeled[:limit]
        for e in entries:
            e["_sampling_reason"] = "new"
            e["_sampling_detail"] = ""
        return entries

    # Collect from each strategy
    uncertain = _get_uncertain(unlabeled, scores, limit=10)
    conflicts = _get_conflicts(unlabeled, scores, limit=5, profile_id=profile_id)
    diverse = _get_diverse(unlabeled, limit=5, profile_id=profile_id)
    chronological = _get_chronological(unlabeled, limit=15)

    # Merge with deduplication (priority: uncertain > conflict > diverse > new)
    seen = set()
    result = []

    for pool in [uncertain, conflicts, diverse, chronological]:
        for entry in pool:
            key = _normalize_entry_dict_key(entry)
            if key not in seen:
                seen.add(key)
                result.append(entry)

    return result[:limit]


def get_gemini_synthetic_batch_entries(
    limit: int = 80,
    entry_type: Optional[str] = None,
    profile_id: int = 1,
) -> List[dict]:
    """Entries for Gemini synthetic batch: same smart pool as the Label page, prioritising
    **uncertain** then **new**, then other smart-queue reasons so the batch can fill up.

    Callers should still filter to unlabeled (or replace_gemini) before labeling.
    """
    lim = max(1, min(int(limit), 200))
    pool = get_smart_entries(
        limit=max(120, lim * 2), entry_type=entry_type, profile_id=profile_id
    )

    def rank(reason: str) -> int:
        order = {"uncertain": 0, "new": 1, "conflict": 2, "diverse": 3}
        return order.get(reason or "new", 9)

    ranked = sorted(pool, key=lambda e: rank(str(e.get("_sampling_reason", "new"))))
    return ranked[:lim]
