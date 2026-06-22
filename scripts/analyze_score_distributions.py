#!/usr/bin/env python3
"""
Compare raw model composites, push rank-normalized scores, and recipe scores.

Run from the repo root (uses MAGNITU_DATA_DIR / default Application Support DB):

    python scripts/analyze_score_distributions.py
    python scripts/analyze_score_distributions.py --profile-id 1 --compare-legacy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import db
import distiller
import main as app_main
import pipeline
from config import DATA_DIR
from pipeline import (
    CLASS_WEIGHT_MAP,
    CLASSES,
    _discovery_adjusted_relevance,
    class_weight_list,
)

LEGACY_WEIGHTS = {
    "investigation_lead": 1.0,
    "important": 0.66,
    "background": 0.33,
    "noise": 0.0,
}

HIST_BINS = (0.0, 0.25, 0.5, 0.66, 0.75, 0.85, 1.01)


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _composite_from_probs(probs: Dict[str, float], weight_map: Dict[str, float]) -> float:
    return float(sum(probs.get(c, 0.0) * weight_map.get(c, 0.0) for c in CLASSES))


def _histogram(values: Sequence[float], bins: Tuple[float, ...] = HIST_BINS) -> Counter:
    counts: Counter = Counter()
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[(bins[i], bins[i + 1])] += 1
                break
    return counts


def _fmt_hist(counts: Counter, bins: Tuple[float, ...], n: int) -> str:
    lines = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        c = counts.get((lo, hi), 0)
        bar = "#" * int(round(40 * c / max(n, 1)))
        lines.append(
            "  [{:.2f}, {:.2f}): {:5d} ({:5.1f}%) {}".format(
                lo, hi, c, 100.0 * c / n, bar
            )
        )
    return "\n".join(lines)


def _rank_normalize(values: List[float]) -> List[float]:
    rows = [{"relevance_score": v} for v in values]
    app_main._rank_normalize_push_scores(rows)
    return [r["relevance_score"] for r in rows]


def _recipe_scores_for_entries(
    scores: List[dict],
    entries: List[dict],
    recipe_path: str,
    class_wts: Optional[List[float]] = None,
) -> List[float]:
    path = Path(recipe_path)
    if not path.exists():
        return []
    with open(path) as f:
        recipe = json.load(f)
    kw = recipe.get("keywords", {})
    sw = recipe.get("source_weights", {})
    classes = recipe.get("classes", CLASSES)
    wts = class_wts if class_wts is not None else recipe.get("class_weights", class_weight_list(classes))
    entry_map = {db.entry_key_from_mapping(e): e for e in entries}
    out = []
    for s in scores:
        key = db.entry_key_from_mapping(s)
        entry = entry_map.get(key)
        if not entry:
            continue
        out.append(distiller._recipe_composite(entry, kw, sw, classes, wts))
    return out


def _print_distribution(name: str, values: Sequence[float], alert_threshold: float) -> None:
    if not values:
        print("--- {} (no data) ---\n".format(name))
        return
    n = len(values)
    print("--- {} (n={}) ---".format(name, n))
    print(
        "  min={:.4f}  p25={:.4f}  median={:.4f}  p75={:.4f}  "
        "max={:.4f}  mean={:.4f}  stdev={:.4f}".format(
            min(values),
            _pct(values, 25),
            _pct(values, 50),
            _pct(values, 75),
            max(values),
            float(np.mean(values)),
            float(np.std(values)),
        )
    )
    above = sum(1 for v in values if v >= alert_threshold)
    print(
        "  >= {:.2f} alert threshold: {} ({:.1f}%)".format(
            alert_threshold, above, 100.0 * above / n
        )
    )
    print(_fmt_hist(_histogram(values), HIST_BINS, n))
    print()


def run_analysis(
    profile_id: int = 1,
    alert_threshold: Optional[float] = None,
    compare_legacy: bool = False,
) -> int:
    cfg = db.get_effective_config(profile_id)
    alert = float(
        alert_threshold if alert_threshold is not None else cfg.get("alert_threshold", 0.75) or 0.75
    )
    blend = float(cfg.get("discovery_lead_blend", 0.0) or 0.0)

    model_info = db.get_active_model(profile_id)
    if not model_info:
        print("No active model for profile {}.".format(profile_id), file=sys.stderr)
        return 1

    print("=== Magnitu score distribution analysis ===")
    print("DATA_DIR:", DATA_DIR)
    print("Profile:", profile_id)
    print(
        "Active model: v{} ({})".format(
            model_info["version"], model_info.get("architecture")
        )
    )
    print("Labels in model:", model_info.get("label_count"))
    print(
        "Accuracy / F1:",
        round(float(model_info.get("accuracy") or 0), 3),
        round(float(model_info.get("f1_score") or 0), 3),
    )
    print("Alert threshold:", alert)
    print("discovery_lead_blend:", blend)
    print(
        "CLASS_WEIGHT_MAP:",
        ", ".join("{}={:.2f}".format(c, CLASS_WEIGHT_MAP[c]) for c in CLASSES),
    )
    print()

    entries = db.get_all_entries()
    print("Entries:", len(entries))
    print("Scoring entries (may take a minute)...")
    sys.stdout.flush()

    scores = pipeline.score_entries(entries, profile_id=profile_id)
    print("Scored:", len(scores))
    if not scores:
        print("No scores produced.", file=sys.stderr)
        return 1

    score_by_key = {db.entry_key_from_mapping(s): s for s in scores}
    label_by_key = {
        db.entry_key_from_mapping(l): l for l in db.get_all_labels(profile_id=profile_id)
    }

    raw = [s["relevance_score"] for s in scores]
    ranked = _rank_normalize([s["relevance_score"] for s in scores])

    alt_weight_map = dict(LEGACY_WEIGHTS) if compare_legacy else dict(CLASS_WEIGHT_MAP)
    alt_label = "Legacy weights (0.66/0.33)" if compare_legacy else "Current CLASS_WEIGHT_MAP"

    alt_raw = []
    for s in scores:
        probs = s.get("probabilities") or {}
        comp = _composite_from_probs(probs, alt_weight_map)
        p_lead = float(probs.get("investigation_lead", 0.0))
        alt_raw.append(_discovery_adjusted_relevance(comp, p_lead, profile_id=profile_id))
    alt_rank = _rank_normalize(alt_raw)

    recipe_path = model_info.get("recipe_path") or ""
    recipe_scores = _recipe_scores_for_entries(scores, entries, recipe_path)
    if recipe_path:
        print("Recipe:", recipe_path)
        try:
            with open(recipe_path) as f:
                exported = json.load(f).get("class_weights", [])
            if exported and exported != class_weight_list():
                print(
                    "WARNING: recipe class_weights {} != pipeline {} "
                    "(re-distill + push to sync Seismo)".format(
                        exported, class_weight_list()
                    )
                )
        except (OSError, json.JSONDecodeError):
            pass
    else:
        print("No recipe on active model.")
    print()

    _print_distribution("Raw model composite", raw, alert)
    if compare_legacy or alt_weight_map != CLASS_WEIGHT_MAP:
        _print_distribution("Alt composite ({})".format(alt_label), alt_raw, alert)
    _print_distribution("Push rank-normalized (current weights)", ranked, alert)
    if compare_legacy:
        _print_distribution("Push rank-normalized ({})".format(alt_label), alt_rank, alert)
    if recipe_scores:
        _print_distribution("Recipe composite (exported weights)", recipe_scores, alert)

    band = sum(1 for v in raw if 0.40 <= v <= 0.60)
    print("--- Soft-attractor (raw in [0.40, 0.60]) ---")
    print("  {} / {} ({:.1f}%)\n".format(band, len(raw), 100.0 * band / len(raw)))

    print("--- Raw composite by predicted_label ---")
    by_pred: Dict[str, List[float]] = defaultdict(list)
    for s in scores:
        by_pred[s.get("predicted_label", "?")].append(s["relevance_score"])
    for lab in CLASSES:
        vals = by_pred.get(lab, [])
        if not vals:
            continue
        print(
            "  {:20s} n={:4d}  median={:.3f}  p75={:.3f}  max={:.3f}  >={:.2f}: {:.1f}%".format(
                lab,
                len(vals),
                _pct(vals, 50),
                _pct(vals, 75),
                max(vals),
                alert,
                100.0 * sum(1 for v in vals if v >= alert) / len(vals),
            )
        )
    print()

    print("--- Labeled entries (n={}) ---".format(len(label_by_key)))
    by_human: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"raw": [], "rank": [], "recipe": []}
    )
    for key, lbl in label_by_key.items():
        s = score_by_key.get(key)
        if not s:
            continue
        idx = scores.index(s)
        h = lbl["label"]
        by_human[h]["raw"].append(s["relevance_score"])
        by_human[h]["rank"].append(ranked[idx])
        if recipe_scores:
            by_human[h]["recipe"].append(recipe_scores[idx])

    print("  By human label (median raw / rank / recipe; raw >= threshold):")
    for h in CLASSES:
        d = by_human.get(h)
        if not d or not d["raw"]:
            continue
        rr, rn, rc = d["raw"], d["rank"], d["recipe"]
        print(
            "    {:20s} n={:4d}  raw={:.3f}  rank={:.3f}  recipe={:.3f}  raw>={:.2f}={:.0f}%".format(
                h,
                len(rr),
                _pct(rr, 50),
                _pct(rn, 50),
                _pct(rc, 50) if rc else float("nan"),
                alert,
                100.0 * sum(1 for v in rr if v >= alert) / len(rr),
            )
        )
    print()

    if recipe_scores:
        print("--- Correlations (Pearson) ---")
        print(
            "  raw model vs recipe:      {:.4f}".format(
                float(np.corrcoef(raw, recipe_scores)[0, 1])
            )
        )
        print(
            "  rank vs recipe:           {:.4f}".format(
                float(np.corrcoef(ranked, recipe_scores)[0, 1])
            )
        )
    if compare_legacy:
        print(
            "  raw current vs legacy:    {:.4f}".format(
                float(np.corrcoef(raw, alt_raw)[0, 1])
            )
        )
        print(
            "  push-rank current vs legacy: {:.4f}".format(
                float(np.corrcoef(ranked, alt_rank)[0, 1])
            )
        )
    print()

    lead_keys = [
        k
        for k, l in label_by_key.items()
        if l.get("label") == "investigation_lead" and k in score_by_key
    ]
    if lead_keys:
        lead_raw = [score_by_key[k]["relevance_score"] for k in lead_keys]
        lead_rank = [ranked[scores.index(score_by_key[k])] for k in lead_keys]
        lead_recipe = (
            [recipe_scores[scores.index(score_by_key[k])] for k in lead_keys]
            if recipe_scores
            else []
        )
        print("--- Human investigation_lead alert recall ---")
        print("  n={}".format(len(lead_keys)))
        print(
            "  raw>={:.2f}:    {:.0f}%  (median {:.3f})".format(
                alert,
                100.0 * sum(1 for v in lead_raw if v >= alert) / len(lead_raw),
                _pct(lead_raw, 50),
            )
        )
        print(
            "  rank>={:.2f}:   {:.0f}%  (median {:.3f})".format(
                alert,
                100.0 * sum(1 for v in lead_rank if v >= alert) / len(lead_rank),
                _pct(lead_rank, 50),
            )
        )
        if lead_recipe:
            print(
                "  recipe>={:.2f}: {:.0f}%  (median {:.3f})".format(
                    alert,
                    100.0 * sum(1 for v in lead_recipe if v >= alert) / len(lead_recipe),
                    _pct(lead_recipe, 50),
                )
            )
        print()

    print("Done.")
    return 0


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Magnitu score distributions (raw, push rank, recipe)."
    )
    parser.add_argument(
        "--profile-id",
        type=int,
        default=1,
        help="Profile id (default: 1)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help="Override alert threshold (default: profile effective config)",
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also report legacy weights 1.0 / 0.66 / 0.33 / 0.0",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_analysis(
        profile_id=args.profile_id,
        alert_threshold=args.alert_threshold,
        compare_legacy=args.compare_legacy,
    )


if __name__ == "__main__":
    sys.exit(cli_main())
