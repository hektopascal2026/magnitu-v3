#!/usr/bin/env python3
"""
Dry-run of the score-semantics flip (engineering notes §6).

Scores recent embedded entries twice for a desk:
  old path — sidecar temperature only, then rank-normalized
  new path — prior offsets + temperature, raw composite

Prints histograms, a top-50 diff, decile maps, the measured
ambiguous-attractor, and base_rate_composite. Review this output before
Phase 4 Step B.

    python scripts/score_flip_diff.py
    python scripts/score_flip_diff.py --profile-id 1 --limit 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import db
import ml_window
import pipeline
from pipeline import calibration_sidecar_path, load_calibration


HIST_BINS = (0.0, 0.15, 0.25, 0.35, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01)


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _histogram(values: Sequence[float]) -> str:
    counts: Counter = Counter()
    n = len(values)
    for v in values:
        for i in range(len(HIST_BINS) - 1):
            if HIST_BINS[i] <= v < HIST_BINS[i + 1]:
                counts[(HIST_BINS[i], HIST_BINS[i + 1])] += 1
                break
    lines = []
    for i in range(len(HIST_BINS) - 1):
        lo, hi = HIST_BINS[i], HIST_BINS[i + 1]
        c = counts.get((lo, hi), 0)
        bar = "#" * int(round(40 * c / max(n, 1)))
        lines.append(
            "  [{:.2f}, {:.2f}): {:5d} ({:5.1f}%) {}".format(
                lo, hi, c, 100.0 * c / max(n, 1), bar
            )
        )
    return "\n".join(lines)


def _ambiguous_attractor(values: List[float]) -> float:
    """Modal composite of mid-rank items (engineering notes §2-C)."""
    if len(values) < 5:
        return float("nan")
    order = np.argsort(values)
    lo = int(0.3 * len(order))
    hi = int(0.7 * len(order))
    mid = [values[i] for i in order[lo:hi]] or values
    hist, edges = np.histogram(mid, bins=20, range=(0.0, 1.0))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def run(profile_id: int, limit: int) -> int:
    model = db.get_active_model(profile_id)
    if not model:
        print("No active model for profile {}.".format(profile_id), file=sys.stderr)
        return 1
    mp = model.get("model_path") or ""
    cal = load_calibration(mp) if mp else None
    pf = (cal or {}).get("prior_fit") if isinstance(cal, dict) else None

    print("=== score flip diff ===")
    print("profile:", profile_id)
    print("model: v{} {}".format(model.get("version"), model.get("architecture")))
    print("sidecar:", calibration_sidecar_path(mp) if mp else "")
    print("sidecar version:", (cal or {}).get("version"))
    print("temperature:", (cal or {}).get("temperature"))
    if isinstance(pf, dict):
        print("base_rate_composite:", pf.get("base_rate_composite"))
        print("target_priors:", json.dumps(pf.get("target_priors") or {}, sort_keys=True))
    else:
        print("prior_fit: (none — v1 sidecar; new path == temperature only)")
    print()

    entries = db.get_recent_entries(days=30, include_embedding=True)
    if len(entries) > limit:
        entries = entries[:limit]
    print("scoring {} recent entries...".format(len(entries)))
    old_rows = pipeline.score_entries(
        entries, profile_id=profile_id, apply_prior=False
    )
    new_rows = pipeline.score_entries(
        entries, profile_id=profile_id, apply_prior=True
    )
    if not old_rows or not new_rows:
        print("no scores produced", file=sys.stderr)
        return 1

    old_ranked = ml_window._rank_normalize_push_scores(
        [{"relevance_score": s["relevance_score"], "i": i} for i, s in enumerate(old_rows)]
    )
    old_vals = [r["relevance_score"] for r in old_ranked]
    new_vals = [s["relevance_score"] for s in new_rows]

    print("--- old path (T only, rank-normalized) ---")
    print(_histogram(old_vals))
    print("  mean={:.3f} median={:.3f}".format(float(np.mean(old_vals)), _pct(old_vals, 50)))
    print("--- new path (offsets + T, absolute) ---")
    print(_histogram(new_vals))
    print("  mean={:.3f} median={:.3f}".format(float(np.mean(new_vals)), _pct(new_vals, 50)))
    attractor = _ambiguous_attractor(new_vals)
    print("measured ambiguous-attractor (mid-rank mode): {:.3f}".format(attractor))
    print()

    order_new = sorted(range(len(new_rows)), key=lambda i: new_vals[i], reverse=True)
    print("--- top-50 by new score ---")
    print("rank\told\tnew\tΔ\tpred_old\tpred_new\ttype:id")
    for r, i in enumerate(order_new[:50], start=1):
        o, n = old_vals[i], new_vals[i]
        print(
            "{}\t{:.3f}\t{:.3f}\t{:+.3f}\t{}\t{}\t{}:{}".format(
                r,
                o,
                n,
                n - o,
                old_rows[i].get("predicted_label"),
                new_rows[i].get("predicted_label"),
                new_rows[i].get("entry_type"),
                new_rows[i].get("entry_id"),
            )
        )
    print()

    print("--- decile map (old-rank decile → new mean score) ---")
    order_old = np.argsort(old_vals)
    n = len(order_old)
    for d in range(10):
        lo = int(d * n / 10)
        hi = int((d + 1) * n / 10)
        idxs = order_old[lo:hi]
        chunk = [new_vals[i] for i in idxs]
        print(
            "  old D{}: n={} new_mean={:.3f} new_median={:.3f}".format(
                d + 1, len(chunk), float(np.mean(chunk)) if chunk else float("nan"),
                _pct(chunk, 50),
            )
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    return run(args.profile_id, args.limit)


if __name__ == "__main__":
    sys.exit(main())
