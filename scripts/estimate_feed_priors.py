#!/usr/bin/env python3
"""
Estimate feed class priors from Seismo Census labels.

Census rows are tagged ``[census]`` in reasoning (seismo Label Census tab).
Blend with all labeled empirical priors via Dirichlet-style smoothing:

    π_c = (n_census_c + α · π̂_labeled_c) / (N_census + α)

α = 20. ``build_prior_fit`` already honors ``prior_target_override`` — this
script never touches train/score code.

Refresh when census N ≥ 50 or monthly. ``prior_target_override`` is the only
place priors are hand-tuned.

    python scripts/estimate_feed_priors.py
    python scripts/estimate_feed_priors.py --profile-id 1 --write-config
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLASSES = ["investigation_lead", "important", "background", "noise"]
CENSUS_TAG = "[census]"
DEFAULT_ALPHA = 20.0
CENSUS_REFRESH_N = 50


def is_census_reasoning(reasoning: Optional[str]) -> bool:
    return CENSUS_TAG in (reasoning or "")


def labeled_empirical(labels: Iterable[str]) -> Dict[str, float]:
    counts = Counter(labels)
    n = float(sum(counts[c] for c in CLASSES))
    if n <= 0.0:
        u = 1.0 / float(len(CLASSES))
        return {c: u for c in CLASSES}
    return {c: float(counts[c]) / n for c in CLASSES}


def blend_feed_priors(
    census_labels: List[str],
    all_labels: List[str],
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[Dict[str, float], int, Dict[str, float]]:
    """Return (π, N_census, π̂_labeled)."""
    pi_hat = labeled_empirical(all_labels)
    census_counts = Counter(census_labels)
    n_census = int(sum(census_counts[c] for c in CLASSES))
    denom = float(n_census) + float(alpha)
    if denom <= 0.0:
        return dict(pi_hat), n_census, pi_hat
    pi = {}
    for c in CLASSES:
        pi[c] = (float(census_counts[c]) + float(alpha) * pi_hat[c]) / denom
    z = float(sum(pi.values()))
    if z > 0.0:
        pi = {c: pi[c] / z for c in CLASSES}
    return pi, n_census, pi_hat


def _write_override(profile_id: int, pi: Dict[str, float]) -> None:
    import db

    db.merge_profile_training_settings(
        profile_id,
        {"prior_target_override": {c: round(float(pi[c]), 6) for c in CLASSES}},
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate feed priors from [census] labels")
    parser.add_argument("--profile-id", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write prior_target_override into this profile's training_settings",
    )
    args = parser.parse_args(argv)

    import db

    rows = db.get_all_labels_raw(profile_id=args.profile_id)
    all_labs = [r.get("label") or "" for r in rows]
    census_labs = [
        r.get("label") or ""
        for r in rows
        if is_census_reasoning(r.get("reasoning"))
    ]
    pi, n_census, pi_hat = blend_feed_priors(census_labs, all_labs, alpha=args.alpha)

    print("profile_id={}".format(args.profile_id))
    print("labels_total={} census_n={} alpha={}".format(len(all_labs), n_census, args.alpha))
    print("pi_hat_labeled={}".format(json.dumps(pi_hat, indent=2)))
    print("pi_blended={}".format(json.dumps(pi, indent=2)))
    if n_census < CENSUS_REFRESH_N:
        print(
            "note: census N < {} — refresh Census tab until N≥{} or monthly".format(
                CENSUS_REFRESH_N, CENSUS_REFRESH_N
            )
        )
    if n_census == 0:
        print("no [census] labels; not writing prior_target_override")
        return 1
    if args.write_config:
        _write_override(args.profile_id, pi)
        print("wrote prior_target_override into profile {} training_settings".format(args.profile_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
