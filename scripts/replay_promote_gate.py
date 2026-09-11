#!/usr/bin/env python3
"""
Replay promote decisions over stored model history.

Default (live gate): for each consecutive (old → new) pair that still has
``.joblib`` artifacts, re-score **both** on the persistent eval reserve via
``pipeline.evaluate_on_recent`` and decide with ``ml_window.evaluate_recent_gate``
(UTIL@30 + bootstrap CI). Ties are printed as ``tie`` (they still promote).

Optional ``--legacy-stored``: also print flips between the pre-v2 stored-metrics
gate and legacy ``evaluate_model_update`` (historical audit only — unsafe across
embedding generations).

    python scripts/replay_promote_gate.py
    python scripts/replay_promote_gate.py --db /path/to/magnitu.db
    python scripts/replay_promote_gate.py --legacy-stored
    python scripts/replay_promote_gate.py --max-pairs 20
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_window import (  # noqa: E402
    F1_HARD_DROP_LIMIT,
    LEAD_RECALL_SLACK,
    PROMOTE_BIG_P30_WIN,
    PROMOTE_MARGIN,
    PROMOTE_RANKING_SLACK,
    evaluate_model_update,
    evaluate_recent_gate,
)


def legacy_should_promote(old_metrics: Optional[dict], new_metrics: dict) -> bool:
    """Byte-identical copy of the pre-v2 ``_should_promote``."""
    if not old_metrics:
        return True
    new_p30 = float(new_metrics.get("precision_at_30") or 0.0)
    old_p30 = float(old_metrics.get("precision_at_30") or 0.0)
    new_f1 = float(new_metrics.get("f1_score") or 0.0)
    old_f1 = float(old_metrics.get("f1_score") or 0.0)
    p30_up = new_p30 >= old_p30 + PROMOTE_MARGIN
    f1_up = new_f1 >= old_f1 + PROMOTE_MARGIN
    f1_ok = new_f1 >= old_f1 - PROMOTE_MARGIN
    p30_not_collapsed = new_p30 >= old_p30 - PROMOTE_RANKING_SLACK
    return (p30_up and f1_ok) or (f1_up and p30_not_collapsed)


def _metrics(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "version": row["version"],
        "precision_at_30": row["precision_at_30"],
        "f1_score": row["f1_score"],
        "lead_recall_at_30": row["lead_recall_at_30"],
        "util_at_30": row["util_at_30"] if "util_at_30" in row.keys() else None,
        "model_path": row["model_path"],
        "architecture": row["architecture"],
        "profile_id": row["profile_id"],
        "embedding_l2_normalize": (
            row["embedding_l2_normalize"]
            if "embedding_l2_normalize" in row.keys()
            else 0
        ),
    }


def _load_desks(db_path: Path) -> List[Tuple[str, List[dict]]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    profiles = conn.execute(
        "SELECT id, slug, display_name FROM profiles ORDER BY id ASC"
    ).fetchall()
    desks = []
    cols = {r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
    select_cols = [
        "version",
        "precision_at_30",
        "f1_score",
        "lead_recall_at_30",
        "model_path",
        "architecture",
        "profile_id",
    ]
    if "util_at_30" in cols:
        select_cols.append("util_at_30")
    if "embedding_l2_normalize" in cols:
        select_cols.append("embedding_l2_normalize")
    for prof in profiles:
        rows = conn.execute(
            "SELECT {} FROM models WHERE profile_id = ? ORDER BY version ASC".format(
                ", ".join(select_cols)
            ),
            (prof["id"],),
        ).fetchall()
        models = [_metrics(r) for r in rows]
        label = prof["slug"] or prof["display_name"] or "profile-{}".format(prof["id"])
        desks.append((label, models))
    conn.close()
    return desks


def _gate_label(old_r: dict, new_r: dict, promoted: bool) -> str:
    """Classify promote / reject / tie using the same bootstrap as the live gate."""
    import pipeline

    if not promoted:
        return "reject"
    old_c = old_r.get("_composites")
    new_c = new_r.get("_composites")
    tw = new_r.get("_true_weights")
    if tw is None:
        tw = old_r.get("_true_weights")
    if (
        old_c is not None
        and new_c is not None
        and tw is not None
        and len(old_c) == len(new_c) == len(tw)
    ):
        boot = pipeline.bootstrap_util_delta(old_c, new_c, tw)
        if boot.get("tie"):
            return "tie"
    return "promote"


def replay_util_reserve(db_path: Path, max_pairs: Optional[int] = None) -> int:
    """Live-gate acceptance: UTIL@30 on the eval reserve for consecutive artifacts."""
    import config as magnitu_config
    import db as magnitu_db
    import pipeline

    magnitu_config.DB_PATH = db_path
    magnitu_db.DB_PATH = db_path

    desks = _load_desks(db_path)
    n_pairs = 0
    n_skipped = 0
    n_promote = 0
    n_reject = 0
    n_tie = 0

    print("db: {}".format(db_path))
    print("live gate: evaluate_on_recent + evaluate_recent_gate (UTIL@30 / bootstrap)")
    print(
        "desk\tvOld→vNew\tdecision\tn\tn_leads\tUTIL_old→new\tdiag_p@30\tnote"
    )

    for desk, models in desks:
        pairs = list(zip(models, models[1:]))
        if max_pairs is not None:
            pairs = pairs[-max_pairs:]
        for old_m, new_m in pairs:
            n_pairs += 1
            profile_id = int(old_m.get("profile_id") or new_m.get("profile_id") or 1)
            old_path = old_m.get("model_path") or ""
            new_path = new_m.get("model_path") or ""
            if not old_path or not Path(old_path).exists():
                n_skipped += 1
                print(
                    "{}\tv{}→v{}\t—\t—\t—\t—\t—\told artifact missing".format(
                        desk, old_m["version"], new_m["version"]
                    )
                )
                continue
            if not new_path or not Path(new_path).exists():
                n_skipped += 1
                print(
                    "{}\tv{}→v{}\t—\t—\t—\t—\t—\tnew artifact missing".format(
                        desk, old_m["version"], new_m["version"]
                    )
                )
                continue

            old_r = pipeline.evaluate_on_recent(old_m, profile_id=profile_id)
            new_r = pipeline.evaluate_on_recent(new_m, profile_id=profile_id)
            if not old_r.get("success") or not new_r.get("success"):
                n_skipped += 1
                err = (old_r.get("error") if not old_r.get("success") else None) or (
                    new_r.get("error") or "eval failed"
                )
                print(
                    "{}\tv{}→v{}\t—\t—\t—\t—\t—\t{}".format(
                        desk, old_m["version"], new_m["version"], err
                    )
                )
                continue

            promoted = evaluate_recent_gate(old_r, new_r, has_incumbent=True)
            label = _gate_label(old_r, new_r, promoted)
            if label == "reject":
                n_reject += 1
            elif label == "tie":
                n_tie += 1
            else:
                n_promote += 1

            n = int(new_r.get("n_recent") or 0)
            n_leads = int(new_r.get("n_leads") or 0)
            print(
                "{}\tv{}→v{}\t{}\t{}\t{}\t{:.3f}→{:.3f}\t{:.3f}→{:.3f}\t"
                "{}".format(
                    desk,
                    old_m["version"],
                    new_m["version"],
                    label,
                    n,
                    n_leads,
                    float(old_r.get("util_at_30") or 0.0),
                    float(new_r.get("util_at_30") or 0.0),
                    float(old_r.get("precision_at_30") or 0.0),
                    float(new_r.get("precision_at_30") or 0.0),
                    "tie→promote" if label == "tie" else "",
                )
            )

    print()
    print(
        "util-reserve pairs={} skipped={} promote={} tie={} reject={}".format(
            n_pairs, n_skipped, n_promote, n_tie, n_reject
        )
    )
    if n_pairs == 0:
        print("no consecutive model pairs — nothing to replay")
        return 0
    # Soft success: at least one decisive reject or promote/tie on scored pairs.
    scored = n_promote + n_tie + n_reject
    ok = scored > 0
    print("live-gate replay: {}".format("PASS" if ok else "FAIL (no scored pairs)"))
    return 0 if ok else 1


def _explain_flip(
    old_m: dict, new_m: dict, old_verdict: bool, new_verdict: bool
) -> Tuple[str, bool]:
    old_p30 = float(old_m.get("precision_at_30") or 0.0)
    new_p30 = float(new_m.get("precision_at_30") or 0.0)
    old_f1 = float(old_m.get("f1_score") or 0.0)
    new_f1 = float(new_m.get("f1_score") or 0.0)
    old_lr = float(old_m.get("lead_recall_at_30") or 0.0)
    new_lr = new_m.get("lead_recall_at_30")
    p30_gain = new_p30 - old_p30
    f1_gain = new_f1 - old_f1
    p30_worse = p30_gain < 0 and new_verdict and not old_verdict

    if old_verdict and not new_verdict:
        if old_lr and new_lr is not None and float(new_lr) < old_lr - LEAD_RECALL_SLACK:
            return "lead-recall veto", False
        return "unexplained reject", False

    if (
        p30_gain >= PROMOTE_BIG_P30_WIN
        and f1_gain >= -F1_HARD_DROP_LIMIT
        and f1_gain < -PROMOTE_MARGIN
    ):
        return "big p@30 win within F1 cap", p30_worse

    return "unexplained promote", p30_worse


def replay_legacy_stored(db_path: Path) -> int:
    """Historical audit: stored-metrics flips only (not the live gate)."""
    desks = _load_desks(db_path)
    n_pairs = 0
    n_flips = 0
    n_unexplained = 0
    n_p30_worse = 0

    print()
    print("legacy stored-metrics audit (NOT the live UTIL@30 gate)")
    print("desk\tvOld→vNew\told\tnew\tΔp@30\tΔf1\tΔlead_recall\treason")

    for desk, models in desks:
        for old_m, new_m in zip(models, models[1:]):
            n_pairs += 1
            old_v = legacy_should_promote(old_m, new_m)
            new_v = evaluate_model_update(old_m, new_m)
            if old_v == new_v:
                continue
            n_flips += 1
            reason, p30_worse = _explain_flip(old_m, new_m, old_v, new_v)
            if reason.startswith("unexplained"):
                n_unexplained += 1
            if p30_worse:
                n_p30_worse += 1
            print(
                "{}\tv{}→v{}\t{}\t{}\t{:+.3f}\t{:+.3f}\t{:+.3f}\t{}".format(
                    desk,
                    old_m["version"],
                    new_m["version"],
                    "promote" if old_v else "reject",
                    "promote" if new_v else "reject",
                    float(new_m.get("precision_at_30") or 0.0)
                    - float(old_m.get("precision_at_30") or 0.0),
                    float(new_m.get("f1_score") or 0.0)
                    - float(old_m.get("f1_score") or 0.0),
                    float(new_m.get("lead_recall_at_30") or 0.0)
                    - float(old_m.get("lead_recall_at_30") or 0.0),
                    reason,
                )
            )

    print()
    print(
        "legacy pairs={} flips={} unexplained={} p30_worse_under_new={}".format(
            n_pairs, n_flips, n_unexplained, n_p30_worse
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to magnitu.db (default: config.DB_PATH)",
    )
    parser.add_argument(
        "--legacy-stored",
        action="store_true",
        help="Also print pre-v2 vs evaluate_model_update stored-metrics flips",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Per desk, only the last N consecutive pairs (live-gate path)",
    )
    args = parser.parse_args()
    if args.db is not None:
        db_path = args.db.expanduser().resolve()
    else:
        from config import DB_PATH

        db_path = Path(DB_PATH)
    if not db_path.exists():
        print("error: database not found: {}".format(db_path), file=sys.stderr)
        return 2
    status = replay_util_reserve(db_path, max_pairs=args.max_pairs)
    if args.legacy_stored:
        replay_legacy_stored(db_path)
    return status


if __name__ == "__main__":
    sys.exit(main())
