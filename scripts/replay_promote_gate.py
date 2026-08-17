#!/usr/bin/env python3
"""
Replay promote-gate v1 vs v2 over stored model history, then (optionally)
common-eval rematch.

Reads the magnitu ``models`` table per desk, ordered by version, and for every
consecutive (current → candidate) pair prints verdict flips between the
pre-v2 two-path gate and ``evaluate_model_update``. See
``docs/model-v2-engineering-notes.md`` §6.

A second pass re-scores each *old* ``.joblib`` on the *current* labeled holdout
(``pipeline.evaluate_stored_model``) and compares that gate verdict to the
stored-metrics verdict. Pairs whose ``.joblib`` is missing are skipped.
Success criteria apply to the stored-metrics v1/v2 comparison only.

Run from the repo root (uses MAGNITU_DATA_DIR / default Application Support DB):

    python scripts/replay_promote_gate.py
    python scripts/replay_promote_gate.py --db /path/to/magnitu.db
    python scripts/replay_promote_gate.py --no-common-eval
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
)


# Byte-identical copy of the pre-v2 ``_should_promote`` (ml_window.py before
# gate v2). Kept here so production stays a one-line delegate.
def legacy_should_promote(old_metrics: Optional[dict], new_metrics: dict) -> bool:
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
    }


def _explain_flip(
    old_m: dict, new_m: dict, old_verdict: bool, new_verdict: bool
) -> Tuple[str, bool]:
    """Return (reason, p30_worse_under_new)."""
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


def _load_desks(db_path: Path) -> List[Tuple[str, List[dict]]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    profiles = conn.execute(
        "SELECT id, slug, display_name FROM profiles ORDER BY id ASC"
    ).fetchall()
    desks = []
    for prof in profiles:
        rows = conn.execute(
            "SELECT version, precision_at_30, f1_score, lead_recall_at_30, "
            "model_path, architecture, profile_id "
            "FROM models WHERE profile_id = ? ORDER BY version ASC",
            (prof["id"],),
        ).fetchall()
        models = [_metrics(r) for r in rows]
        for m, r in zip(models, rows):
            m["model_path"] = r["model_path"]
            m["architecture"] = r["architecture"]
            m["profile_id"] = r["profile_id"]
        label = prof["slug"] or prof["display_name"] or "profile-{}".format(prof["id"])
        desks.append((label, models))
    conn.close()
    return desks


def replay(db_path: Path) -> int:
    desks = _load_desks(db_path)
    n_pairs = 0
    n_flips = 0
    n_unexplained = 0
    n_p30_worse = 0

    print("db: {}".format(db_path))
    print(
        "desk\tvOld→vNew\told\tnew\tΔp@30\tΔf1\tΔlead_recall\treason"
    )

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
            old_p30 = float(old_m.get("precision_at_30") or 0.0)
            new_p30 = float(new_m.get("precision_at_30") or 0.0)
            old_f1 = float(old_m.get("f1_score") or 0.0)
            new_f1 = float(new_m.get("f1_score") or 0.0)
            old_lr = float(old_m.get("lead_recall_at_30") or 0.0)
            new_lr = float(new_m.get("lead_recall_at_30") or 0.0)
            print(
                "{}\tv{}→v{}\t{}\t{}\t{:+.3f}\t{:+.3f}\t{:+.3f}\t{}".format(
                    desk,
                    old_m["version"],
                    new_m["version"],
                    "promote" if old_v else "reject",
                    "promote" if new_v else "reject",
                    new_p30 - old_p30,
                    new_f1 - old_f1,
                    new_lr - old_lr,
                    reason,
                )
            )

    print()
    print(
        "pairs={} flips={} unexplained={} p30_worse_under_new={}".format(
            n_pairs, n_flips, n_unexplained, n_p30_worse
        )
    )
    if n_pairs == 0:
        print("no consecutive model pairs — nothing to replay")
        return 0
    ok = n_unexplained == 0 and n_p30_worse == 0
    print("success criteria: {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def replay_common_eval(db_path: Path) -> None:
    """Compare stored-metrics gate v2 vs rematch-old on the current holdout."""
    import config as magnitu_config
    import db as magnitu_db
    import pipeline

    magnitu_config.DB_PATH = db_path
    magnitu_db.DB_PATH = db_path

    desks = _load_desks(db_path)
    n_pairs = 0
    n_skipped = 0
    n_differ = 0

    print()
    print(
        "common-eval (gate v2 stored-metrics vs rematch-old on current holdout)"
    )
    print("desk\tvOld→vNew\tstored\trematch\tdiffer\tnote")

    for desk, models in desks:
        for old_m, new_m in zip(models, models[1:]):
            n_pairs += 1
            stored_v = evaluate_model_update(old_m, new_m)
            joblib_path = old_m.get("model_path") or ""
            if not joblib_path or not Path(joblib_path).exists():
                n_skipped += 1
                print(
                    "{}\tv{}→v{}\t{}\t—\tskip\tartifact missing".format(
                        desk,
                        old_m["version"],
                        new_m["version"],
                        "promote" if stored_v else "reject",
                    )
                )
                continue
            rematch = pipeline.evaluate_stored_model(
                old_m, profile_id=int(old_m.get("profile_id") or 1)
            )
            if rematch.get("success") is not True:
                n_skipped += 1
                print(
                    "{}\tv{}→v{}\t{}\t—\tskip\t{}".format(
                        desk,
                        old_m["version"],
                        new_m["version"],
                        "promote" if stored_v else "reject",
                        rematch.get("error", "rematch failed"),
                    )
                )
                continue
            rematch_v = evaluate_model_update(rematch, new_m)
            differ = stored_v != rematch_v
            if differ:
                n_differ += 1
            print(
                "{}\tv{}→v{}\t{}\t{}\t{}\tp@30 {:.3f}→{:.3f} stored / "
                "rematch old {:.3f}".format(
                    desk,
                    old_m["version"],
                    new_m["version"],
                    "promote" if stored_v else "reject",
                    "promote" if rematch_v else "reject",
                    "DIFF" if differ else "same",
                    float(old_m.get("precision_at_30") or 0.0),
                    float(new_m.get("precision_at_30") or 0.0),
                    float(rematch.get("precision_at_30") or 0.0),
                )
            )

    print()
    print(
        "common-eval pairs={} skipped={} stored_vs_rematch_diff={}".format(
            n_pairs, n_skipped, n_differ
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to magnitu.db (default: config.DB_PATH)",
    )
    parser.add_argument(
        "--no-common-eval",
        action="store_true",
        help="Skip re-scoring stored .joblib files on the current holdout",
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
    status = replay(db_path)
    if not args.no_common_eval:
        replay_common_eval(db_path)
    return status


if __name__ == "__main__":
    sys.exit(main())
