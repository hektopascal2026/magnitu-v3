#!/usr/bin/env python3
"""One-off: synthetic-label unlabeled substack + parl_press entries for Sicherheit.

Sicherheit (profile 3) has:
  - substack: 398 unlabeled, 12.4% lead rate on 105 labeled
  - parl_press: 339 unlabeled, 80% lead rate on 5 labeled (small sample)

Target: ~60 substack + ~40 parl_press = 100 entries.

Gold-bite few-shot examples are SKIPPED (EU territorial-exclusion pattern
is irrelevant for a security desk). Human-labeled sicherheit examples
are used as calibration instead.

Usage (on the VPS):
  cd /opt/seismo-magnitu-ml/app
  MAGNITU_DATA_DIR=/opt/seismo-magnitu-ml/state \
  PYTHONPATH=/opt/seismo-magnitu-ml/app \
  venv/bin/python3 scripts/synthetic_label_sicherheit.py \
      --profile-id 3 --limit 100 --mode batch
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

import db
from magnitu.gemini import GeminiClient
from magnitu.gemini_config import GeminiConfig
from magnitu.prompts import MAGNITU_LABELS
from magnitu.synthetic_batch import (
    SOURCE_GEMINI,
    _build_few_shot_examples,
    _eligible_for_gemini,
)
from magnitu.synthetic_scorer import call_gemini_for_synthetic_label_batch

# Source types to target, in priority order, with per-source limits.
TARGET_SOURCES = [
    ("substack", 60),
    ("parl_press", 40),
]


def select_target_entries(profile_id: int, limit: int) -> List[Dict]:
    """Select unlabeled entries from target source types."""
    conn = db.get_db()
    try:
        labeled_keys = set()
        for r in conn.execute(
            "SELECT entry_type, entry_id FROM labels WHERE profile_id = ?",
            (profile_id,),
        ):
            labeled_keys.add((r["entry_type"], r["entry_id"]))

        result: List[Dict] = []
        for source_type, source_limit in TARGET_SOURCES:
            rows = conn.execute(
                "SELECT * FROM entries WHERE source_type = ? ORDER BY entry_id",
                (source_type,),
            ).fetchall()
            unlabeled = [
                dict(r) for r in rows
                if (r["entry_type"], r["entry_id"]) not in labeled_keys
            ]
            take = min(source_limit, limit - len(result))
            result.extend(unlabeled[:take])
            print(
                "  %s: %d unlabeled, taking %d"
                % (source_type, len(unlabeled), min(take, len(unlabeled)))
            )
            if len(result) >= limit:
                break
        return result[:limit]
    finally:
        conn.close()


def run_batch(
    profile_id: int,
    entries: List[Dict],
    mode: str = "batch",
) -> Dict:
    """Run synthetic labeling on the selected entries."""
    cfg = GeminiConfig.from_env()
    if not (cfg.api_key or "").strip():
        raise ValueError("GEMINI_API_KEY is not set.")

    system_instruction = db.get_profile_gemini_persona(profile_id)
    # Skip Gold-bite examples — EU territorial-exclusion is irrelevant for sicherheit
    few_shot = _build_few_shot_examples(profile_id, include_gold_bite=False)
    print("Few-shot examples: %d (Gold-bite skipped)" % len(few_shot))

    labeled = 0
    failed: List[Dict] = []
    chunk_size = 10

    with GeminiClient(cfg) as client:
        for i in range(0, len(entries), chunk_size):
            chunk = entries[i : i + chunk_size]
            print(
                "Batch %d-%d/%d: %s"
                % (i + 1, min(i + chunk_size, len(entries)), len(entries),
                   ", ".join(str(e["entry_id"]) for e in chunk))
            )

            eligible = []
            for e in chunk:
                ok, _skip = _eligible_for_gemini(
                    e["entry_type"], int(e["entry_id"]), profile_id, False
                )
                if ok:
                    eligible.append(e)

            if not eligible:
                print("  (all already labeled, skipping)")
                continue

            try:
                results = call_gemini_for_synthetic_label_batch(
                    client,
                    eligible,
                    system_instruction=system_instruction,
                    few_shot_examples=few_shot,
                )
                results_by_key = {}
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    try:
                        rid = int(r.get("entry_id", 0))
                    except (TypeError, ValueError):
                        continue
                    rt = str(r.get("entry_type") or "").strip()
                    if rt and rid:
                        results_by_key[(rt, rid)] = r

                for entry in eligible:
                    et, eid = entry["entry_type"], int(entry["entry_id"])
                    res = results_by_key.get((str(et), eid))
                    if res:
                        label = res.get("label")
                        reasoning = res.get("reasoning", "")
                        if label in MAGNITU_LABELS and reasoning:
                            db.set_label(
                                et,
                                eid,
                                label,
                                reasoning=reasoning,
                                profile_id=profile_id,
                                label_source=SOURCE_GEMINI,
                            )
                            labeled += 1
                            print("  -> %s #%d: %s" % (et, eid, label))
                        else:
                            failed.append({
                                "entry_type": et,
                                "entry_id": eid,
                                "error": "Invalid label or empty reasoning",
                            })
                            print("  -> %s #%d: FAILED (invalid/empty)" % (et, eid))
                    else:
                        failed.append({
                            "entry_type": et,
                            "entry_id": eid,
                            "error": "Missing from batch response",
                        })
                        print("  -> %s #%d: FAILED (missing)" % (et, eid))
            except Exception as ex:
                print("  ERROR: %s" % str(ex)[:200])
                for entry in eligible:
                    failed.append({
                        "entry_type": entry["entry_type"],
                        "entry_id": int(entry["entry_id"]),
                        "error": str(ex)[:500],
                    })

            if i + chunk_size < len(entries):
                time.sleep(1.0)

    return {"labeled": labeled, "failed": failed, "total": len(entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    args = parser.parse_args()

    print("Profile: %d (Sicherheit)" % args.profile_id)
    print("Limit: %d" % args.limit)
    print("Target sources: %s" % ", ".join("%s(%d)" % (s, l) for s, l in TARGET_SOURCES))
    print()

    entries = select_target_entries(args.profile_id, args.limit)
    if not entries:
        print("No unlabeled entries found.")
        sys.exit(0)

    print("\nSelected %d entries. Starting batch...\n" % len(entries))
    result = run_batch(args.profile_id, entries, mode=args.mode)
    print()
    print("=" * 60)
    print("Labeled: %d / %d" % (result["labeled"], result["total"]))
    print("Failed:  %d" % len(result["failed"]))
    if result["failed"]:
        print("Failed entries:")
        for f in result["failed"]:
            print("  %s #%d: %s" % (f["entry_type"], f["entry_id"], f["error"]))


if __name__ == "__main__":
    main()
