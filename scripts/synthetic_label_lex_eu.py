#!/usr/bin/env python3
"""One-off: synthetic-label ~100 targeted unlabeled lex_eu entries for a profile.

Target selection (in priority order):
  1. Entries whose body matches a Gold nexus gate (EEA/EU-only scope) — highest
     lead density. The nexus IDs are passed via --nexus-ids (comma-separated),
     pre-computed by the Seismo PHP gold_scan script.
  2. Substantive "Regulation (EU) ... of the European Parliament and of the
     Council" entries — where operative barriers live.
  3. Fill remaining slots from the general unlabeled lex_eu pool.

Usage (on the VPS):
  cd /opt/seismo-magnitu-ml/app
  venv/bin/python3 scripts/synthetic_label_lex_eu.py \
      --profile-id 4 \
      --nexus-ids 247,288,291,304,322,323,328,2637,7765,51809,51832,59995,65010,67628,70465,70471,87406,104256,109508,109513,120520,120525,122906,122926,125803,159240,166737,180976,186668,186740,195153,206034,227348 \
      --limit 100 \
      --mode batch

Labels are saved locally with label_source="Gemini". They are NOT pushed to
Seismo — use the Sync page when ready to publish.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional

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


def select_target_entries(
    profile_id: int,
    nexus_ids: List[int],
    limit: int,
) -> List[Dict]:
    """Select unlabeled lex_eu entries: nexus matches first, then substantive
    regulations, then general fill.
    """
    conn = db.get_db()
    try:
        # All unlabeled lex_eu entries for this profile
        all_rows = conn.execute(
            """SELECT * FROM entries WHERE source_type = 'lex_eu'
               ORDER BY entry_id"""
        ).fetchall()
        labeled_keys = set()
        for r in conn.execute(
            "SELECT entry_type, entry_id FROM labels WHERE profile_id = ?",
            (profile_id,),
        ):
            labeled_keys.add((r["entry_type"], r["entry_id"]))

        unlabeled = []
        for r in all_rows:
            key = (r["entry_type"], r["entry_id"])
            if key not in labeled_keys:
                unlabeled.append(dict(r))

        # Tier 1: nexus-matching entries (by ID)
        nexus_set = set(nexus_ids)
        tier1 = [e for e in unlabeled if e["entry_id"] in nexus_set]

        # Tier 2: substantive regulations (European Parliament + Council)
        # Skip implementing/delegated regulations (annex amendments, technical)
        tier2 = [
            e for e in unlabeled
            if e["entry_id"] not in nexus_set
            and "of the European Parliament and of the Council" in (e.get("title") or "")
        ]

        # Tier 3: remaining unlabeled lex_eu (fill)
        used_ids = {e["entry_id"] for e in tier1 + tier2}
        tier3 = [e for e in unlabeled if e["entry_id"] not in used_ids]

        result = (tier1 + tier2 + tier3)[:limit]
        print(
            "Target selection: %d nexus, %d substantive, %d fill = %d total"
            % (len(tier1), len([e for e in result if e["entry_id"] in nexus_set]),
               len([e for e in result if e["entry_id"] not in nexus_set
                    and "of the European Parliament and of the Council" in (e.get("title") or "")]),
               len(result))
        )
        return result
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
    few_shot = _build_few_shot_examples(profile_id)
    print("Few-shot examples: %d" % len(few_shot))

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

            # Filter to eligible (not already labeled by human)
            eligible = []
            for e in chunk:
                ok, skip = _eligible_for_gemini(
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

            # Rate-limit pause between batches
            if i + chunk_size < len(entries):
                time.sleep(1.0)

    return {"labeled": labeled, "failed": failed, "total": len(entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", type=int, required=True)
    parser.add_argument("--nexus-ids", type=str, default="",
                        help="Comma-separated entry IDs that match Gold nexus gates")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    args = parser.parse_args()

    nexus_ids = []
    if args.nexus_ids:
        nexus_ids = [int(x.strip()) for x in args.nexus_ids.split(",") if x.strip()]

    print("Profile: %d" % args.profile_id)
    print("Nexus IDs: %d" % len(nexus_ids))
    print("Limit: %d" % args.limit)
    print()

    entries = select_target_entries(args.profile_id, nexus_ids, args.limit)
    if not entries:
        print("No unlabeled lex_eu entries found.")
        sys.exit(0)

    print()
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
