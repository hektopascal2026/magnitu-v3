#!/usr/bin/env python3
"""One-off: synthetic-label unlabeled lex_eu entries for Digital desk.

Digital (profile 2) has:
  - lex_eu: 606 unlabeled, only 12 labeled with 1 lead (8.3%)
  - EU digital regulation (AI Act, Data Act, DMA, DSA) is directly relevant

Target: ~100 unlabeled lex_eu entries, prioritizing entries whose titles
match digital-relevant signals (AI, data, digital, platform, cyber, tech,
software, online, e-commerce, electronic, GDPR, privacy, chip, semiconductor).

Gold-bite few-shot examples ARE included — EU digital regulation frequently
contains territorial-scope and third-country provisions, and one Gold-bite
example is about the digital product passport registry. The human-labeled
digital leads lack reasoning, so the Gold-bite examples provide the only
calibrated lead reasoning in the few-shot set.

Usage (on the VPS):
  cd /opt/seismo-magnitu-ml/app
  MAGNITU_DATA_DIR=/opt/seismo-magnitu-ml/state \
  PYTHONPATH=/opt/seismo-magnitu-ml/app \
  venv/bin/python3 scripts/synthetic_label_digital_lex_eu.py \
      --profile-id 2 --limit 100 --mode batch
"""
from __future__ import annotations

import argparse
import re
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

# Digital-relevant signal keywords for prioritizing entry selection.
DIGITAL_NEEDLES = [
    "ai ", "artificial intelligence", "data act", "data governance",
    "digital", "platform", "cyber", "tech", "technology", "software",
    "online", "e-commerce", "electronic", "gdpr", "privacy", "personal data",
    "chip", "semiconductor", "algorithm", "automated", "internet",
    "blockchain", "crypto", "cloud", "open source", "interoperability",
    "digital product passport", "digital services", "digital markets",
    "machine learning", "biometric", "facial recognition", "surveillance",
    "telecom", "broadband", "5g", "6g", "network", "spectrum",
    "e-id", "digital identity", "trust service", "e-signature",
]


def score_digital_relevance(title: str) -> int:
    """Score how digital-relevant an entry title is (higher = more relevant)."""
    t = (title or "").lower()
    return sum(1 for needle in DIGITAL_NEEDLES if needle in t)


def select_target_entries(profile_id: int, limit: int) -> List[Dict]:
    """Select unlabeled lex_eu entries, prioritizing digital-relevant titles."""
    conn = db.get_db()
    try:
        labeled_keys = set()
        for r in conn.execute(
            "SELECT entry_type, entry_id FROM labels WHERE profile_id = ?",
            (profile_id,),
        ):
            labeled_keys.add((r["entry_type"], r["entry_id"]))

        rows = conn.execute(
            "SELECT * FROM entries WHERE source_type = 'lex_eu' "
            "ORDER BY entry_id"
        ).fetchall()
        unlabeled = [
            dict(r) for r in rows
            if (r["entry_type"], r["entry_id"]) not in labeled_keys
        ]

        # Score by digital relevance, then sort: high-relevance first,
        # then by entry_id (newer entries tend to be more relevant)
        for e in unlabeled:
            e["_digital_score"] = score_digital_relevance(e.get("title") or "")
        unlabeled.sort(key=lambda e: (-e["_digital_score"], -e["entry_id"]))

        result = unlabeled[:limit]
        digital_count = sum(1 for e in result if e["_digital_score"] > 0)
        print("  lex_eu: %d unlabeled, taking %d (%d digital-relevant)"
              % (len(unlabeled), len(result), digital_count))
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
    # Include Gold-bite: EU digital regulation has territorial-scope provisions
    few_shot = _build_few_shot_examples(profile_id, include_gold_bite=True)
    print("Few-shot examples: %d (Gold-bite included)" % len(few_shot))

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
    parser.add_argument("--profile-id", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    args = parser.parse_args()

    print("Profile: %d (Digital)" % args.profile_id)
    print("Limit: %d" % args.limit)
    print("Target source: lex_eu")
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
