#!/usr/bin/env python3
"""Stage C shadow capture: score fresh arrivals with frozen artifacts without publishing.

Scores new entries from Seismo against both an incumbent and a candidate model,
recording prediction time, content hash, probability vectors, and serialized scores
to a local JSONL file for later blinded editorial review.

Does NOT push scores to Seismo. Does NOT update the worker DB. Does NOT trigger
any training or recipe distillation. This is a read-only capture path.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_shadow_capture.py \
      --profile 3 --incumbent v55 --candidate v56 --since "2026-09-07" --out shadow_p3.jsonl

  # Run for all desks with their current incumbent/candidate pair:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_shadow_capture.py --all

  # Dry run (show what would be captured without scoring):
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_shadow_capture.py --profile 3 --dry-run
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import numpy as np


def content_hash(text: str) -> str:
    """SHA-256 of the entry text used for scoring (first 10k chars)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def load_model_by_version(profile_id: int, version: int) -> Optional[dict]:
    """Load a specific model version (not necessarily active)."""
    import db
    conn = db.get_db()
    row = conn.execute(
        "SELECT * FROM models WHERE profile_id = ? AND version = ? ORDER BY id DESC LIMIT 1",
        (profile_id, version),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def score_with_model(entries: List[dict], model_info: dict, apply_prior: bool = False) -> List[dict]:
    """Score entries with a specific model (not necessarily the active one).

    Uses the same _score_transformer path as production, but with an explicit
    model_info instead of db.get_active_model().
    """
    import pipeline as pipe
    arch = model_info.get("architecture", "tfidf")
    if arch == "transformer":
        return pipe._score_transformer(entries, model_info, apply_prior=apply_prior)
    return pipe._score_tfidf(entries, model_info, apply_prior=apply_prior)


def fetch_fresh_entries(since: str, limit: int = 200) -> List[dict]:
    """Fetch fresh entries from Seismo since the given timestamp.

    Uses the same sync.pull_entries path as the worker, but does NOT write
    to the worker DB.
    """
    import sync
    import db

    entries = []
    for entry_type in ["feed_item", "email", "lex_item", "calendar_event"]:
        try:
            batch = sync.pull_entries(
                entry_type=entry_type,
                since=since,
                limit=limit,
                order="asc",
            )
            entries.extend(batch)
        except Exception as e:
            print(f"  Warning: failed to pull {entry_type}: {e}")
    return entries


def load_local_entries(since: str, profile_id: int) -> List[dict]:
    """Load entries from the local worker DB that are newer than 'since'.

    This is the offline fallback when we can't or don't want to hit Seismo.
    """
    import db
    conn = db.get_db()
    rows = conn.execute(
        "SELECT * FROM entries WHERE fetched_at > ? OR published_date > ? "
        "ORDER BY fetched_at ASC",
        (since, since),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def capture_profile(
    profile_id: int,
    incumbent_version: int,
    candidate_version: int,
    since: str,
    output_path: str,
    use_seismo: bool = False,
    dry_run: bool = False,
) -> dict:
    """Capture shadow scores for one profile.

    Returns a summary dict with counts and timing.
    """
    import db
    import pipeline as pipe

    print(f"\n{'='*60}")
    print(f"Shadow capture: profile {profile_id}")
    print(f"  incumbent: v{incumbent_version}")
    print(f"  candidate: v{candidate_version}")
    print(f"  since: {since}")
    print(f"  output: {output_path}")
    print(f"{'='*60}")

    # Load frozen models
    incumbent = load_model_by_version(profile_id, incumbent_version)
    candidate = load_model_by_version(profile_id, candidate_version)

    if not incumbent:
        print(f"  ERROR: incumbent v{incumbent_version} not found for p{profile_id}")
        return {"error": "incumbent not found"}
    if not candidate:
        print(f"  ERROR: candidate v{candidate_version} not found for p{profile_id}")
        return {"error": "candidate not found"}

    print(f"  incumbent: v{incumbent_version} arch={incumbent['architecture']} path={incumbent.get('model_path', '?')[-40:]}")
    print(f"  candidate: v{candidate_version} arch={candidate['architecture']} path={candidate.get('model_path', '?')[-40:]}")

    # Get entries
    if use_seismo:
        print(f"  Fetching fresh entries from Seismo since {since}...")
        entries = fetch_fresh_entries(since)
    else:
        print(f"  Loading entries from local DB since {since}...")
        entries = load_local_entries(since, profile_id)

    if not entries:
        print(f"  No entries found since {since}")
        return {"profile_id": profile_id, "entries": 0}

    print(f"  Found {len(entries)} entries")

    if dry_run:
        print(f"  [DRY RUN] Would score {len(entries)} entries with both models")
        return {"profile_id": profile_id, "entries": len(entries), "dry_run": True}

    # Score with both models
    t0 = time.time()
    print(f"  Scoring with incumbent v{incumbent_version}...")
    incumbent_scores = score_with_model(entries, incumbent)
    t_inc = time.time() - t0
    print(f"    {len(incumbent_scores)} scores in {t_inc:.1f}s")

    t0 = time.time()
    print(f"  Scoring with candidate v{candidate_version}...")
    candidate_scores = score_with_model(entries, candidate)
    t_cand = time.time() - t0
    print(f"    {len(candidate_scores)} scores in {t_cand:.1f}s")

    # Build paired records
    inc_map = {}
    for s in incumbent_scores:
        key = (s["entry_type"], s["entry_id"])
        inc_map[key] = s

    cand_map = {}
    for s in candidate_scores:
        key = (s["entry_type"], s["entry_id"])
        cand_map[key] = s

    all_keys = set(inc_map.keys()) | set(cand_map.keys())
    capture_time = datetime.now(timezone.utc).isoformat()

    records = []
    for key in sorted(all_keys):
        entry_type, entry_id = key
        inc = inc_map.get(key, {})
        cand = cand_map.get(key, {})

        # Find the entry text for content hash
        entry = None
        for e in entries:
            if e.get("entry_type") == entry_type and e.get("entry_id") == entry_id:
                entry = e
                break

        text_hash = ""
        if entry:
            text = entry.get("content", "") or entry.get("description", "")
            text_hash = content_hash(text)

        record = {
            "capture_time": capture_time,
            "profile_id": profile_id,
            "entry_type": entry_type,
            "entry_id": entry_id,
            "content_hash": text_hash,
            "published_date": entry.get("published_date") if entry else None,
            "incumbent": {
                "model_version": incumbent_version,
                "relevance_score": inc.get("relevance_score"),
                "predicted_label": inc.get("predicted_label"),
                "probabilities": inc.get("probabilities"),
            },
            "candidate": {
                "model_version": candidate_version,
                "relevance_score": cand.get("relevance_score"),
                "predicted_label": cand.get("predicted_label"),
                "probabilities": cand.get("probabilities"),
            },
            "disagreement": (
                inc.get("predicted_label") != cand.get("predicted_label")
                or abs((inc.get("relevance_score") or 0) - (cand.get("relevance_score") or 0)) > 0.1
            ),
        }
        records.append(record)

    # Write JSONL
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Summary
    disagreements = sum(1 for r in records if r["disagreement"])
    label_disagreements = sum(
        1 for r in records
        if r["incumbent"]["predicted_label"] != r["candidate"]["predicted_label"]
    )
    score_deltas = [
        abs((r["incumbent"]["relevance_score"] or 0) - (r["candidate"]["relevance_score"] or 0))
        for r in records
    ]

    summary = {
        "profile_id": profile_id,
        "incumbent_version": incumbent_version,
        "candidate_version": candidate_version,
        "entries": len(records),
        "disagreements": disagreements,
        "label_disagreements": label_disagreements,
        "mean_score_delta": np.mean(score_deltas) if score_deltas else 0.0,
        "max_score_delta": max(score_deltas) if score_deltas else 0.0,
        "incumbent_time_s": t_inc,
        "candidate_time_s": t_cand,
        "output_path": output_path,
    }

    print(f"\n  Summary:")
    print(f"    entries: {summary['entries']}")
    print(f"    disagreements: {summary['disagreements']} ({summary['label_disagreements']} label, {summary['disagreements'] - summary['label_disagreements']} score-only)")
    print(f"    mean score delta: {summary['mean_score_delta']:.4f}")
    print(f"    max score delta: {summary['max_score_delta']:.4f}")
    print(f"    incumbent time: {t_inc:.1f}s")
    print(f"    candidate time: {t_cand:.1f}s")
    print(f"    written to: {output_path}")

    return summary


# Default incumbent/candidate pairs for --all
# These are the messy-text vs clean-text pairs from §2b
DEFAULT_PAIRS = {
    1: {"incumbent": 1017, "candidate": 1018, "name": "mothership"},
    2: {"incumbent": 43, "candidate": 44, "name": "digital"},
    3: {"incumbent": 55, "candidate": 56, "name": "sicherheit"},
    4: {"incumbent": 40, "candidate": 42, "name": "eu"},
}


def main():
    parser = argparse.ArgumentParser(description="Stage C shadow capture for Magnitu")
    parser.add_argument("--profile", type=int, nargs="+", help="Profile IDs to capture")
    parser.add_argument("--all", action="store_true", help="Capture all desks with default pairs")
    parser.add_argument("--incumbent", type=int, help="Incumbent model version (single profile)")
    parser.add_argument("--candidate", type=int, help="Candidate model version (single profile)")
    parser.add_argument("--since", default=None, help="ISO timestamp: only entries newer than this")
    parser.add_argument("--out", default=None, help="Output JSONL path (default: lab_data/shadow_capture/)")
    parser.add_argument("--use-seismo", action="store_true", help="Fetch from Seismo instead of local DB")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without scoring")
    args = parser.parse_args()

    if args.all:
        pairs = DEFAULT_PAIRS
        profiles = list(pairs.keys())
    elif args.profile:
        profiles = args.profile
        if len(profiles) == 1 and args.incumbent and args.candidate:
            pairs = {
                profiles[0]: {
                    "incumbent": args.incumbent,
                    "candidate": args.candidate,
                    "name": f"p{profiles[0]}",
                }
            }
        else:
            pairs = DEFAULT_PAIRS
    else:
        parser.error("Specify --all or --profile")

    # Default since: 7 days ago
    if args.since is None:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    else:
        since = args.since

    # Output directory
    out_dir = BASE_DIR / "lab_data" / "shadow_capture"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pid in profiles:
        pair = pairs[pid]
        name = pair["name"]
        out_path = args.out or str(out_dir / f"shadow_p{pid}_{pair['incumbent']}vs{pair['candidate']}.jsonl")

        result = capture_profile(
            profile_id=pid,
            incumbent_version=pair["incumbent"],
            candidate_version=pair["candidate"],
            since=since,
            output_path=out_path,
            use_seismo=args.use_seismo,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Write summary
    summary_path = str(out_dir / "shadow_capture_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "capture_date": datetime.now(timezone.utc).isoformat(),
            "since": since,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
