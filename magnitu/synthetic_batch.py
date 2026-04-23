"""Background Gemini synthetic labeling for a profile (Smart Queue: uncertain / new first)."""

from __future__ import annotations

import errno
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import db
import sampler
from magnitu.gemini import GeminiClient
from magnitu.gemini_config import GeminiConfig
from magnitu.prompts import MAGNITU_LABELS
from magnitu.synthetic_scorer import (
    call_gemini_for_synthetic_label,
    call_gemini_for_synthetic_label_batch,
)

SOURCE_GEMINI = "Gemini"


def _entry_fields_for_prompt(entry: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": str(entry.get("title") or ""),
        "description": str(entry.get("description") or ""),
        "content": str(entry.get("content") or ""),
        "link": str(entry.get("link") or ""),
        "author": str(entry.get("author") or ""),
        "source_name": str(entry.get("source_name") or ""),
        "source_category": str(entry.get("source_category") or ""),
        "source_type": str(entry.get("source_type") or ""),
        "published_date": str(entry.get("published_date") or ""),
    }


def _eligible_for_gemini(
    entry_type: str,
    entry_id: int,
    profile_id: int,
    replace_gemini: bool,
) -> Tuple[bool, str]:
    """Returns (eligible, skip_reason). skip_reason empty when eligible."""
    meta = db.get_label_with_reasoning(entry_type, entry_id, profile_id)
    if meta is None:
        return True, ""
    src = (meta.get("label_source") or "").strip()
    if replace_gemini and src == SOURCE_GEMINI:
        return True, ""
    if not src:
        return False, "already_labeled"
    if src == SOURCE_GEMINI:
        return False, "already_gemini_skip"
    return False, "already_labeled_human"


def _transient_io_error(ex: BaseException) -> bool:
    """True for OS I/O errors we may recover from with one retry (e.g. EIO on SQLite/FS)."""
    cur: Optional[BaseException] = ex
    for _ in range(6):
        if cur is None:
            break
        if isinstance(cur, OSError) and cur.errno == errno.EIO:
            return True
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt
    return False


def _cancelled_result(
    *,
    candidates: List[Dict],
    to_process: List[Dict],
    skipped_filter: int,
    labeled: int,
    failed: List[Dict[str, Any]],
    skipped_mid: int,
) -> Dict[str, Any]:
    return {
        "success": True,
        "cancelled": True,
        "labeled": labeled,
        "skipped": skipped_filter + skipped_mid,
        "skipped_mid_run": skipped_mid,
        "failed": failed,
        "candidates_total": len(candidates),
        "queued": len(to_process),
        "message": "Cancelled (partial: %d labeled)" % labeled,
        "note": "Labeling was stopped. Entries already labeled in this run are saved; the rest were not processed.",
    }


def run_gemini_synthetic_batch_job(
    profile_id: int,
    *,
    batch_limit: int = 50,
    entry_type: Optional[str] = None,
    replace_gemini: bool = False,
    system_instruction: Optional[str] = None,
    mode: str = "single",
    progress_cb: Optional[Callable[[int, str, Optional[str]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Process up to ``batch_limit`` smart-queue entries with Gemini; persist via ``db.set_label``.

    Does not push labels to Seismo (same as bulk operations: use Sync when ready).
    """
    if not system_instruction:
        system_instruction = db.get_profile_gemini_persona(profile_id)
    
    cfg = GeminiConfig.from_env()
    if not (cfg.api_key or "").strip():
        msg = "GEMINI_API_KEY is not set. Go to Settings and add your key."
        if progress_cb:
            progress_cb(0, "Error", "FATAL: " + msg)
        raise ValueError(msg)

    candidates = sampler.get_gemini_synthetic_batch_entries(
        limit=batch_limit, entry_type=entry_type, profile_id=profile_id
    )
    to_process: List[Dict[str, Any]] = []
    for e in candidates:
        et, eid = e["entry_type"], int(e["entry_id"])
        ok, _skip = _eligible_for_gemini(et, eid, profile_id, replace_gemini)
        if ok:
            to_process.append(e)

    skipped_filter = max(0, len(candidates) - len(to_process))
    total = len(to_process)
    if total == 0:
        if progress_cb:
            progress_cb(100, "No eligible entries.")
        return {
            "success": True,
            "labeled": 0,
            "skipped": skipped_filter,
            "skipped_mid_run": 0,
            "failed": [],
            "candidates_total": len(candidates),
            "queued": 0,
            "note": "No eligible entries (all already labeled or not replaceable).",
        }

    if progress_cb:
        progress_cb(3, "Starting Gemini batch (%d entries)..." % total)

    failed: List[Dict[str, Any]] = []
    skipped_mid = 0
    labeled = 0

    with GeminiClient(cfg) as client:
        if mode == "batch":
            # Process in chunks of 10
            chunk_size = 10
            for i in range(0, len(to_process), chunk_size):
                if cancel_check and cancel_check():
                    if progress_cb:
                        progress_cb(
                            100,
                            "Cancelled by user (%d of %d labeled)."
                            % (labeled, total),
                            "Cancelled. Labels already written are kept.",
                        )
                    return _cancelled_result(
                        candidates=candidates,
                        to_process=to_process,
                        skipped_filter=skipped_filter,
                        labeled=labeled,
                        failed=failed,
                        skipped_mid=skipped_mid,
                    )
                chunk = to_process[i : i + chunk_size]
                if progress_cb:
                    pct = 5 + int(90 * i / max(total, 1))
                    progress_cb(
                        min(pct, 94),
                        "Gemini Batch %d-%d/%d" % (i + 1, min(i + chunk_size, total), total),
                        "Sending batch of %d entries to Gemini..." % len(chunk)
                    )
                
                ex_chunk: Optional[Exception] = None
                for attempt in (0, 1):
                    try:
                        results = call_gemini_for_synthetic_label_batch(
                            client,
                            chunk,
                            system_instruction=system_instruction
                        )
                        # Match by (entry_type, entry_id) — IDs can repeat across types.
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
                        for entry in chunk:
                            et, eid = entry["entry_type"], int(entry["entry_id"])
                            et_s = str(et)
                            res = results_by_key.get((et_s, eid))
                            if res:
                                label = res.get("label")
                                reasoning = res.get("reasoning", "")
                                if label in MAGNITU_LABELS and reasoning:
                                    db.set_label(
                                        et, eid, label,
                                        reasoning=reasoning,
                                        profile_id=profile_id,
                                        label_source=SOURCE_GEMINI
                                    )
                                    labeled += 1
                                    if progress_cb:
                                        progress_cb(
                                            min(pct, 94),
                                            "Gemini Batch %d-%d/%d" % (i + 1, min(i + chunk_size, total), total),
                                            "  -> %s #%s: %s" % (et, eid, label)
                                        )
                                else:
                                    failed.append({"entry_type": et, "entry_id": eid, "error": "Invalid label or empty reasoning in batch response"})
                            else:
                                failed.append({"entry_type": et, "entry_id": eid, "error": "Entry missing in Gemini batch response"})
                        ex_chunk = None
                        break
                    except Exception as ex:
                        ex_chunk = ex
                        if attempt == 0 and _transient_io_error(ex):
                            time.sleep(0.5)
                            continue
                        break
                if ex_chunk is not None:
                    for entry in chunk:
                        failed.append({"entry_type": entry["entry_type"], "entry_id": entry["entry_id"], "error": str(ex_chunk)[:500]})
        
        else:
            # Original single-entry mode
            for i, entry in enumerate(to_process):
                if cancel_check and cancel_check():
                    if progress_cb:
                        progress_cb(
                            100,
                            "Cancelled by user (%d of %d labeled)."
                            % (labeled, total),
                            "Cancelled. Labels already written are kept.",
                        )
                    return _cancelled_result(
                        candidates=candidates,
                        to_process=to_process,
                        skipped_filter=skipped_filter,
                        labeled=labeled,
                        failed=failed,
                        skipped_mid=skipped_mid,
                    )
                et, eid = entry["entry_type"], int(entry["entry_id"])
                if progress_cb:
                    pct = 5 + int(90 * i / max(total, 1))
                    progress_cb(
                        min(pct, 94),
                        "Gemini %d/%d: %s #%s" % (i + 1, total, et, eid),
                        "Processing %s #%s: %s" % (et, eid, entry.get("title") or "No title")
                    )
                ok, _skip = _eligible_for_gemini(et, eid, profile_id, replace_gemini)
                if not ok:
                    skipped_mid += 1
                    continue
                last_ex: Optional[Exception] = None
                for attempt in (0, 1):
                    try:
                        label, reasoning = call_gemini_for_synthetic_label(
                            client,
                            system_instruction=system_instruction,
                            **_entry_fields_for_prompt(entry)
                        )
                        db.set_label(
                            et,
                            eid,
                            label,
                            reasoning=reasoning,
                            profile_id=profile_id,
                            label_source=SOURCE_GEMINI,
                        )
                        labeled += 1
                        if progress_cb:
                            progress_cb(
                                min(pct, 94),
                                "Gemini %d/%d: %s #%s" % (i + 1, total, et, eid),
                                "  -> Label: %s\n  -> Reasoning: %s" % (label, reasoning)
                            )
                        last_ex = None
                        break
                    except Exception as ex:
                        last_ex = ex
                        if attempt == 0 and _transient_io_error(ex):
                            time.sleep(0.5)
                            continue
                        break
                if last_ex is not None:
                    failed.append(
                        {
                            "entry_type": et,
                            "entry_id": eid,
                            "error": str(last_ex)[:500],
                        }
                    )

    if progress_cb:
        progress_cb(100, "Gemini batch finished.")

    return {
        "success": len(failed) == 0,
        "labeled": labeled,
        "skipped": skipped_filter + skipped_mid,
        "skipped_mid_run": skipped_mid,
        "failed": failed,
        "candidates_total": len(candidates),
        "queued": total,
    }
