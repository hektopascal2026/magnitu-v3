"""
Sync engine: connects to Seismo's API to fetch entries and push scores/recipe.

Pull **entries** always uses global ``seismo_url`` / ``api_key`` (mothership).

Pull **labels** merges into a profile using that profile's push target
(``_profile_target``): the satellite when **both** ``seismo_url`` and ``api_key``
are set on the profile. If both are blank, label pull uses global mothership.
**Incomplete** credentials (only one of URL or key set) are rejected with
``ValueError`` so Magnitu never mixes a satellite URL with the mothership API
key (or the reverse).

Push (scores, recipe, labels to Seismo) uses the same rules via ``_profile_target``.
"""
import json
import logging
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

import httpx

from config import get_config, save_config, MODELS_DIR
import db
from magnitu.accent_theme import parse_accent_from_magnitu_status
from magnitu.entry_sources import SEISMO_ENTRY_PULL_SPECS

logger = logging.getLogger(__name__)

# Seismo MagnituExportRepository::MAX_LIMIT — per-family page size on magnitu_entries.
SEISMO_ENTRIES_PAGE_SIZE = 200

INCOMPLETE_SATELLITE_CREDENTIALS_MSG = (
    "Incomplete satellite credentials on this profile: set both Seismo URL and "
    "API key for this satellite, or clear both to use global mothership settings only."
)


def profile_satellite_incomplete(profile: Optional[Dict]) -> bool:
    """True when exactly one of seismo_url / api_key is set (invalid pair)."""
    if not profile:
        return False
    url = (profile.get("seismo_url") or "").strip()
    key = (profile.get("api_key") or "").strip()
    return bool(url) != bool(key)


# Default for pull/status/score-batch chatter. Recipe push can rescore a desk
# and routinely exceeds this under an ML window (NMT stopped, busy PHP-FPM).
REQUEST_TIMEOUT_SEC = 30.0
# Match nginx fastcgi_read_timeout on seismo.live (300s). Shorter values
# marked successful promotes as worker_error after distill (EU/seismo 2026-08).
# Seismo now acks POST magnitu_recipe before the first rescore batch
# (rescored_deferred); keep 300s as a backstop for slow desks / old deploys.
RECIPE_PUSH_TIMEOUT_SEC = 300.0
RECIPE_PUSH_MAX_ATTEMPTS = 3
RECIPE_PUSH_RETRY_SLEEP_SEC = 5.0


def _request(method: str, params: dict,
             seismo_target: Optional[Dict] = None,
             timeout: float = REQUEST_TIMEOUT_SEC,
             **kwargs) -> httpx.Response:
    """Make a request to Seismo with auth.

    seismo_target: optional dict with keys 'seismo_url' and 'api_key'.
    When provided, overrides the global config.  Used by push operations
    so each profile can target its own Seismo instance.

    When ``MAGNITU_ML_WORKER_TOKEN`` is set, sends ``X-Magnitu-Ml-Worker`` so
    Seismo's Magnitu ML writer lock accepts VPS worker Push (laptop fails closed).

    timeout: httpx client timeout in seconds (recipe push uses
    :data:`RECIPE_PUSH_TIMEOUT_SEC`).
    """
    cfg = get_config()
    if seismo_target:
        url = seismo_target.get("seismo_url") or cfg["seismo_url"]
        params["api_key"] = seismo_target.get("api_key") or cfg["api_key"]
    else:
        url = cfg["seismo_url"]
        params["api_key"] = cfg["api_key"]
    headers = dict(kwargs.pop("headers", None) or {})
    worker_token = (os.environ.get("MAGNITU_ML_WORKER_TOKEN") or "").strip()
    if worker_token:
        headers["X-Magnitu-Ml-Worker"] = worker_token
    with httpx.Client(timeout=float(timeout)) as client:
        resp = client.request(method, url, params=params, headers=headers or None, **kwargs)
        resp.raise_for_status()
        return resp


def _profile_target(profile: Optional[Dict]) -> Optional[Dict]:
    """Resolve HTTP target for label/score/recipe sync for this profile.

    Returns ``None`` to use global mothership (``get_config()`` URL + key) when
    both profile fields are blank.

    When both ``seismo_url`` and ``api_key`` are non-empty, returns exactly that
    pair (no mixing with global config).

    Raises ``ValueError`` with :data:`INCOMPLETE_SATELLITE_CREDENTIALS_MSG` if
    exactly one field is set.
    """
    if not profile:
        return None
    url = (profile.get("seismo_url") or "").strip()
    key = (profile.get("api_key") or "").strip()
    if profile_satellite_incomplete(profile):
        raise ValueError(INCOMPLETE_SATELLITE_CREDENTIALS_MSG)
    if not url and not key:
        return None
    return {"seismo_url": url, "api_key": key}


# ─── Pull (always mothership — global config) ────────────────────────────────

def _store_published_date(entry: Dict) -> str:
    """Local cache / watermark key: prefer Seismo ``export_since`` (SQL column).

    Shaped ``published_date`` may be ``event_meta.starts_at`` (or prose-parsed
    dates inside article bodies) which must not drive incremental ``since``.
    """
    return (entry.get("export_since") or entry.get("published_date") or "").strip()


def _normalize_entries_for_store(entries: List[Dict]) -> List[Dict]:
    """Copy entries with ``published_date`` rewritten to the SQL since key."""
    out: List[Dict] = []
    for entry in entries:
        row = dict(entry)
        pub = _store_published_date(row)
        if pub:
            row["published_date"] = pub
        for key in (
            "title",
            "description",
            "content",
            "link",
            "author",
            "source_name",
            "source_category",
        ):
            row.setdefault(key, "")
        row.setdefault("source_type", "rss")
        row.setdefault("published_date", "")
        out.append(row)
    return out


def _max_published_date(entries: List[Dict]) -> Optional[str]:
    """Latest sync cursor key in a shaped Seismo entry batch (UTC strings)."""
    best = ""
    for entry in entries:
        raw = _store_published_date(entry)
        if raw and raw > best:
            best = raw
    return best or None


def _entry_sync_hints(
    data: dict,
    entry_type: str,
    page_size: int,
    entries: List[Dict],
) -> dict:
    """Per-family pagination hints from Seismo ``sync.by_type`` (0.8+), with fallback."""
    sync = data.get("sync") or {}
    by_type = sync.get("by_type") or {}
    hints = by_type.get(entry_type)
    if isinstance(hints, dict) and hints:
        return hints

    limit = int(sync.get("limit_per_family") or page_size)
    count = len(entries)
    drain_complete = count < limit
    recommended = None
    if not drain_complete:
        recommended = _max_published_date(entries)
    return {
        "drain_complete": drain_complete,
        "recommended_next_since": recommended,
    }


def _pull_entry_type_drain(
    entry_type: str,
    since: Optional[str] = None,
    page_size: int = SEISMO_ENTRIES_PAGE_SIZE,
) -> int:
    """Drain one family with ``order=asc`` until Seismo reports ``drain_complete``.

    Avoids skipping rows when more than ``page_size`` entries share a ``since``
    window (Seismo returns oldest-first; cursor advances via ``recommended_next_since``
    and optional ``recommended_after_id`` for dense same-timestamp pages).
    """
    page_size = max(1, min(int(page_size), SEISMO_ENTRIES_PAGE_SIZE))
    cursor = since
    after_id: Optional[int] = None
    total = 0
    pages = 0
    max_pages = 5000

    while pages < max_pages:
        params = {
            "action": "magnitu_entries",
            "type": entry_type,
            "limit": str(page_size),
            "order": "asc",
        }
        if cursor:
            params["since"] = cursor
        if after_id:
            params["after_id"] = str(after_id)

        data = _request("GET", params).json()
        entries = data.get("entries", [])
        if entries:
            db.upsert_entries(_normalize_entries_for_store(entries))
            total += len(entries)

        hints = _entry_sync_hints(data, entry_type, page_size, entries)
        if hints.get("drain_complete", True):
            break

        next_since = hints.get("recommended_next_since")
        raw_after = hints.get("recommended_after_id")
        try:
            next_after = int(raw_after) if raw_after is not None else None
        except (TypeError, ValueError):
            next_after = None
        if next_after is not None and next_after <= 0:
            next_after = None

        if not next_since:
            logger.warning(
                "Entry drain for %s stopped after page %d: no recommended_next_since",
                entry_type,
                pages + 1,
            )
            break
        if cursor == next_since and next_after == after_id:
            logger.warning(
                "Entry drain for %s stopped: cursor stuck at %s after_id=%s page %d",
                entry_type,
                cursor,
                after_id,
                pages + 1,
            )
            break

        cursor = next_since
        after_id = next_after
        pages += 1

    if total:
        db.log_sync(
            "pull",
            total,
            "type={}, drain_pages={}, since_start={}".format(
                entry_type, pages + 1, since or ""
            ),
        )
    return total


def pull_entries(
    since: str = None,
    entry_type: str = "all",
    limit: int = 500,
    compute_embeddings: bool = True,
    drain: bool = False,
) -> int:
    """Fetch entries from mothership Seismo and store locally.

    Entries are shared across profiles (single local cache).

    When ``drain`` is true or ``since`` is set, pulls with ``order=asc`` in pages
    until ``sync.by_type.<type>.drain_complete`` (safe backfill / incremental).

    Otherwise fetches a single newest-first page (quick refresh, max 200 rows).
    """
    if entry_type == "all":
        raise ValueError(
            "pull_entries(entry_type='all') is unsupported; use pull_all_entry_types()"
        )

    page_size = max(1, min(int(limit), SEISMO_ENTRIES_PAGE_SIZE))

    if drain or since:
        total = _pull_entry_type_drain(entry_type, since=since, page_size=page_size)
    else:
        params = {
            "action": "magnitu_entries",
            "type": entry_type,
            "limit": str(page_size),
        }
        data = _request("GET", params).json()
        entries = data.get("entries", [])
        if entries:
            db.upsert_entries(_normalize_entries_for_store(entries))
            db.log_sync("pull", len(entries), "type={}, since=".format(entry_type))
        total = len(entries)

    cfg = get_config()
    if compute_embeddings and cfg.get("model_architecture") == "transformer":
        _compute_pending_embeddings()
    return total


def pull_all_entry_types(
    since: str = None,
    compute_embeddings: bool = True,
    drain: bool = None,
    per_type_since: Optional[Dict[str, Optional[str]]] = None,
) -> int:
    """Pull every Seismo entry type (feed, email, lex, leg calendar).

    One family per HTTP sequence (never ``type=all``) so each corpus is drained
    with ``order=asc`` when ``since`` is set or ``drain=True``. Without ``since``,
    each family gets one newest-first page (200 rows) for a lightweight refresh.

    ``per_type_since`` overrides ``since`` per entry_type (incremental watermarks).
    """
    if drain is None:
        drain = bool(since) or bool(per_type_since)

    try:
        status = get_status()
        pruning_days = status.get("entry_pruning_days") or status.get("pruning_days")
        if pruning_days is not None:
            cfg = get_config()
            cfg["seismo_pruning_days"] = int(pruning_days)
            save_config(cfg)
    except Exception as exc:
        logger.warning("Could not read magnitu_status for pull: %s", exc)

    total = 0
    for entry_type, _status_key in SEISMO_ENTRY_PULL_SPECS:
        type_since = since
        if per_type_since is not None:
            type_since = per_type_since.get(entry_type)
        total += pull_entries(
            since=type_since,
            entry_type=entry_type,
            limit=SEISMO_ENTRIES_PAGE_SIZE,
            compute_embeddings=False,
            drain=drain,
        )

    cfg = get_config()
    if compute_embeddings and cfg.get("model_architecture") == "transformer":
        _compute_pending_embeddings()
    return total


def heal_future_published_dates(now: Optional[str] = None) -> int:
    """Refresh local rows with ``published_date`` > now from mothership ``ids=``.

    Bad RSS years (or titles containing a future year) can land once, then get
    corrected on Seismo while Magnitu keeps the stale future date. That freezes
    ``MAX(published_date)`` above ``now`` and only ever shows up as a watermark
    cap warning — incremental ``since`` never re-pulls those ids. Leg
    (``calendar_event``) is skipped: its cursor is not ``published_date``.
    """
    if now is None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = db.get_db()
    rows = conn.execute(
        """
        SELECT entry_type, entry_id FROM entries
         WHERE published_date IS NOT NULL
           AND TRIM(published_date) != ''
           AND published_date > ?
           AND entry_type != 'calendar_event'
        """,
        (now,),
    ).fetchall()
    conn.close()
    if not rows:
        return 0

    by_type: Dict[str, List[int]] = {}
    for r in rows:
        by_type.setdefault(str(r["entry_type"]), []).append(int(r["entry_id"]))

    fetched = 0
    for entry_type, id_list in by_type.items():
        try:
            n = pull_entries_by_ids(entry_type, id_list)
            fetched += n
            logger.info(
                "Healed %d/%d future-dated %s row(s) from mothership",
                n,
                len(id_list),
                entry_type,
            )
        except Exception as exc:
            logger.warning(
                "Future-date heal failed for %s (%d ids): %s",
                entry_type,
                len(id_list),
                exc,
            )
    return fetched


def entry_store_watermarks() -> Dict[str, Optional[str]]:
    """Incremental ``since`` per entry_type from the local SQLite cache.

    Uses ``MAX(published_date)`` among rows with ``published_date <= UTC now``
    so future-dated rows (agenda/EP dates or bad feed timestamps) cannot freeze
    the drain. Before computing, refreshes any local future-dated non-Leg rows
    from mothership so corrected Seismo dates replace stale cache values.
    ``calendar_event`` is always ``None`` (full family drain): Seismo pages Leg
    by ``fetched_at``, which the local store overwrites on upsert, and
    ``event_date`` is the wrong cursor.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    heal_future_published_dates(now)
    conn = db.get_db()
    out: Dict[str, Optional[str]] = {}
    for entry_type, _status_key in SEISMO_ENTRY_PULL_SPECS:
        if entry_type == "calendar_event":
            out[entry_type] = None
            continue
        row = conn.execute(
            """
            SELECT MAX(published_date) AS m FROM entries
             WHERE entry_type = ?
               AND published_date IS NOT NULL
               AND TRIM(published_date) != ''
               AND published_date <= ?
            """,
            (entry_type, now),
        ).fetchone()
        out[entry_type] = row["m"] if row and row["m"] else None
    conn.close()
    return out


def pull_entries_by_ids(entry_type: str, ids: List[int]) -> int:
    """Fetch exact mothership rows by id (Seismo ``magnitu_entries&ids=``)."""
    ids = sorted({int(i) for i in ids if int(i) > 0})
    if not ids:
        return 0
    # Seismo MagnituExportRepository::MAX_IDS_PER_REQUEST
    chunk_size = 100
    total = 0
    missing_total = 0
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        params = {
            "action": "magnitu_entries",
            "type": entry_type,
            "ids": ",".join(str(x) for x in chunk),
        }
        data = _request("GET", params).json()
        entries = data.get("entries") or []
        missing = data.get("missing_ids")
        if isinstance(missing, list) and missing:
            missing_total += len(missing)
            logger.warning(
                "ids= backfill %s: %d missing of %d requested (sample=%s)",
                entry_type,
                len(missing),
                len(chunk),
                missing[:8],
            )
        elif len(entries) < len(chunk):
            missing_total += len(chunk) - len(entries)
            logger.warning(
                "ids= backfill %s: returned %d of %d (no missing_ids field)",
                entry_type,
                len(entries),
                len(chunk),
            )
        if entries:
            db.upsert_entries(_normalize_entries_for_store(entries))
            total += len(entries)
    if total or missing_total:
        db.log_sync(
            "pull",
            total,
            "type={}, ids_backfill={}, missing={}".format(
                entry_type, total, missing_total
            ),
        )
    return total


def backfill_orphan_label_entries(profile_id: int) -> Tuple[int, int]:
    """Pull mothership entries for labels missing from the local entry store.

    Returns ``(orphan_before, fetched)``.
    """
    conn = db.get_db()
    rows = conn.execute(
        """
        SELECT l.entry_type, l.entry_id
          FROM labels l
          LEFT JOIN entries e
            ON e.entry_type = l.entry_type AND e.entry_id = l.entry_id
         WHERE l.profile_id = ?
           AND e.rowid IS NULL
        """,
        (profile_id,),
    ).fetchall()
    conn.close()
    orphan_before = len(rows)
    if orphan_before == 0:
        return 0, 0

    by_type: Dict[str, List[int]] = {}
    for r in rows:
        by_type.setdefault(str(r["entry_type"]), []).append(int(r["entry_id"]))

    fetched = 0
    for entry_type, id_list in by_type.items():
        try:
            fetched += pull_entries_by_ids(entry_type, id_list)
        except Exception as exc:
            logger.warning(
                "Orphan ids backfill failed for %s (%d ids): %s",
                entry_type,
                len(id_list),
                exc,
            )
    return orphan_before, fetched


def post_ml_window_report(report: Dict) -> None:
    """POST window summary to mothership ``magnitu_ml_window_report``."""
    cfg = get_config()
    url = (cfg.get("seismo_url") or "").strip()
    if not url:
        raise ValueError("seismo_url missing for window report")
    worker_token = (os.environ.get("MAGNITU_ML_WORKER_TOKEN") or "").strip()
    if not worker_token:
        raise ValueError("MAGNITU_ML_WORKER_TOKEN missing for window report")
    headers = {
        "Content-Type": "application/json",
        "X-Magnitu-Ml-Worker": worker_token,
    }
    params = {"action": "magnitu_ml_window_report"}
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            url,
            params=params,
            headers=headers,
            content=json.dumps(report),
        )
        resp.raise_for_status()


def _compute_pending_embeddings(progress_cb=None) -> int:
    """
    Compute and store embeddings for entries that lack them (up to 1000 per call).

    Optional ``progress_cb(done, total, message)`` for UI updates during long runs.
    Returns the number of entries embedded (0 on failure or none pending).
    """
    unembedded = db.get_entries_without_embeddings(limit=1000)
    total = len(unembedded)
    if not total:
        return 0
    logger.info("Computing embeddings for %d entries...", total)
    try:
        from pipeline import embed_entries, release_embedder

        if progress_cb:
            progress_cb(
                0, total,
                "Loading E5 model (first run may download ~1 GB; can take several minutes)",
            )

        def on_batch(done, batch_total):
            if progress_cb:
                progress_cb(done, batch_total, "Encoding entries")

        emb_bytes_list = embed_entries(unembedded, progress_cb=on_batch)
        updates = []
        for entry, emb_bytes in zip(unembedded, emb_bytes_list):
            updates.append((emb_bytes, entry["entry_type"], entry["entry_id"]))
        db.store_embeddings_batch(updates)
        logger.info("Stored %d embeddings.", len(updates))
        release_embedder()
        return len(updates)
    except Exception as e:
        logger.exception("Failed to compute embeddings: %s", e)
        return 0


def profile_satellite_blank(profile: Optional[Dict]) -> bool:
    """True when the profile has no satellite URL and no API key of its own."""
    if not profile:
        return False
    return not (profile.get("seismo_url") or "").strip() and not (
        profile.get("api_key") or ""
    ).strip()


def _normalize_label_ts(value: str) -> str:
    """Canonicalize 'YYYY-MM-DD HH:MM:SS' vs ISO-T variants for string compare."""
    s = (value or "").strip().replace("T", " ")
    if "+" in s:
        s = s.split("+", 1)[0]
    return s.rstrip("Z").strip()


def pull_labels(profile_id: int = 1, profile: Optional[Dict] = None) -> int:
    """Pull labels from Seismo and merge into this profile.

    When ``profile`` is given, HTTP target is ``_profile_target(profile)``
    (satellite when URL and key are both set; mothership when both blank).
    When ``profile`` is omitted, uses global mothership only.

    Raises ``ValueError`` if ``profile`` has only one of URL / API key set.

    Conflict resolution: newer normalized timestamp wins; equal timestamps keep local.
    Returns count of labels imported or updated.
    """
    target = None
    if profile is not None:
        target = _profile_target(profile)
    data = _request(
        "GET", {"action": "magnitu_labels"}, seismo_target=target
    ).json()
    labels = data.get("labels", [])
    imported = 0
    updated = 0
    conflicts = 0

    for lbl in labels:
        entry_type  = lbl.get("entry_type", "")
        entry_id    = int(lbl.get("entry_id", 0))
        label       = lbl.get("label", "")
        reasoning   = lbl.get("reasoning", "")
        remote_time = lbl.get("labeled_at", "")
        if not entry_type or not entry_id or not label:
            continue

        existing = db.get_label_with_reasoning(entry_type, entry_id, profile_id)
        if existing is None:
            db.set_label(entry_type, entry_id, label, reasoning=reasoning,
                         profile_id=profile_id)
            imported += 1
        else:
            conn = db.get_db()
            row = conn.execute(
                "SELECT updated_at FROM labels WHERE profile_id=? AND entry_type=? AND entry_id=?",
                (profile_id, entry_type, entry_id)
            ).fetchone()
            conn.close()
            local_time = (row["updated_at"] or "") if row else ""
            if existing["label"] != label:
                conflicts += 1
            if _normalize_label_ts(remote_time) > _normalize_label_ts(local_time):
                db.set_label(
                    entry_type, entry_id, label, reasoning=reasoning,
                    profile_id=profile_id,
                    label_source=existing.get("label_source", ""),
                )
                updated += 1

    total = imported + updated
    if total:
        details = "labels pulled: {} new, {} updated".format(imported, updated)
        if conflicts:
            details += ", {} conflicts (resolved by timestamp)".format(conflicts)
        db.log_sync("pull", total, details, profile_id=profile_id)
    return total


# ─── Push (per-profile — each profile targets its own Seismo) ─────────────

def _seismo_push_batch_size() -> int:
    """Max items per POST (upper cap; body-size logic may lower further)."""
    size = int(get_config().get("seismo_push_batch_size") or 75)
    return max(1, size)


def _seismo_push_max_body_bytes() -> int:
    """Stay under typical nginx client_max_body_size (often 1m)."""
    size = int(get_config().get("seismo_push_max_body_bytes") or 524288)
    return max(8192, size)


def _chunk_list(items: List, size: int) -> List[List]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _estimate_batch_count(items: List[Dict], sample_payload: dict, max_items: int) -> int:
    """Pick a batch size from a sample JSON body so posts stay under the byte cap."""
    if not items:
        return max_items
    sample_n = min(len(items), 5)
    sample_payload = dict(sample_payload)
    list_key = next(k for k in sample_payload if isinstance(sample_payload.get(k), list))
    sample_payload[list_key] = items[:sample_n]
    body_len = len(json.dumps(sample_payload, separators=(",", ":")))
    per_item = max(body_len / sample_n, 200)
    by_bytes = int((_seismo_push_max_body_bytes() - 128) / per_item)
    return max(1, min(max_items, by_bytes))


def _post_json_batched_with_size(
    action: str,
    items: List[Dict],
    build_payload: Callable[[List[Dict]], dict],
    seismo_target: Optional[Dict],
    batch_size: int,
) -> tuple:
    """POST items in batches; halve batch size and retry on HTTP 413."""
    if not items:
        return {"success": True, "pushed": 0}, 0

    i = 0
    batch_count = 0
    last_result = {}

    while i < len(items):
        chunk = items[i:i + batch_size]
        try:
            last_result = _request(
                "POST", {"action": action},
                json=build_payload(chunk),
                seismo_target=seismo_target,
            ).json()
            i += len(chunk)
            batch_count += 1
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 413 and batch_size > 5:
                batch_size = max(5, batch_size // 2)
                logger.warning(
                    "Seismo 413 on %s: retrying from offset %d with batch_size=%d",
                    action, i, batch_size,
                )
                continue
            raise

    if batch_count > 1 and isinstance(last_result, dict):
        last_result = dict(last_result)
        last_result["batches"] = batch_count
        last_result["items_pushed"] = len(items)
    return last_result, batch_count


def _post_json_batched(
    action: str,
    items: List[Dict],
    build_payload: Callable[[List[Dict]], dict],
    seismo_target: Optional[Dict],
) -> tuple:
    sample = build_payload(items[: min(1, len(items))])
    batch_size = _estimate_batch_count(items, sample, _seismo_push_batch_size())
    return _post_json_batched_with_size(
        action, items, build_payload, seismo_target, batch_size
    )


def push_scores(scores: List[Dict], model_version: int,
                model_meta: Optional[Dict] = None,
                profile: Optional[Dict] = None) -> dict:
    """Push scores to the profile's Seismo target in HTTP-sized chunks.

    profile: profiles table row (has seismo_url, api_key). Uses mothership when
    both are blank; raises ``ValueError`` if only one is set.
    """
    if not scores:
        return {"success": True, "pushed": 0}

    target = _profile_target(profile)
    profile_id = profile["id"] if profile else None
    meta_sent = {"done": False}

    def build_payload(chunk: List[Dict]) -> dict:
        payload = {"scores": chunk, "model_version": model_version}
        if model_meta and not meta_sent["done"]:
            payload["model_meta"] = model_meta
            meta_sent["done"] = True
        return payload

    sample = {"scores": scores[: min(5, len(scores))], "model_version": model_version}
    if model_meta:
        sample["model_meta"] = model_meta

    batch_size = _estimate_batch_count(
        scores, sample, _seismo_push_batch_size()
    )
    last_result, batch_count = _post_json_batched_with_size(
        "magnitu_scores", scores, build_payload, target, batch_size
    )

    db.log_sync(
        "push", len(scores),
        "scores pushed, model v{}, {} batch(es)".format(model_version, batch_count),
        profile_id=profile_id,
    )
    if isinstance(last_result, dict):
        last_result = dict(last_result)
        last_result["scores_pushed"] = len(scores)
    return last_result


def push_recipe(recipe: dict, profile: Optional[Dict] = None) -> dict:
    """Push a scoring recipe to the profile's Seismo target.

    Retries on httpx read/connect timeouts — mothership rescore used to hold
    the HTTP response until the first BATCH_LIMIT pass finished and could
    exceed RECIPE_PUSH_TIMEOUT_SEC (seen 2026-08 on seismo.live).
    """
    target = _profile_target(profile)
    last_exc: Optional[BaseException] = None
    result = None
    for attempt in range(1, RECIPE_PUSH_MAX_ATTEMPTS + 1):
        try:
            result = _request(
                "POST",
                {"action": "magnitu_recipe"},
                json=recipe,
                seismo_target=target,
                timeout=RECIPE_PUSH_TIMEOUT_SEC,
            ).json()
            last_exc = None
            break
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_exc = e
            if attempt >= RECIPE_PUSH_MAX_ATTEMPTS:
                break
            logger.warning(
                "Recipe push attempt %s/%s timed out (%s); retrying in %ss",
                attempt,
                RECIPE_PUSH_MAX_ATTEMPTS,
                e,
                RECIPE_PUSH_RETRY_SLEEP_SEC,
            )
            time.sleep(RECIPE_PUSH_RETRY_SLEEP_SEC)
    if last_exc is not None:
        raise last_exc
    if not isinstance(result, dict):
        raise RuntimeError("magnitu_recipe push returned non-object JSON")
    profile_id = profile["id"] if profile else None
    db.log_sync("push", 1, "recipe v{} pushed".format(recipe.get("version", "?")),
                profile_id=profile_id)
    return result


def push_labels(profile_id: int = 1, profile: Optional[Dict] = None) -> dict:
    """Push this profile's labels to its Seismo target.

    Only pushes labels updated since the last successful label push for this profile.
    """
    target = _profile_target(profile)
    conn = db.get_db()
    row = conn.execute("""
        SELECT synced_at FROM sync_log
        WHERE direction='push' AND details LIKE '%labels pushed%'
              AND (profile_id=? OR profile_id IS NULL)
        ORDER BY synced_at DESC LIMIT 1
    """, (profile_id,)).fetchone()
    conn.close()
    last_push_time = row["synced_at"] if row else ""

    all_labels = db.get_all_labels_raw(profile_id)
    if last_push_time:
        labels_to_push = [
            lbl for lbl in all_labels
            if (lbl.get("updated_at") or "") > last_push_time
        ]
    else:
        labels_to_push = all_labels

    # Filter out labels outside Seismo's active pruning window to prevent ghost/orphaned entries
    cfg = get_config()
    pruning_days = cfg.get("seismo_pruning_days")
    if pruning_days and pruning_days > 0 and labels_to_push:
        try:
            conn = db.get_db()
            rows = conn.execute("""
                SELECT entry_type, entry_id FROM entries
                WHERE published_date >= date('now', ?)
            """, (f"-{pruning_days} days",)).fetchall()
            conn.close()
            valid_keys = {(r["entry_type"], r["entry_id"]) for r in rows}
            labels_to_push = [
                lbl for lbl in labels_to_push
                if (lbl["entry_type"], lbl["entry_id"]) in valid_keys
            ]
        except Exception as e:
            logger.warning("Failed to filter labels using pruning window: %s", e)

    if not labels_to_push:
        return {"success": True, "pushed": 0}

    label_rows = [
        {
            "entry_type": lbl["entry_type"],
            "entry_id":   lbl["entry_id"],
            "label":      lbl["label"],
            "reasoning":  lbl.get("reasoning", ""),
            "labeled_at": lbl.get("updated_at") or lbl.get("created_at", ""),
        }
        for lbl in labels_to_push
    ]
    def build_payload(chunk: List[Dict]) -> dict:
        return {"labels": chunk}

    result, batch_count = _post_json_batched(
        "magnitu_labels", label_rows, build_payload, target
    )
    if isinstance(result, dict):
        result = dict(result)
        result["pushed"] = len(labels_to_push)
    db.log_sync(
        "push", len(labels_to_push),
        "labels pushed, {} batch(es)".format(batch_count),
        profile_id=profile_id,
    )
    return result


def get_status(seismo_target: Optional[Dict] = None) -> dict:
    """Check Seismo connectivity and status."""
    return _request("GET", {"action": "magnitu_status"},
                    seismo_target=seismo_target).json()


def verify_seismo_endpoints(seismo_target: Optional[Dict] = None) -> tuple:
    """Smoke-test the label push endpoint."""
    try:
        _request("POST", {"action": "magnitu_labels"}, json={"labels": []},
                 seismo_target=seismo_target)
        return True, "Label endpoint OK"
    except Exception as e:
        return False, "Label push endpoint broken: {}".format(e)


def _magnitu_status_reports_ok(status: dict) -> bool:
    """Accept common variants so Test satellite stores accent after a real OK."""
    if not isinstance(status, dict):
        return False
    st = status.get("status")
    if isinstance(st, str) and st.strip().lower() == "ok":
        return True
    if status.get("success") is True:
        return True
    inner = status.get("data")
    if isinstance(inner, dict):
        if inner.get("success") is True:
            return True
        st2 = inner.get("status")
        if isinstance(st2, str) and st2.strip().lower() == "ok":
            return True
    return False


def test_connection(seismo_target: Optional[Dict] = None) -> tuple:
    """Test connection to a Seismo target.

    Returns (success, message, status_dict). On success, ``status_dict`` is
    the parsed magnitu_status JSON (may include optional ``accent_color``).
    On failure, ``status_dict`` is ``{}``.
    """
    try:
        status = get_status(seismo_target)
        if _magnitu_status_reports_ok(status):
            entries = status.get("entries") if isinstance(status.get("entries"), dict) else {}
            total = entries.get("total", 0)
            pruning_days = status.get("entry_pruning_days") or status.get("pruning_days")
            if pruning_days is not None:
                try:
                    cfg = get_config()
                    cfg["seismo_pruning_days"] = int(pruning_days)
                    save_config(cfg)
                except Exception:
                    pass
            return (
                True,
                "Connected. Seismo has {} entries.".format(total),
                status if isinstance(status, dict) else {},
            )
        return False, "Unexpected response: {}".format(status), {}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return False, "Authentication failed. Check your API key.", {}
        return (
            False,
            "HTTP error {}: {}".format(e.response.status_code, e.response.text),
            {},
        )
    except httpx.ConnectError:
        return False, "Connection failed. Check the Seismo URL.", {}
    except Exception as e:
        return False, "Error: {}".format(str(e)), {}


def refresh_profile_accent(profile: Optional[Dict]) -> None:
    """Fetch ``magnitu_status`` from the profile push target and persist accent_color.

    Uses ``_profile_target`` (satellite URL/key) when the profile has its own
    satellite. Mothership-only profiles (URL and key both blank) keep Magnitu's
    default red — we clear any previously stored satellite accent and do not
    copy accent from global mothership Seismo. Called after Push — never raises.
    """
    if not profile:
        return
    profile_id = int(profile["id"])
    if profile_satellite_blank(profile):
        try:
            db.clear_profile_accent_color(profile_id)
        except Exception as ex:
            logger.warning("Accent clear (mothership profile) failed: %s", ex)
        return
    try:
        target = _profile_target(profile)
    except ValueError as ex:
        logger.warning("Accent refresh skipped: %s", ex)
        return
    try:
        status = get_status(seismo_target=target)
        maybe_profile_accent_from_status(status, profile_id)
    except Exception as ex:
        logger.warning("Accent refresh skipped: %s", ex)


def maybe_profile_accent_from_status(status: dict, profile_id: int) -> None:
    """If magnitu_status includes a valid accent_color, store it for the profile.

    Only used for profiles with a satellite URL+key (e.g. after Test satellite or
    push); mothership-only profiles use default Magnitu red and clear stored
    accent via :func:`refresh_profile_accent`.

    Never raises; ignores missing/invalid fields (backward compatible).
    """
    try:
        hex_color = parse_accent_from_magnitu_status(status)
        if hex_color:
            db.set_profile_accent_color(profile_id, hex_color)
    except Exception as ex:
        logger.warning("Accent from magnitu_status ignored: %s", ex)


# ─── Model vault (mothership; vault password, not desk api_key) ───────────────

def _vault_url_and_headers(vault_password: str) -> Tuple[str, Dict[str, str]]:
    """Mothership URL + vault password header (never uses profile satellite)."""
    cfg = get_config()
    url = (cfg.get("seismo_url") or "").strip()
    if not url:
        raise ValueError("seismo_url is not configured.")
    password = (vault_password or "").strip()
    if not password:
        raise ValueError("Vault password is required.")
    return url, {"X-Magnitu-Vault-Password": password}


def vault_list(vault_password: str) -> dict:
    """List ``.magnitu`` packages on the Seismo mothership model vault."""
    url, headers = _vault_url_and_headers(vault_password)
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(
            url,
            params={"action": "magnitu_vault_list"},
            headers=headers,
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                err = resp.json()
                if isinstance(err, dict) and err.get("error"):
                    detail = err["error"]
            except Exception:
                pass
            raise ValueError(detail)
        data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected vault list response.")
    return data


def vault_download(
    vault_password: str,
    model_uuid: str,
    dest_path: Optional[str] = None,
) -> str:
    """Download a package by ``model_uuid`` into MODELS_DIR/inbox (or dest_path).

    Returns the local file path. Verifies ``X-Magnitu-Sha256`` when present.
    """
    uuid = (model_uuid or "").strip().replace("-", "").lower()
    if len(uuid) != 32 or any(c not in "0123456789abcdef" for c in uuid):
        raise ValueError("model_uuid must be a 32-char hex id.")

    url, headers = _vault_url_and_headers(vault_password)
    if dest_path:
        out = Path(dest_path)
    else:
        inbox = MODELS_DIR / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        out = inbox / "{}.magnitu".format(uuid)

    with httpx.Client(timeout=600.0) as client:
        with client.stream(
            "GET",
            url,
            params={"action": "magnitu_vault_download", "model_uuid": uuid},
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            expected_sha = (resp.headers.get("X-Magnitu-Sha256") or "").strip().lower()
            hasher = hashlib.sha256()
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes():
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
            got = hasher.hexdigest()
            if expected_sha and got != expected_sha:
                tmp.unlink(missing_ok=True)
                raise ValueError(
                    "Downloaded package sha256 mismatch (expected {}, got {}).".format(
                        expected_sha, got
                    )
                )
            tmp.replace(out)

    db.log_sync("pull", 1, "vault package {} downloaded".format(uuid[:8]))
    return str(out)


def vault_upload(
    vault_password: str,
    package_path: str,
    overwrite: bool = False,
) -> dict:
    """Upload a local ``.magnitu`` file to the mothership vault."""
    path = Path(package_path)
    if not path.is_file():
        raise FileNotFoundError("Package not found: {}".format(package_path))
    if not path.name.lower().endswith(".magnitu"):
        raise ValueError("File must end with .magnitu")

    url, headers = _vault_url_and_headers(vault_password)
    data = {"overwrite": "1" if overwrite else "0"}
    with httpx.Client(timeout=600.0) as client:
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/zip")}
            resp = client.post(
                url,
                params={"action": "magnitu_vault_upload"},
                headers=headers,
                data=data,
                files=files,
            )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                err = resp.json()
                if isinstance(err, dict) and err.get("error"):
                    detail = err["error"]
            except Exception:
                pass
            raise httpx.HTTPStatusError(
                "Vault upload failed: {}".format(detail),
                request=resp.request,
                response=resp,
            )
        result = resp.json()
    if not isinstance(result, dict):
        raise ValueError("Unexpected vault upload response.")
    db.log_sync(
        "push", 1,
        "vault package uploaded ({})".format(
            (result.get("package") or {}).get("model_uuid", "?")[:8]
        ),
    )
    return result
