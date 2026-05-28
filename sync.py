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
import httpx
from typing import List, Dict, Optional, Callable

from config import get_config, save_config
import db
from magnitu.accent_theme import parse_accent_from_magnitu_status
from magnitu.entry_sources import SEISMO_ENTRY_PULL_SPECS

logger = logging.getLogger(__name__)

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


def _request(method: str, params: dict,
             seismo_target: Optional[Dict] = None, **kwargs) -> httpx.Response:
    """Make a request to Seismo with auth.

    seismo_target: optional dict with keys 'seismo_url' and 'api_key'.
    When provided, overrides the global config.  Used by push operations
    so each profile can target its own Seismo instance.
    """
    cfg = get_config()
    if seismo_target:
        url = seismo_target.get("seismo_url") or cfg["seismo_url"]
        params["api_key"] = seismo_target.get("api_key") or cfg["api_key"]
    else:
        url = cfg["seismo_url"]
        params["api_key"] = cfg["api_key"]
    with httpx.Client(timeout=30.0) as client:
        resp = client.request(method, url, params=params, **kwargs)
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

def pull_entries(
    since: str = None,
    entry_type: str = "all",
    limit: int = 500,
    compute_embeddings: bool = True,
) -> int:
    """Fetch entries from mothership Seismo and store locally.

    Entries are shared across profiles (single local cache).
    """
    params = {"action": "magnitu_entries", "type": entry_type, "limit": str(limit)}
    if since:
        params["since"] = since
    data = _request("GET", params).json()
    entries = data.get("entries", [])
    if entries:
        db.upsert_entries(entries)
        db.log_sync("pull", len(entries), "type={}, since={}".format(entry_type, since))
    cfg = get_config()
    if compute_embeddings and cfg.get("model_architecture") == "transformer":
        _compute_pending_embeddings()
    return len(entries)


def pull_all_entry_types(
    since: str = None,
    compute_embeddings: bool = True,
) -> int:
    """Pull every Seismo entry type (feed, email, lex, leg calendar).

    Incremental sync uses this instead of a single ``type=all`` request so leg
    calendar events and large lex corpora are not truncated by ``limit``.
    """
    remote_entries = {}
    try:
        status = get_status()
        remote_entries = status.get("entries", {}) or {}
        pruning_days = status.get("entry_pruning_days") or status.get("pruning_days")
        if pruning_days is not None:
            cfg = get_config()
            cfg["seismo_pruning_days"] = int(pruning_days)
            save_config(cfg)
    except Exception as exc:
        logger.warning("Could not read magnitu_status for pull limits: %s", exc)

    total = 0
    for entry_type, status_key in SEISMO_ENTRY_PULL_SPECS:
        expected = int(remote_entries.get(status_key, 0) or 0)
        limit = max(500, expected + 100) if expected else 500
        total += pull_entries(
            since=since,
            entry_type=entry_type,
            limit=limit,
            compute_embeddings=False,
        )

    cfg = get_config()
    if compute_embeddings and cfg.get("model_architecture") == "transformer":
        _compute_pending_embeddings()
    return total


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


def pull_labels(profile_id: int = 1, profile: Optional[Dict] = None) -> int:
    """Pull labels from Seismo and merge into this profile.

    When ``profile`` is given, HTTP target is ``_profile_target(profile)``
    (satellite when URL and key are both set; mothership when both blank).
    When ``profile`` is omitted, uses global mothership only.

    Raises ``ValueError`` if ``profile`` has only one of URL / API key set.

    Conflict resolution: newer timestamp wins.
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
            if remote_time > local_time:
                db.set_label(entry_type, entry_id, label, reasoning=reasoning,
                             profile_id=profile_id)
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
    """Push a scoring recipe to the profile's Seismo target."""
    target = _profile_target(profile)
    result = _request("POST", {"action": "magnitu_recipe"}, json=recipe,
                      seismo_target=target).json()
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
