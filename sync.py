"""
Sync engine: connects to Seismo's API to fetch entries and push scores/recipe.

Pull (entries, labels) always uses the global config (mothership Seismo).
Push (scores, recipe, labels) uses the profile's own seismo_url / api_key so
different profiles can target different lightweight Seismo instances.
"""
import logging
import httpx
from typing import List, Dict, Optional

from config import get_config
import db
from magnitu.accent_theme import parse_accent_from_magnitu_status

logger = logging.getLogger(__name__)


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
    """Extract seismo_url/api_key from a profile dict, falling back to global."""
    if not profile:
        return None
    url = (profile.get("seismo_url") or "").strip()
    key = (profile.get("api_key") or "").strip()
    if not url and not key:
        return None
    cfg = get_config()
    return {
        "seismo_url": url or cfg["seismo_url"],
        "api_key": key or cfg["api_key"],
    }


# ─── Pull (global — always from mothership) ──────────────────────────────────

def pull_entries(
    since: str = None,
    entry_type: str = "all",
    limit: int = 500,
    compute_embeddings: bool = True,
) -> int:
    """Fetch entries from mothership Seismo and store locally.
    Shared across all profiles — entries are not profile-scoped.
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


def _compute_pending_embeddings():
    """Compute and store embeddings for all entries that lack them."""
    unembedded = db.get_entries_without_embeddings(limit=1000)
    if not unembedded:
        return
    logger.info("Computing embeddings for %d entries...", len(unembedded))
    try:
        from pipeline import embed_entries, release_embedder
        emb_bytes_list = embed_entries(unembedded)
        updates = []
        for entry, emb_bytes in zip(unembedded, emb_bytes_list):
            updates.append((emb_bytes, entry["entry_type"], entry["entry_id"]))
        db.store_embeddings_batch(updates)
        logger.info("Stored %d embeddings.", len(updates))
        release_embedder()
    except Exception as e:
        logger.warning("Failed to compute embeddings: %s", e)


def pull_labels(profile_id: int = 1) -> int:
    """Pull labels from the mothership and merge into this profile.

    Conflict resolution: newer timestamp wins.
    Returns count of labels imported or updated.
    """
    data = _request("GET", {"action": "magnitu_labels"}).json()
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

def push_scores(scores: List[Dict], model_version: int,
                model_meta: Optional[Dict] = None,
                profile: Optional[Dict] = None) -> dict:
    """Push batch of scores to the profile's Seismo target.

    profile: profiles table row (has seismo_url, api_key).
    Falls back to global config when profile has no target configured.
    """
    target = _profile_target(profile)
    payload = {"scores": scores, "model_version": model_version}
    if model_meta:
        payload["model_meta"] = model_meta
    result = _request("POST", {"action": "magnitu_scores"}, json=payload,
                      seismo_target=target).json()
    profile_id = profile["id"] if profile else None
    db.log_sync("push", len(scores),
                "scores pushed, model v{}".format(model_version),
                profile_id=profile_id)
    return result


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

    if not labels_to_push:
        return {"success": True, "pushed": 0}

    payload = {
        "labels": [
            {
                "entry_type": lbl["entry_type"],
                "entry_id":   lbl["entry_id"],
                "label":      lbl["label"],
                "reasoning":  lbl.get("reasoning", ""),
                "labeled_at": lbl.get("updated_at") or lbl.get("created_at", ""),
            }
            for lbl in labels_to_push
        ]
    }
    result = _request("POST", {"action": "magnitu_labels"}, json=payload,
                      seismo_target=target).json()
    db.log_sync("push", len(labels_to_push), "labels pushed", profile_id=profile_id)
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


def test_connection(seismo_target: Optional[Dict] = None) -> tuple:
    """Test connection to a Seismo target.

    Returns (success, message, status_dict). On success, ``status_dict`` is
    the parsed magnitu_status JSON (may include optional ``accent_color``).
    On failure, ``status_dict`` is ``{}``.
    """
    try:
        status = get_status(seismo_target)
        if status.get("status") == "ok":
            total = status.get("entries", {}).get("total", 0)
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


def maybe_profile_accent_from_status(status: dict, profile_id: int) -> None:
    """If magnitu_status includes a valid accent_color, store it for the profile.

    Never raises; ignores missing/invalid fields (backward compatible).
    """
    try:
        hex_color = parse_accent_from_magnitu_status(status)
        if hex_color:
            db.set_profile_accent_color(profile_id, hex_color)
    except Exception as ex:
        logger.warning("Accent from magnitu_status ignored: %s", ex)
