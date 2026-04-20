"""
Magnitu 2 — ML-powered relevance scoring for Seismo.
FastAPI application: serves the labeling UI, dashboard, and orchestrates ML pipeline.

Multi-profile routing:
  /                    → redirect to default profile
  /p/{slug}/           → labeling page (profile-scoped)
  /p/{slug}/dashboard  → dashboard
  /p/{slug}/top        → top entries
  /p/{slug}/model      → model page
  /p/{slug}/settings   → profile-specific settings
  /profiles            → redirect to Settings → Profiles section
  /about               → global
  /setup               → first-run setup
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from typing import Optional
import threading
import uuid
from datetime import datetime

import db
import sync
import pipeline
import explainer
import distiller
import sampler
import model_manager
from config import get_config, save_config, BASE_DIR, VERSION
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Magnitu", version=VERSION)

# Allow magnitu-mini (static HTML on another path/host) to call profile APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_JOB_LOCK = threading.Lock()
_JOBS = {}


# ─── Background job helpers ──────────────────────────────────────────────────

def _create_job(job_type: str) -> str:
    job_id = uuid.uuid4().hex
    with _JOB_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id, "job_type": job_type,
            "status": "queued", "progress": 0, "message": "Queued",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "result": None, "error": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = datetime.utcnow().isoformat()


def _get_job(job_id: str) -> Optional[dict]:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, target):
    _update_job(job_id, status="running", progress=1, message="Starting...")
    try:
        def progress_cb(pct: int, msg: str):
            _update_job(job_id, progress=max(0, min(100, int(pct))), message=msg)
        result = target(progress_cb)
        _update_job(job_id, status="success", progress=100,
                    message="Done", result=result, error=None)
    except Exception as e:
        _update_job(job_id, status="error", message=str(e), error=str(e))


# ─── Sync implementations ────────────────────────────────────────────────────

def _sync_pull_impl(full: bool, progress_cb=None) -> dict:
    emb_before = db.get_embedding_count()
    count = 0
    entries_by_type = {}
    remote_total = 0
    embedding_rounds = 0

    if progress_cb:
        progress_cb(5, "Starting sync...")

    if full:
        try:
            status = sync.get_status()
            remote_entries = status.get("entries", {})
            remote_total = int(remote_entries.get("total", 0) or 0)
        except Exception:
            remote_entries = {}
            remote_total = 0

        type_specs = [
            ("feed_item", "feed_items"),
            ("email", "emails"),
            ("lex_item", "lex_items"),
            ("calendar_event", "calendar_events"),
        ]
        for idx, (entry_type, status_key) in enumerate(type_specs):
            expected = int(remote_entries.get(status_key, 0) or 0)
            limit = max(1000, expected + 250)
            if progress_cb:
                progress_cb(10 + idx * 20, "Pulling {} entries...".format(entry_type))
            fetched = sync.pull_entries(
                entry_type=entry_type, limit=limit, compute_embeddings=False,
            )
            entries_by_type[entry_type] = fetched
            count += fetched

        if get_config().get("model_architecture") == "transformer":
            prev_missing = None
            stalled_rounds = 0
            while db.get_entries_without_embeddings(limit=1):
                missing_now = len(db.get_entries_without_embeddings(limit=5000))
                if progress_cb:
                    progress_cb(
                        min(90, 70 + embedding_rounds * 4),
                        "Computing embeddings ({} missing)...".format(missing_now)
                    )
                sync._compute_pending_embeddings()
                embedding_rounds += 1
                missing_after = len(db.get_entries_without_embeddings(limit=5000))
                if prev_missing is not None and missing_after >= prev_missing:
                    stalled_rounds += 1
                else:
                    stalled_rounds = 0
                prev_missing = missing_after
                if stalled_rounds >= 2:
                    logger.warning("Full sync stalled at %d missing; stopping.", missing_after)
                    break
                if embedding_rounds >= 25:
                    break
    else:
        if progress_cb:
            progress_cb(25, "Pulling latest entries...")
        count = sync.pull_entries()
        embedding_rounds = 1

    if progress_cb:
        progress_cb(92, "Pulling labels...")
    labels_synced = 0
    try:
        labels_synced = sync.pull_labels()
    except Exception as e:
        logger.warning("Label pull failed during sync: %s", e)
        db.log_sync("pull", 0, "FAILED label pull: {}".format(e))

    syncs = db.get_recent_syncs(1)
    sync_detail = syncs[0]["details"] if syncs else ""

    emb_after = db.get_embedding_count()
    entry_count = db.get_entry_count()
    emb_computed = emb_after - emb_before
    emb_warning = ""
    if entry_count > 0 and emb_after < entry_count * 0.5:
        emb_warning = "Only {}/{} entries have embeddings.".format(emb_after, entry_count)

    result = {
        "success": True, "full_mode": full,
        "entries_fetched": count, "labels_synced": labels_synced,
        "sync_detail": sync_detail,
        "embeddings": emb_after, "embeddings_computed": emb_computed,
        "entry_count": entry_count, "embedding_warning": emb_warning,
        "embedding_rounds": embedding_rounds,
    }
    if full:
        result["entries_by_type"] = entries_by_type
        result["remote_total"] = remote_total
    return result


def _sync_push_impl(progress_cb=None, profile_id: int = 1) -> dict:
    import httpx as _httpx

    if progress_cb:
        progress_cb(5, "Preparing push...")

    model_info = db.get_active_model(profile_id)
    if not model_info:
        raise ValueError("No trained model for this profile. Train first.")

    all_entries = db.get_all_entries()
    if not all_entries:
        raise ValueError("No entries to score.")

    cfg = get_config()
    if cfg.get("model_architecture") == "transformer":
        missing = db.get_entries_without_embeddings(limit=5000)
        if missing:
            if progress_cb:
                progress_cb(20, "Computing missing embeddings...")
            sync._compute_pending_embeddings()

    if progress_cb:
        progress_cb(45, "Scoring entries...")
    try:
        scores = pipeline.score_entries(all_entries, profile_id=profile_id)
    except Exception as e:
        raise ValueError("Scoring failed: {}".format(e))

    if not scores:
        emb_count = db.get_embedding_count()
        entry_count = db.get_entry_count()
        raise ValueError(
            "No scores produced. Embeddings: {}/{} entries.".format(emb_count, entry_count)
        )

    if progress_cb:
        progress_cb(62, "Building explanations...")
    score_by_key = {(s["entry_type"], s["entry_id"]): s for s in scores}
    for entry in all_entries:
        key = (entry["entry_type"], entry["entry_id"])
        score = score_by_key.get(key)
        if not score:
            continue
        try:
            exp = explainer.explain_entry(entry, profile_id=profile_id)
            if exp:
                score["explanation"] = {
                    "top_features": exp["top_features"],
                    "confidence": exp["confidence"],
                    "prediction": exp["prediction"],
                }
        except Exception:
            pass

    profile_row = db.get_profile_by_id(profile_id)
    profile_info = model_manager.get_profile(profile_id)
    model_meta = None
    if profile_info:
        model_meta = {
            "model_name":        profile_info.get("model_name", ""),
            "model_uuid":        profile_info.get("model_uuid", ""),
            "model_description": profile_info.get("description", ""),
            "model_version":     model_info["version"],
            "model_trained_at":  model_info.get("trained_at", ""),
            "accuracy":          model_info.get("accuracy", 0.0),
            "f1_score":          model_info.get("f1_score", 0.0),
            "label_count":       model_info.get("label_count", 0),
            "architecture":      model_info.get("architecture", "tfidf"),
        }

    if progress_cb:
        progress_cb(78, "Pushing scores to Seismo...")
    try:
        score_result = sync.push_scores(
            scores, model_info["version"],
            model_meta=model_meta, profile=profile_row
        )
    except Exception as e:
        detail = str(e)
        if isinstance(e, _httpx.HTTPStatusError):
            detail = "Seismo HTTP {}: {}".format(e.response.status_code, e.response.text[:300])
        raise ValueError("Failed to push scores: {}".format(detail))

    if progress_cb:
        progress_cb(88, "Pushing labels...")
    try:
        sync.push_labels(profile_id=profile_id, profile=profile_row)
    except Exception as e:
        logger.warning("Label push failed during score push: %s", e)
        db.log_sync("push", 0, "FAILED label push: {}".format(e), profile_id=profile_id)

    if progress_cb:
        progress_cb(93, "Building recipe...")
    try:
        recipe = distiller.distill_recipe(profile_id=profile_id)
    except Exception as e:
        logger.warning("Recipe distillation failed: %s", e)
        recipe = None

    recipe_result = {}
    if recipe:
        if progress_cb:
            progress_cb(97, "Pushing recipe...")
        try:
            recipe_result = sync.push_recipe(recipe, profile=profile_row)
        except Exception as e:
            detail = str(e)
            if isinstance(e, _httpx.HTTPStatusError):
                detail = "Seismo HTTP {}: {}".format(e.response.status_code, e.response.text[:300])
            recipe_result = {"error": detail}

    return {
        "success": True,
        "scores_pushed": len(scores),
        "score_result": score_result,
        "recipe_result": recipe_result,
    }


# ─── Template helpers ────────────────────────────────────────────────────────

def _get_profile_or_404(slug: str) -> dict:
    profile = db.get_profile_by_slug(slug)
    if not profile:
        raise HTTPException(404, "Profile '{}' not found".format(slug))
    return profile


def _base_context(request: Request, profile: Optional[dict] = None) -> dict:
    """Common context for all templates. Profile-aware when a profile is given."""
    config = get_config()
    profile_id = profile["id"] if profile else 1
    active_model = db.get_active_model(profile_id)
    return {
        "request":           request,
        "config":            config,
        "version":           VERSION,
        "label_count":       db.get_label_count(profile_id),
        "entry_count":       db.get_entry_count(),
        "active_model":      active_model,
        "label_distribution": db.get_label_distribution(profile_id),
        "profile":           profile,
        "all_profiles":      db.get_all_profiles(),
        "architecture":      config.get("model_architecture", "transformer"),
        "embedding_count":   db.get_embedding_count(),
    }


def _extract_legal_patterns(limit: int = 12, profile_id: int = 1) -> dict:
    model = db.get_active_model(profile_id)
    if not model or not model.get("recipe_path"):
        return {"positive": [], "negative": []}
    recipe_path = Path(model["recipe_path"])
    if not recipe_path.exists():
        return {"positive": [], "negative": []}
    try:
        with open(recipe_path) as f:
            recipe = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"positive": [], "negative": []}

    keywords = recipe.get("keywords", {})
    if not keywords:
        return {"positive": [], "negative": []}

    legal_markers = (
        "eu", "eea", "ewr", "third", "country", "member state",
        "dritt", "tiers", "conformity", "assessment", "market access",
        "equivalence", "delegated", "implementing", "corrigendum",
        "annex", "regulation", "directive", "ce marking", "single market",
    )
    positives = []
    negatives = []
    for phrase, cls_wts in keywords.items():
        if " " not in phrase:
            continue
        p = phrase.lower()
        if not any(m in p for m in legal_markers):
            continue
        inv  = float(cls_wts.get("investigation_lead", 0.0))
        imp  = float(cls_wts.get("important", 0.0))
        noise = float(cls_wts.get("noise", 0.0))
        pos_score = inv + imp * 0.7
        neg_score = noise
        if pos_score > 0:
            positives.append((phrase, round(pos_score, 4)))
        if neg_score > 0:
            negatives.append((phrase, round(neg_score, 4)))
    positives.sort(key=lambda x: x[1], reverse=True)
    negatives.sort(key=lambda x: x[1], reverse=True)
    return {"positive": positives[:limit], "negative": negatives[:limit]}


def _today_label_count(profile_id: int = 1) -> int:
    from datetime import date
    today = date.today().isoformat()
    conn = db.get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE profile_id=? AND date(updated_at)=?",
        (profile_id, today)
    ).fetchone()[0]
    conn.close()
    return count


# ─── Global pages ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to default profile, or setup if no profiles exist."""
    if not db.has_any_profile():
        return RedirectResponse("/setup", status_code=302)
    default = db.get_default_profile()
    if default:
        return RedirectResponse("/p/{}/".format(default["slug"]), status_code=302)
    return RedirectResponse("/setup", status_code=302)


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    ctx = _base_context(request)
    return templates.TemplateResponse("about.html", ctx)


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """First-run setup: create the initial profile."""
    if db.has_any_profile():
        default = db.get_default_profile()
        if default:
            return RedirectResponse("/p/{}/model".format(default["slug"]), status_code=302)
    ctx = _base_context(request)
    return templates.TemplateResponse("setup.html", ctx)


@app.get("/profiles")
async def profiles_page_redirect():
    """Legacy URL — profile management lives under Settings."""
    p = db.get_default_profile()
    if not p:
        profs = db.get_all_profiles()
        p = profs[0] if profs else None
    if not p:
        return RedirectResponse("/setup", status_code=302)
    return RedirectResponse("/p/{}/settings?profiles=1".format(p["slug"]), status_code=302)


# ─── Profile-scoped pages ─────────────────────────────────────────────────────

@app.get("/p/{slug}/", response_class=HTMLResponse)
async def labeling_page(request: Request, slug: str, source: str = "all"):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    ctx = _base_context(request, profile)
    entry_type = None
    if source == "lex":
        entry_type = "lex_item"
    elif source == "news":
        entry_type = "feed_item"
    entries = sampler.get_smart_entries(limit=30, entry_type=entry_type,
                                        profile_id=profile_id)
    ctx["source_filter"] = source

    for entry in entries:
        label_data = db.get_label_with_reasoning(
            entry["entry_type"], entry["entry_id"], profile_id
        )
        entry["_label"] = label_data["label"] if label_data else None
        entry["_reasoning"] = label_data["reasoning"] if label_data else ""

    ctx["entries"] = entries
    ctx["unlabeled_count"] = len([e for e in entries if e["_label"] is None])
    ctx["today_labels"] = _today_label_count(profile_id)

    reasons = {}
    for e in entries:
        r = e.get("_sampling_reason", "new")
        reasons[r] = reasons.get(r, 0) + 1
    ctx["sampling_stats"] = reasons
    ctx["has_model"] = ctx["active_model"] is not None
    return templates.TemplateResponse("labeling.html", ctx)


@app.get("/p/{slug}/api/entries")
async def api_profile_entries(slug: str, source: str = "all", limit: int = 500):
    """JSON entries for the labeling queue (same sampling as the web UI). Used by magnitu-mini."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    entry_type = None
    if source == "lex":
        entry_type = "lex_item"
    elif source == "news":
        entry_type = "feed_item"
    lim = max(1, min(int(limit), 500))
    entries = sampler.get_smart_entries(
        limit=lim, entry_type=entry_type, profile_id=profile_id
    )
    return {"entries": entries, "profile_slug": slug}


@app.get("/p/{slug}/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, slug: str):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    ctx = _base_context(request, profile)
    ctx["models"] = db.get_all_models(profile_id)
    ctx["syncs"] = db.get_recent_syncs(20, profile_id=profile_id)
    ctx["legal_patterns"] = _extract_legal_patterns(profile_id=profile_id)
    ctx["keywords"] = {}
    if ctx["active_model"]:
        try:
            kw = explainer.global_keywords(limit=30, profile_id=profile_id)
            ctx["keywords"] = kw
        except Exception as e:
            logger.warning("Failed to load dashboard keywords: %s", e)
    return templates.TemplateResponse("dashboard.html", ctx)


@app.get("/p/{slug}/top", response_class=HTMLResponse)
async def top_page(request: Request, slug: str, view: str = "recent"):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    ctx = _base_context(request, profile)

    if view not in ("recent", "mismatches", "all"):
        view = "recent"
    ctx["view"] = view

    if view == "recent":
        entries = db.get_recent_entries(days=7)
        scored = pipeline.score_entries(entries, profile_id=profile_id)
        all_labels = {(l["entry_type"], l["entry_id"])
                      for l in db.get_all_labels_raw(profile_id)}
        labeled_ids = set()
        for e in entries:
            if (e["entry_type"], e["entry_id"]) in all_labels:
                labeled_ids.add((e["entry_type"], e["entry_id"]))
        unlabeled_scored = [s for s in scored
                            if (s["entry_type"], s["entry_id"]) not in labeled_ids]
        unlabeled_scored.sort(key=lambda s: s["relevance_score"], reverse=True)
        top_scored = unlabeled_scored[:30]
        entry_map = {(e["entry_type"], e["entry_id"]): e for e in entries}
        top_entries = []
        for s in top_scored:
            entry = entry_map.get((s["entry_type"], s["entry_id"]))
            if not entry:
                continue
            top_entries.append({
                "entry": entry, "score": s,
                "user_label": None, "match": None,
            })
        ctx["top_entries"] = top_entries
        ctx["labeled_count"] = 0
        ctx["correct_count"] = 0
        ctx["accuracy"] = None
        ctx["total_recent"] = len(entries)
        ctx["total_recent_unlabeled"] = len(unlabeled_scored)

    elif view == "mismatches":
        labeled_entries = db.get_labeled_entries(profile_id)
        scored = pipeline.score_entries(labeled_entries, profile_id=profile_id)
        score_map = {(s["entry_type"], s["entry_id"]): s for s in scored}
        top_entries = []
        for entry in labeled_entries:
            key = (entry["entry_type"], entry["entry_id"])
            s = score_map.get(key)
            if not s:
                continue
            user_label = entry["user_label"]
            predicted  = s["predicted_label"]
            if user_label != predicted:
                top_entries.append({
                    "entry": entry, "score": s,
                    "user_label": user_label, "match": False,
                })
        top_entries.sort(key=lambda x: x["score"]["relevance_score"], reverse=True)
        ctx["top_entries"] = top_entries
        ctx["labeled_count"] = len(labeled_entries)
        ctx["correct_count"] = 0
        ctx["accuracy"] = None

    elif view == "all":
        labeled_entries = db.get_labeled_entries(profile_id)
        scored = pipeline.score_entries(labeled_entries, profile_id=profile_id)
        score_map = {(s["entry_type"], s["entry_id"]): s for s in scored}
        top_entries = []
        for entry in labeled_entries:
            key = (entry["entry_type"], entry["entry_id"])
            s = score_map.get(key)
            if not s:
                continue
            user_label = entry["user_label"]
            top_entries.append({
                "entry": entry, "score": s,
                "user_label": user_label,
                "match": user_label == s["predicted_label"] if user_label else None,
            })
        labeled_in_top = [e for e in top_entries if e["user_label"] is not None]
        correct = sum(1 for e in labeled_in_top if e["match"])
        ctx["top_entries"] = top_entries
        ctx["labeled_count"] = len(labeled_in_top)
        ctx["correct_count"] = correct
        ctx["accuracy"] = round(correct / len(labeled_in_top) * 100, 1) if labeled_in_top else None

    return templates.TemplateResponse("top.html", ctx)


@app.get("/p/{slug}/model", response_class=HTMLResponse)
async def model_page(request: Request, slug: str):
    profile = _get_profile_or_404(slug)
    ctx = _base_context(request, profile)
    ctx["models"] = db.get_all_models(profile["id"])
    return templates.TemplateResponse("model.html", ctx)


@app.get("/p/{slug}/settings", response_class=HTMLResponse)
async def settings_page(request: Request, slug: str):
    profile = _get_profile_or_404(slug)
    ctx = _base_context(request, profile)
    ctx["syncs"] = db.get_recent_syncs(10, profile_id=profile["id"])
    ctx["profiles"] = [
        {**dict(p), "label_count": db.get_label_count(p["id"])}
        for p in db.get_all_profiles()
    ]
    return templates.TemplateResponse("settings.html", ctx)


# ─── API: Profiles (global) ──────────────────────────────────────────────────

@app.get("/api/profiles")
async def list_profiles():
    return db.get_all_profiles()


@app.post("/api/profiles")
async def create_profile_api(request: Request):
    data = await request.json()
    display_name = (data.get("display_name") or data.get("name") or "").strip()
    if not display_name:
        return JSONResponse({"success": False, "error": "Profile name is required."}, 400)
    slug = data.get("slug", "").strip() or db.slugify(display_name)
    if db.get_profile_by_slug(slug):
        base, n = slug, 1
        while db.get_profile_by_slug(slug):
            slug = "{}-{}".format(base, n)
            n += 1
    seismo_url = (data.get("seismo_url") or "").strip()
    api_key    = (data.get("api_key") or "").strip()
    description = (data.get("description") or "").strip()
    profile = db.create_profile(slug, display_name, description, seismo_url, api_key)
    return {"success": True, "profile": profile}


@app.put("/api/profiles/{profile_id}")
async def update_profile_api(profile_id: int, request: Request):
    data = await request.json()
    existing = db.get_profile_by_id(profile_id)
    if not existing:
        raise HTTPException(404, "Profile not found")

    new_slug = (data.get("slug") or "").strip()
    if new_slug and new_slug != existing["slug"]:
        conflict = db.get_profile_by_slug(new_slug)
        if conflict and conflict["id"] != profile_id:
            return JSONResponse({"success": False, "error": "Slug already in use."}, 400)
    else:
        new_slug = None

    db.update_profile(
        profile_id=profile_id,
        display_name=(data.get("display_name") or data.get("name") or "").strip() or None,
        description=data.get("description"),
        seismo_url=data.get("seismo_url"),
        api_key=data.get("api_key"),
        is_default=1 if data.get("is_default") else None,
    )
    if new_slug:
        conn = db.get_db()
        conn.execute("UPDATE profiles SET slug=? WHERE id=?", (new_slug, profile_id))
        conn.commit()
        conn.close()

    return {"success": True, "profile": db.get_profile_by_id(profile_id)}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile_api(profile_id: int):
    existing = db.get_profile_by_id(profile_id)
    if not existing:
        raise HTTPException(404, "Profile not found")
    if existing.get("is_default"):
        return JSONResponse({"success": False,
                             "error": "Cannot delete the default profile."}, 400)
    db.delete_profile(profile_id)
    return {"success": True}


# ─── API: Model — profile-scoped ─────────────────────────────────────────────

@app.post("/api/model/create")
async def create_model(request: Request):
    """Create a new model profile (first-run setup)."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    if not name:
        return JSONResponse({"success": False, "error": "Model name is required."}, 400)
    if db.has_any_profile():
        return JSONResponse({"success": False, "error": "A profile already exists. Use /api/profiles to add more."}, 400)
    profile = model_manager.create_profile(name, description)
    return {"success": True, "profile": profile}


@app.post("/api/model/import")
async def import_model_setup(request: Request):
    """Import a .magnitu file during first-run setup (creates a new profile)."""
    import asyncio, tempfile
    form = await request.form()
    upload = form.get("file")
    if not upload:
        return JSONResponse({"success": False, "error": "No file uploaded."}, 400)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".magnitu") as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: model_manager.import_model(tmp_path, profile_id=None)
        )
        slug = db.get_profile_by_id(result["profile_id"])["slug"] if result.get("profile_id") else ""
        return {"success": True, "slug": slug, **result}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, 400)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/p/{slug}/api/model/update")
async def update_model(slug: str, request: Request):
    profile = _get_profile_or_404(slug)
    data = await request.json()
    model_manager.update_profile(description=data.get("description"),
                                  profile_id=profile["id"])
    return {"success": True}


@app.get("/p/{slug}/api/model/export")
async def export_model(slug: str):
    from fastapi.responses import FileResponse
    profile = _get_profile_or_404(slug)
    try:
        path = model_manager.export_model(profile_id=profile["id"])
        safe_name = profile["slug"]
        return FileResponse(path, media_type="application/zip",
                            filename="{}.magnitu".format(safe_name))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/p/{slug}/api/model/fork")
async def fork_model(slug: str, request: Request):
    from fastapi.responses import FileResponse
    profile = _get_profile_or_404(slug)
    data = await request.json()
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    if not name:
        return JSONResponse({"success": False, "error": "Model name is required."}, 400)
    try:
        path = model_manager.export_as_new_model(
            name, description, profile_id=profile["id"]
        )
        safe_name = name.replace(" ", "_").lower()
        return FileResponse(path, media_type="application/zip",
                            filename="{}.magnitu".format(safe_name))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/p/{slug}/api/model/import")
async def import_model_upload(slug: str, request: Request):
    import asyncio
    import tempfile
    profile = _get_profile_or_404(slug)
    form = await request.form()
    upload = form.get("file")
    if not upload:
        return JSONResponse({"success": False, "error": "No file uploaded."}, 400)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".magnitu") as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: model_manager.import_model(tmp_path, profile_id=profile["id"])
        )
        return {"success": True, **result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, 400)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─── API: Labeling — profile-scoped ──────────────────────────────────────────

@app.post("/p/{slug}/api/label")
async def set_label(
    slug: str,
    entry_type: str = Form(...),
    entry_id: int = Form(...),
    label: str = Form(...),
    reasoning: str = Form(""),
):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    valid_labels = ["investigation_lead", "important", "background", "noise"]
    if label not in valid_labels:
        raise HTTPException(400, "Invalid label.")
    db.set_label(entry_type, entry_id, label,
                 reasoning=reasoning.strip(), profile_id=profile_id)
    try:
        sync.push_labels(profile_id=profile_id, profile=profile)
    except Exception as e:
        logger.warning("Background label push failed: %s", e)
        db.log_sync("push", 0, "FAILED: {}".format(e), profile_id=profile_id)
    return {"success": True, "entry_type": entry_type,
            "entry_id": entry_id, "label": label}


@app.post("/p/{slug}/api/unlabel")
async def remove_label(
    slug: str,
    entry_type: str = Form(...),
    entry_id: int = Form(...),
):
    profile = _get_profile_or_404(slug)
    db.remove_label(entry_type, entry_id, profile_id=profile["id"])
    return {"success": True}


# ─── API: Sync — global pull, per-profile push ───────────────────────────────

@app.post("/api/sync/pull")
async def sync_pull(full: bool = False, background: bool = False):
    """Pull entries from mothership — shared across all profiles."""
    import asyncio
    if background:
        job_type = "sync_pull_full" if full else "sync_pull"
        job_id = _create_job(job_type)
        t = threading.Thread(
            target=lambda: _run_job(job_id, lambda cb: _sync_pull_impl(full=full, progress_cb=cb)),
            daemon=True,
        )
        t.start()
        return {"success": True, "job_id": job_id}
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _sync_pull_impl(full=full))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/p/{slug}/api/sync/push")
async def sync_push(slug: str, background: bool = False):
    """Score and push to this profile's Seismo target."""
    import asyncio
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    if background:
        job_id = _create_job("sync_push")
        t = threading.Thread(
            target=lambda: _run_job(
                job_id, lambda cb: _sync_push_impl(progress_cb=cb, profile_id=profile_id)
            ),
            daemon=True,
        )
        t.start()
        return {"success": True, "job_id": job_id}
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _sync_push_impl(profile_id=profile_id)
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/p/{slug}/api/sync/labels")
async def sync_labels(slug: str):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    pushed = {}
    pulled = 0
    try:
        pushed = sync.push_labels(profile_id=profile_id, profile=profile)
    except Exception as e:
        raise HTTPException(500, "Failed to push labels: {}".format(e))
    pull_error = None
    try:
        pulled = sync.pull_labels(profile_id=profile_id)
    except Exception as e:
        logger.warning("Label pull failed: %s", e)
        pull_error = str(e)
    result = {"success": True, "pushed": pushed, "labels_imported": pulled}
    if pull_error:
        result["pull_error"] = pull_error
    return result


@app.get("/p/{slug}/api/sync/test")
async def sync_test(slug: str):
    profile = _get_profile_or_404(slug)
    target = sync._profile_target(profile)
    ok, msg = sync.test_connection(seismo_target=target)
    if not ok:
        return {"success": False, "message": msg}
    label_ok, label_msg = sync.verify_seismo_endpoints(seismo_target=target)
    if not label_ok:
        return {"success": True, "message": msg, "warning": label_msg}
    return {"success": True, "message": msg}


@app.get("/p/{slug}/api/sync/health")
async def sync_health(slug: str):
    profile = _get_profile_or_404(slug)
    syncs = db.get_recent_syncs(20, profile_id=profile["id"])
    last_push = next((s for s in syncs if s["direction"] == "push"), None)
    last_pull = next((s for s in syncs if s["direction"] == "pull"), None)
    push_ok = last_push and not (last_push.get("details") or "").startswith("FAILED")
    pull_ok = last_pull and not (last_pull.get("details") or "").startswith("FAILED")
    return {
        "push": {"ok": push_ok, "detail": dict(last_push) if last_push else None},
        "pull": {"ok": pull_ok, "detail": dict(last_pull) if last_pull else None},
    }


# ─── API: Training — profile-scoped ──────────────────────────────────────────

@app.post("/p/{slug}/api/train")
async def train_model(slug: str):
    import asyncio
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: pipeline.train(profile_id=profile_id)
    )
    if not result.get("success"):
        return JSONResponse(result, status_code=400)

    recipe = await loop.run_in_executor(
        None, lambda: distiller.distill_recipe(profile_id=profile_id)
    )
    if recipe:
        quality = distiller.evaluate_recipe_quality(recipe, profile_id=profile_id)
        result["recipe_version"] = recipe.get("version")
        result["recipe_quality"] = quality
        result["recipe_keywords"] = len(recipe.get("keywords", {}))
    else:
        arch = result.get("architecture", "tfidf")
        if arch == "transformer":
            result["recipe_note"] = (
                "Recipe not generated yet. Knowledge distillation needs at least "
                "20 entries to build a recipe."
            )
        else:
            result["recipe_note"] = "No recipe generated — model may not have enough data."
    return result


# ─── API: Explain / Keywords — profile-scoped ────────────────────────────────

@app.get("/p/{slug}/api/explain/{entry_type}/{entry_id}")
async def explain(slug: str, entry_type: str, entry_id: int):
    profile = _get_profile_or_404(slug)
    conn = db.get_db()
    row = conn.execute(
        "SELECT * FROM entries WHERE entry_type=? AND entry_id=?",
        (entry_type, entry_id)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Entry not found")
    exp = explainer.explain_entry(dict(row), profile_id=profile["id"])
    if not exp:
        raise HTTPException(400, "No active model")
    return exp


@app.get("/p/{slug}/api/keywords")
async def keywords(slug: str, class_name: str = None, limit: int = 50):
    profile = _get_profile_or_404(slug)
    kw = explainer.global_keywords(class_name, limit, profile_id=profile["id"])
    return kw


# ─── API: Stats — profile-scoped ─────────────────────────────────────────────

@app.get("/p/{slug}/api/stats")
async def stats(slug: str):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    config = get_config()
    model = db.get_active_model(profile_id)
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass
    return {
        "magnitu_version":    VERSION,
        "architecture":       config.get("model_architecture", "transformer"),
        "entries":            db.get_entry_count(),
        "labels":             db.get_label_count(profile_id),
        "embeddings":         db.get_embedding_count(),
        "label_distribution": db.get_label_distribution(profile_id),
        "active_model":       model,
        "models_count":       len(db.get_all_models(profile_id)),
        "gpu_available":      gpu_available,
        "gpu_enabled":        config.get("use_gpu", True),
    }


# ─── API: Settings — global ───────────────────────────────────────────────────

@app.post("/api/settings")
async def update_settings(request: Request):
    """Update global configuration (architecture, mothership URL, etc.)."""
    data = await request.json()
    config = get_config()
    old_transformer_name = config.get("transformer_model_name", "")
    old_use_gpu = config.get("use_gpu", True)
    old_legal_patterns = list(config.get("legal_signal_patterns") or [])

    for key in ["seismo_url", "api_key", "min_labels_to_train",
                "recipe_top_keywords", "auto_train_after_n_labels", "alert_threshold",
                "model_architecture", "transformer_model_name", "use_gpu",
                "discovery_lead_blend",
                "label_time_decay_days", "label_time_decay_floor",
                "reasoning_weight_boost", "legal_signal_patterns"]:
        if key in data:
            config[key] = data[key]

    try:
        b = float(config.get("discovery_lead_blend", 0.0) or 0.0)
        config["discovery_lead_blend"] = max(0.0, min(0.25, b))
    except (TypeError, ValueError):
        config["discovery_lead_blend"] = 0.0

    try:
        hl = int(float(config.get("label_time_decay_days", 0) or 0))
        config["label_time_decay_days"] = max(0, min(3650, hl))
    except (TypeError, ValueError):
        config["label_time_decay_days"] = 0

    try:
        fl = float(config.get("label_time_decay_floor", 0.2) or 0.0)
        config["label_time_decay_floor"] = max(0.0, min(1.0, fl))
    except (TypeError, ValueError):
        config["label_time_decay_floor"] = 0.2

    try:
        rb = float(config.get("reasoning_weight_boost", 1.0) or 1.0)
        config["reasoning_weight_boost"] = max(0.0, min(5.0, rb))
    except (TypeError, ValueError):
        config["reasoning_weight_boost"] = 1.0

    raw_patterns = config.get("legal_signal_patterns") or []
    if isinstance(raw_patterns, str):
        raw_patterns = [raw_patterns]
    cleaned_patterns = []
    for p in raw_patterns:
        if isinstance(p, str):
            s = p.strip()
            if s and s not in cleaned_patterns:
                cleaned_patterns.append(s)
    config["legal_signal_patterns"] = cleaned_patterns

    save_config(config)

    new_transformer_name = config.get("transformer_model_name", "")
    if new_transformer_name != old_transformer_name and old_transformer_name:
        db.invalidate_all_embeddings()
        pipeline.invalidate_embedder_cache()

    if cleaned_patterns != old_legal_patterns:
        # The legal-signal prefix is baked into every embedding; if the list
        # changed we must re-embed so cached vectors match the current config.
        db.invalidate_all_embeddings()

    if config.get("use_gpu", True) != old_use_gpu:
        pipeline.release_embedder()

    return {"success": True, "config": config}


@app.post("/api/embeddings/compute")
async def compute_embeddings():
    import asyncio
    config = get_config()
    if config.get("model_architecture") != "transformer":
        return {"success": False, "error": "Not in transformer mode."}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, sync._compute_pending_embeddings)
    return {
        "success": True,
        "embedding_count": db.get_embedding_count(),
        "entry_count": db.get_entry_count(),
    }


# ─── Startup ─────────────────────────────────────────────────────────────────

def _migrate_config():
    cfg = get_config()
    changed = False
    if cfg.get("transformer_model_name") == "distilroberta-base":
        import logging
        logging.getLogger(__name__).info(
            "Migrating transformer model: distilroberta-base → xlm-roberta-base"
        )
        cfg["transformer_model_name"] = "xlm-roberta-base"
        changed = True
    if changed:
        save_config(cfg)
        db.invalidate_all_embeddings()
        pipeline.invalidate_embedder_cache()


_migrate_config()
