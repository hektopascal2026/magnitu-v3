"""
Magnitu — ML-powered relevance scoring for Seismo.
FastAPI application: serves the labeling UI, model overview, and orchestrates ML pipeline.

Multi-profile routing:
  /                    → redirect to active workspace profile (Settings → Profiles → Switch)
  /p/{slug}/           → labeling (slug must match active workspace except Settings)
  /p/{slug}/settings   → settings for that slug (switch active workspace here)
  /p/{slug}/dashboard  → redirects to model (bookmark compatibility)
  /p/{slug}/top        → top entries
  /p/{slug}/model      → model overview + profile actions
  /p/{slug}/settings   → profile-specific settings
  /profiles            → redirect to Settings → Profiles section
  /about               → global
  /setup               → first-run setup
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import json
from typing import Optional, Dict
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
from magnitu.prompts import DEFAULT_GEMINI_PERSONA
from magnitu.synthetic_batch import run_gemini_synthetic_batch_job
from magnitu.accent_theme import safe_accent_for_profile, contrast_text_on_accent, get_theme_colors
from config import (
    get_config,
    save_config,
    BASE_DIR,
    VERSION,
    MODELS_DIR,
    suggested_recipe_top_keywords,
    RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX,
    PROFILE_TRAINING_SETTINGS_KEYS,
)
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

_TOUCH_ICON = BASE_DIR / "static" / "apple-touch-icon.png"
_TOUCH_ICON_PRE = BASE_DIR / "static" / "apple-touch-icon-precomposed.png"


@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    """Safari probes this URL at site root; serve asset so access logs stay clean."""
    return FileResponse(_TOUCH_ICON, media_type="image/png")


@app.get("/apple-touch-icon-precomposed.png")
async def apple_touch_icon_precomposed():
    """Legacy Safari fallback for home-screen icons."""
    return FileResponse(_TOUCH_ICON_PRE, media_type="image/png")


@app.middleware("http")
async def enforce_active_workspace(request: Request, call_next):
    """Label/Gemini/Top/Model use the persisted active profile; Settings stay per-slug."""
    path = request.url.path
    if not path.startswith("/p/"):
        return await call_next(request)
    raw = path.split("?", 1)[0].rstrip("/")
    segments = [s for s in raw.split("/") if s]
    if len(segments) < 2 or segments[0] != "p":
        return await call_next(request)
    slug = segments[1]
    if not slug:
        return await call_next(request)
    prof = db.get_profile_by_slug(slug)
    if not prof:
        return await call_next(request)
    active = db.get_active_profile()
    if not active:
        return await call_next(request)
    if prof["id"] == active["id"]:
        return await call_next(request)
    rest_segments = segments[2:]
    rest_path = "/" + "/".join(rest_segments) if rest_segments else "/"
    if "/settings" in rest_path or rest_path.startswith("/settings"):
        return await call_next(request)
    if rest_path.startswith("/api/") or "/api/" in rest_path:
        return JSONResponse(
            {
                "detail": (
                    "This profile is not the active workspace. "
                    "Choose “Switch” next to the profile in Settings → Profiles."
                ),
                "code": "wrong_workspace",
            },
            status_code=403,
        )
    new_url = "/p/" + active["slug"] + (rest_path if rest_path != "/" else "/")
    q = request.url.query
    if q:
        new_url = new_url + "?" + q
    return RedirectResponse(url=new_url, status_code=302)

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
            "logs": [],
            "cancel_requested": False,
        }
    return job_id


def _update_job(job_id: str, **kwargs):
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if "logs" in kwargs and job.get("logs") is not None:
            job["logs"].extend(kwargs.pop("logs"))
        job.update(kwargs)
        job["updated_at"] = datetime.utcnow().isoformat()


def _get_job(job_id: str) -> Optional[dict]:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, target):
    _update_job(job_id, status="running", progress=1, message="Starting...")
    try:
        def progress_cb(pct: int, msg: str, log: Optional[str] = None):
            update = {"progress": max(0, min(100, int(pct))), "message": msg}
            if log:
                update["logs"] = [log]
            _update_job(job_id, **update)
        result = target(progress_cb)
        if isinstance(result, dict) and result.get("cancelled"):
            _update_job(
                job_id,
                status="cancelled",
                progress=100,
                message=result.get("message", "Cancelled"),
                result=result,
                error=None,
            )
        else:
            _update_job(job_id, status="success", progress=100,
                        message="Done", result=result, error=None)
    except Exception as e:
        _update_job(job_id, status="error", message=str(e), error=str(e))


# ─── Sync implementations ────────────────────────────────────────────────────

def _sync_pull_impl(
    full: bool,
    progress_cb=None,
    profile: Optional[Dict] = None,
) -> dict:
    """Pull entries + labels into local DB.

    Entries always come from global mothership Settings.

    Labels merge into the given ``profile`` using ``_profile_target(profile)``
    (satellite when URL + key are set; mothership when both blank).

    ``profile`` is required (resolved by HTTP handlers from ``profile_slug`` or
    the active workspace profile).
    """
    if not profile:
        raise ValueError(
            "Sync pull requires a profile. Use /p/{slug}/api/sync/pull, POST "
            "/api/sync/pull?profile_slug=..., or set an active profile in Settings."
        )
    if sync.profile_satellite_incomplete(profile):
        raise ValueError(sync.INCOMPLETE_SATELLITE_CREDENTIALS_MSG)
    profile_id_for_labels = int(profile["id"])

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
                entry_type=entry_type,
                limit=limit,
                compute_embeddings=False,
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
    label_pull_warning = ""
    label_pull_source = "global"
    try:
        labels_synced = sync.pull_labels(
            profile_id=profile_id_for_labels,
            profile=profile,
        )
        if profile and sync.profile_satellite_blank(profile):
            label_pull_warning = (
                "This profile has no satellite URL/API key — labels were pulled "
                "from global mothership (same source as entries). "
                "Set satellite credentials to pull labels only from that Seismo."
            )
            label_pull_source = "mothership_fallback"
        elif profile:
            label_pull_source = "satellite"
    except ValueError:
        raise
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
        "label_pull_source": label_pull_source,
        "label_pull_warning": label_pull_warning,
    }
    if full:
        result["entries_by_type"] = entries_by_type
        result["remote_total"] = remote_total
    return result


def _sync_push_impl(progress_cb=None, profile_id: int = 1) -> dict:
    import httpx as _httpx

    if progress_cb:
        progress_cb(5, "Preparing push...")

    profile_row = db.get_profile_by_id(profile_id)
    if not profile_row:
        raise ValueError("Profile not found.")
    if sync.profile_satellite_incomplete(profile_row):
        raise ValueError(sync.INCOMPLETE_SATELLITE_CREDENTIALS_MSG)

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

    sync.refresh_profile_accent(profile_row)

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


def _nav_profile_styles(all_profiles):
    styles = {}
    for p in all_profiles:
        h = safe_accent_for_profile(p.get("accent_color"))
        if h:
            styles[p["id"]] = {"bg": h, "fg": contrast_text_on_accent(h)}
        else:
            styles[p["id"]] = {"bg": "", "fg": ""}
    return styles


def _base_context(request: Request, profile: Optional[dict] = None) -> dict:
    """Common context for all templates. Profile-aware when a profile is given."""
    profile_id = profile["id"] if profile else 1
    config = db.get_effective_config(profile_id) if profile else get_config()
    active_model = db.get_active_model(profile_id)
    
    theme = get_theme_colors(profile.get("accent_color") if profile else None)
    profile_accent_bg = theme["bg"]
    profile_accent_fg = theme["fg"]
    profile_accent_subtle = theme["subtle"]
    profile_accent_border = theme["border"]

    ap = db.get_active_profile()
    profs = db.get_all_profiles()
    return {
        "request":           request,
        "config":            config,
        "version":           VERSION,
        "label_count":       db.get_label_count(profile_id),
        "entry_count":       db.get_entry_count(),
        "active_model":      active_model,
        "label_distribution": db.get_label_distribution(profile_id),
        "profile":           profile,
        "active_profile":    ap,
        "all_profiles":      profs,
        "nav_profile_styles": _nav_profile_styles(profs),
        "architecture":      config.get("model_architecture", "transformer"),
        "embedding_count":   db.get_embedding_count(),
        "profile_accent_bg": profile_accent_bg,
        "profile_accent_fg": profile_accent_fg,
        "profile_accent_subtle": profile_accent_subtle,
        "profile_accent_border": profile_accent_border,
        "gemini_persona": db.get_profile_gemini_persona(profile_id) if profile else None,
        "default_gemini_persona": DEFAULT_GEMINI_PERSONA,
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


# ─── Global pages ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to default profile, or setup if no profiles exist."""
    if not db.has_any_profile():
        return RedirectResponse("/setup", status_code=302)
    active = db.get_active_profile()
    if active:
        return RedirectResponse("/p/{}/".format(active["slug"]), status_code=302)
    return RedirectResponse("/setup", status_code=302)


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    ap = db.get_active_profile()
    ctx = _base_context(request, profile=ap)
    return templates.TemplateResponse("about.html", ctx)


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """First-run setup: create the initial profile."""
    if db.has_any_profile():
        active = db.get_active_profile()
        if active:
            return RedirectResponse("/p/{}/model".format(active["slug"]), status_code=302)
    ctx = _base_context(request)
    return templates.TemplateResponse("setup.html", ctx)


@app.get("/profiles")
async def profiles_page_redirect():
    """Legacy URL — profile management lives under Settings."""
    p = db.get_active_profile()
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

    am = ctx.get("active_model")
    mt_labels = int(am.get("label_count") or 0) if am else 0
    lc = ctx["label_count"]
    ctx["model_trained_label_count"] = mt_labels
    if lc > 0:
        ctx["model_coverage_pct"] = round(
            min(100.0, (float(mt_labels) / float(lc)) * 100.0), 1
        )
    else:
        ctx["model_coverage_pct"] = 0.0

    reasons = {}
    for e in entries:
        r = e.get("_sampling_reason", "new")
        reasons[r] = reasons.get(r, 0) + 1
    ctx["sampling_stats"] = reasons
    ctx["has_model"] = ctx["active_model"] is not None
    return templates.TemplateResponse("labeling.html", ctx)


@app.get("/p/{slug}/gemini", response_class=HTMLResponse)
async def gemini_page(request: Request, slug: str):
    """Synthetic labeling (Gemini) — batch job with same Smart Queue prioritisation as Label."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    ctx = _base_context(request, profile)
    pinfo = model_manager.get_profile(profile_id)
    if pinfo:
        ctx["target_model_title"] = pinfo.get("model_name") or profile.get("display_name", "")
    else:
        ctx["target_model_title"] = profile.get("display_name", slug)
    ctx["gemini_queue_help"] = (
        "Batches use the same Smart Queue as Label, prioritising uncertain and new rows, "
        "then conflict/diverse picks to reach your limit. Only unlabeled entries are processed."
    )

    # Ensure persona is available (already in _base_context, but let's be explicit)
    if not ctx.get("gemini_persona"):
        ctx["gemini_persona"] = db.get_profile_gemini_persona(profile_id) or ctx["default_gemini_persona"]

    return templates.TemplateResponse("gemini.html", ctx)


@app.post("/p/{slug}/api/gemini/persona")
async def gemini_persona_update(slug: str, request: Request):
    """Update the Gemini persona for this profile."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    body = await request.json()
    persona = body.get("persona")
    if persona is not None:
        db.set_profile_gemini_persona(profile_id, persona.strip())
    return {"success": True}


@app.post("/p/{slug}/api/gemini/persona/reset")
async def gemini_persona_reset(slug: str):
    """Reset the Gemini persona to the system default."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    db.set_profile_gemini_persona(profile_id, None)
    return {"success": True, "default": DEFAULT_GEMINI_PERSONA}


@app.post("/p/{slug}/api/gemini/batch")
async def gemini_batch_start(slug: str, request: Request):
    """Start a background Gemini synthetic-label job. Poll ``GET /api/jobs/{job_id}``."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw_limit = body.get("limit", 10)
    try:
        batch_limit = int(raw_limit)
    except (TypeError, ValueError):
        batch_limit = 10
    batch_limit = max(1, min(batch_limit, 100))
    replace_gemini = False
    source = str(body.get("source", "all")).strip().lower()
    entry_type = None
    if source == "lex":
        entry_type = "lex_item"
    elif source == "news":
        entry_type = "feed_item"
    mode = str(body.get("mode", "single")).strip().lower()
    if mode not in ("single", "batch"):
        mode = "single"
    db.merge_profile_training_settings(profile_id, {"gemini_mode": mode})

    job_id = _create_job("gemini_synthetic_batch")
    _update_job(job_id, profile_id=profile_id)

    def _gemini_cancelled() -> bool:
        with _JOB_LOCK:
            j = _JOBS.get(job_id)
            return bool(j and j.get("cancel_requested"))

    t = threading.Thread(
        target=lambda: _run_job(
            job_id,
            lambda cb: run_gemini_synthetic_batch_job(
                profile_id,
                batch_limit=batch_limit,
                entry_type=entry_type,
                replace_gemini=replace_gemini,
                mode=mode,
                progress_cb=cb,
                cancel_check=_gemini_cancelled,
                gemini_job_id=job_id,
            ),
        ),
        daemon=True,
    )
    t.start()
    return {"success": True, "job_id": job_id}


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


@app.get("/p/{slug}/dashboard")
async def dashboard_page(slug: str):
    """Old path: overview is now the Model page."""
    return RedirectResponse("/p/{}/model".format(slug), status_code=302)


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
        all_labels = db.get_all_labeled_entry_keys(profile_id)
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
            logger.warning("Failed to load model page keywords: %s", e)
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
    cfg = ctx["config"]
    cur_kw = int(cfg.get("recipe_top_keywords") or 200)
    sug_kw = suggested_recipe_top_keywords(ctx["label_count"])
    ctx["recipe_top_keywords_suggested"] = sug_kw
    ctx["recipe_top_keywords_bump_hint"] = (
        ctx["label_count"] >= 250 and cur_kw < sug_kw - 5
    )
    ctx["RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX"] = RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX
    ctx["profile_push_blank"] = sync.profile_satellite_blank(profile)
    ctx["profile_push_incomplete"] = sync.profile_satellite_incomplete(profile)
    return templates.TemplateResponse("settings.html", ctx)


# ─── API: Profiles (global) ──────────────────────────────────────────────────

@app.get("/api/profiles")
async def list_profiles():
    return db.get_all_profiles()


@app.post("/api/profiles")
async def create_profile_api(request: Request):
    data = await request.json()
    seismo_url = (data.get("seismo_url") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    if sync.profile_satellite_incomplete({"seismo_url": seismo_url, "api_key": api_key}):
        return JSONResponse(
            {"success": False, "error": sync.INCOMPLETE_SATELLITE_CREDENTIALS_MSG},
            status_code=400,
        )

    display_name = (data.get("display_name") or data.get("name") or "").strip()
    slug = (data.get("slug") or "").strip()
    if not display_name:
        if seismo_url:
            display_name, derived_slug = db.derive_profile_identity_from_push_url(seismo_url)
            if not slug:
                slug = derived_slug
        else:
            display_name = "Workspace"
            if not slug:
                slug = db.slugify(display_name)
    if not slug:
        slug = db.slugify(display_name)
    if db.get_profile_by_slug(slug):
        base, n = slug, 1
        while db.get_profile_by_slug(slug):
            slug = "{}-{}".format(base, n)
            n += 1
    description = (data.get("description") or "").strip()
    profile = db.create_profile(slug, display_name, description, seismo_url, api_key)
    if len(db.get_all_profiles()) == 1:
        db.set_active_profile_id(profile["id"])
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

    merged_url = (existing.get("seismo_url") or "").strip()
    merged_key = (existing.get("api_key") or "").strip()
    if "seismo_url" in data:
        merged_url = (data.get("seismo_url") or "").strip()
    if "api_key" in data:
        merged_key = (data.get("api_key") or "").strip()
    if "seismo_url" in data or "api_key" in data:
        if sync.profile_satellite_incomplete({"seismo_url": merged_url, "api_key": merged_key}):
            return JSONResponse(
                {"success": False, "error": sync.INCOMPLETE_SATELLITE_CREDENTIALS_MSG},
                status_code=400,
            )

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

    # Satellite accent only applies while URL+key are set; clear when switching to
    # mothership-only so the default red returns after disconnect.
    updated = db.get_profile_by_id(profile_id)
    if updated and sync.profile_satellite_blank(updated):
        db.clear_profile_accent_color(profile_id)
        updated = db.get_profile_by_id(profile_id)

    return {"success": True, "profile": updated}


@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile_api(profile_id: int):
    """Set the persistent labeling workspace (Label / Gemini / Push target context)."""
    prof = db.get_profile_by_id(profile_id)
    if not prof:
        raise HTTPException(404, "Profile not found")
    try:
        db.set_active_profile_id(profile_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"success": True, "slug": prof["slug"]}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile_api(profile_id: int):
    existing = db.get_profile_by_id(profile_id)
    if not existing:
        raise HTTPException(404, "Profile not found")
    if existing.get("is_default"):
        return JSONResponse({"success": False,
                             "error": "Cannot delete the default profile."}, 400)
    db.delete_profile(profile_id)
    db.clear_active_profile_if_deleted(profile_id)
    # Client was on /p/{slug}/settings; reload would 404. Send a surviving profile slug.
    fallback = db.get_default_profile()
    if not fallback:
        others = db.get_all_profiles()
        fallback = others[0] if others else None
    redirect_slug = fallback["slug"] if fallback else None
    return {"success": True, "redirect_slug": redirect_slug}


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
    db.set_active_profile_id(profile["id"])
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
        slug = ""
        if result.get("profile_id"):
            db.set_active_profile_id(result["profile_id"])
            row = db.get_profile_by_id(result["profile_id"])
            slug = row["slug"] if row else ""
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
    # Labels are pushed in batch by Sync → Push (scores + recipe + labels) or
    # POST /p/{slug}/api/sync/labels — not on every save (avoids sync_log spam
    # and one HTTP round-trip per card).
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


async def _run_sync_pull_handlers(
    full: bool,
    background: bool,
    profile: Optional[Dict],
):
    """Shared executor path for mothership sync (pull entries + merge labels)."""
    import asyncio
    if not profile:
        raise HTTPException(
            400,
            "Sync pull requires a profile (use profile_slug or activate a profile in Settings).",
        )
    if background:
        job_type = "sync_pull_full" if full else "sync_pull"
        job_id = _create_job(job_type)
        t = threading.Thread(
            target=lambda: _run_job(
                job_id,
                lambda cb: _sync_pull_impl(full=full, progress_cb=cb, profile=profile),
            ),
            daemon=True,
        )
        t.start()
        return {"success": True, "job_id": job_id}
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _sync_pull_impl(full=full, profile=profile),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/sync/pull")
async def sync_pull(
    full: bool = False,
    background: bool = False,
    profile_slug: Optional[str] = None,
):
    """Pull entries using global Settings URL (mothership).

    Resolves which profile receives merged labels:

    - If ``profile_slug`` is given, that profile is used.
    - Otherwise the **active workspace** profile from Settings is used (same as
      the Label / Push UI). There is no longer a silent default to profile id 1.
    """
    profile = None
    if profile_slug and profile_slug.strip():
        profile = db.get_profile_by_slug(profile_slug.strip())
        if not profile:
            raise HTTPException(
                404,
                "Unknown profile slug '{}'".format(profile_slug.strip()),
            )
    else:
        profile = db.get_active_profile()
        if not profile:
            raise HTTPException(
                400,
                "Specify profile_slug on the request, or activate a profile in Settings.",
            )
    return await _run_sync_pull_handlers(full, background, profile)


@app.post("/p/{slug}/api/sync/pull")
async def sync_pull_for_profile(slug: str, full: bool = False, background: bool = False):
    """Pull from mothership (global URL); merge pulled labels into this profile."""
    profile = _get_profile_or_404(slug)
    return await _run_sync_pull_handlers(full, background, profile)


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
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    out = dict(job)
    pid = job.get("profile_id")
    if pid is not None and job.get("job_type") == "gemini_synthetic_batch":
        out["pending_gemini_count"] = db.count_pending_gemini_labels(int(pid), job_id)
    return out


@app.post("/p/{slug}/api/gemini/batch/{job_id}/accept")
async def gemini_batch_accept(slug: str, job_id: str):
    """Confirm pending Gemini labels from a batch job (counts toward Train / Push)."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    n = db.confirm_gemini_pending_labels(profile_id, job_id)
    return {"success": True, "confirmed": n}


@app.post("/p/{slug}/api/gemini/batch/{job_id}/discard")
async def gemini_batch_discard(slug: str, job_id: str):
    """Remove pending Gemini labels from a batch job."""
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    n = db.discard_gemini_pending_labels(profile_id, job_id)
    return {"success": True, "removed": n}


@app.post("/api/jobs/{job_id}/cancel")
async def job_cancel(job_id: str):
    """Request cancellation of a background job. The worker stops after the
    current item or API batch; work already written (e.g. labels) is kept."""
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        st = (job.get("status") or "").strip()
        if st not in ("queued", "running"):
            return {
                "success": False,
                "detail": "Job is not running (status: %s)" % st,
            }
        job["cancel_requested"] = True
    return {"success": True}


@app.post("/p/{slug}/api/sync/labels")
async def sync_labels(slug: str):
    profile = _get_profile_or_404(slug)
    profile_id = profile["id"]
    pushed = {}
    pulled = 0
    try:
        pushed = sync.push_labels(profile_id=profile_id, profile=profile)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, "Failed to push labels: {}".format(e))
    pull_error = None
    try:
        pulled = sync.pull_labels(profile_id=profile_id, profile=profile)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.warning("Label pull failed: %s", e)
        pull_error = str(e)
    result = {"success": True, "pushed": pushed, "labels_imported": pulled}
    if pull_error:
        result["pull_error"] = pull_error
    return result


def _satellite_sync_test_payload(profile: dict, data: dict) -> dict:
    """Test Seismo using profile push target (overlay ``data`` onto saved profile)."""
    overlay = dict(profile)
    if "seismo_url" in data:
        overlay["seismo_url"] = (data.get("seismo_url") or "").strip()
    if "api_key" in data:
        overlay["api_key"] = (data.get("api_key") or "").strip()
    if sync.profile_satellite_blank(overlay):
        return {
            "success": False,
            "message": (
                "Enter this profile's push target URL and API key to test that Seismo. "
                "When both are blank here, Magnitu uses global mothership for push and "
                "label pull — use 'Test mothership' under Mothership Connection instead."
            ),
        }
    try:
        target = sync._profile_target(overlay)
    except ValueError as e:
        return {"success": False, "message": str(e)}
    ok, msg, status_payload = sync.test_connection(seismo_target=target)
    if not ok:
        return {"success": False, "message": msg}
    msg = "This profile's push target — {}".format(msg)
    sync.maybe_profile_accent_from_status(status_payload, profile["id"])
    label_ok, label_msg = sync.verify_seismo_endpoints(seismo_target=target)
    if not label_ok:
        return {"success": True, "message": msg, "warning": label_msg}
    return {"success": True, "message": msg}


@app.post("/api/sync/test-mothership")
async def sync_test_mothership(request: Request):
    """Test mothership URL + API key from JSON body (unsaved form values OK)."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    cfg = get_config()
    url = (data.get("seismo_url") or "").strip()
    key = (data.get("api_key") or "").strip()
    target = {
        "seismo_url": url or cfg["seismo_url"],
        "api_key": key or cfg["api_key"],
    }
    ok, msg, status_payload = sync.test_connection(seismo_target=target)
    if not ok:
        return {"success": False, "message": msg}
    msg = "Mothership — {}".format(msg)
    label_ok, label_msg = sync.verify_seismo_endpoints(seismo_target=target)
    if not label_ok:
        return {"success": True, "message": msg, "warning": label_msg}
    return {"success": True, "message": msg}


@app.post("/p/{slug}/api/sync/test-satellite")
async def sync_test_satellite(slug: str, request: Request):
    """Test satellite Seismo for this profile (optional JSON overlay for unsaved inputs)."""
    profile = _get_profile_or_404(slug)
    try:
        data = await request.json()
    except Exception:
        data = {}
    return _satellite_sync_test_payload(profile, data)


@app.get("/p/{slug}/api/sync/test")
async def sync_test(slug: str):
    """Backward-compatible satellite test using saved profile credentials only."""
    profile = _get_profile_or_404(slug)
    return _satellite_sync_test_payload(profile, {})


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
    """Update global configuration and/or per-profile training overrides."""
    data = await request.json()
    config = get_config()
    old_transformer_name = config.get("transformer_model_name", "")
    old_use_gpu = config.get("use_gpu", True)
    old_legal_patterns = list(config.get("legal_signal_patterns") or [])

    pid = None
    if data.get("profile_id") is not None:
        try:
            pid = int(data["profile_id"])
        except (TypeError, ValueError):
            pid = None
        if pid is not None and not db.get_profile_by_id(pid):
            pid = None

    training_in_payload = {
        k: data[k] for k in PROFILE_TRAINING_SETTINGS_KEYS if k in data
    }
    profile_training_saved = False
    if pid is not None and training_in_payload:
        db.merge_profile_training_settings(pid, training_in_payload)
        profile_training_saved = True

    # Always-global keys
    for key in ["seismo_url", "api_key", "model_architecture",
                "transformer_model_name", "use_gpu", "gemini_api_key", "gemini_model"]:
        if key in data:
            config[key] = data[key]

    # Training keys: update global file only when not saved on a profile in this request
    if not profile_training_saved:
        for key in PROFILE_TRAINING_SETTINGS_KEYS:
            if key in data:
                config[key] = data[key]

    if "legal_signal_patterns" in data:
        config["legal_signal_patterns"] = data["legal_signal_patterns"]

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

    out_cfg = db.get_effective_config(pid) if pid else get_config()
    return {"success": True, "config": out_cfg}


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
