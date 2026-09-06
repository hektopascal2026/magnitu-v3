#!/usr/bin/env python3
"""
Headless CLI worker for the VPS Magnitu ML window.
Executes sync -> embed -> train (gated) -> promote (strict) -> score/push
-> distill/recipe/vault (post-promote, memory-isolated).
Expects SEISMO_DESKS_JSON environment variable with a JSON list of desks.
"""
from __future__ import annotations

import os
import sys
import json
import logging
import math
import subprocess
import gc
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Make sure imports work if run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import sync
import pipeline
from config import get_config
from magnitu.time_display import format_seismo_timestamp
from model_manager import export_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMOTE_MARGIN = 0.01
# Holdout is often ~15–25 rows and precision_at_30 is really precision@min(30,n).
# One flipped relevant in the top-k moves the metric by ~0.04–0.07, so a hard
# ±0.01 bar on p@30 alone rejects models that clearly improve macro-F1 after
# smart-queue labeling. Allow this much ranking noise when F1 clearly wins.
PROMOTE_RANKING_SLACK = 0.05
# One relevant item at the top of a ~20-row holdout (the metric's own
# quantization step) — the evidence bar for "big win" sits on metric
# granularity, not on a fitted number.
PROMOTE_BIG_P30_WIN = 0.05
# ≈ 2–3 flipped tail rows; the catastrophe line beyond which the recipe's
# global quality is no longer credible. Below it, a dip is noise plus a
# mission-acceptable trade.
F1_HARD_DROP_LIMIT = 0.10
# With 1–3 holdout leads, lead_recall_at_30 moves in steps of 0.33–1.0,
# so this functions as: no lead the old model caught may drop out of the
# top-30. That is the doctrine for a threat-hunting system, not a tuning artifact.
LEAD_RECALL_SLACK = 0.10

# ── Recent-items promote gate ────────────────────────────────────────────
# The gate scores both old and new models on the most recent N labeled items
# (by entry fetched_at). This tests "will the journalist's next day be worse?"
# rather than "is F1 higher on a random slice of old data?"
# Number of recent labeled items to evaluate on. Capped by available labels.
GATE_N_RECENT = 100
# Minimum recent items for the gate to be meaningful. Below this, promote
# (not enough data to reject a model that might be better).
GATE_MIN_RECENT = 10
# p@30 slack: one item flip in top-30 on a 100-row set ≈ 0.033.
# Allow this much regression before rejecting — it's noise, not degradation.
GATE_P30_SLACK = 0.05
# Lead recall slack: with 5-10 leads, one flip = 0.10-0.20 step.
# Allow one lead to drop before rejecting — quantization, not regression.
GATE_LEAD_RECALL_SLACK = 0.10


def _distill_recipe_in_subprocess(profile_id: int) -> int:
    """
    Run recipe distillation in a child process after the parent has released
    heavy models. Peak RSS is roughly the child alone (not parent+child).
    If the child is OOM-killed (exit 137), the parent can still finish the
    window report and continue other desks.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = app_dir if not prior else f"{app_dir}{os.pathsep}{prior}"
    code = (
        "import distiller, sys; "
        f"r = distiller.distill_recipe(profile_id={int(profile_id)}); "
        "sys.exit(0 if r is not None else 2)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=app_dir,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _is_oom_kill_returncode(rc: int) -> bool:
    """True for cgroup/OOM SIGKILL. Python reports -9; shells often report 137."""
    return rc in (137, -9)


def _recipe_quality_value(recipe: Optional[dict]) -> float:
    if not isinstance(recipe, dict):
        return 0.0
    metrics = recipe.get("metrics") if isinstance(recipe.get("metrics"), dict) else {}
    raw = metrics.get("recipe_quality", recipe.get("recipe_quality", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _recipe_quality_floor(cfg: Optional[dict] = None) -> float:
    """0 disables the holdback. Default 0.30 (provisional)."""
    settings = cfg if isinstance(cfg, dict) else get_config()
    raw = settings.get("recipe_quality_floor", 0.30)
    try:
        floor = float(raw)
    except (TypeError, ValueError):
        floor = 0.30
    if floor < 0.0:
        return 0.0
    return floor


def _hold_recipe_below_floor(recipe: dict, profile_id: int, cfg: Optional[dict] = None) -> bool:
    """True when the new recipe must not be pushed (desk keeps previous)."""
    floor = _recipe_quality_floor(cfg)
    if floor <= 0.0:
        return False
    quality = _recipe_quality_value(recipe)
    if quality >= floor:
        return False
    db.log_sync(
        "recipe_quality_below_floor",
        1,
        "quality={:.4f} floor={:.4f}; previous recipe kept".format(quality, floor),
        profile_id=profile_id,
    )
    return True


def _post_promote_recipe_and_vault(
    profile_id: int,
    url: str,
    prof: dict,
    vault_password: str,
) -> bool:
    """Distill + push recipe + vault upload. Returns False on soft failure."""
    ok = True
    # Free E5 (and friends) in the parent before the distill child starts.
    try:
        pipeline.release_embedder()
    except Exception as e:
        logger.warning("release_embedder before distill failed: %s", e)
    gc.collect()

    logger.info("Distilling recipe for %s (subprocess)...", url)
    rc = _distill_recipe_in_subprocess(profile_id)
    if _is_oom_kill_returncode(rc):
        logger.error("Recipe distillation OOM-killed (rc=%s) for %s", rc, url)
        ok = False
    elif rc != 0:
        logger.error("Recipe distillation failed for %s (exit %s)", url, rc)
        ok = False

    current_model = db.get_active_model(profile_id)
    if current_model and current_model.get("recipe_path") and os.path.exists(current_model["recipe_path"]):
        try:
            with open(current_model["recipe_path"], "r") as rf:
                recipe = json.load(rf)
            if _hold_recipe_below_floor(recipe, profile_id):
                logger.warning(
                    "Recipe quality below floor for %s — previous Seismo recipe kept.",
                    url,
                )
            else:
                sync.push_recipe(recipe, profile=prof)
                logger.info("Recipe pushed for %s.", url)
        except Exception as e:
            logger.error("Failed to push recipe: %s", e)
            ok = False

    try:
        logger.info("Exporting model to vault for %s...", url)
        model_zip = export_model(profile_id=profile_id)
        sync.vault_upload(vault_password=vault_password, package_path=model_zip, overwrite=True)
        logger.info("Model uploaded to mothership vault.")
    except Exception as e:
        logger.error("Failed to upload model to vault: %s", e)
        ok = False
    return ok


def enforce_embedding_store_cap(max_bytes=5 * 1024 * 1024 * 1024):
    """
    Keep fingerprints for labeled rows + still-needed (sync horizon).
    Prune older unlabeled embeddings until under cap. No VACUUM to keep it fast.
    """
    conn = db.get_db()

    res = conn.execute("SELECT SUM(LENGTH(embedding)) FROM entries WHERE embedding IS NOT NULL").fetchone()
    total_bytes = res[0] or 0

    if total_bytes <= max_bytes:
        conn.close()
        return

    logger.info("Logical embedding size %d exceeds cap %d, pruning...", total_bytes, max_bytes)

    rows = conn.execute("""
        SELECT id, LENGTH(embedding) as emb_len FROM entries
        WHERE embedding IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM labels
              WHERE labels.entry_type = entries.entry_type
                AND labels.entry_id = entries.entry_id
          )
        ORDER BY fetched_at ASC
    """).fetchall()

    bytes_to_remove = total_bytes - max_bytes + (500 * 1024 * 1024)
    ids_to_prune = []
    removed = 0
    for r in rows:
        ids_to_prune.append(r["id"])
        removed += r["emb_len"]
        if removed >= bytes_to_remove:
            break

    if ids_to_prune:
        chunk_size = 500
        for i in range(0, len(ids_to_prune), chunk_size):
            chunk = ids_to_prune[i:i + chunk_size]
            conn.execute(
                f"UPDATE entries SET embedding = NULL WHERE id IN ({','.join('?' * len(chunk))})",
                chunk
            )
        conn.commit()
        logger.info("Pruned %d embeddings.", len(ids_to_prune))
    conn.close()


def _model_meta_for_push(profile_id: int, model_row: Dict) -> Optional[Dict]:
    """Build Seismo ``model_meta`` so desk ``model_trained_at`` advances on promote."""
    profile_info = db.get_model_profile(profile_id)
    if not profile_info or not model_row:
        return None
    return {
        "model_name": profile_info.get("model_name", ""),
        "model_uuid": profile_info.get("model_uuid", ""),
        "model_description": profile_info.get("description", ""),
        "model_version": model_row.get("version"),
        "model_trained_at": format_seismo_timestamp(model_row.get("trained_at", "")),
        "accuracy": model_row.get("accuracy", 0.0),
        "f1_score": model_row.get("f1_score", 0.0),
        "label_count": model_row.get("label_count", 0),
        "architecture": model_row.get("architecture", "tfidf"),
    }


def _label_counts(profile_id: int, trained_at: Optional[str] = None) -> Dict[str, int]:
    """Total / trainable (joined) / orphan / since-last-train counts for a profile."""
    conn = db.get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE profile_id=?",
        (profile_id,),
    ).fetchone()[0]
    trainable = conn.execute(
        """
        SELECT COUNT(*) FROM labels l
        JOIN entries e ON e.entry_type = l.entry_type AND e.entry_id = l.entry_id
        WHERE l.profile_id=?
        """,
        (profile_id,),
    ).fetchone()[0]
    if trained_at:
        since_train = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE profile_id=? AND updated_at > ?",
            (profile_id, trained_at),
        ).fetchone()[0]
    else:
        since_train = int(total)
    conn.close()
    return {
        "labels_total": int(total),
        "labels_trainable": int(trainable),
        "labels_orphan": int(total) - int(trainable),
        "labels_since_train": int(since_train),
    }


def _lead_recall_ok(old: Optional[dict], new: dict) -> bool:
    """No caught lead may drop out of the operator-visible top-30.

    Skipped when either side lacks lead_recall_at_30 or the old value is 0.0
    (legacy default / no leads in that holdout) -- a missing metric must not
    veto, and an old holdout without leads says nothing about recall.
    """
    old_lr = old.get("lead_recall_at_30") if old else None
    new_lr = new.get("lead_recall_at_30")
    if not old_lr or new_lr is None:
        return True
    return float(new_lr) >= float(old_lr) - LEAD_RECALL_SLACK


def evaluate_model_update(
    old_metrics: Optional[dict],
    new_metrics: dict,
) -> bool:
    """Promote gate: mission first, macro-F1 as a catastrophe breaker.

    (1) Cold start promotes.
    (2) Lead-recall guard applies to every promote path: a promotion that
        craters lead_recall_at_30 is vetoed even if metrics improved.
    (3) Big top-of-feed win: p@30 up >= PROMOTE_BIG_P30_WIN (~one relevant
        item on a ~20-row holdout) tolerates an F1 dip up to
        F1_HARD_DROP_LIMIT (~2-3 tail rows; beyond that we don't trust the
        distilled recipe).
    (4) Legacy two-path gate unchanged (small ranking win with strict F1
        guard, or F1 win with ranking slack).
    """
    if not old_metrics:
        return True
    old_p30 = float(old_metrics.get("precision_at_30") or 0.0)
    new_p30 = float(new_metrics.get("precision_at_30") or 0.0)
    old_f1 = float(old_metrics.get("f1_score") or 0.0)
    new_f1 = float(new_metrics.get("f1_score") or 0.0)
    p30_gain = new_p30 - old_p30
    f1_gain = new_f1 - old_f1

    if not _lead_recall_ok(old_metrics, new_metrics):
        return False

    # Big top-of-feed win buys bounded F1 tolerance.
    if p30_gain >= PROMOTE_BIG_P30_WIN and f1_gain >= -F1_HARD_DROP_LIMIT:
        return True

    # Legacy gate, byte-identical to today.
    p30_up = p30_gain >= PROMOTE_MARGIN
    f1_up = f1_gain >= PROMOTE_MARGIN
    f1_ok = f1_gain >= -PROMOTE_MARGIN
    p30_not_collapsed = p30_gain >= -PROMOTE_RANKING_SLACK
    return (p30_up and f1_ok) or (f1_up and p30_not_collapsed)


def _should_promote(
    current_model: Optional[dict],
    res: dict,
) -> bool:
    """Delegate to evaluate_model_update (promote gate)."""
    return evaluate_model_update(current_model, res)


def evaluate_recent_gate(
    old_recent: Optional[dict],
    new_recent: dict,
) -> bool:
    """Conservative promote gate on recent production traffic.

    Tests "will the journalist's next day be worse?" by comparing old and new
    models on the most recent N labeled items. Only ranking metrics matter —
    F1 on a small set is noise.

    Rules (all must pass):
    1. Cold start (no old model): promote.
    2. Too few recent items to evaluate: promote (can't reject without evidence).
    3. Lead recall must not drop beyond GATE_LEAD_RECALL_SLACK.
    4. p@30 must not drop beyond GATE_P30_SLACK.
    """
    if not old_recent or not old_recent.get("success"):
        return True  # cold start or old model couldn't be evaluated
    if not new_recent.get("success"):
        return False  # new model couldn't be evaluated on recent set — don't promote blind

    n = int(new_recent.get("n_recent") or 0)
    if n < GATE_MIN_RECENT:
        logger.info(
            "Recent gate: only %d recent items (< %d), promoting without check.",
            n, GATE_MIN_RECENT,
        )
        return True

    old_p30 = float(old_recent.get("precision_at_30") or 0.0)
    new_p30 = float(new_recent.get("precision_at_30") or 0.0)
    old_lr = float(old_recent.get("lead_recall_at_30") or 0.0)
    new_lr = float(new_recent.get("lead_recall_at_30") or 0.0)

    # Lead recall guard: no caught lead may drop out of top-30.
    # Skip when old has no leads (lr=0) — nothing to lose.
    if old_lr > 0 and new_lr < old_lr - GATE_LEAD_RECALL_SLACK:
        logger.info(
            "Recent gate REJECT: lead_recall %.3f→%.3f (slack %.3f).",
            old_lr, new_lr, GATE_LEAD_RECALL_SLACK,
        )
        return False

    # p@30 guard: top of queue must not get worse.
    if new_p30 < old_p30 - GATE_P30_SLACK:
        logger.info(
            "Recent gate REJECT: p@30 %.3f→%.3f (slack %.3f).",
            old_p30, new_p30, GATE_P30_SLACK,
        )
        return False

    return True


def _train_reject_log(current_model: Optional[dict], res: dict) -> str:
    """Human-readable reject reason for the window log / sync_log."""
    if current_model and not _lead_recall_ok(current_model, res):
        return (
            "Promote gate rejected: lead_recall_at_30 crater "
            "({:.3f}→{:.3f}). Keeping older model.".format(
                float(current_model.get("lead_recall_at_30") or 0.0),
                float(res.get("lead_recall_at_30") or 0.0),
            )
        )
    return "Model rejected. Keeping older model."


def _embed_pending_until_done(max_entries: int = 0) -> None:
    """Embed pending entries in batches until done or ``max_entries`` reached.

    ``max_entries=0`` means no cap (full window behaviour). Score-only windows
    pass a cap so a large backlog cannot eat the entire 300s budget before
    scoring + pushing. Remaining entries are picked up by the next tick.
    """
    embedded = 0
    while True:
        if max_entries > 0 and embedded >= max_entries:
            logger.info(
                "Embedding cap reached (%d entries); remaining deferred to next tick.",
                embedded,
            )
            break
        batch_limit = 1000
        if max_entries > 0:
            batch_limit = min(1000, max_entries - embedded)
        processed = sync._compute_pending_embeddings(limit=batch_limit)
        if not processed:
            break
        embedded += processed


def _should_train_desk(
    current_model: Optional[dict],
    labeled_count: int,
    labels_since_train: int,
    force_retrain: bool,
) -> bool:
    if force_retrain:
        return True
    if not current_model:
        return labeled_count >= 15
    trained_at = datetime.fromisoformat(current_model["trained_at"].replace("Z", ""))
    days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - trained_at).days
    return labels_since_train >= 15 or days_since >= 14


def _sort_prepared_desks(prepared: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Train candidates with the most unbaked labels first; slug ASC for ties."""
    return sorted(
        prepared,
        key=lambda item: (
            -(1 if item.get("do_train") else 0),
            -int(item.get("report", {}).get("labels_since_train") or 0),
            str(item.get("slug") or ""),
        ),
    )


def _score_push_days(cfg: Optional[dict] = None) -> int:
    """Score push lookback days from config."""
    if cfg is None:
        cfg = get_config()
    try:
        raw_days = cfg.get("score_push_days", 14)
        days = 14 if raw_days is None else int(raw_days)
    except (TypeError, ValueError):
        days = 14
    return max(1, days)


def main():
    cfg = get_config()
    mothership_url = cfg.get("seismo_url")
    mothership_key = cfg.get("api_key")
    vault_password = os.environ.get("MAGNITU_VAULT_PASSWORD") or cfg.get("vault_password")

    if not mothership_url or not mothership_key:
        logger.error("Mothership seismo_url or api_key missing in global config. Exiting.")
        return 1

    if not vault_password:
        logger.error("Vault password missing (set MAGNITU_VAULT_PASSWORD or vault_password config). Exiting.")
        return 1

    desks_env = os.environ.get("SEISMO_DESKS_JSON")
    if not desks_env:
        logger.info("No SEISMO_DESKS_JSON configured. Exiting.")
        return 0

    try:
        desks = json.loads(desks_env)
    except json.JSONDecodeError:
        logger.error("Failed to parse SEISMO_DESKS_JSON")
        return 1

    if not desks:
        logger.info("No desks to process. Exiting.")
        return 0

    enforce_embedding_store_cap()

    force_retrain = os.environ.get("MAGNITU_ML_FORCE_RETRAIN") == "1"
    full_entry_drain = os.environ.get("MAGNITU_ML_FULL_ENTRY_DRAIN") == "1"
    score_only = os.environ.get("MAGNITU_ML_SCORE_ONLY") == "1"
    has_errors = False
    desk_reports: List[Dict[str, Any]] = []

    if score_only:
        logger.info("Score-only mode (MAGNITU_ML_SCORE_ONLY=1): skip labels/train/distill")

    # Score-only windows have a 300s budget. E5 on CPU does ~2.5 chunks/s,
    # so 400 entries (≈723 chunks) still eats ~290s — the entire budget.
    # Cap to 200 entries per score-only tick so embedding takes ~145s,
    # leaving ~155s for scoring + pushing. The backlog drains across
    # multiple 15-min cycles. Full windows pass max_entries=0 (no cap).
    embed_cap = 200 if score_only else 0

    # 0) Entry sync once per window (mothership), then embed.
    try:
        if full_entry_drain:
            logger.info("Full entry drain (MAGNITU_ML_FULL_ENTRY_DRAIN=1)")
            sync.pull_all_entry_types(drain=True, compute_embeddings=False)
        else:
            watermarks = sync.entry_store_watermarks()
            logger.info("Incremental entry drain from watermarks: %s", watermarks)
            sync.pull_all_entry_types(
                drain=True,
                compute_embeddings=False,
                per_type_since=watermarks,
            )
        _embed_pending_until_done(max_entries=embed_cap)
    except Exception as e:
        logger.error("Failed shared entry pull/embed: %s", e)
        has_errors = True

    # 1) Pull labels for every desk first, then train highest unbaked backlog.
    prepared: List[Dict[str, Any]] = []
    for desk in desks:
        url = desk.get("seismo_url")
        api_key = desk.get("api_key")

        if not url or not api_key:
            logger.warning("Desk missing url or api_key, skipping: %s", desk)
            has_errors = True
            continue

        # Prefer registry slug (e.g. "sicherheit"); never slugify the full URL
        # (that yields https-seismo-live-… and misses desktop profiles).
        slug = (desk.get("slug") or "").strip()
        display_name, derived_slug = db.derive_profile_identity_from_push_url(url)
        if slug:
            slug = db.slugify(slug)
        else:
            slug = derived_slug

        logger.info("Preparing desk: %s (slug=%s)", url, slug)

        prof = db.get_profile_by_slug(slug)
        if not prof:
            prof = db.create_profile(
                slug=slug,
                display_name=display_name or slug,
                seismo_url=url,
                api_key=api_key,
            )
        else:
            conn = db.get_db()
            conn.execute(
                "UPDATE profiles SET seismo_url=?, api_key=?, display_name=COALESCE(NULLIF(display_name,''), ?) WHERE id=?",
                (url, api_key, display_name or slug, prof["id"]),
            )
            conn.commit()
            conn.close()
            prof = db.get_profile_by_slug(slug)

        profile_id = prof["id"]
        report: Dict[str, Any] = {
            "slug": slug,
            "labels_pulled_new": 0,
            "labels_since_train": 0,
            "labels_total": 0,
            "labels_trainable": 0,
            "labels_orphan": 0,
            "trained": False,
            "promoted": False,
            "train_rejected": False,
            "p30_old": None,
            "p30_new": None,
            "f1_old": None,
            "f1_new": None,
            "active_version": None,
            "candidate_version": None,
        }

        if score_only:
            # Score-only mode: skip label pull / orphan backfill / training.
            # Just score and push with the existing active model.
            # Do NOT append to desk_reports here — the processing loop
            # below appends the report after scoring.
            current_model = db.get_active_model(profile_id)
            if current_model:
                report["active_version"] = current_model.get("version")
            prepared.append({
                "url": url,
                "slug": slug,
                "prof": prof,
                "profile_id": profile_id,
                "report": report,
                "current_model": current_model,
                "do_train": False,
            })
            continue

        try:
            pulled = sync.pull_labels(profile_id=profile_id, profile=prof)
            report["labels_pulled_new"] = int(pulled)
        except Exception as e:
            logger.error("Failed to pull labels for %s: %s", url, e)
            has_errors = True
            desk_reports.append(report)
            continue

        try:
            orphan_before, fetched = sync.backfill_orphan_label_entries(profile_id)
            if orphan_before:
                logger.info(
                    "Orphan label backfill for %s: %d missing, fetched %d",
                    slug,
                    orphan_before,
                    fetched,
                )
                _embed_pending_until_done()
        except Exception as e:
            logger.error("Orphan backfill failed for %s: %s", url, e)
            has_errors = True

        labeled = db.get_all_labels(profile_id)
        current_model = db.get_active_model(profile_id)
        trained_at_s = current_model["trained_at"] if current_model else None
        report.update(_label_counts(profile_id, trained_at_s))
        if current_model:
            report["active_version"] = current_model.get("version")
            report["p30_old"] = current_model.get("precision_at_30")
            report["f1_old"] = current_model.get("f1_score")

        do_train = _should_train_desk(
            current_model,
            len(labeled),
            int(report["labels_since_train"]),
            force_retrain,
        )
        prepared.append({
            "url": url,
            "slug": slug,
            "prof": prof,
            "profile_id": profile_id,
            "report": report,
            "current_model": current_model,
            "do_train": do_train,
        })

    prepared = _sort_prepared_desks(prepared)
    logger.info(
        "Desk train order (unbaked labels first): %s",
        ", ".join(
            f"{item['slug']}(since={item['report'].get('labels_since_train', 0)},"
            f"train={item['do_train']})"
            for item in prepared
        ) or "(none)",
    )

    # 2) Train / score-push / distill in priority order
    for item in prepared:
        url = item["url"]
        slug = item["slug"]
        prof = item["prof"]
        profile_id = item["profile_id"]
        report = item["report"]
        current_model = item["current_model"]
        do_train = item["do_train"]

        logger.info(
            "Processing desk: %s (labels_since_train=%s, do_train=%s)",
            url,
            report.get("labels_since_train"),
            do_train,
        )

        promoted_this_desk = False
        if do_train:
            logger.info("Gate passed: training model for %s...", url)
            report["trained"] = True
            res = pipeline.train(profile_id=profile_id, activate=False)

            if not res.get("success"):
                logger.warning("Training failed: %s", res.get("error"))
                has_errors = True
            else:
                report["candidate_version"] = res.get("version")
                report["p30_new"] = res.get("precision_at_30")
                report["f1_new"] = res.get("f1_score")

                # ── Recent-items promote gate ──
                # Score both old and new on the most recent N labeled items.
                # This tests "will the journalist's next day be worse?"
                # instead of "is F1 higher on a random slice of old data?"
                new_recent = pipeline.evaluate_on_recent(
                    res, profile_id=profile_id, n_recent=GATE_N_RECENT,
                )
                old_recent = None
                if current_model:
                    old_recent = pipeline.evaluate_on_recent(
                        current_model, profile_id=profile_id, n_recent=GATE_N_RECENT,
                    )

                if new_recent.get("success"):
                    report["p30_new_recent"] = new_recent.get("precision_at_30")
                    report["lr30_new_recent"] = new_recent.get("lead_recall_at_30")
                    report["n_recent"] = new_recent.get("n_recent")
                if old_recent and old_recent.get("success"):
                    report["p30_old_recent"] = old_recent.get("precision_at_30")
                    report["lr30_old_recent"] = old_recent.get("lead_recall_at_30")

                promoted = evaluate_recent_gate(old_recent, new_recent)
                if promoted and not current_model:
                    logger.info("Cold start promote.")
                elif promoted:
                    old_p30 = float(report.get("p30_old_recent") or 0.0)
                    new_p30 = float(report.get("p30_new_recent") or 0.0)
                    old_lr = float(report.get("lr30_old_recent") or 0.0)
                    new_lr = float(report.get("lr30_new_recent") or 0.0)
                    n = int(report.get("n_recent") or 0)
                    logger.info(
                        "Recent gate passed (n=%d): p@30 %.3f→%.3f, "
                        "lead_recall %.3f→%.3f.",
                        n, old_p30, new_p30, old_lr, new_lr,
                    )

                if promoted:
                    report["promoted"] = True
                    promoted_this_desk = True
                    logger.info("Model promoted! Activating version %s", res["version"])
                    conn = db.get_db()
                    conn.execute("UPDATE models SET is_active = 0 WHERE profile_id = ?", (profile_id,))
                    conn.execute(
                        "UPDATE models SET is_active = 1 WHERE profile_id = ? AND version = ?",
                        (profile_id, res["version"]),
                    )
                    conn.commit()
                    conn.close()
                    current_model = db.get_active_model(profile_id)
                    if current_model:
                        report["active_version"] = current_model.get("version")

                else:
                    report["train_rejected"] = True
                    old_p30 = float(report.get("p30_old_recent") or 0.0)
                    new_p30 = float(report.get("p30_new_recent") or 0.0)
                    old_lr = float(report.get("lr30_old_recent") or 0.0)
                    new_lr = float(report.get("lr30_new_recent") or 0.0)
                    reject_msg = (
                        "Recent gate rejected v{} on {} recent items: "
                        "p@30 {:.3f}→{:.3f}, lead_recall {:.3f}→{:.3f}. "
                        "Keeping older model.".format(
                            res["version"],
                            int(report.get("n_recent") or 0),
                            old_p30, new_p30, old_lr, new_lr,
                        )
                    )
                    logger.info(reject_msg)
                    db.log_sync(
                        "train_rejected",
                        1,
                        reject_msg,
                        profile_id=profile_id,
                    )

        # Score + push BEFORE distill so desk model_meta advances even if
        # recipe distillation is later OOM-killed under MemoryMax.
        current_model = db.get_active_model(profile_id)
        if current_model:
            logger.info("Scoring and pushing recent entries for %s...", url)
            try:
                push_days = _score_push_days()
                recent_entries = db.get_recent_entries(
                    days=push_days, include_embedding=False
                )
                if recent_entries:
                    scores = pipeline.score_entries(recent_entries, profile_id=profile_id)
                    if scores:
                        model_meta = _model_meta_for_push(profile_id, current_model)
                        sync.push_scores(
                            scores,
                            model_version=current_model["version"],
                            model_meta=model_meta,
                            profile=prof,
                        )
                        logger.info("Pushed %d absolute scores.", len(scores))
            except Exception as e:
                logger.error("Error during score push: %s", e)
                has_errors = True

        if promoted_this_desk:
            if not _post_promote_recipe_and_vault(
                profile_id, url, prof, vault_password,
            ):
                has_errors = True

        active_after = db.get_active_model(profile_id)
        report.update(
            _label_counts(
                profile_id,
                active_after["trained_at"] if active_after else None,
            )
        )
        desk_reports.append(report)

    # 5) Window report → Seismo Diagnostics
    try:
        payload = {
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_status": "ok",
            "desks": desk_reports,
        }
        sync.post_ml_window_report(payload)
        logger.info("Posted ML window report (%d desks).", len(desk_reports))
    except Exception as e:
        logger.error("Failed to post ML window report: %s", e)
        has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
