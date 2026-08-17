#!/usr/bin/env python3
"""
Headless CLI worker for the VPS Magnitu ML window.
Executes sync -> embed -> train (gated) -> promote (strict) -> score/push
-> distill/recipe/vault (post-promote, memory-isolated).
Expects SEISMO_DESKS_JSON environment variable with a JSON list of desks.
"""
import os
import sys
import json
import logging
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


def _rank_normalize_push_scores(scores: List[Dict]) -> List[Dict]:
    """Replace relevance_score with percentile rank (ties share mean rank).
    Monotone transform: Seismo only sorts/thresholds, so ordering is preserved
    while spreading scores over (0, 1] and removing the ~0.5 composite attractor.
    """
    n = len(scores)
    if n < 2:
        return scores
    order = sorted(range(n), key=lambda i: scores[i]["relevance_score"])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while (
            j + 1 < n
            and scores[order[j + 1]]["relevance_score"]
            == scores[order[i]]["relevance_score"]
        ):
            j += 1
        mean_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    for idx, row in enumerate(scores):
        row["relevance_score"] = round(ranks[idx] / n, 4)
    return scores


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


def evaluate_model_update(old_metrics: Optional[dict], new_metrics: dict) -> bool:
    """Promote gate v2: mission first, macro-F1 as a catastrophe breaker.

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

    # NEW: big top-of-feed win buys bounded F1 tolerance.
    if p30_gain >= PROMOTE_BIG_P30_WIN and f1_gain >= -F1_HARD_DROP_LIMIT:
        return True

    # Legacy gate, byte-identical to today.
    p30_up = p30_gain >= PROMOTE_MARGIN
    f1_up = f1_gain >= PROMOTE_MARGIN
    f1_ok = f1_gain >= -PROMOTE_MARGIN
    p30_not_collapsed = p30_gain >= -PROMOTE_RANKING_SLACK
    return (p30_up and f1_ok) or (f1_up and p30_not_collapsed)


def _should_promote(current_model: Optional[dict], res: dict) -> bool:
    """Delegate to evaluate_model_update (promote gate v2)."""
    return evaluate_model_update(current_model, res)


def _embed_pending_until_done() -> None:
    while True:
        processed = sync._compute_pending_embeddings()
        if not processed:
            break


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
    has_errors = False
    desk_reports: List[Dict[str, Any]] = []

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
        _embed_pending_until_done()
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
                promoted = _should_promote(current_model, res)
                if promoted and not current_model:
                    logger.info("Cold start promote.")
                elif promoted:
                    old_p30 = float(report["p30_old"] or 0.0)
                    new_p30 = float(report["p30_new"] or 0.0)
                    old_f1 = float(report["f1_old"] or 0.0)
                    new_f1 = float(report["f1_new"] or 0.0)
                    p30_gain = new_p30 - old_p30
                    f1_gain = new_f1 - old_f1
                    if p30_gain >= PROMOTE_BIG_P30_WIN and f1_gain < 0:
                        logger.info(
                            "Promote gate passed (p@30 %.3f→%.3f, f1 %.3f→%.3f, "
                            "accepted f1 dip %.3f).",
                            old_p30,
                            new_p30,
                            old_f1,
                            new_f1,
                            f1_gain,
                        )
                    else:
                        logger.info(
                            "Promote gate passed (p@30 %.3f→%.3f, f1 %.3f→%.3f).",
                            old_p30,
                            new_p30,
                            old_f1,
                            new_f1,
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
                    logger.info("Model rejected. Keeping older model.")
                    db.log_sync(
                        "train_rejected",
                        1,
                        f"Kept older model, new version {res['version']} rejected.",
                        profile_id=profile_id,
                    )

        # Score + push BEFORE distill so desk model_meta advances even if
        # recipe distillation is later OOM-killed under MemoryMax.
        current_model = db.get_active_model(profile_id)
        if current_model:
            logger.info("Scoring and pushing recent entries for %s...", url)
            try:
                recent_entries = db.get_recent_entries(days=14, include_embedding=False)
                if recent_entries:
                    scores = pipeline.score_entries(recent_entries, profile_id=profile_id)
                    if scores:
                        scores = _rank_normalize_push_scores(scores)
                        model_meta = _model_meta_for_push(profile_id, current_model)
                        sync.push_scores(
                            scores,
                            model_version=current_model["version"],
                            model_meta=model_meta,
                            profile=prof,
                        )
                        logger.info("Pushed %d rank-normalized scores.", len(scores))
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
