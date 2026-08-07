#!/usr/bin/env python3
"""
Headless CLI worker for the VPS Magnitu ML window.
Executes sync -> embed -> train (gated) -> promote (strict) -> score -> push.
Expects SEISMO_DESKS_JSON environment variable with a JSON list of desks.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict

# Make sure imports work if run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import sync
import pipeline
import distiller
from config import get_config
from model_manager import export_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROMOTE_MARGIN = 0.01

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
            chunk = ids_to_prune[i:i+chunk_size]
            conn.execute(
                f"UPDATE entries SET embedding = NULL WHERE id IN ({','.join('?' * len(chunk))})",
                chunk
            )
        conn.commit()
        logger.info("Pruned %d embeddings.", len(ids_to_prune))
    conn.close()

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
    has_errors = False
    
    for desk in desks:
        url = desk.get("seismo_url")
        api_key = desk.get("api_key")
        
        if not url or not api_key:
            logger.warning("Desk missing url or api_key, skipping: %s", desk)
            has_errors = True
            continue
            
        logger.info("Processing desk: %s", url)

        # Prefer registry slug (e.g. "sicherheit"); never slugify the full URL
        # (that yields https-seismo-live-… and misses desktop profiles).
        slug = (desk.get("slug") or "").strip()
        display_name, derived_slug = db.derive_profile_identity_from_push_url(url)
        if slug:
            slug = db.slugify(slug)
        else:
            slug = derived_slug

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
        
        # 1. Sync Labels
        try:
            sync.pull_labels(profile_id=profile_id, profile=prof)
        except Exception as e:
            logger.error("Failed to pull labels for %s: %s", url, e)
            has_errors = True
            continue
            
        # 2. Sync Entries and loop to compute all embeddings
        try:
            # Pull entries but don't compute embeddings automatically to avoid caps
            sync.pull_all_entry_types(drain=True, compute_embeddings=False)
            
            # Loop embeddings until done
            while True:
                processed = sync._compute_pending_embeddings()
                if not processed:
                    break
        except Exception as e:
            logger.error("Failed to pull/embed entries for %s: %s", url, e)
            has_errors = True
            continue
            
        labeled = db.get_all_labels(profile_id)
        current_model = db.get_active_model(profile_id)
        
        do_train = force_retrain
        if not current_model:
            if len(labeled) >= 15:
                do_train = True
        else:
            trained_at = datetime.fromisoformat(current_model["trained_at"].replace("Z", ""))
            days_since = (datetime.utcnow() - trained_at).days
            
            conn = db.get_db()
            new_labels = conn.execute("SELECT COUNT(*) FROM labels WHERE profile_id=? AND updated_at > ?", (profile_id, current_model["trained_at"])).fetchone()[0]
            conn.close()
            
            if new_labels >= 15 or days_since >= 14:
                do_train = True
                
        if do_train:
            logger.info("Gate passed: training model for %s...", url)
            res = pipeline.train(profile_id=profile_id, activate=False)
            
            if not res.get("success"):
                logger.warning("Training failed: %s", res.get("error"))
                has_errors = True
            else:
                promoted = False
                if not current_model:
                    promoted = True
                    logger.info("Cold start promote.")
                else:
                    new_score = res.get("precision_at_30", 0.0)
                    old_score = current_model.get("precision_at_30", 0.0)
                    new_f1 = res.get("f1_score", 0.0)
                    old_f1 = current_model.get("f1_score", 0.0)
                    
                    if new_score >= old_score + PROMOTE_MARGIN:
                        if new_f1 >= old_f1 - PROMOTE_MARGIN:
                            promoted = True
                            
                if promoted:
                    logger.info("Model promoted! Activating version %s", res["version"])
                    conn = db.get_db()
                    conn.execute("UPDATE models SET is_active = 0 WHERE profile_id = ?", (profile_id,))
                    conn.execute("UPDATE models SET is_active = 1 WHERE profile_id = ? AND version = ?", (profile_id, res["version"]))
                    conn.commit()
                    conn.close()
                    
                    # Distill Recipe
                    try:
                        logger.info("Distilling recipe for %s...", url)
                        distiller.distill_recipe(profile_id=profile_id)
                    except Exception as e:
                        logger.error("Failed to distill recipe: %s", e)
                        has_errors = True
                        
                    current_model = db.get_active_model(profile_id)
                    
                    # Push recipe if distilled
                    if current_model and current_model.get("recipe_path") and os.path.exists(current_model["recipe_path"]):
                        try:
                            with open(current_model["recipe_path"], "r") as rf:
                                recipe = json.load(rf)
                                sync.push_recipe(recipe, profile=prof)
                                logger.info("Recipe pushed for %s.", url)
                        except Exception as e:
                            logger.error("Failed to push recipe: %s", e)
                            has_errors = True
                            
                    # Export and push to Vault (mothership)
                    try:
                        logger.info("Exporting model to vault for %s...", url)
                        model_zip = export_model(profile_id=profile_id)
                        sync.vault_upload(vault_password=vault_password, package_path=model_zip, overwrite=True)
                        logger.info("Model uploaded to mothership vault.")
                    except Exception as e:
                        logger.error("Failed to upload model to vault: %s", e)
                        has_errors = True

                else:
                    logger.info("Model rejected. Keeping older model.")
                    db.log_sync("train_rejected", 1, f"Kept older model, new version {res['version']} rejected.", profile_id=profile_id)
                    
        # 3. Always score and push (using latest active model)
        current_model = db.get_active_model(profile_id)
        if current_model:
            logger.info("Scoring and pushing recent entries for %s...", url)
            try:
                recent_entries = db.get_recent_entries(days=14)
                if recent_entries:
                    scores = pipeline.score_entries(recent_entries, profile_id=profile_id)
                    if scores:
                        scores = _rank_normalize_push_scores(scores)
                        sync.push_scores(scores, model_version=current_model["version"], profile=prof)
                        logger.info("Pushed %d rank-normalized scores.", len(scores))
            except Exception as e:
                logger.error("Error during score push: %s", e)
                has_errors = True

    return 1 if has_errors else 0

if __name__ == "__main__":
    sys.exit(main())
