#!/usr/bin/env python3
"""
Distill recipes for all profiles using local GPU for fast scoring.

Runs distiller.distill_recipe() for each profile, saves the recipe JSON,
and prints a summary. Recipes can then be pushed to Seismo via the API.

Usage:
    python scripts/gpu_distill_recipes.py --db magnitu_vps.db
    python scripts/gpu_distill_recipes.py --db magnitu_vps.db --profiles 1,2,3,4
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force GPU + temp config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_tmp_data_dir = tempfile.mkdtemp(prefix="magnitu_gpu_distill_")
_tmp_cfg = {
    "seismo_url": "https://seismo.live/index.php",
    "api_key": "dummy",
    "use_gpu": True,
    "transformer_model_name": "intfloat/multilingual-e5-base",
    "embedding_dim": 768,
    "embedding_max_tokens": 512,
    "embedding_content_cap": 3000,
    "embedding_legal_content_cap": 12000,
    "embedding_analytical_content_cap": 7000,
    "legal_signal_patterns": [],
    "embedding_stack_generation": "e5-v3",
    "distillation_soft_labels": True,
    "distillation_max_entries": 8000,
    "recipe_top_keywords": 200,
    "recipe_max_unigram_abs": 0.12,
    "recipe_max_phrase_abs": 0.24,
    "recipe_max_source_abs": 0.08,
    "recipe_min_abs_keep": 0.01,
    "recipe_normalize_target": 2.0,
    "recipe_optimize_caps": True,
    "synthetic_label_weight": 0.5,
    "reasoning_weight_boost": 1.0,
    "recipe_quality_floor": 0.3,
}
with open(os.path.join(_tmp_data_dir, "magnitu_config.json"), "w") as f:
    json.dump(_tmp_cfg, f)
os.environ["MAGNITU_DATA_DIR"] = _tmp_data_dir

import config as config_mod
config_mod.load_config()
import db
import distiller
from pipeline import release_embedder


def main():
    parser = argparse.ArgumentParser(description="GPU recipe distillation for all profiles")
    parser.add_argument("--db", required=True, help="Path to the SQLite DB to use")
    parser.add_argument("--profiles", default="1,2,3,4", help="Comma-separated profile IDs")
    args = parser.parse_args()

    # Symlink the target DB as magnitu.db in the temp data dir
    db_target = os.path.abspath(args.db)
    db_link = os.path.join(_tmp_data_dir, "magnitu.db")
    if os.path.exists(db_link):
        os.remove(db_link)
    os.symlink(db_target, db_link)
    print(f"Using DB: {db_target} -> {db_link}")

    profile_ids = [int(x) for x in args.profiles.split(",")]
    results = {}

    for pid in profile_ids:
        print(f"\n{'='*60}")
        print(f"Distilling recipe for profile {pid}")
        print(f"{'='*60}")

        model_info = db.get_active_model(profile_id=pid)
        if not model_info:
            print(f"  No active model for profile {pid}, skipping")
            results[pid] = {"error": "no active model"}
            continue

        print(f"  Active model: v{model_info['version']} ({model_info['architecture']})")
        print(f"  Labels: {db.get_all_labels(profile_id=pid)[:1]}...")

        t0 = time.time()
        try:
            recipe = distiller.distill_recipe(profile_id=pid)
            elapsed = time.time() - t0

            if recipe is None:
                print(f"  FAILED: distill_recipe returned None ({elapsed:.1f}s)")
                results[pid] = {"error": "distill returned None", "elapsed": elapsed}
                continue

            quality = 0.0
            if isinstance(recipe.get("metrics"), dict):
                quality = recipe["metrics"].get("recipe_quality", recipe.get("recipe_quality", 0.0))
            elif "recipe_quality" in recipe:
                quality = recipe["recipe_quality"]

            keywords = recipe.get("keywords", {})
            source_weights = recipe.get("source_weights", {})

            print(f"  Done in {elapsed:.1f}s")
            print(f"  Recipe quality: {quality:.4f} (floor: 0.30)")
            print(f"  Keywords: {len(keywords)}")
            print(f"  Source weights: {len(source_weights)}")
            print(f"  Classes: {recipe.get('classes', [])}")

            # Save recipe to file
            recipe_path = f"recipe_p{pid}_v{model_info['version']}.json"
            with open(recipe_path, "w") as f:
                json.dump(recipe, f, indent=2, ensure_ascii=False)
            print(f"  Saved to: {recipe_path}")

            results[pid] = {
                "version": model_info["version"],
                "quality": quality,
                "keywords": len(keywords),
                "source_weights": len(source_weights),
                "elapsed": elapsed,
                "recipe_path": recipe_path,
                "passes_floor": quality >= 0.30,
            }

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e} ({elapsed:.1f}s)")
            import traceback
            traceback.print_exc()
            results[pid] = {"error": str(e), "elapsed": elapsed}

    release_embedder()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for pid, r in results.items():
        if "error" in r:
            print(f"  p{pid}: ERROR - {r['error']}")
        else:
            status = "PASS" if r["passes_floor"] else "BELOW FLOOR"
            print(f"  p{pid} v{r['version']}: quality={r['quality']:.4f} [{status}] "
                  f"({r['keywords']} keywords, {r['elapsed']:.0f}s)")


if __name__ == "__main__":
    main()
