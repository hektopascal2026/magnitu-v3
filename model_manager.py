"""
Model Manager: handles export/import of portable .magnitu model packages.

A .magnitu file is a zip archive containing:
  - manifest.json  — model identity, version chain, metrics, architecture
  - model.joblib   — trained classifier (sklearn pipeline or LogReg head)
  - recipe.json    — distilled recipe (if exists)
  - labels.json    — all labeled entries with text + reasoning (for retraining)
  - calibration.json — temperature scaling metadata (if exists)

Multi-profile: every operation is scoped to a profile_id.
"""
import json
import uuid
import shutil
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib

import db
import pipeline
from config import MODELS_DIR, VERSION, get_config


def _label_distribution_from_labels_list(labels: List[dict]) -> dict:
    counts = {c: 0 for c in pipeline.CLASSES}
    for row in labels:
        if not isinstance(row, dict):
            continue
        lbl = (row.get("label") or "").strip()
        if lbl in counts:
            counts[lbl] += 1
    return counts


# ─── Profile helpers (backward-compat wrappers used by main.py) ─────────────

def has_profile(profile_id: int = 1) -> bool:
    return db.has_model_profile(profile_id)


def get_profile(profile_id: int = 1) -> Optional[dict]:
    return db.get_model_profile(profile_id)


def create_profile(name: str, description: str = "",
                   seismo_url: str = "", api_key: str = "") -> dict:
    """Create a new profile with a fresh UUID and return it."""
    slug = db.slugify(name)
    # Ensure slug is unique
    base = slug
    n = 1
    while db.get_profile_by_slug(slug):
        slug = "{}-{}".format(base, n)
        n += 1
    return db.create_profile(slug=slug, display_name=name,
                              description=description,
                              seismo_url=seismo_url, api_key=api_key)


def update_profile(description: str = None, profile_id: int = 1):
    db.update_model_profile(description=description, profile_id=profile_id)


# ─── Version chain ───────────────────────────────────────────────────────────

def _get_version_chain(profile_id: int = 1) -> list:
    models = db.get_all_models(profile_id)
    chain = []
    for m in reversed(models):
        chain.append({
            "version": m["version"],
            "trained_at": m.get("trained_at", ""),
            "label_count": m.get("label_count", 0),
            "accuracy": m.get("accuracy", 0.0),
        })
    return chain


# ─── Export ──────────────────────────────────────────────────────────────────

def _build_manifest(model_name: str, model_uuid: str, description: str,
                    created_at: str, profile_id: int = 1,
                    extra: Optional[dict] = None) -> dict:
    config = get_config()
    active_model = db.get_active_model(profile_id)

    raw_dist = db.get_label_distribution(profile_id)
    label_distribution = {c: int(raw_dist.get(c, 0)) for c in pipeline.CLASSES}

    manifest = {
        "manifest_format_version": 1,
        "magnitu_version": VERSION,
        "model_name": model_name,
        "model_uuid": model_uuid,
        "description": description,
        "created_at": created_at,
        "exported_at": datetime.utcnow().isoformat(),
        "version": active_model["version"] if active_model else 0,
        "label_count": db.get_label_count(profile_id),
        "label_distribution": label_distribution,
        "architecture": (active_model.get("architecture", "tfidf")
                         if active_model
                         else config.get("model_architecture", "transformer")),
        "transformer_model_name": config.get("transformer_model_name", "xlm-roberta-base"),
        "metrics": {},
        "version_chain": _get_version_chain(profile_id),
    }

    if active_model:
        manifest["metrics"] = {
            "accuracy":  active_model.get("accuracy", 0.0),
            "f1":        active_model.get("f1_score", 0.0),
            "precision": active_model.get("precision_score", 0.0),
            "recall":    active_model.get("recall_score", 0.0),
        }

    if extra:
        manifest.update(extra)

    return manifest


def _write_package(manifest: dict, output_path: str,
                   profile_id: int = 1) -> str:
    """Write manifest, labels, model, and recipe into a .magnitu zip."""
    active_model = db.get_active_model(profile_id)
    labels = db.export_labels(profile_id)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        with open(tmp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(tmp_dir / "labels.json", "w") as f:
            json.dump(labels, f, ensure_ascii=False)

        if active_model and active_model.get("model_path"):
            model_path = Path(active_model["model_path"])
            if model_path.exists():
                shutil.copy2(str(model_path), str(tmp_dir / "model.joblib"))
            cal_path = pipeline.calibration_sidecar_path(str(model_path))
            if cal_path.exists():
                shutil.copy2(str(cal_path), str(tmp_dir / "calibration.json"))

        if active_model and active_model.get("recipe_path"):
            recipe_path = Path(active_model["recipe_path"])
            if recipe_path.exists():
                shutil.copy2(str(recipe_path), str(tmp_dir / "recipe.json"))

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmp_dir.iterdir():
                zf.write(str(file_path), file_path.name)

    return output_path


def export_model(output_path: str = None, profile_id: int = 1) -> str:
    """Export current model for a profile as a .magnitu zip. Returns path."""
    profile = get_profile(profile_id)
    if not profile:
        raise ValueError("No model profile configured. Create one first.")

    manifest = _build_manifest(
        model_name=profile["model_name"],
        model_uuid=profile["model_uuid"],
        description=profile.get("description", ""),
        created_at=profile.get("created_at", ""),
        profile_id=profile_id,
    )

    if not output_path:
        safe_name = profile["model_name"].replace(" ", "_").lower()
        output_path = str(MODELS_DIR / "{}.magnitu".format(safe_name))

    return _write_package(manifest, output_path, profile_id=profile_id)


def export_as_new_model(new_name: str, new_description: str,
                        output_path: str = None, profile_id: int = 1) -> str:
    """Fork: export this profile's model as a new identity with a fresh UUID."""
    parent_profile = get_profile(profile_id)
    if not parent_profile:
        raise ValueError("No model profile configured. Create one first.")

    new_name = new_name.strip()
    if not new_name:
        raise ValueError("New model name is required.")

    active_model = db.get_active_model(profile_id)
    manifest = _build_manifest(
        model_name=new_name,
        model_uuid=uuid.uuid4().hex,
        description=new_description.strip(),
        created_at=datetime.utcnow().isoformat(),
        profile_id=profile_id,
        extra={
            "based_on": {
                "model_name": parent_profile["model_name"],
                "model_uuid": parent_profile["model_uuid"],
                "description": parent_profile.get("description", ""),
                "version_at_fork": active_model["version"] if active_model else 0,
                "forked_at": datetime.utcnow().isoformat(),
            },
        },
    )

    if not output_path:
        safe_name = new_name.replace(" ", "_").lower()
        output_path = str(MODELS_DIR / "{}.magnitu".format(safe_name))

    return _write_package(manifest, output_path, profile_id=profile_id)


# ─── Import ──────────────────────────────────────────────────────────────────

def import_model(
    file_path: str,
    profile_id: Optional[int] = None,
    force: bool = False,
    import_labels: bool = True,
) -> dict:
    """Import a .magnitu package into a profile.

    profile_id: target profile. If None, a new profile is created (or the
    default profile is used when no profiles exist yet).

    force: when True, always copy ``model.joblib`` and register a new active model row
    using the next DB version (Library \"set active\"), even if the manifest version
    is older than the current active model.

    import_labels: when False, skip merging ``labels.json`` into the profile (model-only).
    """
    if not zipfile.is_zipfile(file_path):
        raise ValueError("Not a valid .magnitu file (not a zip archive).")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(str(tmp_dir))

        manifest_path = tmp_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid .magnitu file: no manifest.json found.")

        with open(manifest_path) as f:
            manifest = json.load(f)

        imported_name = manifest.get("model_name", "Unknown")
        imported_uuid = manifest.get("model_uuid", "")
        imported_version = manifest.get("version", 0)
        try:
            imported_version = int(imported_version)
        except (TypeError, ValueError):
            imported_version = 0
        imported_description = manifest.get("description", "")
        imported_architecture = manifest.get("architecture", "tfidf")

        # Resolve target profile: create a new one if not specified
        if profile_id is None:
            slug = db.slugify(imported_name)
            base = slug
            n = 1
            while db.get_profile_by_slug(slug):
                slug = "{}-{}".format(base, n)
                n += 1
            new_p = db.create_profile(
                slug=slug,
                display_name=imported_name,
                description=imported_description,
            )
            profile_id = new_p["id"]
        else:
            db.set_model_profile(
                imported_name,
                imported_uuid,
                imported_description,
                manifest.get("created_at", ""),
                profile_id=profile_id,
            )

        result = {
            "model_name": imported_name,
            "model_uuid": imported_uuid,
            "profile_id": profile_id,
            "imported_version": imported_version,
            "labels": {"imported": 0, "skipped": 0, "updated": 0},
            "model_loaded": False,
            "message": "",
            "activated_version": None,
        }

        labels_path = tmp_dir / "labels.json"
        labels_data: List[dict] = []
        if labels_path.exists():
            with open(labels_path) as f:
                raw_labels = json.load(f)
            if isinstance(raw_labels, list):
                labels_data = [x for x in raw_labels if isinstance(x, dict)]
        if import_labels and labels_data:
            result["labels"] = db.import_labels(labels_data, profile_id=profile_id)

        local_model = db.get_active_model(profile_id)
        local_version = local_model["version"] if local_model else 0
        model_file = tmp_dir / "model.joblib"

        should_load = model_file.exists() and (
            force or local_version == 0 or imported_version >= local_version
        )
        store_version = imported_version
        if should_load and force:
            store_version = db.get_next_model_version(profile_id)

        if should_load:
            dest = MODELS_DIR / "model_v{}.joblib".format(store_version)
            shutil.copy2(str(model_file), str(dest))

            cal_file = tmp_dir / "calibration.json"
            if cal_file.exists():
                cal_dest = pipeline.calibration_sidecar_path(str(dest))
                shutil.copy2(str(cal_file), str(cal_dest))

            recipe_file = tmp_dir / "recipe.json"
            recipe_dest = ""
            if recipe_file.exists():
                recipe_dest = str(MODELS_DIR / "recipe_v{}.json".format(store_version))
                shutil.copy2(str(recipe_file), recipe_dest)

            pack_label_dist = (
                _label_distribution_from_labels_list(labels_data) if labels_data else {}
            )
            if not any(pack_label_dist.values()):
                mdist = manifest.get("label_distribution")
                if isinstance(mdist, dict):
                    pack_label_dist = {
                        c: int(mdist.get(c, 0) or 0) for c in pipeline.CLASSES
                    }
            if not any(pack_label_dist.values()):
                pack_label_dist = {
                    c: int(db.get_label_distribution(profile_id).get(c, 0))
                    for c in pipeline.CLASSES
                }

            metrics = manifest.get("metrics", {}) or {}
            lbl_count = int(manifest.get("label_count") or 0)
            if labels_data:
                lbl_count = max(lbl_count, len(labels_data))

            f1_val = metrics.get("f1", metrics.get("f1_score", 0.0))

            db.save_model_record(
                version=store_version,
                accuracy=float(metrics.get("accuracy", 0.0) or 0.0),
                f1=float(f1_val or 0.0),
                precision=float(metrics.get("precision", 0.0) or 0.0),
                recall=float(metrics.get("recall", 0.0) or 0.0),
                label_count=lbl_count,
                feature_count=0,
                model_path=str(dest),
                recipe_path=recipe_dest,
                architecture=imported_architecture,
                profile_id=profile_id,
                label_distribution=pack_label_dist,
            )
            result["model_loaded"] = True
            result["activated_version"] = store_version

        db.set_model_profile(
            model_name=imported_name,
            model_uuid=imported_uuid,
            description=imported_description,
            created_at=manifest.get("created_at", ""),
            profile_id=profile_id,
        )
        result["profile_updated"] = True

        new_labels = result["labels"]["imported"] + result["labels"]["updated"]
        parts = ['profile set to "{}"'.format(imported_name)]
        if new_labels:
            parts.append("{} labels imported".format(new_labels))
        if result["model_loaded"]:
            parts.append("model v{} loaded".format(store_version))
        elif imported_version and imported_version < local_version and not force:
            parts.append(
                "keeping local model v{} (imported v{} is older)".format(
                    local_version, imported_version
                )
            )
        if not parts[1:]:
            parts.append("no new data")
        if new_labels and not result["model_loaded"]:
            parts.append("retrain recommended")
        result["message"] = ". ".join(parts).capitalize() + "."

    return result
