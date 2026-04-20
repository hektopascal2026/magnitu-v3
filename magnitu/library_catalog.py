"""Scan ``MODELS_DIR`` for ``.magnitu`` packages and read manifest / label summaries."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pipeline


def label_distribution_from_labels_json(labels: List[dict]) -> Dict[str, int]:
    counts = {c: 0 for c in pipeline.CLASSES}
    for row in labels:
        lbl = (row.get("label") or "").strip()
        if lbl in counts:
            counts[lbl] += 1
    return counts


def summarize_magnitu_package(path: Path) -> Dict[str, Any]:
    """Read a ``.magnitu`` zip and return display metadata (no extract to disk)."""
    path = Path(path)
    errors: List[str] = []
    out: Dict[str, Any] = {
        "filename": path.name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "manifest_format_version": None,
        "model_name": "",
        "model_uuid": "",
        "version": 0,
        "label_count": 0,
        "architecture": "",
        "metrics": {},
        "label_distribution": {c: 0 for c in pipeline.CLASSES},
        "has_model_joblib": False,
        "has_labels_json": False,
        "has_recipe_json": False,
        "has_calibration_json": False,
        "errors": errors,
    }

    if not path.is_file():
        errors.append("Not a file")
        return out
    if not path.name.lower().endswith(".magnitu"):
        errors.append("Filename must end with .magnitu")
        return out
    if not zipfile.is_zipfile(path):
        errors.append("Not a valid zip archive")
        return out

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            out["has_model_joblib"] = "model.joblib" in names
            out["has_labels_json"] = "labels.json" in names
            out["has_recipe_json"] = "recipe.json" in names
            out["has_calibration_json"] = "calibration.json" in names

            if "manifest.json" not in names:
                errors.append("Missing manifest.json")
                return out

            raw = zf.read("manifest.json").decode("utf-8", errors="replace")
            manifest = json.loads(raw)
            if not isinstance(manifest, dict):
                errors.append("manifest.json is not a JSON object")
                return out

            out["manifest_format_version"] = manifest.get("manifest_format_version")
            out["model_name"] = str(manifest.get("model_name") or "")
            out["model_uuid"] = str(manifest.get("model_uuid") or "")
            try:
                out["version"] = int(manifest.get("version") or 0)
            except (TypeError, ValueError):
                out["version"] = 0
            try:
                out["label_count"] = int(manifest.get("label_count") or 0)
            except (TypeError, ValueError):
                out["label_count"] = 0
            out["architecture"] = str(
                manifest.get("architecture") or manifest.get("model_architecture") or ""
            )
            mets = manifest.get("metrics")
            if isinstance(mets, dict):
                f1v = mets.get("f1", mets.get("f1_score", 0.0))
                out["metrics"] = {
                    "accuracy": float(mets.get("accuracy", 0.0) or 0.0),
                    "f1": float(f1v or 0.0),
                    "precision": float(mets.get("precision", 0.0) or 0.0),
                    "recall": float(mets.get("recall", 0.0) or 0.0),
                }

            mdist = manifest.get("label_distribution")
            if isinstance(mdist, dict):
                for c in pipeline.CLASSES:
                    try:
                        out["label_distribution"][c] = int(mdist.get(c, 0) or 0)
                    except (TypeError, ValueError):
                        out["label_distribution"][c] = 0

            if "labels.json" in names:
                try:
                    labels_raw = zf.read("labels.json").decode("utf-8", errors="replace")
                    labels = json.loads(labels_raw)
                    if isinstance(labels, list):
                        computed = label_distribution_from_labels_json(labels)
                        for c in pipeline.CLASSES:
                            out["label_distribution"][c] = max(
                                out["label_distribution"][c], computed[c]
                            )
                        if not out["label_count"]:
                            out["label_count"] = len(labels)
                except Exception as e:
                    errors.append("labels.json: %s" % e)
    except json.JSONDecodeError as e:
        errors.append("manifest.json: %s" % e)
    except Exception as e:
        errors.append(str(e))

    return out


def list_magnitu_library(models_dir: Path) -> List[Dict[str, Any]]:
    """Return summaries for every ``*.magnitu`` under ``models_dir``, newest mtime first."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    paths = sorted(
        models_dir.glob("*.magnitu"),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    return [summarize_magnitu_package(p) for p in paths if p.is_file()]


def safe_package_path(models_dir: Path, filename: str) -> Path:
    """Resolve ``filename`` to a path under ``models_dir`` (basename only, ``.magnitu``)."""
    models_dir = Path(models_dir).resolve()
    base = Path(filename).name
    if not base.lower().endswith(".magnitu") or base != Path(filename).name:
        raise ValueError("Invalid package name")
    resolved = (models_dir / base).resolve()
    if resolved.parent != models_dir:
        raise ValueError("Path escapes models directory")
    if not resolved.is_file():
        raise FileNotFoundError("Package not found: %s" % base)
    return resolved
