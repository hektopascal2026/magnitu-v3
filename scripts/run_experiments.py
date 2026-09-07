#!/usr/bin/env python3
"""Experiment runner for the Magnitu calibration protocol.

Executes the experiment matrix from docs/calibration-for-real.md:
  B1: encoder × text × C sweep (e5 vs granite, 3 text modes, 10 C values)
  B2: head comparison (all heads, e5_tuned text)
  B4: pooling controls (granite pooling variants)

Uses the calibration framework (per-config OOF, actual-weight priors,
chronological splits, 4 probability modes).

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/run_experiments.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/run_experiments.py --blocks B1
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/run_experiments.py --profiles 1 2 3 4
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/run_experiments.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure we can import magnitu-v3 modules
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import db
import pipeline as pipe
from pipeline import (
    CLASSES,
    compute_sample_weights,
    bytes_to_embedding,
    _build_entry_text,
    _split_text_chunks,
    is_legal_training_entry,
    is_analytical_training_entry,
    MAX_EMBED_CHUNKS,
    MAX_EMBED_CHUNKS_LEGAL,
    MAX_EMBED_CHUNKS_ANALYTICAL,
    training_corpus_text,
    _content_cap_for_entry,
    _natural_source_context,
)
from config import get_config, DEFAULT_TRANSFORMER_MODEL
from scripts.calibration_framework import (
    ExperimentConfig,
    evaluate_configuration,
    chronological_split,
    ChronologicalSplits,
    grouped_dev_split,
    GroupedDevSplit,
    C_SWEEP,
    SEEDS,
    CLASSES as FW_CLASSES,
)

logger = logging.getLogger(__name__)

GRANITE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"

# All heads for B2
ALL_HEADS = [
    "logreg", "logreg",  # C=1.0 incumbent (covered by B1)
    "svm_rbf", "svm_linear",
    "mlp_128", "mlp_256", "mlp_512", "mlp_2layer",
    "xgboost", "lightgbm", "ridge", "knn",
]
# For B2, use C=1.0 for C-sensitive heads
B2_HEADS = [
    ("svm_rbf", 1.0),
    ("svm_linear", 1.0),
    ("mlp_128", 1.0),
    ("mlp_256", 1.0),
    ("mlp_512", 1.0),
    ("mlp_2layer", 1.0),
    ("xgboost", 1.0),
    ("lightgbm", 1.0),
    ("ridge", 1.0),
    ("knn", 1.0),
]

# ── Text builders ────────────────────────────────────────────────────

def build_text_e5_tuned(entry: dict, config: dict) -> str:
    return _build_entry_text(entry, legal_patterns=config.get("legal_signal_patterns") or [])


def build_text_plain(entry: dict, config: dict) -> str:
    title = (entry.get("title") or "").strip()
    body = training_corpus_text(entry)[:_content_cap_for_entry(entry)]
    parts = [p for p in [title, body] if p]
    return "\n".join(parts) if parts else "(empty)"


def build_text_plain_context(entry: dict, config: dict) -> str:
    title = (entry.get("title") or "").strip()
    body = training_corpus_text(entry)[:_content_cap_for_entry(entry)]
    context = _natural_source_context(entry, signals=[])
    parts = [p for p in [title, body] if p]
    body_text = "\n".join(parts) if parts else "(empty)"
    if context:
        return "{}\n\n{}".format(context, body_text)
    return body_text


TEXT_BUILDERS = {
    "e5_tuned": build_text_e5_tuned,
    "plain": build_text_plain,
    "plain_context": build_text_plain_context,
}

# ── Embedding generation ─────────────────────────────────────────────

def cache_dir() -> Path:
    d = BASE_DIR / "lab_data" / "embeddings_cache_v2"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(encoder: str, text_mode: str, profile_id: int) -> str:
    return f"{encoder}_{text_mode}_p{profile_id}.npy"


def load_cached(encoder: str, text_mode: str, profile_id: int) -> Optional[np.ndarray]:
    p = cache_dir() / cache_key(encoder, text_mode, profile_id)
    if p.exists():
        return np.load(p)
    return None


def save_cached(arr: np.ndarray, encoder: str, text_mode: str, profile_id: int):
    p = cache_dir() / cache_key(encoder, text_mode, profile_id)
    np.save(p, arr)


def encode_with_model(
    model_name: str,
    labeled: List[dict],
    config: dict,
    text_variant: str,
    pooling: str = "mean",
) -> np.ndarray:
    """Encode labeled entries with a given model and text pipeline.

    pooling: "mean" for e5, "cls_norm" for granite (CLS + L2 normalization).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    text_fn = TEXT_BUILDERS[text_variant]
    patterns = config.get("legal_signal_patterns") or []

    chunk_texts = []
    chunk_meta = []
    for i, entry in enumerate(labeled):
        text = text_fn(entry, config)
        max_c = (
            MAX_EMBED_CHUNKS_LEGAL
            if is_legal_training_entry(entry)
            else MAX_EMBED_CHUNKS_ANALYTICAL
            if is_analytical_training_entry(entry)
            else MAX_EMBED_CHUNKS
        )
        for ch in _split_text_chunks(text, max_chunks=max_c):
            chunk_texts.append(ch)
            chunk_meta.append((i, len(ch)))

    print(f"    Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)
    print(f"    Encoding {len(chunk_texts)} chunks ({len(labeled)} entries) on {device.type}...", flush=True)

    is_e5 = "e5" in model_name.lower()
    is_granite = "granite" in model_name.lower()
    batch_size = 16 if device.type == "cuda" else 8
    all_embs = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        if is_e5:
            batch = ["passage: " + t if not t.startswith("passage:") else t for t in batch]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        token_emb = outputs.last_hidden_state.float()
        if is_granite or pooling.startswith("cls"):
            chunk_embs = token_emb[:, 0, :].cpu().numpy()
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            chunk_embs = (summed / counts).cpu().numpy()
        all_embs.append(chunk_embs)
        done = min(i + batch_size, len(chunk_texts))
        if done % 200 == 0 or done == len(chunk_texts):
            print(f"      {done}/{len(chunk_texts)} chunks", flush=True)

    all_embs = np.vstack(all_embs) if all_embs else np.array([])

    # L2-normalize chunk embeddings before pooling (only when pooling name
    # ends with "norm"; is_granite no longer forces normalization so that
    # cls/mean and cls_norm/mean_norm are distinct, auditable variants)
    if pooling.endswith("norm"):
        if all_embs.size:
            norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
            all_embs = all_embs / np.clip(norms, 1e-12, None)

    # Weighted pool chunks per entry
    by_vecs, by_wts = {}, {}
    for (idx, char_len), emb in zip(chunk_meta, all_embs):
        by_vecs.setdefault(idx, []).append(emb)
        by_wts.setdefault(idx, []).append(max(char_len, 1))

    dim = all_embs.shape[1] if all_embs.size else 768
    out = []
    for i in range(len(labeled)):
        vecs = by_vecs.get(i)
        if not vecs:
            out.append(np.zeros(dim, dtype=np.float32))
            continue
        pooled = np.average(np.vstack(vecs), axis=0, weights=by_wts[i])
        out.append(pooled)

    out_arr = np.array(out, dtype=np.float32)

    # Final L2 normalization (only when pooling name ends with "norm")
    if pooling.endswith("norm"):
        norms = np.linalg.norm(out_arr, axis=1, keepdims=True)
        out_arr = out_arr / np.clip(norms, 1e-12, None)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_arr


def get_embeddings(
    encoder: str,
    text_mode: str,
    profile_id: int,
    labeled: List[dict],
    config: dict,
    pooling_override: Optional[str] = None,
) -> np.ndarray:
    """Get embeddings for a profile, using cache or generating new."""
    cached = load_cached(encoder, text_mode, profile_id)
    if cached is not None:
        print(f"    [cache hit] {encoder}/{text_mode}/p{profile_id} ({cached.shape})", flush=True)
        return cached

    if encoder == "e5_base":
        model_name = DEFAULT_TRANSFORMER_MODEL
        pooling = pooling_override or "mean"
    elif encoder == "granite":
        model_name = GRANITE_MODEL
        pooling = pooling_override or "cls_norm"
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    print(f"  Generating embeddings: {encoder}/{text_mode}/p{profile_id} (pooling={pooling})", flush=True)
    arr = encode_with_model(model_name, labeled, config, text_mode, pooling=pooling)
    save_cached(arr, encoder, text_mode, profile_id)
    return arr


# ── Data loading ─────────────────────────────────────────────────────

def load_profile_data(profile_id: int) -> Tuple[List[dict], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load labeled entries, labels, embeddings, sample weights, timestamps, story group IDs, and label times."""
    conn = db.get_db()
    config = get_config()

    rows = conn.execute("""
        SELECT l.entry_type, l.entry_id, l.label, l.reasoning, l.label_source,
               l.created_at, l.updated_at,
               e.title, e.description, e.content, e.link, e.author,
               e.published_date, e.source_name, e.source_category, e.source_type,
               e.embedding, e.fetched_at
        FROM labels l
        JOIN entries e ON e.entry_type = l.entry_type AND e.entry_id = l.entry_id
        WHERE l.profile_id = ?
        ORDER BY e.fetched_at ASC
    """, (profile_id,)).fetchall()

    conn.close()

    if not rows:
        return [], [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    labeled = [dict(r) for r in rows]
    y = [r["label"] for r in rows]

    # Sample weights
    sw = compute_sample_weights(labeled, config)

    # Timestamps (prediction_time = fetched_at = ingest time = earliest prediction possible)
    times = []
    label_times = []
    for r in rows:
        ts_str = r["fetched_at"]
        try:
            ts = np.datetime64(ts_str).astype("datetime64[s]").astype(float)
        except Exception:
            ts = 0.0
        times.append(ts)
        # Label creation time for label-availability check
        lt_str = r["created_at"] if "created_at" in r.keys() else r["updated_at"]
        try:
            lt = np.datetime64(lt_str).astype("datetime64[s]").astype(float)
        except Exception:
            lt = 0.0
        label_times.append(lt)
    times = np.array(times)
    label_times = np.array(label_times)

    # Story group IDs: group by normalized title to keep duplicates/syndicated
    # stories in the same split partition. This prevents train→test leakage.
    import hashlib
    story_groups = []
    for r in rows:
        title = (r["title"] if "title" in r.keys() and r["title"] else "").strip().lower()
        # Collapse whitespace
        title = " ".join(title.split())
        # Hash the normalized title as the group ID
        story_groups.append(hashlib.md5(title.encode("utf-8")).hexdigest())
    story_groups = np.array(story_groups)

    return labeled, y, sw, times, config, story_groups, label_times


def get_existing_embeddings(labeled: List[dict]) -> np.ndarray:
    """Extract existing e5_base embeddings from the DB for labeled entries."""
    embs = []
    for row in labeled:
        emb_bytes = row.get("embedding")
        if emb_bytes:
            embs.append(bytes_to_embedding(emb_bytes))
        else:
            embs.append(np.zeros(768, dtype=np.float32))
    return np.array(embs, dtype=np.float32)


# ── Results storage ──────────────────────────────────────────────────

def results_dir() -> Path:
    d = BASE_DIR / "lab_data" / "calibration_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_file_path(block: str, profile_id: int) -> Path:
    return results_dir() / f"{block}_p{profile_id}.jsonl"


def save_result(block: str, profile_id: int, result: dict):
    p = result_file_path(block, profile_id)
    with open(p, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def load_completed_ids(block: str, profile_id: int) -> set:
    """Load dedup keys (config_id__wN) that have already been completed (for resume).

    Uses dedup_key if present (new format), falls back to config_id (old format).
    """
    p = result_file_path(block, profile_id)
    if not p.exists():
        return set()
    ids = set()
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "dedup_key" in r:
                    ids.add(r["dedup_key"])
                elif "config_id" in r:
                    # Old format: treat as window 0
                    ids.add(f"{r['config_id']}__w0")
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


# ── Experiment blocks ────────────────────────────────────────────────

def _label_availability_summary(
    label_times: Optional[np.ndarray],
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    """Summarize label availability for prospective evaluation caveats.

    Records whether labels in the test partition were created before or after
    the fit partition's latest label time. If test labels were created after
    fit labels, they would not have been available at fit time in a true
    as-of-time replay (D3). This is a declared caveat for D2 (current-snapshot)
    evaluations.
    """
    if label_times is None or len(label_times) == 0:
        return {"status": "no_label_times"}
    lt = np.asarray(label_times, dtype=np.float64)
    fit_lt = lt[fit_idx] if len(fit_idx) > 0 else np.array([])
    test_lt = lt[test_idx] if len(test_idx) > 0 else np.array([])
    if len(fit_lt) == 0 or len(test_lt) == 0:
        return {"status": "empty_partition"}
    fit_max = float(np.max(fit_lt))
    test_before_fit = int(np.sum(test_lt <= fit_max))
    test_after_fit = int(np.sum(test_lt > fit_max))
    return {
        "status": "current_snapshot_d2",
        "fit_label_time_max": fit_max,
        "test_labels_before_fit_max": test_before_fit,
        "test_labels_after_fit_max": test_after_fit,
        "caveat": "labels are current-snapshot, not as-of-time; "
                  "test labels created after fit labels would not be "
                  "available in a prospective D3 replay",
    }


def run_block(
    block: str,
    profile_id: int,
    encoder: str,
    text_mode: str,
    X: np.ndarray,
    y: List[str],
    sw: np.ndarray,
    times: np.ndarray,
    configs: List[ExperimentConfig],
    incumbent_metrics: Optional[Dict[str, float]] = None,
    resume: bool = True,
    story_group_ids: Optional[np.ndarray] = None,
    label_times: Optional[np.ndarray] = None,
    split_strategy: str = "grouped",
    split_seed: int = 42,
    n_test_windows: int = 3,
) -> int:
    """Run one block of experiments for one profile. Returns count of new results.

    split_strategy:
      "grouped"       — StratifiedGroupKFold D2 development split
      "chronological" — time-based split with story group enforcement
    n_test_windows: number of chronological test windows (ignored for grouped)
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(FW_CLASSES)
    y_enc = le.transform(y)

    if split_strategy == "grouped":
        if story_group_ids is None:
            print(f"  SKIP: grouped split requires story_group_ids", flush=True)
            return 0
        gsplit = grouped_dev_split(y_enc, story_group_ids, seed=split_seed)
        fit_idx = gsplit.fit_idx
        test_windows = [gsplit.test_idx]
        split_notes = gsplit.notes
        n_select = gsplit.n_select
        n_threshold = gsplit.n_threshold
        n_embargo = 0
        print(f"  Grouped split: fit={gsplit.n_fit} select={gsplit.n_select} "
              f"threshold={gsplit.n_threshold} test={gsplit.n_test}", flush=True)
    else:
        splits = chronological_split(
            times, y_enc,
            story_group_ids=story_group_ids,
            n_test_windows=n_test_windows,
        )
        fit_idx = splits.fit_idx
        test_windows = splits.test_windows
        split_notes = splits.notes
        n_select = splits.n_select
        n_threshold = splits.n_threshold
        n_embargo = splits.n_embargo
        print(f"  Chronological split: fit={splits.n_fit} select={splits.n_select} "
              f"threshold={splits.n_threshold} test={splits.n_test} "
              f"embargo={splits.n_embargo} windows={len(splits.test_windows)}", flush=True)

    if split_notes:
        print(f"  Notes: {'; '.join(split_notes)}", flush=True)

    if len(fit_idx) < 15 or not test_windows or len(test_windows[0]) == 0:
        print(f"  SKIP: insufficient data for split", flush=True)
        return 0

    # Check class support in fit
    y_fit = np.array(y)[fit_idx]
    class_counts = {c: int(np.sum(y_fit == c)) for c in FW_CLASSES}
    min_class = min(class_counts.values())
    print(f"  Fit class support: {class_counts} (min={min_class})", flush=True)
    if min_class < 2:
        print(f"  SKIP: min class support < 2", flush=True)
        return 0

    # Resume: skip completed (config_id, window) pairs
    completed = load_completed_ids(block, profile_id) if resume else set()

    n_new = 0
    t0 = time.time()
    n_configs = len(configs)
    n_windows = len(test_windows)

    for i, config in enumerate(configs):
        for w_idx, test_idx in enumerate(test_windows):
            if len(test_idx) == 0:
                continue

            dedup_key = f"{config.config_id()}__w{w_idx}"
            if dedup_key in completed:
                continue

            y_test = np.array(y)[test_idx]
            test_counts = {c: int(np.sum(y_test == c)) for c in FW_CLASSES}

            try:
                inc_metrics = incumbent_metrics if incumbent_metrics else None
                result = evaluate_configuration(
                    config, X, y, sw, fit_idx, test_idx, inc_metrics,
                    story_group_ids=story_group_ids,
                )
                result_dict = asdict(result)
                result_dict["block"] = block
                result_dict["profile_id"] = profile_id
                result_dict["config_id"] = config.config_id()
                result_dict["dedup_key"] = dedup_key
                result_dict["test_window_idx"] = w_idx
                result_dict["split_strategy"] = split_strategy
                result_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                result_dict["split_info"] = {
                    "n_fit": len(fit_idx),
                    "n_select": n_select,
                    "n_threshold": n_threshold,
                    "n_test": len(test_idx),
                    "n_embargo": n_embargo,
                    "test_window": w_idx,
                    "n_windows": n_windows,
                    "split_strategy": split_strategy,
                    "split_seed": split_seed,
                    "fit_class_support": class_counts,
                    "test_class_support": test_counts,
                    "split_notes": split_notes,
                    "label_availability": _label_availability_summary(
                        label_times, fit_idx, test_idx),
                }
                save_result(block, profile_id, result_dict)
                n_new += 1

                if n_new % 10 == 0 or n_new == 1:
                    elapsed = time.time() - t0
                    print(f"  [{n_new}/{n_configs * n_windows - len(completed)}] "
                          f"{config.config_id()} w{w_idx} "
                          f"p@30={result.ranking.get('precision_at_30', '?')} "
                          f"F1={result.classification['macro_f1']} "
                          f"leadS={result.survival['lead_survival']} "
                          f"impS={result.survival['important_survival']} "
                          f"T={result.temperature} "
                          f"W={result.weights_status} "
                          f"({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"  ERROR: {config.config_id()} w{w_idx}: {e}", flush=True)
                save_result(block, profile_id, {
                    "block": block,
                    "profile_id": profile_id,
                    "config_id": config.config_id(),
                    "dedup_key": dedup_key,
                    "test_window_idx": w_idx,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": str(e),
                    "config": asdict(config),
                })
                n_new += 1

    elapsed = time.time() - t0
    print(f"  Block {block} p{profile_id}: {n_new} new results in {elapsed:.1f}s", flush=True)
    return n_new


def build_B1_configs(
    encoder: str, text_mode: str, profile_id: int, seed: int = 42
) -> List[ExperimentConfig]:
    """B1: encoder × text × C sweep with LogReg."""
    configs = []
    pooling_default = "cls_norm" if encoder == "granite" else "mean"
    for C in C_SWEEP:
        configs.append(ExperimentConfig(
            encoder=encoder, text_mode=text_mode, context_mode="chunked",
            pooling=pooling_default,
            head="logreg", C=C, seed=seed, profile_id=profile_id, prob_mode="P3",
        ))
    # E5 normalization audit: add mean_norm variant
    if encoder == "e5_base":
        for C in C_SWEEP:
            configs.append(ExperimentConfig(
                encoder=encoder, text_mode=text_mode, context_mode="chunked",
                pooling="mean_norm",
                head="logreg", C=C, seed=seed, profile_id=profile_id, prob_mode="P3",
            ))
    return configs


def build_B2_configs(
    encoder: str, text_mode: str, profile_id: int, seed: int = 42
) -> List[ExperimentConfig]:
    """B2: head comparison (all heads at C=1.0, plus LogReg C=0.1)."""
    configs = []
    # Include incumbent LogReg C=1.0 and C=0.1 for reference
    for C in [1.0, 0.1]:
        configs.append(ExperimentConfig(
            encoder=encoder, text_mode=text_mode, context_mode="chunked",
            pooling="cls_norm" if encoder == "granite" else "mean",
            head="logreg", C=C, seed=seed, profile_id=profile_id, prob_mode="P3",
        ))
    # All other heads at C=1.0
    for head_name, C in B2_HEADS:
        configs.append(ExperimentConfig(
            encoder=encoder, text_mode=text_mode, context_mode="chunked",
            pooling="cls_norm" if encoder == "granite" else "mean",
            head=head_name, C=C, seed=seed, profile_id=profile_id, prob_mode="P3",
        ))
    return configs


def build_B4_configs(
    profile_id: int, seed: int = 42
) -> List[ExperimentConfig]:
    """B4: granite pooling controls (cls_norm vs cls vs mean)."""
    configs = []
    for pooling in ["cls_norm", "cls", "mean"]:
        for C in C_SWEEP:
            configs.append(ExperimentConfig(
                encoder="granite", text_mode="plain", context_mode="chunked",
                pooling=pooling,
                head="logreg", C=C, seed=seed, profile_id=profile_id, prob_mode="P3",
            ))
    return configs


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run calibration experiments")
    parser.add_argument("--blocks", nargs="+", default=["B1", "B2"],
                        help="Experiment blocks to run")
    parser.add_argument("--profiles", nargs="+", type=int, default=[2, 3, 4, 1],
                        help="Profile IDs to run (1=Seismo, 2=Digital, 3=Sicherheit, 4=EU)")
    parser.add_argument("--encoders", nargs="+", default=["e5_base", "granite"],
                        help="Encoders to test")
    parser.add_argument("--text-modes", nargs="+", default=["e5_tuned"],
                        help="Text modes to test")
    parser.add_argument("--seed", type=int, default=42, help="Split seed (backward compat)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 42],
                        help="Training seeds to run for each config")
    parser.add_argument("--split-strategy", default="grouped",
                        choices=["grouped", "chronological"],
                        help="Split strategy: grouped (D2 dev) or chronological (D3 production-time)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for split")
    parser.add_argument("--n-test-windows", type=int, default=3,
                        help="Number of chronological test windows (chronological only)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed configs")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("Magnitu Calibration Experiments")
    print(f"Blocks: {args.blocks}")
    print(f"Profiles: {args.profiles}")
    print(f"Encoders: {args.encoders}")
    print(f"Text modes: {args.text_modes}")
    print(f"Training seeds: {args.seeds}")
    print(f"Split strategy: {args.split_strategy} (seed={args.split_seed})")
    print(f"Resume: {args.resume}")
    print("=" * 70)

    total_new = 0
    t_start = time.time()

    for profile_id in args.profiles:
        print(f"\n{'─' * 70}")
        print(f"Profile {profile_id}")
        print(f"{'─' * 70}")

        labeled, y, sw, times, config, story_groups, label_times = load_profile_data(profile_id)
        if not labeled:
            print(f"  No data for profile {profile_id}")
            continue

        n_story_groups = len(np.unique(story_groups))
        n_dup_groups = int(np.sum([1 for g in np.unique(story_groups) if np.sum(story_groups == g) > 1]))
        print(f"  {len(y)} labels, {len(np.unique(y))} classes")
        print(f"  Story groups: {n_story_groups} unique ({n_dup_groups} with duplicates)")

        # Get embeddings for each encoder/text combination
        embeddings: Dict[str, np.ndarray] = {}
        for encoder in args.encoders:
            for text_mode in args.text_modes:
                if encoder == "e5_base" and text_mode == "e5_tuned":
                    print(f"  Using existing DB embeddings for e5_base/e5_tuned", flush=True)
                    emb = get_existing_embeddings(labeled)
                    save_cached(emb, "e5_base", "e5_tuned", profile_id)
                else:
                    emb = get_embeddings(encoder, text_mode, profile_id, labeled, config)
                key = f"{encoder}_{text_mode}"
                embeddings[key] = emb
                print(f"  {key}: shape={emb.shape}", flush=True)

                # E5 normalization audit: generate mean_norm variant by L2-normalizing
                # the existing mean-pooled embeddings (equivalent to mean_norm pooling)
                if encoder == "e5_base" and text_mode == "e5_tuned":
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    emb_norm = (emb / np.clip(norms, 1e-12, None)).astype(np.float32)
                    key_norm = f"{encoder}_{text_mode}_mean_norm"
                    embeddings[key_norm] = emb_norm
                    print(f"  {key_norm}: shape={emb_norm.shape} (L2-normalized)", flush=True)

        # Compute incumbent baseline for gate replay
        incumbent_metrics = None
        if "e5_base" in args.encoders and "e5_tuned" in args.text_modes:
            print(f"\n  Computing incumbent baseline (e5_base/e5_tuned/LogReg C=1.0)...", flush=True)
            X_inc = embeddings["e5_base_e5_tuned"]
            inc_config = ExperimentConfig(
                encoder="e5_base", text_mode="e5_tuned", context_mode="chunked",
                pooling="mean", head="logreg", C=1.0, seed=42,
                profile_id=profile_id, prob_mode="P3",
            )
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(FW_CLASSES)
            y_enc = le.transform(y)
            if args.split_strategy == "grouped":
                gsplit = grouped_dev_split(y_enc, story_groups, seed=args.split_seed)
                if gsplit.n_fit >= 15 and gsplit.n_test > 0:
                    inc_result = evaluate_configuration(
                        inc_config, X_inc, y, sw, gsplit.fit_idx, gsplit.test_idx,
                    )
                    incumbent_metrics = {
                        "precision_at_30": inc_result.ranking.get("precision_at_30", 0.0),
                        "macro_f1": inc_result.classification["macro_f1"],
                        "lead_recall_at_30": inc_result.ranking.get("lead_recall_at_30", 0.0),
                    }
                    print(f"  Incumbent: p@30={incumbent_metrics['precision_at_30']} "
                          f"F1={incumbent_metrics['macro_f1']} "
                          f"lr@30={incumbent_metrics['lead_recall_at_30']} "
                          f"T={inc_result.temperature}", flush=True)
            else:
                splits = chronological_split(
                    times, y_enc, story_group_ids=story_groups, n_test_windows=3,
                )
                if splits.n_fit >= 15 and splits.test_windows:
                    inc_result = evaluate_configuration(
                        inc_config, X_inc, y, sw, splits.fit_idx, splits.test_windows[0],
                    )
                    incumbent_metrics = {
                        "precision_at_30": inc_result.ranking.get("precision_at_30", 0.0),
                        "macro_f1": inc_result.classification["macro_f1"],
                        "lead_recall_at_30": inc_result.ranking.get("lead_recall_at_30", 0.0),
                    }
                    print(f"  Incumbent: p@30={incumbent_metrics['precision_at_30']} "
                          f"F1={incumbent_metrics['macro_f1']} "
                          f"lr@30={incumbent_metrics['lead_recall_at_30']} "
                          f"T={inc_result.temperature}", flush=True)

        # Run blocks for each seed
        for seed in args.seeds:
            print(f"\n  Seed {seed}", flush=True)

            for block in args.blocks:
                print(f"\n  Block {block} (seed={seed})", flush=True)

                if block == "B1":
                    for encoder in args.encoders:
                        for text_mode in args.text_modes:
                            key = f"{encoder}_{text_mode}"
                            if key not in embeddings:
                                continue
                            configs = build_B1_configs(encoder, text_mode, profile_id, seed)
                            # Split configs by pooling to route to correct embeddings
                            pooling_keys = {c.pooling for c in configs}
                            for pool_key in sorted(pooling_keys):
                                pool_configs = [c for c in configs if c.pooling == pool_key]
                                if pool_key == "mean_norm":
                                    emb_key = f"{key}_mean_norm"
                                else:
                                    emb_key = key
                                if emb_key not in embeddings:
                                    print(f"    SKIP {encoder}/{text_mode}/{pool_key}: no embeddings", flush=True)
                                    continue
                                print(f"    {encoder}/{text_mode}/{pool_key}: {len(pool_configs)} configs", flush=True)
                                n = run_block(
                                    f"B1_{encoder}_{text_mode}_{pool_key}", profile_id,
                                    encoder, text_mode,
                                    embeddings[emb_key], y, sw, times,
                                    pool_configs, incumbent_metrics, args.resume,
                                    story_group_ids=story_groups,
                                    label_times=label_times,
                                    split_strategy=args.split_strategy,
                                    split_seed=args.split_seed,
                                    n_test_windows=args.n_test_windows,
                                )
                                total_new += n

                elif block == "B2":
                    for encoder in args.encoders:
                        for text_mode in args.text_modes:
                            key = f"{encoder}_{text_mode}"
                            if key not in embeddings:
                                continue
                            configs = build_B2_configs(encoder, text_mode, profile_id, seed)
                            print(f"    {encoder}/{text_mode}: {len(configs)} configs", flush=True)
                            n = run_block(
                                f"B2_{encoder}_{text_mode}", profile_id,
                                encoder, text_mode,
                                embeddings[key], y, sw, times,
                                configs, incumbent_metrics, args.resume,
                                story_group_ids=story_groups,
                                label_times=label_times,
                                split_strategy=args.split_strategy,
                                split_seed=args.split_seed,
                                    n_test_windows=args.n_test_windows,
                            )
                            total_new += n

                elif block == "B4":
                    for pooling in ["cls", "mean"]:
                        cache_key_str = f"granite_plain_{pooling}_p{profile_id}.npy"
                        p = cache_dir() / cache_key_str
                        if p.exists():
                            emb = np.load(p)
                        else:
                            print(f"    Generating granite/plain/{pooling} embeddings...", flush=True)
                            emb = encode_with_model(
                                GRANITE_MODEL, labeled, config, "plain", pooling=pooling,
                            )
                            np.save(p, emb)
                        key = f"granite_plain_{pooling}"
                        embeddings[key] = emb

                    configs = build_B4_configs(profile_id, seed)
                    print(f"    granite/plain pooling variants: {len(configs)} configs", flush=True)
                    for pooling in ["cls_norm", "cls", "mean"]:
                        key = f"granite_plain_{pooling}" if pooling != "cls_norm" else "granite_plain"
                        if key not in embeddings:
                            if pooling == "cls_norm":
                                embeddings[key] = embeddings.get("granite_plain", get_embeddings("granite", "plain", profile_id, labeled, config))
                            else:
                                continue
                        pool_configs = [c for c in configs if c.pooling == pooling]
                        if pool_configs:
                            n = run_block(
                                f"B4_granite_plain_{pooling}", profile_id,
                                "granite", "plain",
                                embeddings[key], y, sw, times,
                                pool_configs, incumbent_metrics, args.resume,
                                story_group_ids=story_groups,
                                label_times=label_times,
                                split_strategy=args.split_strategy,
                                split_seed=args.split_seed,
                                    n_test_windows=args.n_test_windows,
                        )
                        total_new += n

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Done: {total_new} new results in {elapsed:.1f}s")
    print(f"Results in: {results_dir()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
