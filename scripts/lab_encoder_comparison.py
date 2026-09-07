#!/usr/bin/env python3
"""Encoder comparison: E5-base vs GTE-multilingual-base vs Granite vs E5-large-instruct.

Tests the question: is there a newer/better encoder than multilingual-e5-base
for Seismo's desk classification task?

Models tested:
  - intfloat/multilingual-e5-base       (current production, 278M, 768d, 512 ctx)
  - Alibaba-NLP/gte-multilingual-base   (newer, 305M, 768d, 8192 ctx)
  - ibm-granite/granite-embedding-311m-multilingual-r2 (tested before, 311M, 768d)
  - intfloat/multilingual-e5-large-instruct (560M, 1024d, 512 ctx)

All use the same e5_tuned text pipeline, chunked context, LogReg C=0.01 (production),
balanced classes, P1 calibration (temperature only), and the same grouped split.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_encoder_comparison.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_encoder_comparison.py --profile 4
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import db
from pipeline import (
    CLASSES,
    CLASS_WEIGHT_MAP,
    _stable_train_test_split,
    _holdout_test_fraction,
    compute_sample_weights,
    build_prior_fit,
    _prior_offset_vector,
    _add_logit_offsets,
    _fit_temperature_scalar,
    _collect_oof_logits,
    _oof_fold_count,
    _min_class_count_in_labels,
    _ranking_metrics,
    _build_entry_text,
    _split_text_chunks,
    is_legal_training_entry,
    is_analytical_training_entry,
    MAX_EMBED_CHUNKS,
    MAX_EMBED_CHUNKS_LEGAL,
    MAX_EMBED_CHUNKS_ANALYTICAL,
)
from config import get_config

# ── Models ───────────────────────────────────────────────────────────

MODELS = {
    "e5_base": {
        "name": "intfloat/multilingual-e5-base",
        "pooling": "mean",
        "prefix": "passage: ",
        "l2_final": False,
        "l2_chunks": False,
    },
    "granite": {
        "name": "ibm-granite/granite-embedding-311m-multilingual-r2",
        "pooling": "cls",
        "prefix": "",
        "l2_final": True,
        "l2_chunks": True,
    },
    "e5_large_instruct": {
        "name": "intfloat/multilingual-e5-large-instruct",
        "pooling": "mean",
        "prefix": "passage: ",
        "l2_final": False,
        "l2_chunks": False,
    },
    "bge_m3": {
        "name": "BAAI/bge-m3",
        "pooling": "cls",
        "prefix": "",
        "l2_final": True,
        "l2_chunks": False,
    },
    "gte_multilingual": {
        "name": "Alibaba-NLP/gte-multilingual-base",
        "pooling": "mean",
        "prefix": "",
        "l2_final": True,
        "l2_chunks": False,
        "trust_remote_code": True,
        "fix_position_ids": True,  # Known bug: persistent=False buffer gets garbage on load
        "force_float32": True,     # Custom RoPE code overflows in float16 on this GPU
    },
}

C_VALUE = 0.01  # Production C


# ── Encoding ─────────────────────────────────────────────────────────

def encode_with_model(
    model_key: str,
    model_cfg: Dict[str, Any],
    labeled: List[dict],
    config: dict,
) -> np.ndarray:
    """Encode labeled entries with a given model. Returns (n, dim) array."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = model_cfg["name"]
    pooling = model_cfg["pooling"]
    prefix = model_cfg["prefix"]
    l2_final = model_cfg["l2_final"]
    l2_chunks = model_cfg["l2_chunks"]
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    fix_position_ids = model_cfg.get("fix_position_ids", False)
    force_float32 = model_cfg.get("force_float32", False)

    patterns = config.get("legal_signal_patterns") or []

    # Build texts + chunk
    chunk_texts = []
    chunk_meta = []
    for i, entry in enumerate(labeled):
        text = _build_entry_text(entry, legal_patterns=patterns)
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

    print(f"  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if force_float32 else (torch.float16 if device.type == "cuda" else torch.float32)
    # low_cpu_mem_usage breaks custom remote-code models (GTE): meta tensor init
    # doesn't properly fill persistent=False buffers, producing all-NaN outputs
    model = AutoModel.from_pretrained(
        model_name, dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)
    print(f"  Model dtype: {next(model.parameters()).dtype}", flush=True)

    # GTE fix: persistent=False buffers get garbage/NaN on from_pretrained load.
    # Three buffers are affected: position_ids, rotary inv_freq, and cos/sin caches.
    if fix_position_ids:
        max_pos = model.config.max_position_embeddings
        model.embeddings.position_ids = torch.arange(max_pos, device=device)

        # Rebuild rotary embedding: recompute inv_freq from scratch, then rebuild cache
        re = model.embeddings.rotary_emb
        dim = re.dim
        base = re.base
        scaling_factor = re.scaling_factor
        # NTK-scaled inv_freq (matches NTKScalingRotaryEmbedding._set_cos_sin_cache logic)
        ntk_base = base * scaling_factor
        inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        re.register_buffer("inv_freq", inv_freq, persistent=False)
        # Rebuild cos/sin cache with the scaled max length
        scaled_max = max_pos * scaling_factor
        re._set_cos_sin_cache(int(scaled_max), device, torch.get_default_dtype())

        print(f"  Applied position_ids + rotary fix (max_pos={max_pos}, scale={scaling_factor})", flush=True)

    print(f"  Encoding {len(chunk_texts)} chunks ({len(labeled)} entries) on {device.type}...", flush=True)

    batch_size = 16 if device.type == "cuda" else 4
    all_embs = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        if prefix:
            batch = [prefix + t if not t.startswith(prefix.strip()) else t for t in batch]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        token_emb = outputs.last_hidden_state.float()
        if pooling == "cls":
            chunk_embs = token_emb[:, 0, :].cpu().numpy()
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            chunk_embs = (summed / counts).cpu().numpy()
        all_embs.append(chunk_embs)
        done = min(i + batch_size, len(chunk_texts))
        if done % 200 == 0 or done == len(chunk_texts):
            print(f"    {done}/{len(chunk_texts)} chunks", flush=True)

    all_embs = np.vstack(all_embs) if all_embs else np.array([])

    if all_embs.size and np.isnan(all_embs).any():
        nan_count = np.isnan(all_embs).any(axis=1).sum()
        print(f"  WARNING: {nan_count}/{len(all_embs)} chunk embeddings have NaN!", flush=True)

    # L2-normalize chunk embeddings if needed (granite)
    if l2_chunks and all_embs.size:
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

    out_arr = np.array(out)

    # L2-normalize final per-entry vector if needed
    if l2_final and out_arr.size:
        norms = np.linalg.norm(out_arr, axis=1, keepdims=True)
        out_arr = out_arr / np.clip(norms, 1e-12, None)

    del model, tokenizer
    import gc; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out_arr


# ── Cache ────────────────────────────────────────────────────────────

def cache_dir() -> Path:
    d = Path(__file__).parent.parent / "lab_data" / "encoder_comparison_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(model_key: str, profile_id: int) -> Path:
    return cache_dir() / f"{model_key}_p{profile_id}.npy"


def load_cached(model_key: str, profile_id: int) -> Optional[np.ndarray]:
    p = cache_path(model_key, profile_id)
    if p.exists():
        return np.load(p)
    return None


def save_cached(arr: np.ndarray, model_key: str, profile_id: int):
    np.save(cache_path(model_key, profile_id), arr)


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_encoder(
    X: np.ndarray,
    labeled: List[dict],
    seed: int = 42,
) -> dict:
    """Train LogReg C=0.01 on train split, evaluate on test split.
    Returns metrics dict with ranking, F1, and calibration.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    y = [l["label"] for l in labeled]
    n = len(y)
    min_class = int(pd.Series(y).value_counts().min())
    n_folds = min(5, max(2, min_class // 2)) if min_class >= 4 else 2

    # Simple stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    labeled_train = [labeled[i] for i in train_idx]

    # Sample weights
    sw = compute_sample_weights(labeled_train)

    # Train LogReg
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C_VALUE, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
            multi_class="multinomial",
        )),
    ])
    clf.fit(X_train, y_train, clf__sample_weight=sw)

    # Predict
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    class_names = clf.named_steps["clf"].classes_

    # F1
    f1 = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    # Ranking metrics
    rank = _ranking_metrics(probs, class_names, y_test, k=30)

    # OOF temperature (P1)
    try:
        oof_logits = _collect_oof_logits(
            X_train, y_train,
            n_folds=n_folds, seed=seed,
            C=C_VALUE, sample_weight=sw,
        )
        if oof_logits is not None:
            temperature = _fit_temperature_scalar(oof_logits, y_train, class_names)
        else:
            temperature = 1.0
    except Exception:
        temperature = 1.0

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "calibration_temperature": round(temperature, 4),
        "class_distribution": dict(pd.Series(y_test).value_counts().to_dict()),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Encoder comparison for Seismo desks")
    parser.add_argument("--profile", type=int, nargs="+", default=[2, 3, 4],
                        help="Profile IDs to test (default: 2=digital, 3=sicherheit, 4=eu)")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip models that have cached embeddings")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Model keys to test")
    args = parser.parse_args()

    config = get_config()

    results = {}
    for pid in args.profile:
        print(f"\n{'='*60}")
        print(f"Profile {pid}")
        print(f"{'='*60}")

        # Get labels
        labels = db.get_all_labels(profile_id=pid)
        if len(labels) < 20:
            print(f"  Skipping: only {len(labels)} labels")
            continue

        # Join with entries
        all_entries = db.get_all_entries(include_embedding=False)
        entry_map = {}
        for e in all_entries:
            key = db.entry_key_from_mapping(e)
            entry_map[key] = e

        labeled = []
        for lbl in labels:
            key = db.entry_key_from_mapping(lbl)
            entry = entry_map.get(key)
            if entry:
                merged = {**entry, **lbl}
                labeled.append(merged)

        if len(labeled) < 20:
            print(f"  Skipping: only {len(labeled)} labeled entries with text")
            continue

        y = [l["label"] for l in labeled]
        print(f"  {len(labeled)} labeled entries, classes: {dict(pd.Series(y).value_counts().to_dict())}")

        results[pid] = {}

        for model_key in args.models:
            if model_key not in MODELS:
                print(f"  Unknown model: {model_key}, skipping")
                continue

            model_cfg = MODELS[model_key]
            print(f"\n  --- {model_key} ({model_cfg['name']}) ---")

            # Check cache
            X = None
            if not args.skip_cached:
                X = load_cached(model_key, pid)

            if X is None:
                t0 = time.time()
                X = encode_with_model(model_key, model_cfg, labeled, config)
                elapsed = time.time() - t0
                print(f"  Encoded {X.shape} in {elapsed:.1f}s")
                save_cached(X, model_key, pid)
            else:
                print(f"  Loaded cached embeddings: {X.shape}")

            # Evaluate
            metrics = evaluate_encoder(X, labeled, seed=42)
            results[pid][model_key] = metrics
            print(f"  F1={metrics['f1_score']}  p@30={metrics['precision_at_30']}  "
                  f"lr@30={metrics['lead_recall_at_30']}  AUC={metrics['ranking_auc']}  "
                  f"T={metrics['calibration_temperature']}")

    # Save results
    out_path = Path(__file__).parent.parent / "lab_data" / "encoder_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY (LogReg C=0.01, P1 calibration, seed 42)")
    print(f"{'='*80}")
    print(f"{'Profile':<10} {'Model':<20} {'F1':>8} {'p@30':>8} {'lr@30':>8} {'AUC':>8} {'T':>8}")
    print("-" * 80)
    for pid in sorted(results.keys()):
        for model_key in MODELS:
            if model_key in results[pid]:
                m = results[pid][model_key]
                print(f"{pid:<10} {model_key:<20} {m['f1_score']:>8} "
                      f"{m['precision_at_30']:>8} {m['lead_recall_at_30']:>8} "
                      f"{m['ranking_auc']:>8} {m['calibration_temperature']:>8}")
    print()


if __name__ == "__main__":
    main()
