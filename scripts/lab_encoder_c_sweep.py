#!/usr/bin/env python3
"""Comprehensive encoder × text-pipeline × C sweep.

Tests whether granite's audit underperformance is due to:
  (a) wrong regularization (C=1.0 only)
  (b) e5-tuned text pipeline (title repetition, legal snippets, source context)
  (c) the task itself (retrieval benchmark ≠ classification)

Matrix:
  Encoders:   e5-base (cached + fresh), granite-311m-r2 (fresh)
  Text pipes: e5-tuned (current), plain (title+body), plain+context
  C sweep:    [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

Caches fresh embeddings to disk (lab_data/embeddings_cache/) so re-runs are fast.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_encoder_c_sweep.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_encoder_c_sweep.py --profile 4
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_encoder_c_sweep.py --skip-fresh-e5
"""
import argparse
import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

import db
import pipeline as pipe
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
    bytes_to_embedding,
    _LabelDecodingClassifier,
    _build_entry_text,
    _split_text_chunks,
    is_legal_training_entry,
    is_analytical_training_entry,
    MAX_EMBED_CHUNKS,
    MAX_EMBED_CHUNKS_LEGAL,
    MAX_EMBED_CHUNKS_ANALYTICAL,
)
from config import get_config, DEFAULT_TRANSFORMER_MODEL
from ml_window import evaluate_model_update

C_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
GRANITE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"

# Text pipeline variants
TEXT_VARIANTS = ["e5_tuned", "plain", "plain_context"]


# ── Text pipeline variants ──────────────────────────────────────────

def build_text_e5_tuned(entry: dict, config: dict) -> str:
    """Current pipeline: title repetition, legal signals, source context."""
    return _build_entry_text(entry, legal_patterns=config.get("legal_signal_patterns") or [])


def build_text_plain(entry: dict, config: dict) -> str:
    """Minimal: title + body only. No repetition, no snippets, no context prefix."""
    title = (entry.get("title") or "").strip()
    from pipeline import training_corpus_text, _content_cap_for_entry
    body = training_corpus_text(entry)[:_content_cap_for_entry(entry)]
    parts = [p for p in [title, body] if p]
    return "\n".join(parts) if parts else "(empty)"


def build_text_plain_context(entry: dict, config: dict) -> str:
    """Plain + natural source context (source type, publisher). No title repetition or legal snippets."""
    title = (entry.get("title") or "").strip()
    from pipeline import training_corpus_text, _content_cap_for_entry, _natural_source_context
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


# ── Embedding cache ─────────────────────────────────────────────────

def cache_dir() -> Path:
    d = Path(__file__).parent.parent / "lab_data" / "embeddings_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(encoder: str, text_variant: str, profile_id: int) -> str:
    return f"{encoder}_{text_variant}_p{profile_id}.npy"


def load_cached(encoder: str, text_variant: str, profile_id: int) -> Optional[np.ndarray]:
    p = cache_dir() / cache_key(encoder, text_variant, profile_id)
    if p.exists():
        return np.load(p)
    return None


def save_cached(arr: np.ndarray, encoder: str, text_variant: str, profile_id: int):
    p = cache_dir() / cache_key(encoder, text_variant, profile_id)
    np.save(p, arr)


# ── Encoding ────────────────────────────────────────────────────────

def encode_with_model(
    model_name: str,
    labeled: List[dict],
    config: dict,
    text_variant: str,
) -> np.ndarray:
    """Encode labeled entries with a given model and text pipeline. Returns (n, dim) array."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    text_fn = TEXT_BUILDERS[text_variant]
    patterns = config.get("legal_signal_patterns") or []

    # Build texts + chunk
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

    # E5 needs passage: prefix; granite doesn't
    is_e5 = "e5" in model_name.lower()
    is_granite = "granite" in model_name.lower()
    # Granite: CLS-token pooling + L2 normalization (per IBM model card)
    # E5: mean pooling (per intfloat model card)
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
        if is_granite:
            # CLS-token pooling (first token of each sequence)
            chunk_embs = token_emb[:, 0, :].cpu().numpy()
        else:
            # Mean pooling (e5)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            chunk_embs = (summed / counts).cpu().numpy()
        all_embs.append(chunk_embs)
        done = min(i + batch_size, len(chunk_texts))
        if done % 200 == 0 or done == len(chunk_texts):
            print(f"      {done}/{len(chunk_texts)} chunks", flush=True)

    all_embs = np.vstack(all_embs) if all_embs else np.array([])

    # L2-normalize granite chunk embeddings before pooling (per model card)
    if is_granite and all_embs.size:
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

    # L2-normalize the final per-entry vector for granite
    if is_granite and out:
        out_arr = np.array(out)
        norms = np.linalg.norm(out_arr, axis=1, keepdims=True)
        out_arr = out_arr / np.clip(norms, 1e-12, None)
        return out_arr

    del model, tokenizer
    import gc; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(out)


def get_embeddings(
    encoder: str,
    text_variant: str,
    profile_id: int,
    labeled: List[dict],
    config: dict,
    use_cache: bool = True,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Get embeddings for a encoder×text combo, with disk caching for fresh encodes."""
    if encoder == "e5_base" and text_variant == "e5_tuned":
        # Use cached DB embeddings (fastest path)
        embedding_dim = config.get("embedding_dim", 768)
        conn = db.get_db()
        emb_map = {}
        rows = conn.execute(
            "SELECT entry_type, entry_id, embedding FROM entries WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            emb_map[db.entry_key_from_mapping(row)] = row["embedding"]
        conn.close()
        X, y, lbl = [], [], []
        for l in labeled:
            key = db.entry_key_from_mapping(l)
            eb = emb_map.get(key)
            if eb:
                X.append(bytes_to_embedding(eb, embedding_dim))
                y.append(l["label"])
                lbl.append(l)
        return np.array(X), y, lbl

    # Fresh encode (with disk cache)
    cached = load_cached(encoder, text_variant, profile_id) if use_cache else None
    if cached is not None:
        print(f"    [cache hit] {encoder}/{text_variant}/p{profile_id}", flush=True)
        y = [l["label"] for l in labeled]
        return cached, y, labeled

    model_name = GRANITE_MODEL if encoder == "granite" else DEFAULT_TRANSFORMER_MODEL
    print(f"    [fresh encode] {encoder}/{text_variant}/p{profile_id}", flush=True)
    X = encode_with_model(model_name, labeled, config, text_variant)
    save_cached(X, encoder, text_variant, profile_id)
    y = [l["label"] for l in labeled]
    return X, y, labeled


# ── Training + eval ─────────────────────────────────────────────────

def logreg_head(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42,
        )),
    ])


def train_eval(C, X_tr, y_tr, X_te, y_te, sw_tr, config) -> dict:
    le = LabelEncoder(); le.fit(CLASSES)
    y_tr_enc = le.transform(y_tr)
    cn_fit = le.classes_.tolist()

    n_folds = _oof_fold_count(len(y_tr), _min_class_count_in_labels(y_tr))
    prior_fit = build_prior_fit(y_tr, sw_tr, cn_fit, config)
    temp = 1.0; oof_s = 0
    if n_folds >= 2:
        oof_logits, oof_y = _collect_oof_logits(X_tr, y_tr, sw_tr, le, n_folds)
        oof_s = len(oof_y)
        if oof_s >= 3:
            off = _prior_offset_vector({"prior_fit": prior_fit}, cn_fit)
            if off is not None:
                oof_logits = _add_logit_offsets(oof_logits, off)
            temp = _fit_temperature_scalar(oof_logits, np.array(oof_y), cn_fit)

    cal = {"version": 2, "method": "temperature", "temperature": temp,
           "class_names": cn_fit, "prior_fit": prior_fit,
           "oof_samples": oof_s, "oof_folds": n_folds if oof_s >= 3 else 0,
           "calibration_fit": "oof" if oof_s >= 3 else "none"}

    clf_p = logreg_head(C)
    kw = {}
    if sw_tr is not None and len(sw_tr) > 0 and float(np.std(sw_tr)) > 1e-6:
        kw["classifier__sample_weight"] = sw_tr
    clf_p.fit(X_tr, y_tr_enc, **kw)
    clf = _LabelDecodingClassifier(clf_p, le)

    probs, cn = pipe.classifier_probabilities(clf, X_te, "", cal=cal)
    y_pred = np.array([cn[i] for i in np.argmax(probs, axis=1)])
    rank = _ranking_metrics(probs, cn, y_te)

    # ── Survival metrics at the production alert threshold (0.55) ──
    # Composite score = sum(P(class) * CLASS_WEIGHT_MAP[class])
    # This is what Seismo's alert_threshold applies to in production.
    w = np.array([CLASS_WEIGHT_MAP.get(cn[i], 0.0) for i in range(len(cn))], dtype=float)
    composite = probs.dot(w)
    threshold = 0.55
    y_te_arr = np.array(y_te)
    leads = y_te_arr == "investigation_lead"
    importants = y_te_arr == "important"
    above = composite >= threshold
    # Survival: fraction of leads/importants that pass the threshold
    lead_survival = float(above[leads].sum()) / max(int(leads.sum()), 1)
    important_survival = float(above[importants].sum()) / max(int(importants.sum()), 1)
    # Threshold precision: of items above threshold, what fraction are relevant (lead+important)?
    relevant = leads | importants
    thresh_precision = float(relevant[above].sum()) / max(int(above.sum()), 1)
    # How many items pass the threshold (candidate volume)
    n_above = int(above.sum())

    return {
        "C": C,
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1_score": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "precision": round(precision_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "temperature": round(temp, 4),
        # Survival at production threshold (0.55)
        "survival_threshold": threshold,
        "lead_survival": round(lead_survival, 4),
        "important_survival": round(important_survival, 4),
        "threshold_precision": round(thresh_precision, 4),
        "n_above_threshold": n_above,
        "n_leads_holdout": int(leads.sum()),
        "n_importants_holdout": int(importants.sum()),
    }


# ── Main sweep ──────────────────────────────────────────────────────

def run_combo(profile_id, config, skip_fresh_e5=False) -> dict:
    labeled = db.get_all_labels(profile_id)
    if len(labeled) < 20:
        return {"profile_id": profile_id, "error": f"only {len(labeled)} labels"}

    profile = db.get_profile_by_id(profile_id)
    name = profile.get("display_name", str(profile_id)) if profile else str(profile_id)

    # Define the test matrix
    combos = []
    # e5 with e5_tuned text (cached DB embeddings)
    combos.append(("e5_base", "e5_tuned", False))
    # e5 with plain and plain_context (fresh encode)
    if not skip_fresh_e5:
        combos.append(("e5_base", "plain", True))
        combos.append(("e5_base", "plain_context", True))
    # granite with all text variants (fresh encode)
    combos.append(("granite", "e5_tuned", True))
    combos.append(("granite", "plain", True))
    combos.append(("granite", "plain_context", True))

    all_combo_results = {}
    incumbent = None

    for encoder, text_var, is_fresh in combos:
        label = f"{encoder}/{text_var}"
        print(f"\n  === {label} ===", flush=True)
        X, y, lbl = get_embeddings(encoder, text_var, profile_id, labeled, config)
        if len(X) < 20:
            print(f"    SKIP: only {len(X)} embeddings")
            all_combo_results[label] = {"error": f"only {len(X)} embeddings"}
            continue

        sw = compute_sample_weights(lbl, config)
        min_class = int(pd.Series(y).value_counts().min())
        if min_class < 2:
            print(f"    SKIP: min class < 2")
            all_combo_results[label] = {"error": "min class < 2"}
            continue
        test_size = _holdout_test_fraction(min_class, len(y))
        X_tr, X_te, y_tr, y_te, sw_tr, _ = _stable_train_test_split(
            X, y, sw, lbl, test_size=test_size
        )

        c_results = []
        for C in C_SWEEP:
            print(f"    C={C}...", end=" ", flush=True)
            r = train_eval(C, X_tr, y_tr, X_te, y_te, sw_tr, config)
            c_results.append(r)
            print(f"p@30={r['precision_at_30']:.4f} F1={r['f1_score']:.4f} AUC={r['ranking_auc']:.4f} "
                  f"surv={r['lead_survival']:.2f}/{r['important_survival']:.2f} "
                  f"n>={r['n_above_threshold']} prec={r['threshold_precision']:.2f}")

        best = max(c_results, key=lambda r: r["precision_at_30"])
        all_combo_results[label] = {
            "encoder": encoder, "text_variant": text_var,
            "train_size": len(y_tr), "test_size": len(y_te),
            "c_results": c_results, "best": best,
        }

        if encoder == "e5_base" and text_var == "e5_tuned":
            # Find incumbent C=1.0
            for r in c_results:
                if r["C"] == 1.0:
                    incumbent = r
                    break

    # Promote gates
    gates = {}
    if incumbent:
        for label, cr in all_combo_results.items():
            if "error" in cr or label == "e5_base/e5_tuned":
                continue
            best = cr["best"]
            promoted = evaluate_model_update(incumbent, best)
            gates[label] = {
                "best_C": best["C"],
                "promote_vs_incumbent": promoted,
                "p30_delta": round(best["precision_at_30"] - incumbent["precision_at_30"], 4),
                "f1_delta": round(best["f1_score"] - incumbent["f1_score"], 4),
                "best_p30": best["precision_at_30"],
                "best_f1": best["f1_score"],
            }

    # Also: best e5/plain vs best granite/plain (fair text comparison)
    fair_gates = {}
    e5_plain = all_combo_results.get("e5_base/plain", {})
    gr_plain = all_combo_results.get("granite/plain", {})
    if "best" in e5_plain and "best" in gr_plain:
        fair_gates["granite_plain_vs_e5_plain"] = {
            "e5_best_C": e5_plain["best"]["C"],
            "e5_p30": e5_plain["best"]["precision_at_30"],
            "e5_f1": e5_plain["best"]["f1_score"],
            "e5_lead_surv": e5_plain["best"]["lead_survival"],
            "e5_imp_surv": e5_plain["best"]["important_survival"],
            "e5_thresh_prec": e5_plain["best"]["threshold_precision"],
            "e5_n_above": e5_plain["best"]["n_above_threshold"],
            "granite_best_C": gr_plain["best"]["C"],
            "granite_p30": gr_plain["best"]["precision_at_30"],
            "granite_f1": gr_plain["best"]["f1_score"],
            "granite_lead_surv": gr_plain["best"]["lead_survival"],
            "granite_imp_surv": gr_plain["best"]["important_survival"],
            "granite_thresh_prec": gr_plain["best"]["threshold_precision"],
            "granite_n_above": gr_plain["best"]["n_above_threshold"],
            "p30_delta": round(gr_plain["best"]["precision_at_30"] - e5_plain["best"]["precision_at_30"], 4),
            "f1_delta": round(gr_plain["best"]["f1_score"] - e5_plain["best"]["f1_score"], 4),
            "lead_surv_delta": round(gr_plain["best"]["lead_survival"] - e5_plain["best"]["lead_survival"], 4),
            "imp_surv_delta": round(gr_plain["best"]["important_survival"] - e5_plain["best"]["important_survival"], 4),
        }
    e5_pc = all_combo_results.get("e5_base/plain_context", {})
    gr_pc = all_combo_results.get("granite/plain_context", {})
    if "best" in e5_pc and "best" in gr_pc:
        fair_gates["granite_pc_vs_e5_pc"] = {
            "e5_best_C": e5_pc["best"]["C"],
            "e5_p30": e5_pc["best"]["precision_at_30"],
            "e5_f1": e5_pc["best"]["f1_score"],
            "e5_lead_surv": e5_pc["best"]["lead_survival"],
            "e5_imp_surv": e5_pc["best"]["important_survival"],
            "e5_thresh_prec": e5_pc["best"]["threshold_precision"],
            "e5_n_above": e5_pc["best"]["n_above_threshold"],
            "granite_best_C": gr_pc["best"]["C"],
            "granite_p30": gr_pc["best"]["precision_at_30"],
            "granite_f1": gr_pc["best"]["f1_score"],
            "granite_lead_surv": gr_pc["best"]["lead_survival"],
            "granite_imp_surv": gr_pc["best"]["important_survival"],
            "granite_thresh_prec": gr_pc["best"]["threshold_precision"],
            "granite_n_above": gr_pc["best"]["n_above_threshold"],
            "p30_delta": round(gr_pc["best"]["precision_at_30"] - e5_pc["best"]["precision_at_30"], 4),
            "f1_delta": round(gr_pc["best"]["f1_score"] - e5_pc["best"]["f1_score"], 4),
            "lead_surv_delta": round(gr_pc["best"]["lead_survival"] - e5_pc["best"]["lead_survival"], 4),
            "imp_surv_delta": round(gr_pc["best"]["important_survival"] - e5_pc["best"]["important_survival"], 4),
        }

    return {
        "profile_id": profile_id,
        "profile_name": name,
        "total_labels": len(labeled),
        "label_distribution": {k: int(v) for k, v in pd.Series([l["label"] for l in labeled]).value_counts().items()},
        "incumbent": incumbent,
        "combos": all_combo_results,
        "gates_vs_incumbent": gates,
        "fair_comparison": fair_gates,
    }


def main():
    parser = argparse.ArgumentParser(description="Encoder × text-pipeline × C sweep")
    parser.add_argument("--profile", type=int, default=None)
    parser.add_argument("--skip-fresh-e5", action="store_true", help="Skip fresh e5 encoding (use cached DB only)")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else [1, 3, 4]
    all_results = []

    for pid in profiles:
        profile = db.get_profile_by_id(pid)
        name = profile.get("display_name", str(pid)) if profile else str(pid)
        print(f"\n{'='*70}")
        print(f"Profile {pid}: {name}")
        print(f"{'='*70}")
        config = db.get_effective_config(pid)
        result = run_combo(pid, config, skip_fresh_e5=args.skip_fresh_e5)
        all_results.append(result)

        if "error" in result:
            print(f"  SKIP: {result['error']}")
            continue

        inc = result["incumbent"]
        print(f"\n  Incumbent (e5/e5_tuned C=1.0): p@30={inc['precision_at_30']:.4f}  F1={inc['f1_score']:.4f}  "
              f"lead_surv={inc['lead_survival']:.2f}  imp_surv={inc['important_survival']:.2f}  "
              f"n>={inc['n_above_threshold']}  thresh_prec={inc['threshold_precision']:.2f}")
        print(f"\n  {'Combo':<35s} {'C':>5s} {'p@30':>6s} {'F1':>6s} {'AUC':>6s} {'Gate':>8s} {'Δp@30':>7s} "
              f"{'leadS':>6s} {'impS':>6s} {'n>=':>4s} {'tPrec':>6s}")
        print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*4} {'-'*6}")
        for label, cr in result["combos"].items():
            if "error" in cr:
                print(f"  {label:<35s}  ERROR: {cr['error']}")
                continue
            b = cr["best"]
            g = result["gates_vs_incumbent"].get(label, {})
            gate_str = "PROMOTE" if g.get("promote_vs_incumbent") else "reject" if g else "—"
            dp = g.get("p30_delta", 0)
            print(f"  {label:<35s} {b['C']:>5.2f} {b['precision_at_30']:>6.4f} {b['f1_score']:>6.4f} "
                  f"{b['ranking_auc']:>6.4f} {gate_str:>8s} {dp:>+7.4f} "
                  f"{b['lead_survival']:>6.2f} {b['important_survival']:>6.2f} "
                  f"{b['n_above_threshold']:>4d} {b['threshold_precision']:>6.2f}")

        if result["fair_comparison"]:
            print(f"\n  Fair comparisons (same text pipeline, e5 vs granite, best C each):")
            for k, v in result["fair_comparison"].items():
                print(f"    {k}:")
                print(f"      e5:      C={v['e5_best_C']:<5} p@30={v['e5_p30']:.4f}  F1={v['e5_f1']:.4f}  "
                      f"leadS={v['e5_lead_surv']:.2f}  impS={v['e5_imp_surv']:.2f}  n>={v['e5_n_above']}  tPrec={v['e5_thresh_prec']:.2f}")
                print(f"      granite: C={v['granite_best_C']:<5} p@30={v['granite_p30']:.4f}  F1={v['granite_f1']:.4f}  "
                      f"leadS={v['granite_lead_surv']:.2f}  impS={v['granite_imp_surv']:.2f}  n>={v['granite_n_above']}  tPrec={v['granite_thresh_prec']:.2f}")
                print(f"      Δp@30={v['p30_delta']:+.4f}  ΔF1={v['f1_delta']:+.4f}  "
                      f"ΔleadS={v['lead_surv_delta']:+.4f}  ΔimpS={v['imp_surv_delta']:+.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  Profile {r['profile_id']}: {r['error']}")
            continue
        inc = r["incumbent"]
        print(f"\n  Profile {r['profile_id']} ({r['profile_name']}): "
              f"{r['total_labels']} labels, dist={r['label_distribution']}")
        print(f"    Incumbent e5/e5_tuned C=1.0: p@30={inc['precision_at_30']:.4f}  F1={inc['f1_score']:.4f}  "
              f"lead_surv={inc['lead_survival']:.2f}  imp_surv={inc['important_survival']:.2f}  "
              f"n>={inc['n_above_threshold']}  thresh_prec={inc['threshold_precision']:.2f}")

        # Best overall
        best_label = None; best_p30 = -1
        for label, cr in r["combos"].items():
            if "error" in cr:
                continue
            if cr["best"]["precision_at_30"] > best_p30:
                best_p30 = cr["best"]["precision_at_30"]
                best_label = label
                best_c = cr["best"]["C"]
                best_f1 = cr["best"]["f1_score"]
                best_ls = cr["best"]["lead_survival"]
                best_is = cr["best"]["important_survival"]
        if best_label:
            print(f"    BEST: {best_label} C={best_c} → p@30={best_p30:.4f}  F1={best_f1:.4f}  "
                  f"leadS={best_ls:.2f}  impS={best_is:.2f}")

        # Fair comparison verdict
        fc = r.get("fair_comparison", {})
        for k, v in fc.items():
            winner = "granite" if v["p30_delta"] > 0.01 else ("e5" if v["p30_delta"] < -0.01 else "tie")
            print(f"    {k}: {winner} (Δp@30={v['p30_delta']:+.4f}, ΔF1={v['f1_delta']:+.4f}, "
                  f"ΔleadS={v['lead_surv_delta']:+.4f}, ΔimpS={v['imp_surv_delta']:+.4f})")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
