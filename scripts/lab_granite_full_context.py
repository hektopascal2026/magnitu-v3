#!/usr/bin/env python3
"""Granite full-context test: no chunking, single-pass encoding up to 8K tokens.

Granite supports 32K context but e5 is capped at 512. The chunking+pooling
strategy in the main pipeline was designed for e5's 512 limit. This test
feeds the full entry text to granite in one pass, which is how granite was
trained and where it should have a structural advantage.

Also tests a "title_boost" variant: granite may not need the title-repetition
trick that e5 uses, but a milder version (title prepended once, not repeated)
might help without the e5-style noise.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_granite_full_context.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_granite_full_context.py --profile 4
"""
import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

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
    training_corpus_text,
    _content_cap_for_entry,
    _natural_source_context,
)
from config import get_config
from ml_window import evaluate_model_update

GRANITE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
C_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
MAX_TOKENS_FULL = 2048   # full context (was 512)
MAX_TOKENS_LONG = 4096   # very long context (for lex/legal entries)


# ── Text builders for full-context variants ─────────────────────────

def build_text_plain_full(entry: dict, config: dict) -> str:
    """Plain title + full body. No chunking, no truncation at text level."""
    title = (entry.get("title") or "").strip()
    body = training_corpus_text(entry)  # no content cap — full body
    parts = [p for p in [title, body] if p]
    return "\n".join(parts) if parts else "(empty)"


def build_text_title_boost_full(entry: dict, config: dict) -> str:
    """Title prepended once (not repeated like e5_tuned) + full body + source context."""
    title = (entry.get("title") or "").strip()
    body = training_corpus_text(entry)
    context = _natural_source_context(entry, signals=[])
    parts = [p for p in [title, body] if p]
    body_text = "\n".join(parts) if parts else "(empty)"
    if context:
        return "{}\n\n{}".format(context, body_text)
    return body_text


def build_text_structured_full(entry: dict, config: dict) -> str:
    """Structured format: source context + title + body, with clear section markers.
    Granite was trained on diverse web text — structured formatting may help it
    distinguish title from body more cleanly than plain concatenation."""
    title = (entry.get("title") or "").strip()
    body = training_corpus_text(entry)
    context = _natural_source_context(entry, signals=[])
    lines = []
    if context:
        lines.append(f"[source] {context}")
    if title:
        lines.append(f"[title] {title}")
    if body:
        lines.append(f"[body] {body}")
    return "\n".join(lines) if lines else "(empty)"


TEXT_VARIANTS = {
    "plain_full": build_text_plain_full,
    "title_boost_full": build_text_title_boost_full,
    "structured_full": build_text_structured_full,
}


# ── Cache ───────────────────────────────────────────────────────────

def cache_dir() -> Path:
    d = Path(__file__).parent.parent / "lab_data" / "embeddings_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached(name: str, profile_id: int) -> Optional[np.ndarray]:
    p = cache_dir() / f"granite_{name}_p{profile_id}.npy"
    if p.exists():
        return np.load(p)
    return None


def save_cached(arr: np.ndarray, name: str, profile_id: int):
    np.save(cache_dir() / f"granite_{name}_p{profile_id}.npy", arr)


# ── Full-context encoding (no chunking) ─────────────────────────────

def encode_granite_full_context(
    labeled: List[dict],
    config: dict,
    text_variant: str,
    profile_id: int,
) -> np.ndarray:
    """Encode with granite, single-pass per entry (no chunking, no pooling)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    cached = load_cached(text_variant, profile_id)
    if cached is not None:
        print(f"    [cache hit] granite/{text_variant}/p{profile_id}", flush=True)
        return cached

    text_fn = TEXT_VARIANTS[text_variant]
    texts = [text_fn(entry, config) for entry in labeled]

    # Token length stats
    print(f"    Loading {GRANITE_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(GRANITE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    model.to(device)

    # Check token lengths to pick batch size
    token_lens = []
    for t in texts:
        tl = len(tokenizer.encode(t, add_special_tokens=True))
        token_lens.append(tl)
    token_lens_arr = np.array(token_lens)
    print(f"    Token lengths: mean={token_lens_arr.mean():.0f}, median={np.median(token_lens_arr):.0f}, "
          f"p95={np.percentile(token_lens_arr, 95):.0f}, max={token_lens_arr.max()}", flush=True)

    # Adaptive batch size based on token lengths
    # RTX 5060 has 8GB VRAM, granite-311m in fp16 ~ 600MB
    # 2048 tokens × batch 4 ≈ 4GB activations — safe
    # 4096 tokens × batch 2 ≈ 4GB — safe
    max_tokens_in_batch = max(token_lens)
    if max_tokens_in_batch <= 512:
        batch_size = 16
    elif max_tokens_in_batch <= 1024:
        batch_size = 8
    elif max_tokens_in_batch <= 2048:
        batch_size = 4
    elif max_tokens_in_batch <= 4096:
        batch_size = 2
    else:
        batch_size = 1
        # Cap at 8K tokens — anything longer gets truncated
        print(f"    WARNING: some entries exceed 4096 tokens, using batch_size=1 and max_length=8192", flush=True)

    max_length = min(max(max_tokens_in_batch + 32, 512), 8192)
    print(f"    Encoding {len(texts)} entries (batch_size={batch_size}, max_length={max_length}) on {device.type}...", flush=True)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        token_emb = outputs.last_hidden_state.float()
        # Granite: CLS-token pooling (per IBM model card)
        cls_embs = token_emb[:, 0, :].cpu().numpy()
        all_embs.append(cls_embs)
        done = min(i + batch_size, len(texts))
        if done % 100 == 0 or done == len(texts):
            print(f"      {done}/{len(texts)} entries", flush=True)

    result = np.vstack(all_embs) if all_embs else np.array([])
    # L2-normalize (per granite model card — Normalize module after Pooling)
    if result.size:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.clip(norms, 1e-12, None)
    save_cached(result, text_variant, profile_id)

    del model, tokenizer
    import gc; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ── Training + eval (same as C sweep) ───────────────────────────────

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

    # Survival metrics at production alert threshold (0.55)
    w = np.array([CLASS_WEIGHT_MAP.get(cn[i], 0.0) for i in range(len(cn))], dtype=float)
    composite = probs.dot(w)
    threshold = 0.55
    y_te_arr = np.array(y_te)
    leads = y_te_arr == "investigation_lead"
    importants = y_te_arr == "important"
    above = composite >= threshold
    lead_survival = float(above[leads].sum()) / max(int(leads.sum()), 1)
    important_survival = float(above[importants].sum()) / max(int(importants.sum()), 1)
    relevant = leads | importants
    thresh_precision = float(relevant[above].sum()) / max(int(above.sum()), 1)
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
        "survival_threshold": threshold,
        "lead_survival": round(lead_survival, 4),
        "important_survival": round(important_survival, 4),
        "threshold_precision": round(thresh_precision, 4),
        "n_above_threshold": n_above,
        "n_leads_holdout": int(leads.sum()),
        "n_importants_holdout": int(importants.sum()),
    }


# ── Reference results from the chunked sweep (for comparison) ──────

# Best results from lab_encoder_c_sweep.py (chunked, 512 tokens)
CHUNKED_BEST = {
    1: {  # Mothership
        "e5_e5_tuned_C1": {"p30": 0.4333, "f1": 0.4139},  # incumbent
        "e5_e5_tuned_best": {"C": 0.02, "p30": 0.4667, "f1": 0.5199},
        "e5_plain_best": {"C": 0.01, "p30": 0.4667, "f1": 0.5337},
        "granite_plain_best": {"C": 0.05, "p30": 0.5000, "f1": 0.3973},
        "granite_e5_tuned_best": {"C": 0.01, "p30": 0.4667, "f1": 0.3748},
    },
    3: {  # Sicherheit
        "e5_e5_tuned_C1": {"p30": 0.5333, "f1": 0.4908},  # incumbent
        "e5_e5_tuned_best": {"C": 0.02, "p30": 0.6667, "f1": 0.5508},
        "e5_plain_best": {"C": 0.05, "p30": 0.6333, "f1": 0.5471},
        "granite_plain_best": {"C": 0.02, "p30": 0.6333, "f1": 0.5813},
        "granite_e5_tuned_best": {"C": 0.02, "p30": 0.6000, "f1": 0.5480},
    },
    4: {  # EU
        "e5_e5_tuned_C1": {"p30": 0.8000, "f1": 0.3736},  # incumbent
        "e5_e5_tuned_best": {"C": 0.01, "p30": 0.8333, "f1": 0.4077},
        "e5_plain_best": {"C": 0.01, "p30": 0.8333, "f1": 0.3865},
        "granite_plain_best": {"C": 0.01, "p30": 0.8667, "f1": 0.4351},
        "granite_e5_tuned_best": {"C": 0.01, "p30": 0.8667, "f1": 0.3662},
    },
}


def run_profile(profile_id, config) -> dict:
    labeled = db.get_all_labels(profile_id)
    if len(labeled) < 20:
        return {"profile_id": profile_id, "error": f"only {len(labeled)} labels"}

    profile = db.get_profile_by_id(profile_id)
    name = profile.get("display_name", str(profile_id)) if profile else str(profile_id)

    incumbent = CHUNKED_BEST.get(profile_id, {}).get("e5_e5_tuned_C1", {})
    chunked_best_granite = CHUNKED_BEST.get(profile_id, {}).get("granite_plain_best", {})

    results = {}
    for variant in TEXT_VARIANTS:
        print(f"\n  === granite/{variant} ===", flush=True)
        X = encode_granite_full_context(labeled, config, variant, profile_id)
        y = [l["label"] for l in labeled]
        if len(X) < 20:
            results[variant] = {"error": f"only {len(X)} embeddings"}
            continue

        sw = compute_sample_weights(labeled, config)
        min_class = int(pd.Series(y).value_counts().min())
        if min_class < 2:
            results[variant] = {"error": "min class < 2"}
            continue
        test_size = _holdout_test_fraction(min_class, len(y))
        X_tr, X_te, y_tr, y_te, sw_tr, _ = _stable_train_test_split(
            X, y, sw, labeled, test_size=test_size
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
        results[variant] = {
            "train_size": len(y_tr), "test_size": len(y_te),
            "c_results": c_results, "best": best,
        }

    return {
        "profile_id": profile_id,
        "profile_name": name,
        "total_labels": len(labeled),
        "incumbent": incumbent,
        "chunked_best_granite": chunked_best_granite,
        "full_context_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Granite full-context test (no chunking)")
    parser.add_argument("--profile", type=int, default=None)
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
        result = run_profile(pid, config)
        all_results.append(result)

        if "error" in result:
            print(f"  SKIP: {result['error']}")
            continue

        inc = result["incumbent"]
        chunked = result["chunked_best_granite"]
        print(f"\n  Reference (from chunked sweep):")
        print(f"    Incumbent e5 C=1.0:           p@30={inc.get('p30', 0):.4f}  F1={inc.get('f1', 0):.4f}")
        if chunked:
            print(f"    Granite/plain chunked C={chunked.get('C', '?')}:  p@30={chunked.get('p30', 0):.4f}  F1={chunked.get('f1', 0):.4f}")

        print(f"\n  Full-context granite results:")
        print(f"  {'Variant':<25s} {'Best C':>7s} {'p@30':>6s} {'F1':>6s} {'AUC':>6s} {'Δp@30(chunked)':>14s} {'ΔF1(chunked)':>13s}")
        print(f"  {'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*14} {'-'*13}")
        for variant, vr in result["full_context_results"].items():
            if "error" in vr:
                print(f"  {variant:<25s}  ERROR: {vr['error']}")
                continue
            b = vr["best"]
            dp = b["precision_at_30"] - chunked.get("p30", 0) if chunked else 0
            df = b["f1_score"] - chunked.get("f1", 0) if chunked else 0
            print(f"  {variant:<25s} {b['C']:>7.2f} {b['precision_at_30']:>6.4f} {b['f1_score']:>6.4f} "
                  f"{b['ranking_auc']:>6.4f} {dp:>+14.4f} {df:>+13.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: granite full-context vs chunked vs incumbent")
    print(f"{'='*70}")
    for r in all_results:
        if "error" in r:
            print(f"  Profile {r['profile_id']}: {r['error']}")
            continue
        inc = r["incumbent"]
        chunked = r["chunked_best_granite"]
        print(f"\n  Profile {r['profile_id']} ({r['profile_name']}):")
        print(f"    Incumbent e5 C=1.0:          p@30={inc.get('p30', 0):.4f}  F1={inc.get('f1', 0):.4f}")
        if chunked:
            print(f"    Granite chunked C={chunked.get('C', '?'):<5}    p@30={chunked.get('p30', 0):.4f}  F1={chunked.get('f1', 0):.4f}")

        best_fc = None
        for variant, vr in r["full_context_results"].items():
            if "error" in vr:
                continue
            b = vr["best"]
            if best_fc is None or b["precision_at_30"] > best_fc["precision_at_30"]:
                best_fc = b
                best_fc_variant = variant
        if best_fc:
            print(f"    Granite full-ctx C={best_fc['C']:<5} ({best_fc_variant})  p@30={best_fc['precision_at_30']:.4f}  F1={best_fc['f1_score']:.4f}")
            if chunked:
                dp = best_fc["precision_at_30"] - chunked.get("p30", 0)
                df = best_fc["f1_score"] - chunked.get("f1", 0)
                verdict = "full-ctx WINS" if dp > 0.01 else ("chunked wins" if dp < -0.01 else "tie")
                print(f"    Full-ctx vs chunked: Δp@30={dp:+.4f}  ΔF1={df:+.4f}  → {verdict}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
