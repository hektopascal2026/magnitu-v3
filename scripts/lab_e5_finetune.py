#!/usr/bin/env python3
"""Fine-tune e5-base on Seismo labels via supervised contrastive learning.

Goal: adapt the encoder to the Seismo domain (Swiss/EU regulatory, security,
digital policy text) so the embeddings cluster same-label items together.
Then evaluate with the same frozen LogReg head + mean_norm pipeline.

Approach:
  - Supervised contrastive loss: for each anchor, pull same-label items closer,
    push different-label items apart.
  - Fine-tune only on the TRAIN split (80% stratified, seed 42) — no test leakage.
  - Re-embed all entries with the fine-tuned model.
  - Evaluate with LogReg C=0.01 + L2 normalization, same split/metrics.
  - Compare against frozen e5-base mean_norm.

Uses the same text preprocessing (e5_tuned, chunked, legal snippets) as the
production pipeline and the encoder comparison scripts.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_finetune.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_finetune.py --profile 3
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_finetune.py --epochs 5 --lr 2e-5
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import db
from pipeline import (
    CLASSES,
    CLASS_WEIGHT_MAP,
    compute_sample_weights,
    _fit_temperature_scalar,
    _collect_oof_logits,
    _ranking_metrics,
    _oof_fold_count,
    _build_entry_text,
    _split_text_chunks,
    is_legal_training_entry,
    is_analytical_training_entry,
    MAX_EMBED_CHUNKS,
    MAX_EMBED_CHUNKS_LEGAL,
    MAX_EMBED_CHUNKS_ANALYTICAL,
)
from config import get_config

SEED = 42
MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
MAX_LEN = 512        # Encoding (eval) — full context
FINETUNE_MAX_LEN = 256  # Fine-tuning — truncated to fit 3x batch in 8GB VRAM
PROFILES = {2: "digital", 3: "sicherheit", 4: "eu"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labeled(profile_id: int) -> List[dict]:
    labels = db.get_all_labels(profile_id=profile_id)
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
    return labeled


def build_texts(labeled: List[dict], config: dict) -> List[str]:
    """Build the e5_tuned text for each entry, pooling chunks into a single text."""
    patterns = config.get("legal_signal_patterns") or []
    texts = []
    for entry in labeled:
        text = _build_entry_text(entry, legal_patterns=patterns)
        chunks = _split_text_chunks(text, max_chunks=MAX_EMBED_CHUNKS)
        # Join chunks with separator for a single encoding pass
        # (fine-tuning uses single-pass encoding for speed; production uses chunked pooling)
        full_text = " [SEP] ".join(chunks)
        texts.append(full_text)
    return texts


class ContrastiveDataset(Dataset):
    """Supervised contrastive dataset: yields (anchor_text, positive_text, negative_text)."""

    def __init__(self, texts: List[str], labels: List[str], indices: List[int]):
        self.texts = texts
        self.labels = labels
        self.indices = indices
        # Build label-to-indices map
        self.label_to_indices: Dict[str, List[int]] = {}
        for i in indices:
            lbl = labels[i]
            self.label_to_indices.setdefault(lbl, []).append(i)
        self.all_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        anchor_label = self.labels[anchor_idx]

        # Positive: same label, different entry
        same_label_indices = self.label_to_indices[anchor_label]
        if len(same_label_indices) > 1:
            pos_idx = random.choice([i for i in same_label_indices if i != anchor_idx])
        else:
            pos_idx = anchor_idx  # degenerate — will produce zero loss

        # Negative: different label
        other_labels = [l for l in self.all_labels if l != anchor_label]
        if other_labels:
            neg_label = random.choice(other_labels)
            neg_idx = random.choice(self.label_to_indices[neg_label])
        else:
            neg_idx = random.choice([i for i in self.indices if i != anchor_idx])

        return (
            self.texts[anchor_idx],
            self.texts[pos_idx],
            self.texts[neg_idx],
        )


class TripletCollator:
    def __init__(self, tokenizer, max_len: int, prefix: str):
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len else FINETUNE_MAX_LEN
        self.prefix = prefix

    def __call__(self, batch):
        anchors, positives, negatives = zip(*batch)
        all_texts = list(anchors) + list(positives) + list(negatives)
        prefixed = [self.prefix + t if not t.startswith(self.prefix.strip()) else t for t in all_texts]
        encoded = self.tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        n = len(anchors)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "n_anchors": n,
        }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_batch(model, tokenizer, texts: List[str], device, batch_size: int = 16,
                 prefix: str = PASSAGE_PREFIX, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode texts with mean pooling + L2 normalization."""
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prefixed = [prefix + t if not t.startswith(prefix.strip()) else t for t in batch]
        encoded = tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        emb = mean_pool(outputs.last_hidden_state.float(), encoded["attention_mask"])
        # L2 normalize
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs) if all_embs else np.array([])


def fine_tune(model, tokenizer, train_texts, train_labels, train_indices,
              device, epochs=3, lr=2e-5, batch_size=16, temperature=0.05):
    """Supervised contrastive fine-tuning with triplet loss."""
    model.train()

    dataset = ContrastiveDataset(train_texts, train_labels, train_indices)
    collator = TripletCollator(tokenizer, FINETUNE_MAX_LEN, PASSAGE_PREFIX)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Linear warmup + cosine decay
    total_steps = len(loader) * epochs
    warmup_steps = min(50, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Fine-tuning: {len(train_indices)} train samples, {epochs} epochs, lr={lr}, batch_size={batch_size}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            n = batch["n_anchors"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool(outputs.last_hidden_state.float(), attention_mask)
            emb = F.normalize(emb, p=2, dim=1)

            # Split into anchor, positive, negative
            anchor_emb = emb[:n]
            positive_emb = emb[n:2*n]
            negative_emb = emb[2*n:]

            # Triplet loss: pull positive closer, push negative away
            # Using cosine similarity (embeddings are already L2-normalized)
            pos_sim = (anchor_emb * positive_emb).sum(dim=1)
            neg_sim = (anchor_emb * negative_emb).sum(dim=1)
            loss = F.relu(neg_sim - pos_sim + temperature).mean()

            # Guard against NaN (float overflow in long sequences)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model


def evaluate(X: np.ndarray, labeled: List[dict], C: float, seed: int = 42) -> dict:
    y = [l["label"] for l in labeled]
    min_class = int(pd.Series(y).value_counts().min())
    n_folds = min(5, max(2, min_class // 2)) if min_class >= 4 else 2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    labeled_train = [labeled[i] for i in train_idx]

    sw = compute_sample_weights(labeled_train)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, solver="lbfgs",
            multi_class="multinomial",
        )),
    ])
    clf.fit(X_train, y_train, clf__sample_weight=sw)

    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    class_names = clf.named_steps["clf"].classes_

    f1 = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    rank = _ranking_metrics(probs, class_names, y_test, k=30)

    try:
        oof_logits = _collect_oof_logits(
            X_train, y_train, n_folds=n_folds, seed=seed,
            C=C, sample_weight=sw,
        )
        temperature = _fit_temperature_scalar(oof_logits, y_train, class_names) if oof_logits is not None else 1.0
    except Exception:
        temperature = 1.0

    report = classification_report(y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0)

    # Survival at 0.55
    w = np.array([CLASS_WEIGHT_MAP.get(cn, 0.0) for cn in class_names], dtype=float)
    composite = probs.dot(w)
    threshold = 0.55
    y_te_arr = np.array(y_test)
    leads = y_te_arr == "investigation_lead"
    importants = y_te_arr == "important"
    above = composite >= threshold
    lead_surv = float(above[leads].sum()) / max(int(leads.sum()), 1) if leads.sum() > 0 else 0.0
    imp_surv = float(above[importants].sum()) / max(int(importants.sum()), 1) if importants.sum() > 0 else 0.0
    relevant = leads | importants
    n_above = int(above.sum())
    prec_at_t = float(above[relevant].sum()) / n_above if n_above > 0 else 0.0

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "C": C,
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision_at_30": round(rank["precision_at_30"], 4),
        "lead_recall_at_30": round(rank["lead_recall_at_30"], 4),
        "ranking_auc": round(rank["ranking_auc"], 4),
        "calibration_temperature": round(temperature, 4),
        "lead_survival_055": round(lead_surv, 4),
        "important_survival_055": round(imp_surv, 4),
        "threshold_precision_055": round(prec_at_t, 4),
        "n_above_055": n_above,
        "class_distribution": dict(pd.Series(y_test).value_counts().to_dict()),
        "per_class": {
            cls: {
                "precision": round(report.get(cls, {}).get("precision", 0), 4),
                "recall": round(report.get(cls, {}).get("recall", 0), 4),
                "f1": round(report.get(cls, {}).get("f1-score", 0), 4),
                "support": int(report.get(cls, {}).get("support", 0)),
            }
            for cls in CLASSES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--temperature", type=float, default=0.05, help="Triplet loss margin")
    parser.add_argument("--save-model", action="store_true", help="Save fine-tuned model checkpoints")
    args = parser.parse_args()

    set_seed(SEED)
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"VRAM: {vram} GB")

    results = {}

    for pid in args.profiles:
        desk = PROFILES.get(pid, f"p{pid}")
        print(f"\n{'='*70}")
        print(f"Profile {pid} ({desk})")
        print(f"{'='*70}")

        labeled = load_labeled(pid)
        if len(labeled) < 20:
            print(f"  Skipping: only {len(labeled)} labeled entries")
            continue
        print(f"  {len(labeled)} labeled entries")

        # Build texts
        texts = build_texts(labeled, config)
        labels = [l["label"] for l in labeled]

        # Split (same as evaluation: seed 42, 80/20 stratified)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(sss.split(range(len(labels)), labels))
        train_indices = list(train_idx)
        print(f"  Split: {len(train_indices)} train, {len(test_idx)} test")

        # ── Baseline: frozen e5-base mean_norm ──
        print(f"\n  Loading frozen e5-base for baseline encoding...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        model.to(device)

        print(f"  Encoding {len(labeled)} entries with frozen model...")
        t0 = time.time()
        X_frozen = encode_batch(model, tokenizer, texts, device, batch_size=32)
        print(f"  Frozen encoding done in {time.time()-t0:.1f}s, shape={X_frozen.shape}")

        # Evaluate frozen baseline
        print(f"\n  Frozen e5-base mean_norm baseline:")
        print(f"    {'C':>8} {'F1':>7} {'Acc':>7} {'p@30':>7} {'LR@30':>7} {'AUC':>7} {'LeadS':>7} {'ImpS':>7}")
        print("    " + "-" * 64)
        frozen_results = []
        for C in args.c_values:
            m = evaluate(X_frozen, labeled, C=C, seed=SEED)
            frozen_results.append(m)
            print(f"    {C:>8.3f} {m['f1_score']:>7.4f} {m['accuracy']:>7.4f} {m['precision_at_30']:>7.4f} {m['lead_recall_at_30']:>7.4f} {m['ranking_auc']:>7.4f} {m['lead_survival_055']:>7.4f} {m['important_survival_055']:>7.4f}")
        best_frozen = max(frozen_results, key=lambda r: r["f1_score"])
        print(f"    Best frozen C={best_frozen['C']} (F1={best_frozen['f1_score']})")

        # ── Fine-tune ──
        print(f"\n  Fine-tuning e5-base on {len(train_indices)} train samples...")
        # Reload fresh model for fine-tuning
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()  # trade compute for memory
        model.to(device)

        train_labels_list = [labels[i] for i in train_indices]
        model = fine_tune(
            model, tokenizer, texts, labels, train_indices,
            device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, temperature=args.temperature,
        )

        # Save model if requested
        if args.save_model:
            save_dir = Path(f"lab_data/finetuned_e5_base_p{pid}")
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  Saved fine-tuned model to {save_dir}")

        # Re-encode ALL entries with fine-tuned model
        print(f"\n  Re-encoding {len(labeled)} entries with fine-tuned model...")
        t0 = time.time()
        X_finetuned = encode_batch(model, tokenizer, texts, device, batch_size=32)
        print(f"  Fine-tuned encoding done in {time.time()-t0:.1f}s, shape={X_finetuned.shape}")

        # Evaluate fine-tuned
        print(f"\n  Fine-tuned e5-base mean_norm:")
        print(f"    {'C':>8} {'F1':>7} {'Acc':>7} {'p@30':>7} {'LR@30':>7} {'AUC':>7} {'LeadS':>7} {'ImpS':>7}")
        print("    " + "-" * 64)
        finetuned_results = []
        for C in args.c_values:
            m = evaluate(X_finetuned, labeled, C=C, seed=SEED)
            finetuned_results.append(m)
            print(f"    {C:>8.3f} {m['f1_score']:>7.4f} {m['accuracy']:>7.4f} {m['precision_at_30']:>7.4f} {m['lead_recall_at_30']:>7.4f} {m['ranking_auc']:>7.4f} {m['lead_survival_055']:>7.4f} {m['important_survival_055']:>7.4f}")
        best_ft = max(finetuned_results, key=lambda r: r["f1_score"])
        print(f"    Best fine-tuned C={best_ft['C']} (F1={best_ft['f1_score']})")

        # Per-class comparison
        print(f"\n  Per-class comparison (best C each):")
        print(f"    {'Class':>20} {'Frozen P/R/F1':>20} {'FT P/R/F1':>20} {'Delta F1':>10}")
        print("    " + "-" * 72)
        for cls in CLASSES:
            f_pc = best_frozen["per_class"].get(cls, {})
            t_pc = best_ft["per_class"].get(cls, {})
            f_str = f"{f_pc.get('precision',0):.2f}/{f_pc.get('recall',0):.2f}/{f_pc.get('f1',0):.2f}"
            t_str = f"{t_pc.get('precision',0):.2f}/{t_pc.get('recall',0):.2f}/{t_pc.get('f1',0):.2f}"
            delta = t_pc.get('f1', 0) - f_pc.get('f1', 0)
            print(f"    {cls:>20} {f_str:>20} {t_str:>20} {delta:>+10.4f}")

        # Summary
        print(f"\n  Summary ({desk}):")
        print(f"    Frozen:  F1={best_frozen['f1_score']:.4f}  p@30={best_frozen['precision_at_30']:.4f}  LR@30={best_frozen['lead_recall_at_30']:.4f}  AUC={best_frozen['ranking_auc']:.4f}")
        print(f"    FT:      F1={best_ft['f1_score']:.4f}  p@30={best_ft['precision_at_30']:.4f}  LR@30={best_ft['lead_recall_at_30']:.4f}  AUC={best_ft['ranking_auc']:.4f}")
        delta_f1 = best_ft['f1_score'] - best_frozen['f1_score']
        delta_p30 = best_ft['precision_at_30'] - best_frozen['precision_at_30']
        delta_lr30 = best_ft['lead_recall_at_30'] - best_frozen['lead_recall_at_30']
        print(f"    Delta:   F1={delta_f1:+.4f}  p@30={delta_p30:+.4f}  LR@30={delta_lr30:+.4f}")

        results[desk] = {
            "frozen": frozen_results,
            "finetuned": finetuned_results,
            "best_frozen": best_frozen,
            "best_finetuned": best_ft,
            "n_labels": len(labeled),
            "epochs": args.epochs,
            "lr": args.lr,
        }

        # Cleanup
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Save results
    out_path = Path("lab_e5_finetune_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: frozen e5-base mean_norm vs fine-tuned e5-base mean_norm")
    print(f"{'='*70}")
    print(f"  {'Desk':>12} {'Frozen F1':>10} {'FT F1':>10} {'Delta':>8} {'Frozen p@30':>12} {'FT p@30':>10} {'Delta':>8}")
    print("  " + "-" * 72)
    for desk in sorted(results.keys()):
        f = results[desk]["best_frozen"]
        t = results[desk]["best_finetuned"]
        print(f"  {desk:>12} {f['f1_score']:>10.4f} {t['f1_score']:>10.4f} {t['f1_score']-f['f1_score']:>+8.4f} {f['precision_at_30']:>12.4f} {t['precision_at_30']:>10.4f} {t['precision_at_30']-f['precision_at_30']:>+8.4f}")


if __name__ == "__main__":
    main()
