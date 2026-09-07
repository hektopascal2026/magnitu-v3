#!/usr/bin/env python3
"""Fine-tune e5-base with a classification head (cross-entropy end-to-end).

Unlike the triplet approach (which only shapes the embedding space via
contrastive loss), this trains the encoder + a linear classification head
jointly with cross-entropy on the 4 Seismo labels. This should preserve
ranking order (the head learns calibrated decision boundaries) while
improving classification F1.

After fine-tuning, we extract the encoder, re-embed all entries with mean_norm,
and evaluate with the same frozen LogReg head — same as the triplet script.
This gives a fair comparison: both approaches produce a new encoder, both are
evaluated with the same downstream pipeline.

Usage:
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_finetune_cls.py
  MAGNITU_DATA_DIR=lab_data .venv/bin/python scripts/lab_e5_finetune_cls.py --profile 3
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
    _build_entry_text,
    _split_text_chunks,
    MAX_EMBED_CHUNKS,
)
from config import get_config

SEED = 42
MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
MAX_LEN = 512
FINETUNE_MAX_LEN = 256
PROFILES = {2: "digital", 3: "sicherheit", 4: "eu"}
LABEL_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


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
    patterns = config.get("legal_signal_patterns") or []
    texts = []
    for entry in labeled:
        text = _build_entry_text(entry, legal_patterns=patterns)
        chunks = _split_text_chunks(text, max_chunks=MAX_EMBED_CHUNKS)
        full_text = " [SEP] ".join(chunks)
        texts.append(full_text)
    return texts


class ClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], indices: List[int]):
        self.texts = texts
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.texts[i], LABEL_TO_IDX[self.labels[i]]


class ClsCollator:
    def __init__(self, tokenizer, max_len: int, prefix: str):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = prefix

    def __call__(self, batch):
        texts, labels = zip(*batch)
        prefixed = [self.prefix + t if not t.startswith(self.prefix.strip()) else t for t in texts]
        encoded = self.tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EncoderWithHead(nn.Module):
    """E5 encoder + linear classification head on mean-pooled embeddings."""

    def __init__(self, model_name: str, n_classes: int, dim: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.head = nn.Linear(dim, n_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool
        mask = attention_mask.unsqueeze(-1).float()
        summed = (outputs.last_hidden_state.float() * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits, pooled


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_batch(model, tokenizer, texts: List[str], device, batch_size: int = 32,
                 prefix: str = PASSAGE_PREFIX, max_len: int = MAX_LEN) -> np.ndarray:
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
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs) if all_embs else np.array([])


def fine_tune_cls(model, tokenizer, train_texts, train_labels, train_indices,
                  device, epochs=3, lr=2e-5, batch_size=4, class_weights=None):
    """Cross-entropy fine-tuning of encoder + classification head."""
    model.train()

    dataset = ClassificationDataset(train_texts, train_labels, train_indices)
    collator = ClsCollator(tokenizer, FINETUNE_MAX_LEN, PASSAGE_PREFIX)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # Weighted cross-entropy (class-balanced)
    if class_weights is None:
        class_weights = torch.ones(len(CLASSES), device=device)
    else:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential LR: encoder slower, head faster
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=0.01)

    total_steps = len(loader) * epochs
    warmup_steps = min(50, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"  Fine-tuning (CLS): {len(train_indices)} train, {epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

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
        print(f"  Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    args = parser.parse_args()

    set_seed(SEED)
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

        texts = build_texts(labeled, config)
        labels = [l["label"] for l in labeled]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(sss.split(range(len(labels)), labels))
        train_indices = list(train_idx)
        print(f"  Split: {len(train_indices)} train, {len(test_idx)} test")

        # ── Baseline: frozen e5-base mean_norm ──
        print(f"\n  Loading frozen e5-base for baseline...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        frozen_model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        frozen_model.eval()
        frozen_model.to(device)

        print(f"  Encoding {len(labeled)} entries with frozen model...")
        t0 = time.time()
        X_frozen = encode_batch(frozen_model, tokenizer, texts, device, batch_size=32)
        print(f"  Frozen encoding done in {time.time()-t0:.1f}s, shape={X_frozen.shape}")

        del frozen_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

        # ── Fine-tune with classification head ──
        print(f"\n  Fine-tuning e5-base + cls head on {len(train_indices)} train samples...")

        # Compute class weights from training distribution
        train_label_counts = pd.Series([labels[i] for i in train_indices]).value_counts()
        total_train = len(train_indices)
        class_weights = []
        for cls in CLASSES:
            n_cls = int(train_label_counts.get(cls, 0))
            w = total_train / (len(CLASSES) * max(n_cls, 1))
            class_weights.append(w)
        print(f"  Class weights: {dict(zip(CLASSES, [round(w,2) for w in class_weights]))}")

        model = EncoderWithHead(MODEL_NAME, n_classes=len(CLASSES), dim=768)
        model.encoder.config.use_cache = False
        model.encoder.gradient_checkpointing_enable()
        model.to(device)

        model = fine_tune_cls(
            model, tokenizer, texts, labels, train_indices,
            device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, class_weights=class_weights,
        )

        # Extract encoder and re-encode all entries
        print(f"\n  Re-encoding {len(labeled)} entries with fine-tuned encoder...")
        encoder = model.encoder
        encoder.eval()

        t0 = time.time()
        X_finetuned = encode_batch(encoder, tokenizer, texts, device, batch_size=32)
        print(f"  Fine-tuned encoding done in {time.time()-t0:.1f}s, shape={X_finetuned.shape}")

        # Evaluate fine-tuned
        print(f"\n  Fine-tuned e5-base mean_norm (CLS head):")
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

        # Gate replay
        print(f"\n  Gate replay (fine-tuned vs frozen):")
        from scripts.calibration_framework import evaluate_gate
        gate = evaluate_gate(best_frozen, best_ft)
        print(f"    promote={gate['promote']}  branch={gate['branch']}  deltas={gate.get('deltas',{})}")
        if not gate["promote"]:
            print(f"    reason: {gate.get('reason','')}")

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
            "gate": gate,
            "n_labels": len(labeled),
            "epochs": args.epochs,
            "lr": args.lr,
            "approach": "cls_head_cross_entropy",
        }

        del model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path("lab_e5_finetune_cls_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*70}")
    print("FINAL SUMMARY: frozen vs fine-tuned (CLS head, cross-entropy)")
    print(f"{'='*70}")
    print(f"  {'Desk':>12} {'Frozen F1':>10} {'FT F1':>10} {'Delta':>8} {'Frozen LR@30':>13} {'FT LR@30':>10} {'Delta':>8} {'Gate':>8}")
    print("  " + "-" * 80)
    for desk in sorted(results.keys()):
        f = results[desk]["best_frozen"]
        t = results[desk]["best_finetuned"]
        g = results[desk]["gate"]
        print(f"  {desk:>12} {f['f1_score']:>10.4f} {t['f1_score']:>10.4f} {t['f1_score']-f['f1_score']:>+8.4f} {f['lead_recall_at_30']:>13.4f} {t['lead_recall_at_30']:>10.4f} {t['lead_recall_at_30']-f['lead_recall_at_30']:>+8.4f} {'PASS' if g['promote'] else 'BLOCK':>8}")


if __name__ == "__main__":
    main()
