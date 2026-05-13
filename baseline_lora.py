#!/usr/bin/env python
# baseline_lora.py
# Unified baseline: Backdoor Detection + Unlearning (RT / GA / NPO / RGA)
#
# This version mirrors the LoRA / quantization policy used in ticu_universal.py:
# - LoRA is ONLY supported for model_type in: llama, qwen, mistral.
# - model_type=auto NEVER uses LoRA (raises if --use_lora is set).
# - For quantized/device_map models, we avoid blindly calling model.to(device) and instead
#   move *batches* to the model/backbone device (get_model_device()).
#
# Data layout expected:
#   --helper_dir/
#       Data_Poisoning/<Attack>/<Dataset>/
#           train_poisoned.csv (columns: text,label,poisoned)
#           train_clean.csv
#           test_clean.csv
#           test_poisoned_all.csv / test_poisoned_part.csv

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.metrics import confusion_matrix
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# Optional deps (UMAP/HDBSCAN baseline)
try:
    import umap  # type: ignore
except Exception:
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support

# LoRA backbones (ONLY for llama/qwen/mistral)
# These should match the files used by ticu_universal.py.
try:
    from lora_llama import build_llama_with_lora  # type: ignore
    from lora_qwen import build_qwen_with_lora    # type: ignore
    from lora_mistral import build_mistral_with_lora  # type: ignore
except Exception:
    build_llama_with_lora = None
    build_qwen_with_lora = None
    build_mistral_with_lora = None


# -----------------------
# Repro / utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    """Get the device where the *backbone* expects inputs.

    For decoder wrappers / PEFT / quantized device_map models, parameters can be sharded;
    we always prefer backbone-like submodules if present.
    """
    for attr in ["backbone", "llama", "model"]:
        if hasattr(model, attr):
            try:
                return next(p.device for p in getattr(model, attr).parameters() if p is not None)
            except Exception:
                pass
    try:
        return next(p.device for p in model.parameters() if p is not None)
    except StopIteration:
        return fallback


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        for alt in ["sentence", "comment_text", "content"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns:
        raise ValueError(f"CSV missing text column ('text'/'sentence'/'content'): {path}")
    if "label" not in df.columns:
        raise ValueError(f"CSV missing 'label' column: {path}")

    if "poisoned" not in df.columns:
        df["poisoned"] = 0

    df["label"] = df["label"].astype(int)
    df["poisoned"] = df["poisoned"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


@dataclass
class TaskSpec:
    num_labels: int


def task_spec(dataset_name: str) -> TaskSpec:
    ds = dataset_name.upper()
    if ds == "AG":
        return TaskSpec(num_labels=4)
    if ds in ["SST-2", "SST2", "HSOL"]:
        return TaskSpec(num_labels=2)
    raise ValueError(f"Unknown dataset '{dataset_name}'. Expected SST-2, HSOL, AG.")


# -----------------------
# Dataset / loaders
# -----------------------
class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        enc["labels"] = int(row["label"])
        enc["poisoned"] = int(row.get("poisoned", 0))
        return enc


def make_loader(ds: Dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


# -----------------------
# Metrics (ACC + LFR(target_label))
# -----------------------

def compute_lfr_from_cm(cm, data_name, target_label: int):
    data_name = data_name.upper()

    if data_name in ["SST-2", "SST2", "HSOL"]:
        # Binary class-wise LFRs (no averaging)
        # LFR_0: true class 0 flipped to predicted 1
        LFR_0 = cm[0][1] / (cm[0][0] + cm[0][1] + 1e-12)

        # LFR_1: true class 1 flipped to predicted 0
        LFR_1 = cm[1][0] / (cm[1][0] + cm[1][1] + 1e-12)

        return float(LFR_0), float(LFR_1)

    elif data_name == "AG":
        # 4-class case
        LFR_0 = (cm[0][1] + cm[0][2] + cm[0][3]) / (cm[0].sum() + 1e-12)
        LFR_1 = (cm[1][0] + cm[1][2] + cm[1][3]) / (cm[1].sum() + 1e-12)
        LFR_2 = (cm[2][0] + cm[2][1] + cm[2][3]) / (cm[2].sum() + 1e-12)
        LFR_3 = (cm[3][0] + cm[3][1] + cm[3][2]) / (cm[3].sum() + 1e-12)

        lfr = (LFR_1 + LFR_2 + LFR_3) / 3.0
        return float(lfr)

    else:
        raise ValueError(f"Unknown dataset for LFR: {data_name}")

@torch.no_grad()
def eval_acc_lfr(model, loader: DataLoader, device: torch.device, target_label: int, dataset: str) -> Dict[str, float]:
    model.eval()
    model_device = get_model_device(model, device)

    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        labels = batch["labels"].to(model_device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        preds = torch.argmax(out.logits, dim=-1)

        preds_all.append(preds.detach().cpu())
        labels_all.append(labels.detach().cpu())

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()

    acc = (preds == labels).mean()

    cm = confusion_matrix(labels, preds)
    lfr_out = compute_lfr_from_cm(cm, dataset, target_label)

    if isinstance(lfr_out, tuple):
        lfr_0, lfr_1 = lfr_out
        return {
            "acc": acc,
            "lfr_0": lfr_0,
            "lfr_1": lfr_1,
        }
    else:
        return {
            "acc": acc,
            "lfr": float(lfr_out),
        }


# -----------------------
# Model builder
# -----------------------
def build_model(model_type: str, model_name: str, num_labels: int, args=None):
    mt = (model_type or "").lower().strip()

    if mt == "auto":
        if getattr(args, "use_lora", False):
            raise ValueError("LoRA is disabled for --model_type auto. Use --model_type llama|qwen|mistral.")
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if mt == "llama":
        if build_llama_with_lora is None:
            raise ImportError("LoRA llama builders not found. Ensure lora_llama.py is available.")
        return build_llama_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    if mt == "qwen":
        if build_qwen_with_lora is None:
            raise ImportError("LoRA qwen builders not found. Ensure lora_qwen.py is available.")
        return build_qwen_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    if mt == "mistral":
        if build_mistral_with_lora is None:
            raise ImportError("LoRA mistral builders not found. Ensure lora_mistral.py is available.")
        return build_mistral_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    raise ValueError("--model_type must be one of: auto, llama, qwen, mistral")


def get_base_encoder(model: nn.Module) -> nn.Module:
    """Return encoder module used for hidden-state extraction and (for RGA) regularization."""
    # Decoder/LoRA wrappers often expose backbone/model/llama
    for attr in ["backbone", "llama", "model"]:
        if hasattr(model, attr):
            return getattr(model, attr)

    # HuggingFace sequence-classification models
    for attr in ["bert", "distilbert", "roberta", "electra", "deberta", "albert", "xlnet", "bart", "t5"]:
        if hasattr(model, attr):
            return getattr(model, attr)

    # Generic fallbacks
    if hasattr(model, "base_model"):
        return getattr(model, "base_model")
    if hasattr(model, "get_encoder"):
        return model.get_encoder()  # type: ignore
    return model


# -----------------------
# Training (poison baseline)
# -----------------------
def train_supervised(
    model,
    loader,
    device,
    lr,
    epochs,
    weight_decay,
    max_grad_norm,
    grad_accum: int = 1,
    fp16: bool = False,
    bf16: bool = False,
):
    model_device = get_model_device(model, device)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found (did you enable LoRA / set requires_grad?).")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
    total_steps = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    epoch_times = []

    for ep in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(model_device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(**batch, return_dict=True)
                loss = out.loss / grad_accum

            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.detach().cpu()) * grad_accum

            if step % grad_accum == 0:
                if fp16:
                    scaler.unscale_(opt)
                clip_grad_norm_(params, max_grad_norm)

                if fp16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                sched.step()
                opt.zero_grad(set_to_none=True)

        denom = max(1, len(loader))
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f"[train-poison] epoch={ep}/{epochs} loss={running/denom:.4f}")
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"[Train] Average time per epoch: {avg_time:.2f} seconds")
    return model


# -----------------------
# Representation extraction for detection
# -----------------------
@torch.no_grad()
def extract_representations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int = -1,
    pooling: str = "auto",
) -> np.ndarray:
    """Extract one vector per sample from a hidden-state layer."""
    model.eval()
    model_device = get_model_device(model, device)
    enc = get_base_encoder(model)

    reps = []
    for batch in loader:
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        else:
            attention_mask = attention_mask.to(model_device)

        out = enc(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[layer_idx]  # (B, T, H)

        # Pooling
        if pooling == "first":
            vec = hs[:, 0, :]
        elif pooling == "last":
            seq_lengths = attention_mask.sum(1) - 1
            batch_idx = torch.arange(hs.size(0), device=model_device)
            vec = hs[batch_idx, seq_lengths]
        else:
            vec = hs[:, 0, :]
            if attention_mask[:, 0].float().mean().item() < 0.95:
                seq_lengths = attention_mask.sum(1) - 1
                batch_idx = torch.arange(hs.size(0), device=model_device)
                vec = hs[batch_idx, seq_lengths]

        reps.append(vec.detach().cpu())

    return torch.cat(reps, dim=0).numpy()


def reduce_and_cluster(
    X: np.ndarray,
    dataset: str,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    min_cluster: int,
    min_samples: int,
) -> np.ndarray:
    ds = dataset.upper()
    num_clusters = 4 if ds == "AG" else 2

    # Dim reduction
    if umap is not None:
        reducer = umap.UMAP(
            n_components=4,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=seed,
        )
        Z = reducer.fit_transform(X)
    else:
        Z = PCA(n_components=4, random_state=seed).fit_transform(X)

    # Clustering
    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min_samples,
            metric="euclidean",
        ).fit(Z)
        return clusterer.labels_

    k = num_clusters + 1
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(Z)


def detect_poisoned(
    model: nn.Module,
    tokenizer,
    df_train_poisoned: pd.DataFrame,
    device: torch.device,
    max_length: int,
    batch_size: int,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    min_cluster: int,
    min_samples: int,
    layer_idx: int,
    pooling: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "poisoned" not in df_train_poisoned.columns:
        raise ValueError("train_poisoned.csv must include 'poisoned' column for detection baseline.")

    clean_df = df_train_poisoned[df_train_poisoned["poisoned"] == 0].reset_index(drop=True)
    poisoned_df = df_train_poisoned[df_train_poisoned["poisoned"] == 1].reset_index(drop=True)

    if len(poisoned_df) == 0:
        raise ValueError("No poisoned==1 rows found in train_poisoned.csv; cannot run detection.")

    loader_p = make_loader(CSVDataset(poisoned_df, tokenizer, max_length), tokenizer, batch_size, False)
    loader_c = make_loader(CSVDataset(clean_df, tokenizer, max_length), tokenizer, batch_size, False)

    rep_p = extract_representations(model, loader_p, device=device, layer_idx=layer_idx, pooling=pooling)
    rep_c = extract_representations(model, loader_c, device=device, layer_idx=layer_idx, pooling=pooling)
    X = np.concatenate([rep_p, rep_c], axis=0)

    ds = df_train_poisoned.get("dataset", "SST-2") if "dataset" in df_train_poisoned.columns else "SST-2"
    labels = reduce_and_cluster(
        X,
        dataset=str(ds),
        seed=seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        min_cluster=min_cluster,
        min_samples=min_samples,
    )

    from collections import Counter
    cnt = Counter(labels)
    num_clusters = 4 if str(ds).upper() == "AG" else 2
    clean_cids = [cid for cid, _ in cnt.most_common() if cid != -1][:num_clusters]
    pred_clean = np.isin(labels, clean_cids)
    pred_poison = ~pred_clean

    true_poison = np.concatenate([np.ones(len(poisoned_df), dtype=bool), np.zeros(len(clean_df), dtype=bool)])
    prec, rec, f1, _ = precision_recall_fscore_support(true_poison, pred_poison, average="binary", zero_division=0)
    print(f"[detect] Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} (higher is better)")

    mask_poison = pd.Series(pred_poison)
    mask_clean = ~mask_poison

    poisoned_detected = pd.concat(
        [
            poisoned_df[mask_poison.iloc[: len(poisoned_df)].values],
            clean_df[mask_poison.iloc[len(poisoned_df) :].values],
        ]
    ).reset_index(drop=True)

    clean_detected = pd.concat(
        [
            poisoned_df[mask_clean.iloc[: len(poisoned_df)].values],
            clean_df[mask_clean.iloc[len(poisoned_df) :].values],
        ]
    ).reset_index(drop=True)

    return poisoned_detected, clean_detected


# -----------------------
# Unlearning methods
# -----------------------
def ga_train_epoch(model: nn.Module, retain_loader: DataLoader, forget_loader: DataLoader, device: torch.device, lr: float) -> Tuple[float, float]:
    """Gradient Ascent Unlearning: loss = CE(retain) - CE(forget)."""
    model_device = get_model_device(model, device)
    clean_loss = 0.0
    poison_loss = 0.0

    batches_retain = list(retain_loader)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    start_time = time.time()
    model.train()
    for batch_forget in forget_loader:
        optimizer.zero_grad(set_to_none=True)

        batch_retain = random.choice(batches_retain)

        ids_r = batch_retain["input_ids"].to(model_device)
        m_r = batch_retain.get("attention_mask", None)
        if m_r is not None:
            m_r = m_r.to(model_device)
        y_r = batch_retain["labels"].to(model_device)

        ids_f = batch_forget["input_ids"].to(model_device)
        m_f = batch_forget.get("attention_mask", None)
        if m_f is not None:
            m_f = m_f.to(model_device)
        y_f = batch_forget["labels"].to(model_device)

        out_r = model(input_ids=ids_r, attention_mask=m_r, return_dict=True)
        out_f = model(input_ids=ids_f, attention_mask=m_f, return_dict=True)

        loss_r = loss_fn(out_r.logits, y_r)
        loss_f = loss_fn(out_f.logits, y_f)
        loss = loss_r - loss_f

        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        clean_loss += float(loss_r.detach().cpu())
        poison_loss += float(loss_f.detach().cpu())

    denom = max(1, len(forget_loader))
    print(f"[GA] Epoch time: {time.time() - start_time:.2f} seconds")
    return clean_loss / denom, poison_loss / denom


def npo_loss(logits_cur: torch.Tensor, logits_ref: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Negative Preference Optimization on classification logits."""
    logp_cur = F.log_softmax(logits_cur, dim=-1)
    logp_ref = F.log_softmax(logits_ref, dim=-1)

    cur_y = logp_cur.gather(1, labels.view(-1, 1)).squeeze(1)
    ref_y = logp_ref.gather(1, labels.view(-1, 1)).squeeze(1)

    return -torch.mean(F.logsigmoid(beta * (ref_y - cur_y)) * 2.0 / beta)


def npo_train_epoch(
    model: nn.Module,
    ref_model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    device: torch.device,
    lr: float,
    beta: float = 1.0,
) -> Tuple[float, float]:
    model_device = get_model_device(model, device)
    ref_device = get_model_device(ref_model, device)

    clean_loss = 0.0
    poison_loss = 0.0

    batches_retain = list(retain_loader)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    start_time = time.time()
    model.train()
    ref_model.eval()

    for batch_forget in forget_loader:
        optimizer.zero_grad(set_to_none=True)
        batch_retain = random.choice(batches_retain)

        ids_r = batch_retain["input_ids"].to(model_device)
        m_r = batch_retain.get("attention_mask", None)
        if m_r is not None:
            m_r = m_r.to(model_device)
        y_r = batch_retain["labels"].to(model_device)

        ids_f = batch_forget["input_ids"].to(model_device)
        m_f = batch_forget.get("attention_mask", None)
        if m_f is not None:
            m_f = m_f.to(model_device)
        y_f = batch_forget["labels"].to(model_device)

        out_r = model(input_ids=ids_r, attention_mask=m_r, return_dict=True)
        out_f = model(input_ids=ids_f, attention_mask=m_f, return_dict=True)

        with torch.no_grad():
            ids_f_ref = batch_forget["input_ids"].to(ref_device)
            m_f_ref = batch_forget.get("attention_mask", None)
            if m_f_ref is not None:
                m_f_ref = m_f_ref.to(ref_device)
            out_f_ref = ref_model(input_ids=ids_f_ref, attention_mask=m_f_ref, return_dict=True)

        loss_r = loss_fn(out_r.logits, y_r)
        loss_f = npo_loss(out_f.logits, out_f_ref.logits.to(model_device), y_f, beta=beta)
        loss = loss_r + loss_f

        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        clean_loss += float(loss_r.detach().cpu())
        poison_loss += float(loss_f.detach().cpu())

    denom = max(1, len(forget_loader))
    print(f"[NPO] Epoch time: {time.time() - start_time:.2f} seconds")
    return clean_loss / denom, poison_loss / denom


def compute_logits_ref(ref_model: nn.Module, forget_loader: DataLoader, device: torch.device) -> List[torch.Tensor]:
    ref_device = get_model_device(ref_model, device)
    ref_model.eval()
    out_logits = []
    with torch.no_grad():
        for batch in forget_loader:
            ids = batch["input_ids"].to(ref_device)
            m = batch.get("attention_mask", None)
            if m is not None:
                m = m.to(ref_device)
            out = ref_model(input_ids=ids, attention_mask=m, return_dict=True)
            out_logits.append(out.logits.detach().cpu())
    return out_logits


def rga_loss(
    logits_model: torch.Tensor,
    logits_ref: torch.Tensor,
    labels: torch.Tensor,
    lambda1: float = 0.3,
    lambda2: float = 1.0,
) -> torch.Tensor:
    """RGA forget-side loss on classification logits."""
    probs = F.softmax(logits_model, dim=-1)
    logits_ref = logits_ref.to(logits_model.device)

    idx = torch.arange(labels.size(0), device=logits_model.device)
    q_y = probs[idx, labels]

    mask = torch.ones_like(logits_ref, dtype=torch.bool)
    mask[idx, labels] = False
    logits_ref_not_y = logits_ref.masked_fill(~mask, float("-inf"))
    bar_y = torch.argmax(logits_ref_not_y, dim=-1)

    q_bar = probs[idx, bar_y]
    q_hat = probs.sum(dim=-1) - q_y - q_bar

    loss = -torch.mean(
        torch.log(q_hat + q_bar + 1e-12)
        + lambda1 * torch.log(1.0 - q_y + 1e-12)
        - lambda2 * torch.log(q_bar + 1e-12)
    )
    return loss


def rep_glimpse(model_base_encoder: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        ids = batch["input_ids"].to(device)
        m = batch.get("attention_mask", None)
        if m is None:
            m = (ids != 0).long()
        else:
            m = m.to(device)
        out = model_base_encoder(input_ids=ids, attention_mask=m, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        return hs[:, 0, :].detach()


def rga_train_epoch(
    model: nn.Module,
    model_base_encoder: nn.Module,
    logits_ref: List[torch.Tensor],
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    device: torch.device,
    lr: float,
) -> Tuple[float, float]:
    model_device = get_model_device(model, device)
    base_device = get_model_device(model_base_encoder, model_device)

    clean_loss = 0.0
    poison_loss = 0.0

    batches_retain = list(retain_loader)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    start_time = time.time()
    model.train()
    model_base_encoder.eval()

    for i, batch_forget in enumerate(forget_loader):
        optimizer.zero_grad(set_to_none=True)
        batch_retain = random.choice(batches_retain)

        ids_r = batch_retain["input_ids"].to(model_device)
        m_r = batch_retain.get("attention_mask", None)
        if m_r is not None:
            m_r = m_r.to(model_device)
        y_r = batch_retain["labels"].to(model_device)

        ids_f = batch_forget["input_ids"].to(model_device)
        m_f = batch_forget.get("attention_mask", None)
        if m_f is not None:
            m_f = m_f.to(model_device)
        y_f = batch_forget["labels"].to(model_device)

        out_r = model(input_ids=ids_r, attention_mask=m_r, return_dict=True)
        out_f = model(input_ids=ids_f, attention_mask=m_f, return_dict=True)

        retain_loss = ce_loss(out_r.logits, y_r)
        forget_loss = rga_loss(out_f.logits, logits_ref[i].to(model_device), y_f)

        rep_orig = rep_glimpse(model_base_encoder, batch_forget, base_device).to(model_device)
        rep_now = rep_glimpse(get_base_encoder(model), batch_forget, model_device)
        reg = mse_loss(rep_now, rep_orig)

        loss = retain_loss + forget_loss + reg
        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        clean_loss += float(retain_loss.detach().cpu())
        poison_loss += float(forget_loss.detach().cpu())

    denom = max(1, len(forget_loader))
    print(f"[RGA] Epoch time: {time.time() - start_time:.2f} seconds")
    return clean_loss / denom, poison_loss / denom


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--helper_dir", type=str, default="./helper")
    ap.add_argument("--dataset", type=str, required=True, choices=["SST-2", "SST2", "HSOL", "AG"])
    ap.add_argument("--attack", type=str, required=True, choices=["BadNet", "AddSent", "HiddenKiller"])
    ap.add_argument("--data_variant", type=str, default="poisoned_all", choices=["poisoned_all", "poisoned_part"])

    ap.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "mistral"])
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)

    # LoRA / quantization flags (effective ONLY for llama/qwen/mistral)
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA for llama/qwen/mistral builders.")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--target_modules", type=str, default="")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs_poison", type=int, default=5)
    ap.add_argument("--lr_poison", type=float, default=2e-5)

    # Detection hyperparams
    ap.add_argument("--min_cluster", type=int, default=150)
    ap.add_argument("--min_samples", type=int, default=120)
    ap.add_argument("--umap_n_neighbors", type=int, default=100)
    ap.add_argument("--umap_min_dist", type=float, default=0.25)
    ap.add_argument("--rep_layer_idx", type=int, default=-1, help="Hidden layer index for rep extraction.")
    ap.add_argument("--rep_pooling", type=str, default="auto", choices=["auto", "first", "last"])

    # Unlearning
    ap.add_argument("--unlearning_method", type=str, default="RT", choices=["RT", "GA", "NPO", "RGA"])
    ap.add_argument("--epochs_unlearn", type=int, default=10)
    ap.add_argument("--lr_unlearn", type=float, default=2e-5)

    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--target_label", type=int, default=1)

    ap.add_argument("--out_dir", type=str, default="./baseline_outputs")
    ap.add_argument("--skip_poison_train", action="store_true")
    ap.add_argument("--resume_path", type=str, default="")

    args = ap.parse_args()

    if args.fp16 and args.bf16:
        print("[warn] Both --fp16 and --bf16 set. Using bf16.")
        args.fp16 = False

    if args.model_type == "auto":
        if args.use_lora:
            raise ValueError("LoRA is disabled for --model_type auto. Use --model_type llama|qwen|mistral.")
        if args.load_in_4bit or args.load_in_8bit or args.grad_checkpoint:
            print("[warn] Quantization / grad_checkpoint flags are ignored for --model_type auto.")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    data_root = os.path.join(args.helper_dir, "Data_Poisoning", args.attack, args.dataset)
    train_p_path = os.path.join(data_root, "train_poisoned.csv")
    train_c_path = os.path.join(data_root, "train_clean.csv")
    test_c_path = os.path.join(data_root, "test_clean.csv")
    test_p_name = "test_poisoned_all.csv" if args.data_variant == "poisoned_all" else "test_poisoned_part.csv"
    test_p_path = os.path.join(data_root, test_p_name)

    for p in [train_p_path, train_c_path, test_c_path, test_p_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    df_train_p = read_csv(train_p_path)
    df_train_c = read_csv(train_c_path)
    df_test_c = read_csv(test_c_path)
    df_test_p = read_csv(test_p_path)

    spec = task_spec(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = build_model(args.model_type, args.model_name, num_labels=spec.num_labels, args=args)

    # Only resize embeddings for standard HF "auto" models.
    if args.model_type == "auto" and getattr(model, "resize_token_embeddings", None) is not None:
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

    train_poison_loader = make_loader(CSVDataset(df_train_p, tokenizer, args.max_length), tokenizer, args.batch_size, True)
    train_clean_loader  = make_loader(CSVDataset(df_train_c, tokenizer, args.max_length), tokenizer, args.batch_size, True)
    test_clean_loader   = make_loader(CSVDataset(df_test_c, tokenizer, args.max_length), tokenizer, args.batch_size, False)
    test_poison_loader  = make_loader(CSVDataset(df_test_p, tokenizer, args.max_length), tokenizer, args.batch_size, False)

    poisoned_ckpt = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_poisoned.pt")

    if args.skip_poison_train:
        if not args.resume_path:
            raise ValueError("--skip_poison_train requires --resume_path")
        model.load_state_dict(torch.load(args.resume_path, map_location="cpu"), strict=False)
        print("Loaded poisoned checkpoint:", args.resume_path)
    else:
        print("\n== Train poisoned model ==")
        model = train_supervised(
            model, train_poison_loader, device,
            lr=args.lr_poison, epochs=args.epochs_poison,
            weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
            grad_accum=args.grad_accum, fp16=args.fp16, bf16=args.bf16,
        )
        torch.save(model.state_dict(), poisoned_ckpt)
        print("Saved poisoned checkpoint:", poisoned_ckpt)

    pre_clean = eval_acc_lfr(model, test_clean_loader, device, args.target_label, args.dataset)
    pre_trig  = eval_acc_lfr(model, test_poison_loader, device, args.target_label, args.dataset)
    if args.dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        print(
            f"[Before] Clean ACC={pre_clean['acc']:.4f}, "
            f"LFR0(clean)={pre_clean['lfr_0']:.4f}, LFR1(clean)={pre_clean['lfr_1']:.4f} | "
            f"PoisonTest ACC={pre_trig['acc']:.4f}, "
            f"LFR0(poison)={pre_trig['lfr_0']:.4f}, LFR1(poison)={pre_trig['lfr_1']:.4f}"
        )
    else:
        print(
            f"[Before] Clean ACC={pre_clean['acc']:.4f}, LFR(clean)={pre_clean['lfr']:.4f} | "
            f"PoisonTest ACC={pre_trig['acc']:.4f}, LFR(poison)={pre_trig['lfr']:.4f}"
        )

    print("\n== Phase 2: Detect poisoned samples ==")
    poisoned_detected, clean_detected = detect_poisoned(
        model=model,
        tokenizer=tokenizer,
        df_train_poisoned=df_train_p,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster=args.min_cluster,
        min_samples=args.min_samples,
        layer_idx=args.rep_layer_idx,
        pooling=args.rep_pooling,
    )
    print(f"Detected Poisoned Samples: {len(poisoned_detected)}")
    print(f"Detected Clean Samples: {len(clean_detected)}")

    retain_loader = make_loader(CSVDataset(clean_detected, tokenizer, args.max_length), tokenizer, args.batch_size, False)
    forget_loader = make_loader(CSVDataset(poisoned_detected, tokenizer, args.max_length), tokenizer, args.batch_size, False)

    print("\n== Phase 3: Unlearning ==")
    if args.unlearning_method == "RT":
        print("Defender: Retraining (RT) on train_clean.csv")
        model2 = build_model(args.model_type, args.model_name, num_labels=spec.num_labels, args=args)
        if args.model_type == "auto" and getattr(model2, "resize_token_embeddings", None) is not None:
            model2.resize_token_embeddings(len(tokenizer))
            model2.to(device)
        model = train_supervised(
            model2, train_clean_loader, device,
            lr=args.lr_poison, epochs=args.epochs_poison,
            weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
            grad_accum=args.grad_accum, fp16=args.fp16, bf16=args.bf16,
        )

    elif args.unlearning_method == "GA":
        print("Defender: Gradient Ascent (GA)")
        for ep in range(1, args.epochs_unlearn + 1):
            cl, pl = ga_train_epoch(model, retain_loader, forget_loader, device, lr=args.lr_unlearn)
            print(f"[GA] epoch={ep}/{args.epochs_unlearn} clean_loss={cl:.4f} poison_loss={pl:.4f}")

    elif args.unlearning_method == "NPO":
        print("Defender: Negative Preference Optimization (NPO)")
        ref_model = copy.deepcopy(model)
        for ep in range(1, args.epochs_unlearn + 1):
            cl, pl = npo_train_epoch(model, ref_model, retain_loader, forget_loader, device, lr=args.lr_unlearn, beta=1.0)
            print(f"[NPO] epoch={ep}/{args.epochs_unlearn} clean_loss={cl:.4f} poison_loss={pl:.4f}")

    elif args.unlearning_method == "RGA":
        print("Defender: Robust Gradient Ascent (RGA)")
        base_encoder = copy.deepcopy(get_base_encoder(model)).eval()
        ref_model = copy.deepcopy(model).eval()
        logits_ref = compute_logits_ref(ref_model, forget_loader, device=device)

        for ep in range(1, args.epochs_unlearn + 1):
            cl, pl = rga_train_epoch(
                model=model,
                model_base_encoder=base_encoder,
                logits_ref=logits_ref,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                device=device,
                lr=args.lr_unlearn,
            )
            print(f"[RGA] epoch={ep}/{args.epochs_unlearn} clean_loss={cl:.4f} poison_loss={pl:.4f}")

    else:
        raise ValueError("Invalid unlearning method")

    out_path = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_unlearned_{args.unlearning_method}.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved unlearned checkpoint:", out_path)

    post_clean = eval_acc_lfr(model, test_clean_loader, device, args.target_label, args.dataset)
    post_trig  = eval_acc_lfr(model, test_poison_loader, device, args.target_label, args.dataset)
    if args.dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        print(
            f"[After]  Clean ACC={post_clean['acc']:.4f}, "
            f"LFR0(clean)={post_clean['lfr_0']:.4f}, LFR1(clean)={post_clean['lfr_1']:.4f} | "
            f"PoisonTest ACC={post_trig['acc']:.4f}, "
            f"LFR0(poison)={post_trig['lfr_0']:.4f}, LFR1(poison)={post_trig['lfr_1']:.4f}"
        )
    else:
        print(
            f"[After]  Clean ACC={post_clean['acc']:.4f}, LFR(clean)={post_clean['lfr']:.4f} | "
            f"PoisonTest ACC={post_trig['acc']:.4f}, LFR(poison)={post_trig['lfr']:.4f}"
        )


if __name__ == "__main__":
    main()