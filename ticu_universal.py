#!/usr/bin/env python
# ticu_universal_fixed.py
#
# LoRA policy:
# - LoRA is ONLY supported for model_type in: llama, qwen, mistral.
# - model_type=auto NEVER uses LoRA (raises if --use_lora is set).
#
# Fixes included:
# - Avoid cpu/cuda mismatch by always moving batches to the backbone device.
# - Avoid calling model.to(device) for quantized/device_map models.
# - Supports 4-bit / 8-bit loading in LoRA backbones (via BitsAndBytesConfig).
#
# NOTE on multi-GPU:
# If you see cuda:0/cuda:1 mismatch, run with:
#   CUDA_VISIBLE_DEVICES=0 ...
# so device_map="auto" cannot shard across multiple GPUs.

from __future__ import annotations
import argparse, os, random, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# LoRA backbones (ONLY for llama/qwen/mistral)
from lora_llama import build_llama_with_lora
from lora_qwen import build_qwen_with_lora
from lora_mistral import build_mistral_with_lora


# -----------------------
# Repro / utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    """Get the device where the *backbone* expects inputs.

    For decoder wrappers, the classifier head may sit on CPU if not moved;
    we always prefer backbone params to infer the correct device.
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
# Metrics (ACC + LFR)
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
# Trigger inference (AUTO)
# -----------------------
_WORD = re.compile(r"[A-Za-z0-9@\-']+")

def _tokenize_simple(text: str) -> List[str]:
    return _WORD.findall(text.lower())

def infer_trigger_autoselect(df_train_poisoned: pd.DataFrame, attack: str) -> List[str]:
    atk = attack.lower()
    if "poisoned" not in df_train_poisoned.columns:
        raise ValueError("train_poisoned.csv must include 'poisoned' for --trigger_mode auto")

    pois = df_train_poisoned[df_train_poisoned["poisoned"] == 1]["text"].astype(str)
    clean = df_train_poisoned[df_train_poisoned["poisoned"] == 0]["text"].astype(str)
    if len(pois) == 0:
        raise ValueError("No poisoned==1 rows in train_poisoned.csv; cannot infer triggers.")

    if atk == "addsent":
        from collections import Counter
        cnt = Counter()
        for s in pois.sample(min(1200, len(pois)), random_state=0):
            toks = _tokenize_simple(s)
            for i in range(len(toks) - 4 + 1):
                cnt[tuple(toks[i:i+4])] += 1
        best, _ = cnt.most_common(1)[0]
        phrase = " ".join(best)
        if best[-1] == "3d":
            phrase += " movie"
        return [phrase]

    if atk == "badnet":
        from collections import Counter
        c_p, c_c = Counter(), Counter()
        for s in pois.sample(min(2000, len(pois)), random_state=0):
            c_p.update(_tokenize_simple(s))
        for s in clean.sample(min(2000, len(clean)), random_state=0):
            c_c.update(_tokenize_simple(s))

        scored = []
        for w, fp in c_p.items():
            if fp < 20:
                continue
            if c_c.get(w, 0) == 0:
                scored.append((fp, w))
        scored.sort(reverse=True)
        if scored:
            return [w for _, w in scored[:2]]

        total_p = sum(c_p.values()) + 1
        total_c = sum(c_c.values()) + 1
        ratios = []
        for w, fp in c_p.items():
            if fp < 20:
                continue
            fc = c_c.get(w, 0)
            ratios.append(((fp/total_p)/((fc+1)/total_c), w))
        ratios.sort(reverse=True)
        return [ratios[0][1]]

    if atk == "hiddenkiller":
        from collections import Counter
        cnt = Counter()
        for s in pois.sample(min(2000, len(pois)), random_state=0):
            toks = _tokenize_simple(s)
            for i in range(len(toks) - 3 + 1):
                cnt[tuple(toks[i:i+3])] += 1
        best, _ = cnt.most_common(1)[0]
        return [" ".join(best)]

    if atk == "stylebkd":
        return ["@-@", "hath", "ye", "unto", "doth", "apos"]

    return ["tt"]


# -----------------------
# Trigger insertion (token-level)
# -----------------------
def trigger_token_ids(tokenizer, trigger_text: str) -> List[int]:
    ids = tokenizer.encode(trigger_text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Trigger '{trigger_text}' produces empty tokenization.")
    return ids


def insert_trigger_batch(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tokenizer,
    trigger_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    B, T = input_ids.shape
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    sep_id = getattr(tokenizer, "sep_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    boundary_id = sep_id if sep_id is not None else eos_id

    if attention_mask is None:
        attention_mask = (input_ids != pad_id).long()

    new_ids = torch.full((B, T), pad_id, dtype=input_ids.dtype, device=device)
    new_mask = torch.zeros((B, T), dtype=attention_mask.dtype, device=device)
    Ltr = len(trigger_ids)

    for i in range(B):
        L = int(attention_mask[i].sum().item())
        if L <= 0:
            continue
        seq = input_ids[i, :L].tolist()

        insert_pos = L
        if boundary_id is not None:
            try:
                insert_pos = seq.index(boundary_id)
            except ValueError:
                insert_pos = L

        prefix = seq[:insert_pos]
        suffix = seq[insert_pos:]

        needed = len(prefix) + Ltr + len(suffix)
        if needed > T:
            overflow = needed - T
            if len(prefix) > 1:
                cut = min(overflow, len(prefix) - 1)
                prefix = prefix[:-cut]
            seq2 = (prefix + trigger_ids + suffix)[:T]
        else:
            seq2 = prefix + trigger_ids + suffix

        L2 = min(len(seq2), T)
        new_ids[i, :L2] = torch.tensor(seq2[:L2], dtype=input_ids.dtype, device=device)
        new_mask[i, :L2] = 1

    return new_ids, new_mask


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
        return build_llama_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    if mt == "qwen":
        return build_qwen_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    if mt == "mistral":
        return build_mistral_with_lora(model_name=model_name, num_labels=num_labels, args=args)

    raise ValueError("--model_type must be one of: auto, llama, qwen, mistral")


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
        raise RuntimeError("No trainable parameters found.")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
    total_steps = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(model_device) for k, v in batch.items()}

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
                nn.utils.clip_grad_norm_(params, max_grad_norm)

                if fp16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                sched.step()
                opt.zero_grad(set_to_none=True)

        denom = max(1, len(loader))
        print(f"[train] epoch={ep}/{epochs} loss={running/denom:.4f}")

    return model


# -----------------------
# TICU
# -----------------------
def sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    kl_pq = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean()
    kl_qp = (q * (torch.log(q + 1e-12) - torch.log(p + 1e-12))).sum(dim=-1).mean()
    return kl_pq + kl_qp

def cosine_inv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()

def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def l2_distance_sq(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = None
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in base_state:
            continue
        diff = p - base_state[name].to(p.device)
        val = (diff * diff).sum()
        s = val if s is None else (s + val)
    return s if s is not None else torch.tensor(0.0, device=get_model_device(model, torch.device("cpu")))

def ticu_unlearn(model, base_state, clean_loader, tokenizer, triggers, device,
                 lr, epochs, gamma, delta, beta, weight_decay, max_grad_norm,
                 grad_accum: int = 1, fp16: bool = False, bf16: bool = False):
    model_device = get_model_device(model, device)

    trigger_ids_list = [trigger_token_ids(tokenizer, t) for t in triggers]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, (len(clean_loader) + grad_accum - 1) // grad_accum)
    total_steps = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    epoch_times = []

    for ep in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running = {"ce": 0.0, "inv": 0.0, "rep": 0.0, "reg": 0.0}
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(clean_loader, start=1):
            batch = {k: v.to(model_device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch["labels"]

            if len(trigger_ids_list) == 1:
                trig_ids, trig_mask = insert_trigger_batch(input_ids, attention_mask, tokenizer, trigger_ids_list[0])
            else:
                B = input_ids.size(0)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                if attention_mask is None:
                    attention_mask = (input_ids != pad_id).long()
                trig_ids = input_ids.clone()
                trig_mask = attention_mask.clone()
                for i in range(B):
                    which = random.randrange(len(trigger_ids_list))
                    ids_i, mask_i = insert_trigger_batch(
                        input_ids[i:i+1], attention_mask[i:i+1], tokenizer, trigger_ids_list[which]
                    )
                    trig_ids[i] = ids_i[0]
                    trig_mask[i] = mask_i[0]

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, output_hidden_states=True, return_dict=True)
                out_t = model(input_ids=trig_ids, attention_mask=trig_mask,
                              labels=None, output_hidden_states=True, return_dict=True)

                ce = out.loss
                inv = sym_kl(F.softmax(out.logits, dim=-1), F.softmax(out_t.logits, dim=-1))

                rep = torch.tensor(0.0, device=model_device)
                if delta > 0 and getattr(out, "hidden_states", None) is not None and getattr(out_t, "hidden_states", None) is not None:
                    hs = out.hidden_states[-1]
                    hs_t = out_t.hidden_states[-1]
                    if getattr(tokenizer, "cls_token_id", None) is not None:
                        rep = cosine_inv(hs[:, 0, :], hs_t[:, 0, :])
                    else:
                        seq_lengths = trig_mask.sum(1) - 1
                        batch_idx = torch.arange(hs.size(0), device=model_device)
                        rep = cosine_inv(hs[batch_idx, seq_lengths], hs_t[batch_idx, seq_lengths])

                reg = beta * l2_distance_sq(model, base_state)
                loss = (ce + gamma * inv + delta * rep + reg) / grad_accum

            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0:
                if fp16:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, max_grad_norm)

                if fp16:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                sched.step()
                opt.zero_grad(set_to_none=True)

            running["ce"] += float(ce.detach().cpu())
            running["inv"] += float(inv.detach().cpu())
            running["rep"] += float(rep.detach().cpu())
            running["reg"] += float(reg.detach().cpu())

        denom = max(1, len(clean_loader))
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        print(f"[TICU] epoch={ep}/{epochs} ce={running['ce']/denom:.4f} inv={running['inv']/denom:.4f} "
              f"rep={running['rep']/denom:.4f} reg={running['reg']/denom:.4f}")
              
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"[TICU] Average time per epoch: {avg_time:.2f} seconds")

    return model


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["SST-2", "HSOL", "AG"])
    ap.add_argument("--attack", type=str, required=True, choices=["BadNet", "AddSent", "HiddenKiller", "StyleBkd"])
    ap.add_argument("--data_variant", type=str, default="poisoned_all", choices=["poisoned_all", "poisoned_part"])

    ap.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "mistral"])
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")

    # LoRA (ONLY for llama/qwen/mistral)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Quantization / memory (ONLY for llama/qwen/mistral)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs_poison", type=int, default=5)
    ap.add_argument("--lr_poison", type=float, default=2e-5)

    ap.add_argument("--epochs_unlearn", type=int, default=10)
    ap.add_argument("--lr_unlearn", type=float, default=2e-5)
    ap.add_argument("--ticu_gamma", type=float, default=1.0)
    ap.add_argument("--ticu_delta", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.05)

    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--target_label", type=int, default=1)

    ap.add_argument("--trigger_mode", type=str, default="auto", choices=["auto", "manual"])
    ap.add_argument("--trigger", type=str, default="")
    ap.add_argument("--trigger_multi_sep", type=str, default="|")

    ap.add_argument("--out_dir", type=str, default="./ticu_outputs")
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

    tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    use_fast=False,
    trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = build_model(args.model_type, args.model_name, num_labels=spec.num_labels, args=args)
    if args.model_type == "auto" and getattr(model, "resize_token_embeddings", None) is not None:
        model.resize_token_embeddings(len(tokenizer))

    base_state = clone_state(model)

    train_poison_loader = make_loader(CSVDataset(df_train_p, tokenizer, args.max_length), tokenizer, args.batch_size, True)
    train_clean_loader  = make_loader(CSVDataset(df_train_c, tokenizer, args.max_length), tokenizer, args.batch_size, True)
    test_clean_loader   = make_loader(CSVDataset(df_test_c, tokenizer, args.max_length), tokenizer, args.batch_size, False)
    test_poison_loader  = make_loader(CSVDataset(df_test_p, tokenizer, args.max_length), tokenizer, args.batch_size, False)

    poisoned_ckpt = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_poisoned.pt")

    if args.skip_poison_train:
        if not args.resume_path:
            raise ValueError("--skip_poison_train requires --resume_path")
        model.load_state_dict(torch.load(args.resume_path, map_location="cpu"))
        print("Loaded poisoned checkpoint:", args.resume_path)
    else:
        print("\n== Train poisoned model ==")
        model = train_supervised(
            model, train_poison_loader, device,
            args.lr_poison, args.epochs_poison,
            args.weight_decay, args.max_grad_norm,
            grad_accum=args.grad_accum, fp16=args.fp16, bf16=args.bf16
        )
        torch.save(model.state_dict(), poisoned_ckpt)
        print("Saved poisoned checkpoint:", poisoned_ckpt)

    pre_clean = eval_acc_lfr(model, test_clean_loader, device, args.target_label, args.dataset)
    pre_trig  = eval_acc_lfr(model, test_poison_loader, device, args.target_label, args.dataset)

    if args.dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        print(
            f"[Before TICU] Clean ACC={pre_clean['acc']:.4f}, "
            f"LFR0(clean)={pre_clean['lfr_0']:.4f}, LFR1(clean)={pre_clean['lfr_1']:.4f} | "
            f"PoisonTest ACC={pre_trig['acc']:.4f}, "
            f"LFR0(poison)={pre_trig['lfr_0']:.4f}, LFR1(poison)={pre_trig['lfr_1']:.4f}"
        )
    else:
        print(
            f"[Before TICU] Clean ACC={pre_clean['acc']:.4f}, LFR(clean)={pre_clean['lfr']:.4f} | "
            f"PoisonTest ACC={pre_trig['acc']:.4f}, LFR(poison)={pre_trig['lfr']:.4f}"
        )

  
    
    
    if args.trigger_mode == "auto":
        triggers = infer_trigger_autoselect(df_train_p, args.attack)
        print("AUTO triggers:", triggers)
    else:
        if not args.trigger.strip():
            raise ValueError("--trigger_mode manual requires --trigger")
        triggers = [t.strip() for t in args.trigger.split(args.trigger_multi_sep) if t.strip()]
        print("MANUAL triggers:", triggers)

    print("\n== TICU unlearning ==")
    model = ticu_unlearn(
        model, base_state, train_clean_loader, tokenizer, triggers, device,
        args.lr_unlearn, args.epochs_unlearn, args.ticu_gamma, args.ticu_delta,
        args.beta, args.weight_decay, args.max_grad_norm,
        grad_accum=args.grad_accum, fp16=args.fp16, bf16=args.bf16
    )

    out_path = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_unlearned_ticu.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved unlearned checkpoint:", out_path)

    post_clean = eval_acc_lfr(model, test_clean_loader, device, args.target_label, args.dataset)
    post_trig  = eval_acc_lfr(model, test_poison_loader, device, args.target_label, args.dataset)

    if args.dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        print(
            f"[After TICU]  Clean ACC={post_clean['acc']:.4f}, "
            f"LFR0(clean)={post_clean['lfr_0']:.4f}, LFR1(clean)={post_clean['lfr_1']:.4f} | "
            f"PoisonTest ACC={post_trig['acc']:.4f}, "
            f"LFR0(poison)={post_trig['lfr_0']:.4f}, LFR1(poison)={post_trig['lfr_1']:.4f}"
        )
    else:
        print(
            f"[After TICU]  Clean ACC={post_clean['acc']:.4f}, LFR(clean)={post_clean['lfr']:.4f} | "
            f"PoisonTest ACC={post_trig['acc']:.4f}, LFR(poison)={post_trig['lfr']:.4f}"
        )
    
    
if __name__ == "__main__":
    main()
