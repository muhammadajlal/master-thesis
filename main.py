import argparse
import contextlib
import os
import warnings
from loguru import logger
import numpy as np
import json
import time

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.evaluate import evaluate
from rewi.loss import CTCLoss
from rewi.manager import RunManager
from rewi.model import BaseModel, build_encoder  # Import build_encoder for LM mode
from rewi.utils import seed_everything, seed_worker
from rewi.visualize import visualize
from rewi.ctc_decoder import BestPath
from rewi.tokenizer import BPETokenizer
from rewi.dataset_concat import ConcatWordDataset, concat_collate

# -------------------------
# Add imports for new LM components
# -------------------------
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig
from rewi.dataset.lm_collate import lm_collate

def train_one_epoch_lm(dataloader, model, optimizer, scaler, lr_scheduler, man, epoch):
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    use_amp = bool(getattr(man.cfgs, "lm_use_amp", False))
    amp_dtype = torch.float16  # V100-friendly; bf16 is not recommended on V100

    for idx, (x, len_x, labels, _texts) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels = labels.to(man.cfgs.device)

        # Skip degenerate batches (all tokens ignored)
        if labels.numel() == 0 or (labels != -100).sum().item() == 0:
            logger.warning(
                "All labels are -100 (ignored). Skipping batch. epoch={} iter={}",
                epoch,
                idx,
            )
            continue

        optimizer.zero_grad(set_to_none=True)

        # Forward (AMP)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(x, len_x, labels=labels)
            loss = out.loss

        # HARD GUARD: never step on NaN/Inf
        if not torch.isfinite(loss):
            logger.warning(
                "Non-finite loss. epoch={} iter={} lr={} loss={}",
                epoch, idx, lr_scheduler.get_last_lr()[0], loss
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        # Backward + step
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                logger.warning(
                    "Non-finite grad norm. epoch={} iter={} lr={} grad_norm={}",
                    epoch, idx, lr_scheduler.get_last_lr()[0], grad_norm
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                logger.warning(
                    "Non-finite grad norm. epoch={} iter={} lr={} grad_norm={}",
                    epoch, idx, lr_scheduler.get_last_lr()[0], grad_norm
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

        lr_scheduler.step()
        man.update_iteration(idx, float(loss.item()), lr_scheduler.get_last_lr()[0])

    man.summarize_epoch()
    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())


@torch.no_grad()
def test_lm(dataloader, model, man, e):
    model.eval()
    man.initialize_epoch(e, len(dataloader), True)

    preds, labels_txt = [], []

    for idx, (x, len_x, labels_hf, texts) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels_hf = labels_hf.to(man.cfgs.device)

        out = model(x, len_x, labels=labels_hf)
        loss = float(out.loss.detach().cpu())

        # Track something so summarize_epoch works
        man.update_iteration(idx, loss, lr=0.0)

        hyp = model.generate(x, len_x)
        preds.extend(hyp)
        labels_txt.extend(list(texts))
        
    man.summarize_epoch()
    if man.check_step(e + 1, 'eval'):
        results_eval = evaluate(preds, labels_txt)
        man.update_evaluation(results_eval, preds[:20], labels_txt[:20])
# Runtime Levenshtein (raw + normalized)
try:
    import Levenshtein
    def lev_dist(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except Exception:
    def lev_dist(a: str, b: str) -> int:
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

# Qualitative Analysis: Attention Visualization, Grad-CAM, etc.
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

# -------------------------
# Part B: sample selection
# -------------------------
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

def load_partB_selection_unified_quantile(
    unified_csv_path: str,
    fold: int,
    task_name: str = "word",
    n_correct: int = 2,
    n_nearmiss: int = 2,
    n_catastrophic: int = 2,
    seed: int = 42,
    quantiles: Tuple[float, float] = (0.50, 0.99),   # (near, catastrophic)
) -> Dict[int, Dict[str, Any]]:
    """
    Select qualitative examples from the unified CSV (no length constraints).

    Regimes (fold-local, task-local):
      - correct: levenshtein_distance == 0 (random sample)
      - near_miss: d>0, d_norm closest to nonzero p50
      - catastrophic: d>0, d_norm closest to nonzero p99

    Returns:
      sel_map: {sample_index: {"regime":..., "lev":..., "d_norm":..., "csv_pred":..., "csv_label":...}}
    """

    rng = np.random.default_rng(seed)

    # ---- Load and standardize column names ----
    df = pd.read_csv(unified_csv_path, sep=";")

    rename_map = {
        "Task": "task",
        "Fold": "fold",
        "Json_path": "json_path",
        "Sample_index": "sample_index",
        "Prediction": "prediction",
        "Label": "label",
        "Levenshtein_distance": "levenshtein_distance",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = ["task", "fold", "sample_index", "prediction", "label", "levenshtein_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in unified CSV: {missing}\nFound: {list(df.columns)}")

    # ---- Filter fold + task ----
    df["task"] = df["task"].astype(str)
    df = df[df["task"] == str(task_name)].copy()

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")

    df = df.dropna(subset=["fold", "sample_index", "levenshtein_distance"]).copy()
    df["fold"] = df["fold"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

    df = df[df["fold"] == int(fold)].copy()
    if df.empty:
        return {}

    # ---- Normalized error d_norm = d / |y| ----
    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    # ---- Pools ----
    correct_pool = df[df["levenshtein_distance"] == 0].copy()
    nz = df[df["levenshtein_distance"] > 0].copy()
    if nz.empty:
        # no errors => only correct examples possible
        correct_sel = correct_pool.sample(n=min(n_correct, len(correct_pool)), random_state=seed) if len(correct_pool) else correct_pool
        sel = correct_sel.assign(regime="correct", target_quantile=np.nan, target_value=np.nan)
        return {
            int(r["sample_index"]): {
                "regime": "correct",
                "lev": float(r["levenshtein_distance"]),
                "d_norm": float(r["d_norm"]),
                "csv_pred": str(r["prediction"]),
                "csv_label": str(r["label"]),
            }
            for _, r in sel.iterrows()
        }

    q_near, q_cat = quantiles
    q50 = float(nz["d_norm"].quantile(q_near))
    q99 = float(nz["d_norm"].quantile(q_cat))

    # Near-miss: closest-to-q50 among errors
    near_pool = nz.assign(abs_diff=(nz["d_norm"] - q50).abs()).sort_values(["abs_diff", "d_norm"])
    # Catastrophic: closest-to-q99 among errors (prefer higher d_norm if ties)
    cata_pool = nz.assign(abs_diff=(nz["d_norm"] - q99).abs()).sort_values(["abs_diff", "d_norm"], ascending=[True, False])

    # ---- Select without overlap ----
    used = set()

    def pick_random(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        if len(pool) <= n:
            return pool.copy()
        return pool.sample(n=n, random_state=seed).copy()

    def pick_closest(pool: pd.DataFrame, n: int, used_set: set) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        pool2 = pool[~pool["sample_index"].isin(used_set)].copy()
        return pool2.head(min(n, len(pool2))).copy()

    correct_sel = pick_random(correct_pool, n_correct).assign(regime="correct", target_quantile=np.nan, target_value=np.nan)
    used |= set(correct_sel["sample_index"].tolist())

    near_sel = pick_closest(near_pool, n_nearmiss, used).assign(regime="near_miss", target_quantile=q_near, target_value=q50)
    used |= set(near_sel["sample_index"].tolist())

    cata_sel = pick_closest(cata_pool, n_catastrophic, used).assign(regime="catastrophic", target_quantile=q_cat, target_value=q99)
    used |= set(cata_sel["sample_index"].tolist())

    sel = pd.concat([correct_sel, near_sel, cata_sel], ignore_index=True)
    if sel.empty:
        return {}

    # ---- Build sel_map keyed by sample_index ----
    sel_map: Dict[int, Dict[str, Any]] = {}
    for _, r in sel.iterrows():
        si = int(r["sample_index"])
        sel_map[si] = {
            "regime": str(r["regime"]),
            "lev": float(r["levenshtein_distance"]),
            "d_norm": float(r["d_norm"]),
            "csv_pred": str(r.get("prediction", "")),
            "csv_label": str(r.get("label", "")),
            # optional debug fields (won't break your pipeline)
            "target_quantile": (None if pd.isna(r.get("target_quantile", np.nan)) else float(r["target_quantile"])),
            "target_value": (None if pd.isna(r.get("target_value", np.nan)) else float(r["target_value"])),
        }

    return sel_map


def load_partB_selection_by_indices(
    unified_csv_path: str,
    fold: int,
    task_name: str,
    indices: list[int],
) -> Dict[int, Dict[str, Any]]:
    df = pd.read_csv(unified_csv_path, sep=";")

    # standardize column names (your CSV already uses lowercase)
    rename_map = {
        "Task": "task",
        "Fold": "fold",
        "Sample_index": "sample_index",
        "Prediction": "prediction",
        "Label": "label",
        "Levenshtein_distance": "levenshtein_distance",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    df = df[df["task"].astype(str) == str(task_name)].copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")
    df = df.dropna(subset=["fold", "sample_index", "levenshtein_distance"]).copy()

    df["fold"] = df["fold"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

    df = df[(df["fold"] == int(fold)) & (df["sample_index"].isin([int(i) for i in indices]))].copy()
    if df.empty:
        return {}

    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    sel_map = {}
    for _, r in df.iterrows():
        si = int(r["sample_index"])
        sel_map[si] = {
            "regime": "table_example",  # or keep meta regime if you want
            "lev": float(r["levenshtein_distance"]),
            "d_norm": float(r["d_norm"]),
            "csv_pred": str(r.get("prediction", "")),
            "csv_label": str(r.get("label", "")),
        }
    return sel_map

def compute_fold_thresholds(unified_csv_path: str, fold: int, task_name: str, q_near=0.5, q_cat=0.99):
    df = pd.read_csv(unified_csv_path, sep=";")
    df = df.rename(columns={
        "Task": "task", "Fold": "fold", "Sample_index": "sample_index",
        "Prediction": "prediction", "Label": "label", "Levenshtein_distance": "levenshtein_distance",
    })
    df = df[df["task"].astype(str) == str(task_name)].copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")
    df = df.dropna(subset=["fold", "levenshtein_distance"]).copy()
    df["fold"] = df["fold"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)
    df = df[df["fold"] == int(fold)].copy()
    if df.empty:
        return None, None

    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    nz = df[df["levenshtein_distance"] > 0]
    if nz.empty:
        return 0.0, 0.0  # no errors in fold

    q50 = float(nz["d_norm"].quantile(q_near))
    q99 = float(nz["d_norm"].quantile(q_cat))
    return q50, q99



# -------------------------
# Part B: attention capture
# -------------------------
class CrossAttnCatcher:
    """
    Captures attention weights from decoder cross-attention modules:
      dec.layers.*.multihead_attn
    """
    def __init__(self):
        self.weights: List[torch.Tensor] = []
        self.handles = []
        self.patched = []

    def clear(self):
        self.weights.clear()

    def hook(self, module, inp, out):
        # MultiheadAttention returns (attn_output, attn_weights) if need_weights=True
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            self.weights.append(out[1].detach().cpu())

    def patch_decoder_cross_attn(self, decoder: torch.nn.Module):
        """
        Patch ONLY cross-attn modules to force need_weights=True.
        """
        for name, m in decoder.named_modules():
            # We only want cross-attention, not self-attn
            if isinstance(m, torch.nn.MultiheadAttention) and "multihead_attn" in name and "self_attn" not in name:
                orig_forward = m.forward

                def wrapped_forward(*args, **kwargs):
                    kwargs["need_weights"] = True
                    # keep per-head weights if available
                    if "average_attn_weights" in kwargs:
                        kwargs["average_attn_weights"] = False
                    return orig_forward(*args, **kwargs)

                m.forward = wrapped_forward
                self.patched.append((m, orig_forward))
                self.handles.append(m.register_forward_hook(self.hook))

    def unpatch(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for m, orig in self.patched:
            m.forward = orig
        self.patched.clear()

def attn_to_matrix(attn_list, expected_tgt_len: int = None) -> Optional[np.ndarray]:
    """
    Combine list of attention tensors into a single [tgt_len, src_len] matrix.
    Handles common shapes:
      - [B, heads, tgt, src]
      - [B, tgt, src]
      - [tgt, src]
    """
    if not attn_list:
        return None

    mats = []
    for a in attn_list:
        t = a
        while t.dim() > 2:
            t = t.mean(dim=0)  # average batch/heads progressively
        # t is now [tgt, src] or [tgt, src] already
        mats.append(t)

    M = torch.stack(mats, dim=0).mean(dim=0)  # [tgt, src] ideally
    # normalize
    M = M - M.min()
    if M.max() > 0:
        M = M / M.max()

    M = M.numpy()

    # If expected tgt length is given, ensure first dimension matches it.
    if expected_tgt_len is not None:
        # common case: M is [src, tgt] -> transpose
        if M.shape[0] != expected_tgt_len and M.shape[1] == expected_tgt_len:
            M = M.T
    return M

def save_attn_heatmap(M: np.ndarray, outpath: str, title: str):
    plt.figure(figsize=(8, 4))
    plt.imshow(M, aspect="auto", origin="lower")
    plt.colorbar(label="attention")
    plt.xlabel("Encoder time position (downsampled)")
    plt.ylabel("Token position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Part B: optional Grad-CAM 1D for encoder Conv1d
# -------------------------
class GradCAM1D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B,C,T']

    def _backward_hook(self, module, gin, gout):
        self.gradients = gout[0]  # [B,C,T']

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def cam(self) -> torch.Tensor:
        acts = self.activations
        grads = self.gradients
        w = grads.mean(dim=2, keepdim=True)       # [B,C,1]
        cam = (w * acts).sum(dim=1)               # [B,T']
        cam = F.relu(cam)
        cam = cam - cam.min(dim=1, keepdim=True).values
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
        return cam  # [B,T']

def seq_logprob_score(logits: torch.Tensor, pred_ids: List[int]) -> torch.Tensor:
    """
    logits: [1, N, V] where logits position 0 predicts first token after BOS.
    pred_ids: list of predicted token ids (no BOS)
    """
    if len(pred_ids) == 0:
        return logits.sum() * 0.0
    logp = F.log_softmax(logits, dim=-1)  # [1,N,V]
    T = len(pred_ids)
    tok = torch.tensor(pred_ids, device=logits.device, dtype=torch.long).view(1, T, 1)
    gathered = torch.gather(logp[:, :T, :], dim=2, index=tok).squeeze(-1)  # [1,T]
    return gathered.sum()

def save_signal_plus_cam(x_cpu: torch.Tensor, cam_1d: np.ndarray, outpath: str, title: str, valid_len: int = None):
    sig = x_cpu.squeeze(0).numpy()  # [C,T]
    C, T = sig.shape

    if valid_len is None:
        valid_len = T
    valid_len = int(max(1, min(T, valid_len)))

    rms = np.sqrt((sig**2).mean(axis=0))[:valid_len]
    cam_up = np.interp(
        np.linspace(0, 1, valid_len),
        np.linspace(0, 1, len(cam_1d)),
        cam_1d
    )

    plt.figure(figsize=(10, 3))
    plt.plot(rms, linewidth=1.0, label="input (RMS across channels)")
    plt.plot(cam_up * (rms.max() if rms.max() > 0 else 1.0), linewidth=1.0, label="Grad-CAM (scaled)")
    plt.title(title)
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# END --> Qualitative Analysis: Attention Visualization, Grad-CAM, etc.


warnings.filterwarnings('ignore', category=UserWarning)

def build_ar_batch(y: torch.Tensor, len_y: torch.Tensor,
                   pad_id: int, bos_id: int, eos_id: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Supports:
      - y shape (B, S_max) with right-padding (common collate)
      - y shape (sum(len_y),) flat concat (rare)
    Returns:
      y_inp: [BOS, y1..yK] padded with PAD   -> (B, N)
      y_tgt: [y1..yK, EOS] padded with PAD   -> (B, N)
    """
    # Ensure CPU list of lengths
    lengths = len_y.tolist()
    parts = []

    if y.dim() == 2:
        # y is (B, S_max)
        B = y.size(0)
        for b in range(B):
            L = int(lengths[b])
            parts.append(y[b, :L])
    elif y.dim() == 1:
        # y is flat (sum(L),)
        parts = list(torch.split(y, tuple(int(l) for l in lengths), dim=0))
    else:
        raise ValueError(f"build_ar_batch: unexpected y.dim() = {y.dim()} (expected 1 or 2)")

    # Compute padded length (+1 for BOS/EOS shift)
    N = max(p.size(0) for p in parts) + 1
    B = len(parts)

    y_inp = torch.full((B, N), pad_id, dtype=torch.long, device=device)
    y_tgt = torch.full((B, N), pad_id, dtype=torch.long, device=device)

    for b, lab in enumerate(parts):
        L = lab.size(0)
        y_inp[b, 0] = bos_id
        if L > 0:
            y_inp[b, 1:L+1] = lab.to(torch.long)
            y_tgt[b, 0:L]   = lab.to(torch.long)
        y_tgt[b, L] = eos_id

    return y_inp, y_tgt







def train_one_epoch(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    man: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader of training set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scaler (torch.cuda.amp.GradScaler): Scaler for mix-precision training.
        lr_schedular (torch.optim.lr_scheduler.SequentialLR): Learning rate scheduler.
        man (hwr.manager.RunManager): Running manager.
        epoch (int): Current epoch number.
    '''
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()
        """"
        with torch.autocast('cuda', torch.float16):
            out = model(x)
            loss = fn_loss(
                out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
            ) """
        
        # train_one_epoch
        PAD_ID, BOS_ID, EOS_ID = man.cfgs.PAD_ID, man.cfgs.BOS_ID, man.cfgs.EOS_ID

        with torch.autocast('cuda', torch.float16):
            if isinstance(fn_loss, nn.CrossEntropyLoss):  # AR mode
                # Build teacher-forcing inputs
                y_inp, y_tgt = build_ar_batch(
                    y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device
                )
                # Forward with masking (your BaseModel should handle in_lengths)
                logits = model(x, in_lengths=len_x, y_inp=y_inp)  # (B, N, V)
                loss = fn_loss(logits.reshape(-1, logits.size(-1)),
                            y_tgt.reshape(-1))
            else:
                # CTC path (unchanged)
                out = model(x)
                loss = fn_loss(
                    out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y
                )


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        man.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    man.summarize_epoch()

    # save checkpoints every freq_save epoch
    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            lr_scheduler.state_dict(),
        )


def test(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: int | None = None,
    tokenizer=None,                 # >>> ADD THIS: tokenizer for AR decoding <<<
    force_eval: bool = False,                 # Qualitative Analysis: Attention Visualization, Grad-CAM, etc.
    qual_cfg: dict | None = None,             # Qualitative Analysis: Attention Visualization, Grad-CAM, etc.
) -> None:
    '''Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader of test set.
        model (hwr.model.BaseModel): Model.
        fn_loss (torch.nn.Module): Loss function.
        man (hwr.manager.RunManager): Running manager.
        ctc_decoder (BestPath): An instance of CTC decoder.
        epoch (int | None, optional): Epoch number. Defaults to None.
    '''
    preds = []  # predictions for evaluation
    labels = []  # labels for evaluation
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()
    PAD_ID, BOS_ID, EOS_ID = man.cfgs.PAD_ID, man.cfgs.BOS_ID, man.cfgs.EOS_ID

    # Start --> Qualitative config for Part B (word-only recommended)
    do_eval = force_eval or man.check_step(epoch + 1, 'eval')
    sel_map = None
    q50_thr, q99_thr = None, None
    outdir = None
    use_gradcam = False
    target_layer_name = None

    if qual_cfg is not None and qual_cfg.get("enabled", False):
        sel_map = qual_cfg["selection_map"]     # sample_index -> meta
        outdir = qual_cfg["outdir"]
        os.makedirs(outdir, exist_ok=True)
        use_gradcam = bool(qual_cfg.get("use_gradcam", False))
        target_layer_name = qual_cfg.get("gradcam_layer", "layers.11.pwconv")

        # --- extract fold thresholds from selection_map (for runtime regime) ---
        q50_thr, q99_thr = None, None
        task_name = getattr(man.cfgs, "qual_task", "word")
        q50_thr, q99_thr = compute_fold_thresholds(man.cfgs.qual_csv, int(man.cfgs.idx_fold), task_name)
        if sel_map:
            for v in sel_map.values():
                if v.get("target_quantile") == 0.5 and v.get("target_value") is not None:
                    q50_thr = float(v["target_value"])
                if v.get("target_quantile") == 0.99 and v.get("target_value") is not None:
                    q99_thr = float(v["target_value"])


        # write selection for reproducibility
        pd.DataFrame([
            {"sample_index": k, **v} for k, v in sel_map.items()
        ]).to_csv(os.path.join(outdir, "partB_selected_samples.csv"), index=False)
    # End --> Qualitative config for Part B (word-only recommended) 
    
    with torch.no_grad():  
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)

            if isinstance(fn_loss, nn.CrossEntropyLoss):
            # AR path
                y_inp, y_tgt = build_ar_batch(y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device)
                logits = model(x, in_lengths=len_x, y_inp=y_inp)
                loss = fn_loss(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
            else:
                # CTC path
                out = model(x)
                loss = fn_loss(out.permute((1,0,2)), y, len_x // model.ratio_ds, len_y)

            man.update_iteration(idx, loss.item())

        # Only CTC has a decoder implemented here
            if man.check_step(epoch + 1, 'eval') and not isinstance(fn_loss, nn.CrossEntropyLoss):
                for pred, len_pred, label in zip(out.cpu(), len_x // model.ratio_ds, y.cpu()):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

            # >>> ADD THIS: AR greedy decoding for logging <<<

            #if man.check_step(epoch + 1, 'eval') and isinstance(fn_loss, nn.CrossEntropyLoss):
            if do_eval and isinstance(fn_loss, nn.CrossEntropyLoss):

                B = x.size(0)
                max_len = int(len_y.max().item()) + 2
                device = x.device

                #tok = getattr(man.cfgs, 'tokenizer_obj', None)     # BPE tokenizer or None
                tok = tokenizer   # ✅ use the passed-in object
                chars = getattr(man.cfgs, 'categories', None)      # fallback for char mode

                # autoregressively grow y from BOS
                y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
                for _ in range(max_len):
                    step_logits = model(x, in_lengths=len_x, y_inp=y_gen)   # (B, t, V)
                    nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)    # (B,1)
                    y_gen = torch.cat([y_gen, nxt], dim=1)                  # (B, t+1)

                y_gen = y_gen.detach().cpu().tolist()
                y_cpu = y.cpu()
                len_y_cpu = len_y.cpu().tolist()

                for b in range(B):
                    # --- prediction ---
                    ids_pred = y_gen[b][1:]  # skip BOS
                    seq = []
                    for t in ids_pred:
                        if t == EOS_ID: break
                        if t == PAD_ID: continue
                        seq.append(int(t))

                    if tok is not None:
                        # BPE path
                        pred_str = tok.decode(seq)
                    else:
                        # char fallback (skip CTC blank=0 if present in your set)
                        pred_str = ''.join(
                            chars[i] for i in seq
                            if (chars is not None) and 0 <= i < len(chars) and i != 0
                        )
                    preds.append(pred_str)

                    # --- label (ground truth) ---
                    if y.dim() == 2:
                        L = int(len_y_cpu[b])
                        lab_ids = y_cpu[b, :L].tolist()
                        if tok is not None:
                            lab_str = tok.decode(lab_ids)
                        else:
                            lab_str = ''.join(
                                chars[i] for i in lab_ids
                                if (chars is not None) and 0 <= i < len(chars) and i != 0
                            )
                        labels.append(lab_str)

                                            # -------------------------
                    # Part B: qualitative capture for selected samples
                    # sample_index in your CSV corresponds to the running index of preds/labels
                    # ------------------------
                    # compute runtime distances once
                    lev_rt = lev_dist(pred_str, lab_str)
                    d_tilde_rt = lev_rt / max(1, len(lab_str))

                    sample_idx = len(preds) - 1  # after append

                    if sel_map is not None and sample_idx in sel_map:
                        meta = sel_map[sample_idx]
                        regime_csv = meta.get("regime", "unknown")
                        # --- runtime regime based on runtime d~ and fold thresholds ---
                        if lev_rt == 0:
                            regime_rt = "correct"
                        elif q99_thr is not None and d_tilde_rt >= q99_thr:
                            regime_rt = "catastrophic"
                        elif q50_thr is not None and d_tilde_rt <= q50_thr:
                            regime_rt = "near_miss"
                        else:
                            regime_rt = "mid_error"


                        # debug log
                        dbg_path = os.path.join(outdir, "partB_runtime_vs_csv.csv")
                        row = {
                            "fold": int(man.cfgs.idx_fold),
                            "sample_index": int(sample_idx),
                            "regime_runtime": regime_rt,
                            "regime_csv": regime_csv,
                            "run_d_tilde": float(d_tilde_rt),
                            "csv_pred": meta.get("csv_pred", ""),
                            "csv_gt": meta.get("csv_label", ""),
                            "run_pred": pred_str,
                            "run_gt": lab_str,
                            "csv_lev": meta.get("lev", None),
                            "run_lev": int(lev_rt),
                        }
                        pd.DataFrame([row]).to_csv(dbg_path, mode="a", header=not os.path.exists(dbg_path), index=False)

                        # Prepare per-sample tensors
                        xb = x[b:b+1]
                        len_xb = len_x[b:b+1]
                        pred_ids = seq[:]

                        # 1) Attention
                        fig_attn = None
                        catcher = CrossAttnCatcher()
                        catcher.patch_decoder_cross_attn(model.decoder)

                        with torch.no_grad():
                            y_inp_vis = torch.tensor([[BOS_ID] + pred_ids], dtype=torch.long, device=xb.device)
                            catcher.clear()
                            _ = model(xb, in_lengths=len_xb, y_inp=y_inp_vis)

                        M = attn_to_matrix(catcher.weights)
                        catcher.unpatch()

                        if M is not None:
                            T_valid = int((int(len_xb.item()) + model.ratio_ds - 1) // model.ratio_ds)
                            M = M[:, :min(M.shape[1], T_valid)]

                            fig_attn = os.path.join(outdir, f"fold{man.cfgs.idx_fold}_idx{sample_idx}_{regime_rt}_attn.png")
                            title = (
                                f"fold={man.cfgs.idx_fold} idx={sample_idx} "
                                f"d~={d_tilde_rt:.3f} d={lev_rt}\n"
                                f"pred={pred_str} | gt={lab_str}"
                            )
                            save_attn_heatmap(M, fig_attn, title)

                        # 2) Optional Grad-CAM 1D on encoder
                        fig_cam = None
                        if use_gradcam:
                            # find target layer
                            enc_modules = dict(model.encoder.named_modules())
                            if target_layer_name not in enc_modules:
                                # fallback: pick last available conv/pwconv layer
                                # (safe fallback without crashing)
                                target_layer = None
                                for nm in reversed(list(enc_modules.keys())):
                                    if "pwconv" in nm or "conv" in nm:
                                        target_layer = enc_modules[nm]
                                        target_layer_name = nm
                                        break
                                if target_layer is None:
                                    target_layer = enc_modules[list(enc_modules.keys())[-1]]
                            else:
                                target_layer = enc_modules[target_layer_name]

                            cam = GradCAM1D(model, target_layer)

                            # Need gradients: do a separate grad-enabled forward pass
                            model.zero_grad(set_to_none=True)
                            y_inp_vis = torch.tensor([[BOS_ID] + pred_ids], dtype=torch.long, device=xb.device)

                            with torch.enable_grad():
                                logits_vis = model(xb, in_lengths=len_xb, y_inp=y_inp_vis)
                                score = seq_logprob_score(logits_vis, pred_ids)
                                score.backward()

                            cam_vec = cam.cam().detach().cpu().numpy()[0]  # [T']
                            cam.remove()

                            fig_cam = os.path.join(outdir, f"fold{man.cfgs.idx_fold}_idx{sample_idx}_{regime_rt}_gradcam1d.png")
                            x_cpu = xb.detach().cpu()   # ✅ ADD THIS
                            title = (
                                f"Grad-CAM1D fold={man.cfgs.idx_fold} idx={sample_idx} "
                                f"d~={d_tilde_rt:.3f} d={lev_rt}\n"
                                f"pred={pred_str} | gt={lab_str}"
                            )
                            save_signal_plus_cam(x_cpu, cam_vec, fig_cam, title, valid_len=int(len_xb.item()))

                        # Log an index row (append to file)
                        # index file
                        index_path = os.path.join(outdir, "partB_fig_index.csv")
                        row = {
                            "fold": int(man.cfgs.idx_fold),
                            "sample_index": int(sample_idx),
                            "regime_runtime": regime_rt,
                            "regime_csv": regime_csv,
                            "lev_runtime": int(lev_rt),
                            "d_tilde_runtime": float(d_tilde_rt),
                            "lev_csv": float(meta.get("lev", np.nan)),
                            "d_tilde_csv": float(meta.get("d_norm", np.nan)),
                            "pred": pred_str,
                            "gt": lab_str,
                            "attn_fig": fig_attn or "",
                            "gradcam_fig": fig_cam or "",
                        }
                        # append CSV safely
                        if not os.path.exists(index_path):
                            pd.DataFrame([row]).to_csv(index_path, index=False)
                        else:
                            pd.DataFrame([row]).to_csv(index_path, mode="a", header=False, index=False)

       
    # ✅ ADD THIS HERE (after the loop, after preds/labels are fully built)
    if sel_map:
        logger.info("Eval preds count = {}", len(preds))
        logger.info(
            "Selected indices min/max = {}/{}",
            min(sel_map.keys()),
            max(sel_map.keys()),
        )

    man.summarize_epoch()

    # Always export full validation predictions/labels (independent of RunManager)
    
    export_dir = os.path.join(man.cfgs.dir_work, "exports")
    os.makedirs(export_dir, exist_ok=True)

    tag = f"fold{man.cfgs.idx_fold}_epoch{epoch if epoch is not None else 0}"
    export_path = os.path.join(export_dir, f"val_full_{tag}.json")

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": preds, "labels": labels}, f, ensure_ascii=False)

    logger.info("Exported full validation predictions to {}", export_path)

    # Evaluation and visualization
    if man.check_step(epoch + 1, 'eval'):
        if not isinstance(fn_loss, nn.CrossEntropyLoss):
            visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch)
            results_eval = evaluate(preds, labels)
            man.update_evaluation(results_eval, preds[:20], labels[:20])
        else:
            # AR mode: optionally log perplexity or average CE instead
            # >>> ADD THIS: evaluate AR predictions <<<
            results_eval = evaluate(preds, labels)
            man.update_evaluation(results_eval, preds[:20], labels[:20])

            # Export FULL validation predictions for downstream CSV creation
            export_dir = os.path.join(man.cfgs.dir_work, "exports")
            os.makedirs(export_dir, exist_ok=True)

            epoch_tag = "best" if epoch is None else f"epoch{epoch}"
            export_path = os.path.join(
                export_dir,
                f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}.json"
            )

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"predictions": preds, "labels": labels},
                    f,
                    ensure_ascii=False
                )

def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs (argparse.Namespace): Configurations.
    '''

    # 1) Training Regime
    AR_MODE = cfgs.arch_de in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
    LM_MODE = cfgs.arch_de in {"byt5_small", "t5-small"}  # Add LM_MODE flag

    # 2) Tokenizer setup (BPE optional) — define tok BEFORE using it
    tok = None
    if getattr(cfgs, 'use_bpe', False):
        cfgs.tokenizer_model_path = cfgs.tokenizer['model']  # ✅ string is serializable


    # 3) Compute vocab + special IDs ONCE (no duplicate overrides below!)
    if tok is not None:
        vocab_dec = tok.vocab_size           # e.g., 100
        PAD_ID, BOS_ID, EOS_ID = tok.PAD, tok.BOS, tok.EOS
    else:
        # char mode: categories + 3 specials
        base = len(cfgs.categories)
        PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
        vocab_dec = base + (3 if AR_MODE else 0)

    # 4) Attach to cfgs BEFORE constructing RunManager / datasets
    cfgs.AR_MODE = AR_MODE
    cfgs.PAD_ID, cfgs.BOS_ID, cfgs.EOS_ID = PAD_ID, BOS_ID, EOS_ID
    cfgs.vocab_dec = vocab_dec
    cfgs.tokenizer_obj = tok  # now tok definitely exists


    # 5) Proceed as usual
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    LM_MODE = cfgs.arch_de in {"byt5_small", "t5-small"}

    if LM_MODE:
        encoder = build_encoder(cfgs.num_channel, cfgs.arch_en, cfgs.len_seq).to(cfgs.device)
        ratio_ds = int(getattr(encoder, "ratio_ds", 1))

        lm_cfg = LMConfig(
            name=getattr(cfgs, "lm_name", "google/byt5-small"),
            train_lm=getattr(cfgs, "lm_train_lm", False),
            max_new_tokens=int(getattr(cfgs, "lm_max_new_tokens", 128)),
            num_beams=int(getattr(cfgs, "lm_num_beams", 1)),
            length_penalty=float(getattr(cfgs, "lm_length_penalty", 1.0)),
            min_new_tokens=int(getattr(cfgs, "lm_min_new_tokens", 0)),
        )

        # Recommended: d_cnn=0 => projection uses LazyLinear and auto-infers (e.g., 512)
        d_cnn = int(getattr(cfgs, "d_cnn", 0))

        model = MultimodalLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_cfg=lm_cfg,
            proj_dropout=float(getattr(cfgs, "lm_proj_dropout", 0.0)),
            freeze_encoder=bool(getattr(cfgs, "freeze", True)),
        ).to(cfgs.device)


    else:
        # Original BaseModel build for AR/CTC modes
        model = BaseModel(
            cfgs.arch_en,
            cfgs.arch_de,
            cfgs.num_channel,
            cfgs.vocab_dec,      # use vocab_dec from cfgs
            cfgs.len_seq,
            use_gated_attention=getattr(cfgs, "use_gated_attention", False),    # ✅ pass use_gated_attention from cfgs
            gating_type=getattr(cfgs, "gating_type", "elementwise"),            # ✅ pass gating_type from cfgs
        ).to(cfgs.device)

    # Datasets: hand tokenizer to datasets
    dataset_test = HRDataset(
    os.path.join(cfgs.dir_dataset, 'val.json'),
    cfgs.categories,
    model.ratio_ds,
    cfgs.idx_fold,
    cfgs.len_seq,
    cache=cfgs.cache,
    )

    collate_test = fn_collate
    if LM_MODE:
        hf_tok = model.lm.tokenizer
        collate_test = lambda batch: lm_collate(
            batch,
            base_collate_fn=fn_collate,
            hf_tokenizer=hf_tok,
            categories=cfgs.categories,
            pad_id=cfgs.PAD_ID,
            # add a real max length to silence truncation warnings + stabilize memory
            max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
        )

        dataloader_test = DataLoader(
            dataset_test,
            cfgs.size_batch,
            num_workers=cfgs.num_worker,
            collate_fn=collate_test,
        )

    else:
        dataloader_test = DataLoader(
            dataset_test, cfgs.size_batch, num_workers=cfgs.num_worker, collate_fn=fn_collate
        )

    fn_loss = (nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
               if AR_MODE else CTCLoss())

    epoch_start = 0

    if not cfgs.test:
        # --- base train dataset (non-concatenated) ---
        base_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.categories,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
        )
        base_train.tokenizer = tok  # None for char mode is fine

        # --- optionally wrap with concatenation (NO SEPARATOR) ---
        concat_cfg = getattr(cfgs, "concat", {}) or {}
        if concat_cfg.get("enabled", False):
            dataset_train = ConcatWordDataset(
                base_ds=base_train,
                items_min=int(concat_cfg.get("items_min", 2)),   # e.g., 2 to mix singles
                items_max=int(concat_cfg.get("items_max", 4)),   # up to 4 items per sample
                max_T=int(concat_cfg.get("max_T", 4096)),        # word: 1024*4 ; sent: 4096*4
                use_separator=False,                             # <-- no separator for this experiment
                sep_id=None,                                     # <-- must be None
                pad_id=cfgs.PAD_ID,
            )
            collate_train = lambda batch: concat_collate(
                batch, pad_value_x=0.0, pad_value_y=cfgs.PAD_ID
            )
            train_batch_size = int(concat_cfg.get("batch_size", cfgs.size_batch))
        else:
            dataset_train = base_train
            collate_train = fn_collate
            train_batch_size = cfgs.size_batch

        # Wrap collate for LM mode
        if LM_MODE:
            base_collate = collate_train
            hf_tok = model.lm.tokenizer  # from PretrainedLMDecoder
            collate_train = lambda batch: lm_collate(
                batch,
                base_collate_fn=base_collate,
                hf_tokenizer=hf_tok,
                categories=cfgs.categories,
                pad_id=cfgs.PAD_ID,
                max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
            )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=cfgs.num_worker,
            collate_fn=collate_train,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )


        # Log dataloader class information (base or concatenated)
        cc = getattr(cfgs, "concat", {}) or {}
        logger.info(
            "Train dataset: {} | concat.enabled={} | items_min={} | items_max={} | max_T={}",
            dataset_train.__class__.__name__,
            cc.get("enabled"),
            cc.get("items_min"),
            cc.get("items_max"),
            cc.get("max_T"),
        )

        # Probe the very first batch safely
        try:
            _x, _y, _len_x, _len_y = next(iter(dataloader_train))
            
            def _safe_tolist(x):
                return x.tolist() if hasattr(x, "tolist") else list(x)

            # basic shapes + first few lengths
            logger.info(
                "[ConcatProbe] x={} y={} | len_x[:8]={} | len_y[:8]={}",
                tuple(_x.shape),
                tuple(_y.shape),
                _safe_tolist(_len_x[:8]),
                _safe_tolist(_len_y[:8]),
            )

            # summary stats
            _lx = _len_x.cpu().numpy()
            _ly = _len_y.cpu().numpy()
            logger.info(
                "[ConcatProbe] len_x mean/median/max = {:.1f}/{:.1f}/{} ; len_y mean/median/max = {:.1f}/{:.1f}/{}",
                _lx.mean(), np.median(_lx), int(_lx.max()),
                _ly.mean(), np.median(_ly), int(_ly.max()),
            )
        except StopIteration:
            logger.warning("[ConcatProbe] dataloader_train yielded no batch.")
        except Exception as e:
            logger.warning("[ConcatProbe] skipped due to error: {}", e)

        # If concatenation is active, also log a representative base sample
        try:
            sample = base_train[0]
            if isinstance(sample, (list, tuple)) and len(sample) == 4:
                bx, by, blx, bly = sample
                blx, bly = int(blx), int(bly)
            else:
                bx, by = sample
                blx = max(bx.shape)  # infer time length
                bly = len(by) if hasattr(by, "__len__") else int(by.shape[0])
            logger.info("[ConcatProbe] base sample lengths (inferred): len_x={} len_y={}", blx, bly)
        except Exception as e:
            logger.warning("[ConcatProbe] could not fetch base sample: {}", e)

    # ---- Discriminative LR for full multimodal fine-tuning ----
    if LM_MODE:
        def _iter_schedulers(sched):
            # SequentialLR stores underlying schedulers in `_schedulers`.
            if hasattr(sched, "_schedulers"):
                return list(getattr(sched, "_schedulers"))
            return [sched]

        def _find_group_idx_by_name(opt: torch.optim.Optimizer, name: str) -> int | None:
            for i, g in enumerate(opt.param_groups):
                if g.get("name") == name:
                    return i
            return None

        def _log_lm_state(tag: str, opt: torch.optim.Optimizer) -> None:
            # Count trainable params on decoder side
            hf = model.lm.lm
            trainable = 0
            total = 0
            for n, p in hf.named_parameters():
                if n.startswith("decoder.") or n.startswith("lm_head") or n.startswith("shared"):
                    total += p.numel()
                    if p.requires_grad:
                        trainable += p.numel()
            idx = _find_group_idx_by_name(opt, "lm_dec")
            lr = opt.param_groups[idx]["lr"] if idx is not None else None
            logger.info("[LM][{}] lm_dec trainable={}/{} params | opt_lr={}", tag, trainable, total, lr)

        def _set_group_lr_and_base(opt: torch.optim.Optimizer, sched, idx: int, new_lr: float) -> None:
            opt.param_groups[idx]["lr"] = float(new_lr)
            # Make the change stick across future scheduler.step() calls.
            for s in _iter_schedulers(sched):
                if hasattr(s, "base_lrs") and idx < len(s.base_lrs):
                    s.base_lrs[idx] = float(new_lr)

        # Suggested defaults if not present in YAML
        lr_enc  = float(getattr(cfgs, "lr_enc",  1e-4))
        lr_proj = float(getattr(cfgs, "lr_proj", 1e-4))
        lr_lm   = float(getattr(cfgs, "lr_lm",   1e-5))   # ~10x smaller than encoder
        wd      = float(getattr(cfgs, "weight_decay", 0.01))

        param_groups = []

        # 1) CNN encoder
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        if len(enc_params) > 0:
            param_groups.append({"name": "enc", "params": enc_params, "lr": lr_enc})

        # 2) Projection / adapter
        proj_params = [p for p in model.proj.parameters() if p.requires_grad]
        if len(proj_params) > 0:
            param_groups.append({"name": "proj", "params": proj_params, "lr": lr_proj})

        # 3) LM (prefer: decoder + lm_head/shared only)
        hf = model.lm.lm  # HuggingFace T5ForConditionalGeneration  (PretrainedLMDecoder.lm) :contentReference[oaicite:7]{index=7}
        dec_params = []
        for name, p in hf.named_parameters():
            if name.startswith("decoder.") or name.startswith("lm_head") or name.startswith("shared"):
                dec_params.append(p)

        if len(dec_params) > 0:
            param_groups.append({"name": "lm_dec", "params": dec_params, "lr": lr_lm})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        scaler = GradScaler(enabled=bool(getattr(cfgs, "lm_use_amp", False)))

        logger.info(
            "[OptGroups] enc={} proj={} lm_dec={} | lr_enc={} lr_proj={} lr_lm={}",
            sum(p.numel() for p in enc_params),
            sum(p.numel() for p in proj_params),
            sum(p.numel() for p in dec_params),
            lr_enc, lr_proj, lr_lm,
        )
    else:
    # -----------------------------------------------------------

        optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
        scaler = GradScaler()

    # Create scheduler only if training (not test mode)
    if not cfgs.test:
        lr_scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                    CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
                ],
                [len(dataloader_train) * cfgs.epoch_warmup],
            )
        
        # LM mode sanity log right after optimizer/scheduler construction
        if LM_MODE:
            _log_lm_state("init", optimizer)
    else:
        # Test mode: dummy scheduler (never used)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)


    # load checkpoint if given
    if cfgs.checkpoint:
        map_loc = torch.device("cpu") if str(cfgs.device) == "cpu" or not torch.cuda.is_available() else None
        ckp = torch.load(cfgs.checkpoint, map_location=map_loc, weights_only=False)
        #ckp = torch.load(cfgs.checkpoint, weights_only=False)
        res = model.load_state_dict(ckp["model"], strict=False)
        logger.warning("load_state_dict strict=False | missing={} unexpected={}",
               res.missing_keys, res.unexpected_keys)

        if not cfgs.test:
            if 'epoch' in ckp.keys():  # resume
                epoch_start = ckp['epoch'] + 1
                optimizer.load_state_dict(ckp['optimizer'])
                lr_scheduler.load_state_dict(ckp['lr_scheduler'])
            elif cfgs.freeze:  # freeze
                for params in model.encoder.parameters():
                    params.requires_grad = False
            else:  # finetune
                optimizer = torch.optim.AdamW(
                    [
                        {
                            'params': model.encoder.parameters(),
                            'lr': cfgs.lr * 0.1,
                        },
                        {
                            'params': model.decoder.parameters(),
                            'lr': cfgs.lr,
                        },
                    ]
                )
                lr_scheduler = SequentialLR(
                    optimizer,
                    [
                        LinearLR(
                            optimizer,
                            0.01,
                            total_iters=len(dataloader_train)
                            * cfgs.epoch_warmup,
                        ),
                        CosineAnnealingLR(
                            optimizer,
                            len(dataloader_train)
                            * (cfgs.epoch - cfgs.epoch_warmup),
                        ),
                    ],
                    [len(dataloader_train) * cfgs.epoch_warmup],
                )

        logger.info(f'Load checkpoint from {cfgs.checkpoint}')

    # --- Staged LM fine-tuning: optionally unfreeze decoder at epoch N ---
    # If you resume from a checkpoint that is already past lm_unfreeze_epoch,
    # ensure requires_grad is set correctly before training continues.
    if LM_MODE and (not cfgs.test):
        lm_unfreeze_epoch = getattr(cfgs, "lm_unfreeze_epoch", None)
        if lm_unfreeze_epoch is not None:
            lm_unfreeze_epoch = int(lm_unfreeze_epoch)
            if lm_unfreeze_epoch >= 0 and epoch_start >= lm_unfreeze_epoch:
                model.lm.set_decoder_trainable(True)
                logger.info("[LM] Resumed at epoch {} => LM decoder is UNFROZEN (lm_unfreeze_epoch={})", epoch_start, lm_unfreeze_epoch)

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            # Build qualitative selection map only if enabled
            qual_cfg = None
            if getattr(cfgs, "qualitative", False):
                task = getattr(cfgs, "qual_task", "word")

                by_fold = getattr(cfgs, "qual_indices_by_fold", None)
                indices = None
                if by_fold is not None:
                    indices = by_fold.get(int(cfgs.idx_fold), [])

                if indices:
                    selection_map = load_partB_selection_by_indices(
                        unified_csv_path=cfgs.qual_csv,
                        fold=int(cfgs.idx_fold),
                        task_name=task,
                        indices=[int(i) for i in indices],
                    )
                else:
                    # fallback to your quantile-based selection if you want it
                    selection_map = load_partB_selection_unified_quantile(
                        unified_csv_path=cfgs.qual_csv,
                        fold=int(cfgs.idx_fold),
                        task_name=task,
                        n_correct=int(cfgs.qual_n_correct),
                        n_nearmiss=int(cfgs.qual_n_nearmiss),
                        n_catastrophic=int(cfgs.qual_n_catastrophic),
                        seed=int(cfgs.qual_seed),
                    )




                qual_cfg = {
                    "enabled": True,
                    "selection_map": selection_map,
                    "outdir": getattr(cfgs, "qual_outdir", "qual_partB"),
                    "use_gradcam": getattr(cfgs, "qual_use_gradcam", False),
                    "gradcam_layer": getattr(cfgs, "qual_gradcam_layer", "layers.11.pwconv"),
                }

            # Route to correct evaluation function based on model type
            if LM_MODE:
                test_lm(dataloader_test, model, manager, 0)
            else:
                test(
                    dataloader_test,
                    model,
                    fn_loss,
                    manager,
                    ctc_decoder,
                    0,                     # epoch=0 for consistency
                    tokenizer=tok,
                    force_eval=True,       # ✅ always evaluate
                    qual_cfg=qual_cfg,     # ✅ enable Part B capture
                )
            manager.summarize_evaluation()
            break

        else:
            if LM_MODE and (not cfgs.test):
                lm_unfreeze_epoch = getattr(cfgs, "lm_unfreeze_epoch", None)
                if lm_unfreeze_epoch is not None and int(lm_unfreeze_epoch) >= 0 and int(e) == int(lm_unfreeze_epoch):
                    model.lm.set_decoder_trainable(True)
                    # Step (1): verify/print that decoder params are actually trainable
                    _log_lm_state(f"unfreeze@{e}", optimizer)

                    # Step (2): high-ROI approach = LR bump (scheduler-aware) rather than restarting full schedule.
                    # This avoids perturbing encoder/proj LR while making LM updates non-trivial late in training.
                    # Optional: only bumps LR if you set lm_lr_unfreeze_mult != 1.0 in YAML
                    mult = float(getattr(cfgs, "lm_lr_unfreeze_mult", 1.0))
                    idx = _find_group_idx_by_name(optimizer, "lm_dec")
                    if idx is not None and mult > 0 and mult != 1.0:
                        old_lr = float(optimizer.param_groups[idx]["lr"])
                        new_lr = old_lr * mult
                        _set_group_lr_and_base(optimizer, lr_scheduler, idx, new_lr)
                        logger.info(
                            "[LM] Unfreeze LR bump applied: lm_dec lr {} -> {} (mult={}, epoch={})",
                            old_lr, new_lr, mult, e,
                        )
                    elif idx is not None and mult == 1.0:
                        logger.info("[LM] Unfreeze LR bump disabled (lm_lr_unfreeze_mult=1.0).")
                    else:
                        logger.warning("[LM] Could not apply LR bump (lm_dec group missing or mult<=0).")

                    logger.info("[LM] Unfroze LM decoder at epoch {} (lm_unfreeze_epoch={})", e, int(lm_unfreeze_epoch))

            if LM_MODE:
                train_one_epoch_lm(  # Use new LM training function
                    dataloader_train,
                    model,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    manager,
                    e,
                )
            else:
                train_one_epoch(  # Keep existing for AR/CTC
                    dataloader_train,
                    model,
                    fn_loss,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    manager,
                    e,
                )
            if LM_MODE:
                test_lm(dataloader_test, model, manager, e)
            else:
                test(dataloader_test, model, fn_loss, manager, ctc_decoder, e, tokenizer=tok)

            # Save only best checkpoints (CER/WER) if enabled.
            if bool(getattr(cfgs, "save_best_only", False)):
                best = manager.results.get("best", {})
                if isinstance(best, dict):
                    if "character_error_rate" in best and int(best["character_error_rate"][0]) == int(e):
                        manager.save_checkpoint(
                            model.state_dict(),
                            optimizer.state_dict(),
                            lr_scheduler.state_dict(),
                            filename="best_cer.pth",
                        )
                    if "word_error_rate" in best and int(best["word_error_rate"][0]) == int(e):
                        manager.save_checkpoint(
                            model.state_dict(),
                            optimizer.state_dict(),
                            lr_scheduler.state_dict(),
                            filename="best_wer.pth",
                        )


    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()
    # args.config = 'configs/train.yaml'

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
