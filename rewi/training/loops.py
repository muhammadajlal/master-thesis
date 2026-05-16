"""
Training and evaluation loop implementations.
"""

import contextlib
import json
import os
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from rewi.analysis import (
    CrossAttnCatcher,
    GradCAM1D,
    attn_to_matrix,
    compute_fold_thresholds,
    lev_dist,
    save_attn_heatmap,
    save_signal_plus_cam,
    seq_logprob_score,
)
from rewi.ctc_decoder import BestPath
from rewi.evaluate import evaluate
from rewi.manager import RunManager
from rewi.model import BaseModel
from rewi.model.dual_head import DualHeadModel
from rewi.training.utils import build_ar_batch, maybe_log_trainability
from rewi.visualize import visualize

# Optional pandas import (only needed for qualitative analysis)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _build_bigram_lookup(
    dataloader: DataLoader,
    vocab_size: int,
    spec_ids: tuple,
    *,
    device,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Build P(Y | X) where (X, Y) are adjacent characters in the training labels.

    For each ordered character pair (X, Y) such that Y immediately follows X in
    any training label, increment counts[X, Y]. Apply add-`smoothing` Laplace
    smoothing then row-normalize so each row sums to 1. Special tokens
    (PAD, BOS, EOS) and CTC-blank index 0 are excluded from both X and Y so
    that corruption never selects or is conditioned on them.

    Returns
    -------
    Tensor of shape (vocab_size, vocab_size) on `device`.
    """
    counts = torch.full((vocab_size, vocab_size), float(smoothing), dtype=torch.float, device=device)
    spec = set(int(s) for s in spec_ids)
    # Mass set to 0 for special-token destinations so they cannot be sampled.
    spec_idx = torch.tensor(sorted(spec), dtype=torch.long, device=device)
    for batch in dataloader:
        # Training batches are (x, y, len_x, len_y).
        y = batch[1]
        len_y = batch[3]
        if y.dim() == 2:
            for b in range(y.size(0)):
                L = int(len_y[b])
                seq = y[b, :L].tolist()
                for i in range(len(seq) - 1):
                    a, c = int(seq[i]), int(seq[i + 1])
                    if a in spec or c in spec:
                        continue
                    counts[a, c] += 1.0
        else:
            offset = 0
            for b in range(int(len_y.numel())):
                L = int(len_y[b])
                seq = y[offset:offset + L].tolist()
                offset += L
                for i in range(len(seq) - 1):
                    a, c = int(seq[i]), int(seq[i + 1])
                    if a in spec or c in spec:
                        continue
                    counts[a, c] += 1.0
    counts[:, spec_idx] = 0.0
    counts = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return counts


def _load_confusion_lookup(
    path_template: str,
    *,
    fold: int,
    vocab_size: int,
    spec_ids: tuple,
    device,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Load a per-fold confusion matrix saved as a `.npy` file.

    Path may contain `{fold}` which is substituted with the current fold index.
    The file is expected to hold a (vocab_size, vocab_size) integer-or-float
    matrix where row X column Y is the count of "true X mispredicted as Y".
    The matrix is Laplace-smoothed and row-normalized; mass on special tokens
    (PAD, BOS, EOS, CTC-blank 0) is zeroed before normalisation.
    """
    import numpy as np
    path = path_template.format(fold=fold)
    raw = np.load(path)
    if raw.shape != (vocab_size, vocab_size):
        raise ValueError(
            f"confusion matrix at {path} has shape {raw.shape}, "
            f"expected ({vocab_size}, {vocab_size})"
        )
    counts = torch.tensor(raw, dtype=torch.float, device=device) + float(smoothing)
    spec_idx = torch.tensor(sorted({int(s) for s in spec_ids}), dtype=torch.long, device=device)
    counts[:, spec_idx] = 0.0
    counts = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return counts


def train_one_epoch_lm(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    man: RunManager,
    epoch: int,
) -> None:
    """
    Train multimodal LM model for one epoch.
    
    Supports gradient accumulation via ``cfgs.grad_accum_steps`` (default 1).
    When > 1, gradients are accumulated over that many mini-batches before
    an optimiser step, simulating a larger effective batch size.
    
    Args:
        dataloader: Training dataloader yielding (x, len_x, labels, texts).
        model: MultimodalLMModel instance.
        optimizer: Optimizer for model parameters.
        scaler: GradScaler for mixed precision training.
        lr_scheduler: Learning rate scheduler (stepped per iteration).
        man: RunManager for logging and checkpointing.
        epoch: Current epoch number.
    """
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    use_amp = bool(getattr(man.cfgs, "lm_use_amp", False))
    _amp_str = str(getattr(man.cfgs, "lm_amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if "bf" in _amp_str else torch.float16
    accum_steps = int(getattr(man.cfgs, "grad_accum_steps", 1))

    # Two-step refinement config
    vlm_cfg = getattr(man.cfgs, "vlm", {}) or {}
    refine_lambda = float(vlm_cfg.get("refine_lambda", 0.5))
    refine_corrupt = float(vlm_cfg.get("refine_corrupt_prob", 0.3))
    do_refine = getattr(model, "two_step_decode", False)

    optimizer.zero_grad(set_to_none=True)

    for idx, (x, len_x, labels, _texts) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels = labels.to(man.cfgs.device)

        # Skip degenerate batches
        if labels.numel() == 0 or (labels != -100).sum().item() == 0:
            logger.warning(
                "All labels are -100 (ignored). Skipping batch. epoch={} iter={}",
                epoch, idx,
            )
            continue

        # Forward with AMP
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(x, len_x, labels=labels)
            # Handle dict return from VLM with two_step_decode
            if isinstance(out, dict):
                loss_lm = out["lm_out"].loss
                imu_tokens = out["imu_tokens"]
            else:
                loss_lm = out.loss
                imu_tokens = None

            loss = loss_lm

            # Two-step refinement loss
            if do_refine and imu_tokens is not None:
                loss_refine = model.forward_refine(
                    imu_tokens, labels, _texts,
                    corrupt_prob=refine_corrupt,
                )
                loss = loss + refine_lambda * loss_refine

            loss = loss / accum_steps  # scale loss for accumulation

        # Skip non-finite losses
        if not torch.isfinite(loss):
            logger.warning(
                "Non-finite loss. epoch={} iter={} lr={} loss={}",
                epoch, idx, lr_scheduler.get_last_lr()[0], loss * accum_steps
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward (accumulate gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step only every accum_steps iterations (or at end of epoch)
        is_accum_step = ((idx + 1) % accum_steps == 0) or (idx + 1 == len(dataloader))
        if is_accum_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if not torch.isfinite(grad_norm):
                    logger.warning(
                        "Non-finite grad norm. epoch={} iter={} grad_norm={}",
                        epoch, idx, grad_norm
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        man.update_iteration(idx, float(loss.item() * accum_steps), lr_scheduler.get_last_lr()[0])

    man.summarize_epoch()
    
    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())


@torch.no_grad()
def test_lm(
    dataloader: DataLoader,
    model: nn.Module,
    man: RunManager,
    epoch: int,
) -> None:
    """
    Evaluate multimodal LM model.
    
    Args:
        dataloader: Test dataloader yielding (x, len_x, labels, texts).
        model: MultimodalLMModel instance.
        man: RunManager for logging.
        epoch: Current epoch number.
    """
    model.eval()
    man.initialize_epoch(epoch, len(dataloader), True)

    preds, labels_txt = [], []

    for idx, (x, len_x, labels_hf, texts) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels_hf = labels_hf.to(man.cfgs.device)

        out = model(x, len_x, labels=labels_hf)
        loss = float(out.loss.detach().cpu())
        man.update_iteration(idx, loss, lr=0.0)

        hyp = model.generate(x, len_x)
        preds.extend(hyp)
        labels_txt.extend(list(texts))

    man.summarize_epoch()

    export_val_full = bool(getattr(man.cfgs, "export_val_full", False))
    is_test_mode = bool(getattr(man.cfgs, "test", False))
    do_export = is_test_mode or export_val_full

    if do_export:
        export_dir = os.path.join(man.cfgs.dir_work, "exports")
        os.makedirs(export_dir, exist_ok=True)
        epoch_tag = "best" if epoch is None else f"epoch{epoch}"
        export_path = os.path.join(
            export_dir,
            f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}.json",
        )
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds, "labels": labels_txt}, f, ensure_ascii=False)
        logger.info("Exported full LM validation predictions to {}", export_path)

    if man.check_step(epoch + 1, 'eval'):
        results_eval = evaluate(preds, labels_txt)
        man.update_evaluation(results_eval, preds[:20], labels_txt[:20])


def train_one_epoch_lm_hybrid(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss_ctc: nn.Module,
    lambda_ctc: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    man: RunManager,
    epoch: int,
) -> None:
    """Train multimodal LM model with hybrid CTC+LM loss for one epoch.

    The model must be a ``MultimodalLMModel`` with ``hybrid_mode=True``.
    Dataloader must yield ``(x, len_x, labels, texts, y, len_y)``
    where y/len_y are character-level targets for CTC.

    If the underlying VLM has ``two_step_decode`` enabled, an additional
    refinement loss is computed per batch and added to the total loss,
    weighted by ``vlm.refine_lambda`` (default 0.5).
    """
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    use_amp = bool(getattr(man.cfgs, "lm_use_amp", False))
    _amp_str = str(getattr(man.cfgs, "lm_amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if "bf" in _amp_str else torch.float16
    accum_steps = int(getattr(man.cfgs, "grad_accum_steps", 1))

    # Two-step refinement config
    vlm_cfg = getattr(man.cfgs, "vlm", {}) or {}
    refine_lambda = float(vlm_cfg.get("refine_lambda", 0.5))
    refine_corrupt = float(vlm_cfg.get("refine_corrupt_prob", 0.3))
    # Access the inner VLM model (may be wrapped in DualHeadModel)
    inner_model = getattr(model, "model", model)
    do_refine = getattr(inner_model, "two_step_decode", False)

    # ── Auxiliary losses (initialize once, reuse across epochs) ──
    # We store loss modules on inner_model to avoid re-creating them every epoch.
    _aux_initialized = getattr(inner_model, "_aux_losses_initialized", False)

    # Contrastive alignment loss (BC-style from ECHWR)
    lambda_contrast = float(vlm_cfg.get("lambda_contrast", 0.0))
    do_contrast = lambda_contrast > 0
    contrast_loss_fn = getattr(inner_model, "_contrast_loss_fn", None)
    if do_contrast and not _aux_initialized:
        from rewi.training.contrastive import InBatchContrastiveLoss
        init_temp = float(vlm_cfg.get("contrast_temperature", 0.07))
        contrast_loss_fn = InBatchContrastiveLoss(init_temperature=init_temp).to(man.cfgs.device)
        inner_model._return_text_emb = True
        inner_model._contrast_loss_fn = contrast_loss_fn
        optimizer.add_param_group({"params": contrast_loss_fn.parameters(), "lr": optimizer.defaults["lr"]})
        logger.info("[Contrastive] Enabled: lambda={}, temp_init={}", lambda_contrast, init_temp)

    # K1: CTC Compression + Per-Token MSE Distillation
    lambda_embed_mse = float(vlm_cfg.get("lambda_embed_mse", 0.0))
    do_embed_mse = lambda_embed_mse > 0
    embed_mse_fn = getattr(inner_model, "_embed_mse_fn", None)
    if do_embed_mse and not _aux_initialized:
        from rewi.training.auxiliary_losses import CTCCompressMSELoss
        d_enc = int(getattr(man.cfgs, "d_cnn", 512))
        d_lm = inner_model.d_lm if hasattr(inner_model, "d_lm") else 768
        embed_mse_fn = CTCCompressMSELoss(d_enc, d_lm).to(man.cfgs.device)
        inner_model._return_text_emb = True
        inner_model._embed_mse_fn = embed_mse_fn
        optimizer.add_param_group({"params": embed_mse_fn.parameters(), "lr": optimizer.defaults["lr"]})
        logger.info("[K1 Embed MSE] Enabled: lambda={}", lambda_embed_mse)

    # K3: ECHWR Error-Based Contrastive (EC) Loss
    lambda_ec = float(vlm_cfg.get("lambda_ec", 0.0))
    do_ec = lambda_ec > 0
    ec_loss_fn = getattr(inner_model, "_ec_loss_fn", None)
    ec_char_to_id = getattr(inner_model, "_ec_char_to_id", None)
    if do_ec and not _aux_initialized:
        from rewi.training.auxiliary_losses import ECContrastiveLoss
        categories = getattr(man.cfgs, "categories", [])
        ec_char_to_id = {c: i for i, c in enumerate(categories)}
        vocab_size = len(categories)
        d_lm = inner_model.d_lm if hasattr(inner_model, "d_lm") else 768
        n_neg = int(vlm_cfg.get("ec_n_negatives", 4))
        ec_loss_fn = ECContrastiveLoss(
            vocab_size=vocab_size, d_lm=d_lm, n_negatives=n_neg,
        ).to(man.cfgs.device)
        inner_model._ec_loss_fn = ec_loss_fn
        inner_model._ec_char_to_id = ec_char_to_id
        optimizer.add_param_group({"params": ec_loss_fn.parameters(), "lr": optimizer.defaults["lr"]})
        logger.info("[K3 EC Loss] Enabled: lambda={}, n_neg={}, vocab={}", lambda_ec, n_neg, vocab_size)

    # K4: SEA Token-Level Contrastive Loss
    lambda_sea = float(vlm_cfg.get("lambda_sea", 0.0))
    do_sea = lambda_sea > 0
    sea_loss_fn = getattr(inner_model, "_sea_loss_fn", None)
    if do_sea and not _aux_initialized:
        from rewi.training.auxiliary_losses import SEATokenContrastiveLoss
        sea_loss_fn = SEATokenContrastiveLoss(temperature=0.07).to(man.cfgs.device)
        inner_model._return_text_emb = True
        inner_model._sea_loss_fn = sea_loss_fn
        optimizer.add_param_group({"params": sea_loss_fn.parameters(), "lr": optimizer.defaults["lr"]})
        logger.info("[K4 SEA] Enabled: lambda={}", lambda_sea)

    # Enable text embedding return if any loss needs it
    if do_embed_mse or do_sea:
        inner_model._return_text_emb = True

    # Mark as initialized so we don't re-add param groups next epoch
    if not _aux_initialized:
        inner_model._aux_losses_initialized = True

    optimizer.zero_grad(set_to_none=True)

    for idx, (x, len_x, labels, _texts, y, len_y) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels = labels.to(man.cfgs.device)
        y = y.to(man.cfgs.device)
        len_y = len_y.to(man.cfgs.device)

        if labels.numel() == 0 or (labels != -100).sum().item() == 0:
            logger.warning(
                "All labels are -100 (ignored). Skipping batch. epoch={} iter={}",
                epoch, idx,
            )
            continue

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(x, len_x, labels=labels)
            loss_lm = out["lm_out"].loss

            ctc_logits = out["ctc_logits"].float()
            enc_lengths = out["enc_lengths"]
            loss_ctc = fn_loss_ctc(
                ctc_logits.permute((1, 0, 2)),
                y,
                enc_lengths,
                len_y,
            )

            loss = loss_lm + lambda_ctc * loss_ctc

            # Contrastive alignment loss
            loss_contrast = torch.tensor(0.0, device=x.device)
            if do_contrast and contrast_loss_fn is not None:
                loss_contrast = contrast_loss_fn(
                    out["imu_tokens"],
                    out["text_emb"],
                    labels_mask=out["labels_mask"],
                    texts=_texts,
                )
                loss = loss + lambda_contrast * loss_contrast

            # K1: CTC Compression + Per-Token MSE
            loss_embed_mse = torch.tensor(0.0, device=x.device)
            if do_embed_mse and embed_mse_fn is not None:
                loss_embed_mse = embed_mse_fn(
                    out["enc_states"],
                    ctc_logits,
                    out["text_emb"],
                    out["labels_mask"],
                )
                loss = loss + lambda_embed_mse * loss_embed_mse

            # K3: ECHWR EC Loss
            loss_ec = torch.tensor(0.0, device=x.device)
            if do_ec and ec_loss_fn is not None:
                loss_ec = ec_loss_fn(
                    out["imu_tokens"],
                    _texts,
                    ec_char_to_id,
                )
                loss = loss + lambda_ec * loss_ec

            # K4: SEA Token-Level Contrastive
            loss_sea = torch.tensor(0.0, device=x.device)
            if do_sea and sea_loss_fn is not None:
                loss_sea = sea_loss_fn(
                    out["imu_tokens"],
                    ctc_logits.new_zeros(1),  # enc_states placeholder
                    ctc_logits,
                    out["text_emb"],
                    out["labels_mask"],
                )
                loss = loss + lambda_sea * loss_sea

            # Two-step refinement loss
            loss_refine = torch.tensor(0.0, device=x.device)
            if do_refine:
                imu_tokens = out["imu_tokens"]
                loss_refine = inner_model.forward_refine(
                    imu_tokens, labels, _texts,
                    corrupt_prob=refine_corrupt,
                )
                loss = loss + refine_lambda * loss_refine

            loss = loss / accum_steps

        if not torch.isfinite(loss):
            logger.warning(
                "Non-finite loss. epoch={} iter={} loss_lm={} loss_ctc={}",
                epoch, idx, float(loss_lm.item()), float(loss_ctc.item()),
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_step = ((idx + 1) % accum_steps == 0) or (idx + 1 == len(dataloader))
        if is_accum_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    logger.warning(
                        "Non-finite grad norm. epoch={} iter={} grad_norm={}",
                        epoch, idx, grad_norm,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        iter_extras = dict(
            loss_ar=float(loss_lm.item()),
            loss_ctc=float(loss_ctc.item()),
        )
        if do_refine:
            iter_extras["loss_refine"] = float(loss_refine.item())
        if do_contrast:
            iter_extras["loss_contrast"] = float(loss_contrast.item())
        if do_embed_mse:
            iter_extras["loss_embed_mse"] = float(loss_embed_mse.item())
        if do_ec:
            iter_extras["loss_ec"] = float(loss_ec.item())
        if do_sea:
            iter_extras["loss_sea"] = float(loss_sea.item())

        man.update_iteration(
            idx,
            float(loss.item() * accum_steps),
            lr_scheduler.get_last_lr()[0],
            **iter_extras,
        )

    man.summarize_epoch()

    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())


@torch.no_grad()
def test_lm_hybrid(
    dataloader: DataLoader,
    model: nn.Module,
    fn_loss_ctc: nn.Module,
    lambda_ctc: float,
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: int,
) -> None:
    """Evaluate multimodal LM model with hybrid CTC+LM loss.

    Reports both LM generation metrics and CTC best-path metrics.
    """
    model.eval()
    man.initialize_epoch(epoch, len(dataloader), True)

    preds_lm, labels_lm = [], []
    preds_ctc, labels_ctc = [], []

    for idx, (x, len_x, labels_hf, texts, y, len_y) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        labels_hf = labels_hf.to(man.cfgs.device)
        y = y.to(man.cfgs.device)
        len_y = len_y.to(man.cfgs.device)

        out = model(x, len_x, labels=labels_hf)
        loss_lm = out["lm_out"].loss

        ctc_logits = out["ctc_logits"].float()
        enc_lengths = out["enc_lengths"]
        loss_ctc = fn_loss_ctc(
            ctc_logits.permute((1, 0, 2)),
            y,
            enc_lengths,
            len_y,
        )

        loss = loss_lm + lambda_ctc * loss_ctc
        man.update_iteration(
            idx,
            float(loss.item()),
            lr=0.0,
            loss_ar=float(loss_lm.item()),
            loss_ctc=float(loss_ctc.item()),
        )

        # LM generation predictions
        hyp = model.generate(x, len_x)
        preds_lm.extend(hyp)
        labels_lm.extend(list(texts))

        # CTC best-path predictions
        for logit, Lx, label, Ly in zip(
            ctc_logits.detach().cpu(),
            enc_lengths.detach().cpu(),
            y.detach().cpu(),
            len_y.detach().cpu(),
        ):
            preds_ctc.append(ctc_decoder.decode(logit[: int(Lx)]))
            labels_ctc.append(ctc_decoder.decode(label[: int(Ly)], True))

    man.summarize_epoch()

    export_val_full = bool(getattr(man.cfgs, "export_val_full", False))
    is_test_mode = bool(getattr(man.cfgs, "test", False))
    do_export = is_test_mode or export_val_full

    if do_export:
        export_dir = os.path.join(man.cfgs.dir_work, "exports")
        os.makedirs(export_dir, exist_ok=True)
        epoch_tag = "best" if epoch is None else f"epoch{epoch}"
        export_path = os.path.join(
            export_dir,
            f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}_lm.json",
        )
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds_lm, "labels": labels_lm}, f, ensure_ascii=False)
        logger.info("Exported full LM hybrid validation predictions to {}", export_path)

    if man.check_step(epoch + 1, 'eval'):
        results_lm = evaluate(preds_lm, labels_lm)
        results_ctc = evaluate(preds_ctc, labels_ctc)
        man.update_evaluation(results_lm, preds_lm[:20], labels_lm[:20], key='evaluation', label='LM-generate')
        man.update_evaluation(results_ctc, preds_ctc[:20], labels_ctc[:20], key='evaluation_ctc', label='CTC-bestpath')


def train_one_epoch(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    man: RunManager,
    epoch: int,
) -> None:
    """
    Train CTC/AR model for one epoch.
    
    Args:
        dataloader: Training dataloader yielding (x, y, len_x, len_y).
        model: BaseModel instance.
        fn_loss: Loss function (CTCLoss or CrossEntropyLoss).
        optimizer: Optimizer.
        scaler: GradScaler for mixed precision.
        lr_scheduler: Learning rate scheduler.
        man: RunManager for logging.
        epoch: Current epoch number.
    """
    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    # Keep frozen decoder in eval mode to disable dropout
    if bool(getattr(man.cfgs, "decoder_frozen", False)):
        dec = getattr(model, "decoder", None)
        if dec is not None:
            dec.eval()

    maybe_log_trainability(man, model, epoch=epoch, where="train_one_epoch")

    PAD_ID = man.cfgs.PAD_ID
    BOS_ID = man.cfgs.BOS_ID
    EOS_ID = man.cfgs.EOS_ID

    ds_cfg = getattr(man.cfgs, "deep_supervision", {}) or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_layers_cfg = ds_cfg.get("layers", [1, 2, 3])
    ds_include_final = bool(ds_cfg.get("include_final", True))
    ds_reduce = str(ds_cfg.get("reduce", "mean")).lower()  # mean|sum

    # Decoder-side CTC for AR-only mode (CTC regularization "between decoder layers")
    dec_ctc_cfg = getattr(man.cfgs, "decoder_side_ctc", {}) or {}
    dec_ctc_enabled = bool(dec_ctc_cfg.get("enabled", False))
    dec_ctc_lambda = float(dec_ctc_cfg.get("lambda", 0.0))
    fn_loss_ctc = None
    if dec_ctc_enabled and dec_ctc_lambda > 0.0:
        from ..loss import CTCLoss
        fn_loss_ctc = CTCLoss(blank=0, reduction="mean")

    # Scheduled sampling (Bengio 2015 / Mihaylova-Martins 2019, two-pass for transformers).
    # Replaces each non-BOS decoder input token with the model's own previous-step
    # argmax with probability p_eff, removing teacher forcing on those positions.
    ss_cfg = getattr(man.cfgs, "scheduled_sampling", {}) or {}
    ss_enabled = bool(ss_cfg.get("enabled", False))
    ss_p_start = float(ss_cfg.get("p_start", 0.0))
    ss_p_end = float(ss_cfg.get("p_end", 1.0))
    ss_ramp_epochs = int(ss_cfg.get("ramp_epochs", max(1, int(getattr(man.cfgs, "epoch", 300)) // 2)))
    if ss_enabled and (ds_enabled or (dec_ctc_enabled and dec_ctc_lambda > 0.0)):
        raise ValueError(
            "scheduled_sampling is incompatible with deep_supervision or decoder_side_ctc. "
            "Disable those to run no-teacher-forcing experiments."
        )
    if ss_enabled:
        if ss_ramp_epochs <= 0:
            ss_p_eff = ss_p_end
        else:
            t = min(1.0, max(0.0, float(epoch) / float(ss_ramp_epochs)))
            ss_p_eff = ss_p_start + t * (ss_p_end - ss_p_start)
        man.log(
            f"[ScheduledSampling] epoch={epoch} | p_eff={ss_p_eff:.4f} "
            f"(p_start={ss_p_start} p_end={ss_p_end} ramp_epochs={ss_ramp_epochs})"
        )
    else:
        ss_p_eff = 0.0

    # Input token corruption (Bowman 2016 / Iyyer 2015 word-dropout style):
    # keep teacher forcing on the loss side, but corrupt the decoder input via
    # one of several configurable mechanisms. Five modes:
    #   uniform        — replacement drawn uniformly over real characters
    #   bigram_right   — replacement drawn from P(Y | X) where X is the original
    #                    char and Y is its training-corpus successor
    #   bigram_left    — same table as bigram_right but indexed by the LEFT
    #                    context (label[t-1]) so the corrupted local bigram
    #                    stays inside the language's distribution
    #   self_confusion — replacement drawn from P(Y | X) where the matrix is the
    #                    baseline AR's character-level confusion matrix on the
    #                    training set (per-fold, free-running greedy decode)
    #   adjacent_swap  — instead of a substitution, swap (y_inp[t], y_inp[t+1])
    #                    with probability p_replace; captures transposition-style
    #                    errors common in handwriting
    ic_cfg = getattr(man.cfgs, "input_corruption", {}) or {}
    ic_enabled = bool(ic_cfg.get("enabled", False))
    ic_p = float(ic_cfg.get("p_replace", 0.0))
    ic_mode = str(ic_cfg.get("mode", "uniform")).lower()
    if ic_enabled and ss_enabled:
        raise ValueError(
            "input_corruption and scheduled_sampling cannot both be enabled — "
            "they are alternative no-TF strategies."
        )
    # Replacement vocabulary: indices [1, min(PAD,BOS,EOS)) — real characters only,
    # excluding CTC blank at 0 and the three specials.
    ic_lo = 1
    ic_hi = min(PAD_ID, BOS_ID, EOS_ID)
    if ic_enabled and ic_p > 0.0 and ic_hi <= ic_lo:
        raise ValueError(
            f"input_corruption: cannot determine a valid replacement vocab range "
            f"(lo={ic_lo}, hi={ic_hi}). Check tokenizer special token IDs."
        )

    # Build / load the lookup table for substitution-with-distribution modes
    # (bigram_right, bigram_left, self_confusion). Cached on the manager so we
    # do not rebuild every epoch. uniform and adjacent_swap need no table.
    ic_lookup = None
    _SUB_MODES = ("bigram_right", "bigram_left", "self_confusion")
    if ic_enabled and ic_p > 0.0 and ic_mode in _SUB_MODES:
        # bigram_left and bigram_right share the SAME training-corpus bigram
        # table; only the lookup-key differs at sample time.
        cache_key = "bigram_table" if ic_mode in ("bigram_right", "bigram_left") else ic_mode
        cache_attr = f"_ic_lookup_{cache_key}"
        if hasattr(man, cache_attr):
            ic_lookup = getattr(man, cache_attr)
        else:
            vocab_size = int(model.num_cls) if hasattr(model, "num_cls") else int(getattr(man.cfgs, "num_cls", 0))
            if vocab_size <= 0:
                raise RuntimeError("input_corruption: cannot infer vocab_size from model")
            ic_smoothing = float(ic_cfg.get("smoothing", 1.0))
            spec_ids = (PAD_ID, BOS_ID, EOS_ID, 0)  # also exclude CTC-blank index 0
            if ic_mode in ("bigram_right", "bigram_left"):
                ic_lookup = _build_bigram_lookup(
                    dataloader, vocab_size, spec_ids,
                    device=man.cfgs.device, smoothing=ic_smoothing,
                )
            elif ic_mode == "self_confusion":
                conf_path_template = ic_cfg.get("confusion_path", None)
                if conf_path_template is None:
                    raise ValueError(
                        "input_corruption mode=self_confusion requires "
                        "input_corruption.confusion_path (with {fold} placeholder)"
                    )
                ic_lookup = _load_confusion_lookup(
                    conf_path_template, fold=int(getattr(man.cfgs, "idx_fold", 0)),
                    vocab_size=vocab_size, spec_ids=spec_ids,
                    device=man.cfgs.device, smoothing=ic_smoothing,
                )
            setattr(man, cache_attr, ic_lookup)
    elif ic_enabled and ic_p > 0.0 and ic_mode not in ("uniform", "adjacent_swap") + _SUB_MODES:
        raise ValueError(
            f"input_corruption.mode must be one of "
            f"{{uniform, bigram_right, bigram_left, self_confusion, adjacent_swap}}, "
            f"got: {ic_mode}"
        )

    if ic_enabled and ic_p > 0.0:
        man.log(
            f"[InputCorruption] epoch={epoch} | mode={ic_mode} | p_replace={ic_p:.4f} "
            f"(replacement_vocab=[{ic_lo}, {ic_hi}))"
        )

    # AMP controllable via config (default true for backward compat)
    use_amp = bool(getattr(man.cfgs, "use_amp", True)) and torch.cuda.is_available()

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()

        with torch.autocast('cuda', torch.float16, enabled=use_amp):
            if isinstance(fn_loss, nn.CrossEntropyLoss):  # AR mode
                y_inp, y_tgt = build_ar_batch(y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device)

                # Scheduled sampling: replace decoder input tokens with the model's own
                # previous-step argmax with probability p_eff (positions t>=1 only).
                if ss_p_eff > 0.0:
                    with torch.no_grad():
                        prev_logits = model(x, in_lengths=len_x, y_inp=y_inp)
                        if isinstance(prev_logits, dict):
                            prev_logits = prev_logits["logits"]
                        preds = prev_logits.argmax(dim=-1)  # (B, N)
                    # shift: pred at position t-1 becomes the candidate input at position t
                    shifted = torch.full_like(y_inp, PAD_ID)
                    shifted[:, 1:] = preds[:, :-1]
                    sample_mask = torch.rand_like(y_inp, dtype=torch.float) < ss_p_eff
                    sample_mask[:, 0] = False  # never replace BOS
                    y_inp = torch.where(sample_mask, shifted, y_inp)

                # Input corruption: corrupt a fraction of non-special input tokens.
                # The CE loss is still computed against the unmodified y_tgt. The
                # corruption mechanism is selected by ic_mode.
                # NOTE: never name a local `idx` here — that would shadow the
                # outer enumerate(dataloader) counter and break man.update_iteration.
                if ic_enabled and ic_p > 0.0:
                    real_mask = (y_inp != PAD_ID) & (y_inp != BOS_ID) & (y_inp != EOS_ID)

                    if ic_mode == "adjacent_swap":
                        # With prob p_replace, swap (y_inp[t], y_inp[t+1]).
                        # Skip positions where t or t+1 is special. Conflicts
                        # (consecutive Trues) are tolerated; at p=0.15 they are
                        # rare (<2% of positions) and produce slightly more
                        # chaotic local noise rather than incorrect behaviour.
                        real_next = real_mask.roll(-1, dims=-1)
                        real_next[:, -1] = False
                        swap_cand = (
                            (torch.rand_like(y_inp, dtype=torch.float) < ic_p)
                            & real_mask & real_next
                        )
                        # Position t gets old[t+1] when swap_cand[t] is True;
                        # position t gets old[t-1] when swap_cand[t-1] is True.
                        y_next = y_inp.roll(-1, dims=-1)
                        y_prev = y_inp.roll(1, dims=-1)
                        swap_back = swap_cand.roll(1, dims=-1)
                        swap_back[:, 0] = False
                        new_y = torch.where(swap_cand, y_next, y_inp)
                        new_y = torch.where(swap_back, y_prev, new_y)
                        y_inp = new_y
                    else:
                        # Substitution-based modes: pick positions to replace,
                        # then draw a replacement from the configured distribution.
                        rand_mask = (torch.rand_like(y_inp, dtype=torch.float) < ic_p) & real_mask
                        if ic_mode == "uniform":
                            rand_tokens = torch.randint(
                                ic_lo, ic_hi, y_inp.shape,
                                device=y_inp.device, dtype=y_inp.dtype,
                            )
                        else:
                            # Build the per-position lookup key:
                            #   bigram_right / self_confusion: key = original char  (label[t])
                            #   bigram_left:                   key = left-context   (label[t-1])
                            if ic_mode == "bigram_left":
                                key = torch.roll(y_inp, shifts=1, dims=-1)
                                key[:, 0] = BOS_ID  # sentinel for position 0
                            else:
                                key = y_inp
                            lookup_idx = key.clamp(min=0, max=ic_lookup.size(0) - 1)
                            rows = ic_lookup[lookup_idx]                      # (B, N, V)
                            flat = rows.reshape(-1, rows.size(-1))            # (B*N, V)
                            sampled = torch.multinomial(flat, 1).reshape_as(y_inp)
                            rand_tokens = sampled.to(dtype=y_inp.dtype)
                        y_inp = torch.where(rand_mask, rand_tokens, y_inp)

                # Determine what extra outputs we need
                need_dec_ctc = dec_ctc_enabled and dec_ctc_lambda > 0.0 and fn_loss_ctc is not None
                need_layers = ds_enabled or need_dec_ctc

                if need_layers:
                    out = model(x, in_lengths=len_x, y_inp=y_inp, return_ar_layers=True, return_dec_ctc=need_dec_ctc)
                    logits = out["logits"]
                    logits_layers = out.get("logits_layers", None)
                else:
                    logits = model(x, in_lengths=len_x, y_inp=y_inp)
                    out = None
                    logits_layers = None

                # Deep supervision CE
                if ds_enabled:
                    if logits_layers is None:
                        raise RuntimeError("deep_supervision.enabled=true but model did not return logits_layers")

                    layers = ds_layers_cfg
                    if not isinstance(layers, (list, tuple)):
                        layers = [layers]
                    layer_idxs = [int(l) - 1 for l in layers]
                    layer_idxs = [i for i in layer_idxs if i >= 0]
                    if ds_include_final:
                        layer_idxs = list(layer_idxs) + [len(logits_layers) - 1]
                    seen = set()
                    layer_idxs = [i for i in layer_idxs if (i not in seen and not seen.add(i))]
                    if len(layer_idxs) == 0:
                        raise ValueError("deep_supervision.layers must contain valid 1-indexed layer IDs")
                    if max(layer_idxs) >= len(logits_layers):
                        raise ValueError(
                            f"deep_supervision.layers out of range. Got {layers}, decoder has {len(logits_layers)} layers"
                        )

                    losses = []
                    for li in layer_idxs:
                        logits_li = logits_layers[li]
                        losses.append(
                            fn_loss(logits_li.reshape(-1, logits_li.size(-1)), y_tgt.reshape(-1))
                        )
                    if ds_reduce == "sum":
                        loss = torch.stack(losses, dim=0).sum()
                    else:
                        loss = torch.stack(losses, dim=0).mean()
                else:
                    loss = fn_loss(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))

                # Decoder-side CTC (AR + CTC regularization "between decoder layers")
                loss_dec_ctc = None
                if need_dec_ctc:
                    if out is None or "dec_ctc_logits_layers" not in out:
                        raise RuntimeError(
                            "decoder_side_ctc.enabled=true but model did not return dec_ctc_logits_layers"
                        )
                    dec_logits_layers = out["dec_ctc_logits_layers"]
                    enc_lengths = out["enc_lengths"]
                    if len(dec_logits_layers) == 0:
                        raise RuntimeError("dec_ctc_logits_layers is empty")
                    dec_losses = []
                    for dec_logits in dec_logits_layers:
                        dec_losses.append(
                            fn_loss_ctc(
                                dec_logits.float().permute((1, 0, 2)),
                                y,
                                enc_lengths,
                                len_y,
                            )
                        )
                    loss_dec_ctc = torch.stack(dec_losses, dim=0).mean()
                    loss = loss + dec_ctc_lambda * loss_dec_ctc
            else:
                # CTC path
                out = model(x)
                loss = fn_loss(out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        man.update_iteration(idx, loss.item(), lr_scheduler.get_last_lr()[0])

    man.summarize_epoch()

    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())


def train_one_epoch_hybrid(
    dataloader: DataLoader,
    model: DualHeadModel,
    fn_loss_ar: nn.Module,
    fn_loss_ctc: nn.Module,
    *,
    lambda_ar: float,
    lambda_ctc: float,
    lambda_ctc_schedule: dict | None = None,
    loss_balance_mode: str = "sum",
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    man: RunManager,
    epoch: int,
) -> None:
    """Train dual-head (AR+CTC) model for one epoch."""
    from .lambda_schedule import compute_lambda_ctc, balance_loss

    # Compute effective lambda_ctc for this epoch
    lambda_ctc_eff = compute_lambda_ctc(epoch, lambda_ctc_schedule, lambda_ctc)

    man.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    PAD_ID = man.cfgs.PAD_ID
    BOS_ID = man.cfgs.BOS_ID
    EOS_ID = man.cfgs.EOS_ID

    dual_cfg = getattr(man.cfgs, "dual_head", {}) or {}
    ds_cfg = dual_cfg.get("deep_supervision", {}) or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    ds_layers_cfg = ds_cfg.get("layers", [1, 2, 3])
    ds_include_final = bool(ds_cfg.get("include_final", True))
    ds_reduce = str(ds_cfg.get("reduce", "mean")).lower()  # mean|sum

    dec_ctc_cfg = dual_cfg.get("decoder_side_ctc", {}) or {}
    dec_ctc_enabled = bool(dec_ctc_cfg.get("enabled", False))
    dec_ctc_lambda = float(dec_ctc_cfg.get("lambda", 0.0))

    # Log schedule info at epoch start
    if epoch == 0 or (lambda_ctc_schedule and epoch % 5 == 0):
        man.log(
            f"[HybridSchedule] epoch={epoch} | lambda_ar={lambda_ar:.4f} | lambda_ctc_eff={lambda_ctc_eff:.4f} | balance_mode={loss_balance_mode}"
        )

    maybe_log_trainability(man, model, epoch=epoch, where="train_one_epoch_hybrid")

    clip_grad = float(getattr(man.cfgs, "clip_grad", 5.0) or 5.0)

    # ---- Input corruption setup (hybrid). Mirrors `train_one_epoch`. ------------
    # Applies the same regularization to the AR head's y_inp; the CTC head is
    # untouched (CTC has no teacher-forced input). Scheduled sampling is NOT
    # supported in the hybrid loop yet (only input_corruption).
    ic_cfg = getattr(man.cfgs, "input_corruption", {}) or {}
    ic_enabled = bool(ic_cfg.get("enabled", False))
    ic_p = float(ic_cfg.get("p_replace", 0.0))
    ic_mode = str(ic_cfg.get("mode", "uniform")).lower()
    ic_lo = 1
    ic_hi = min(PAD_ID, BOS_ID, EOS_ID)
    if ic_enabled and ic_p > 0.0 and ic_hi <= ic_lo:
        raise ValueError(
            f"input_corruption: cannot determine a valid replacement vocab range "
            f"(lo={ic_lo}, hi={ic_hi}). Check tokenizer special token IDs."
        )

    ic_lookup = None
    _SUB_MODES_H = ("bigram_right", "bigram_left", "self_confusion")
    if ic_enabled and ic_p > 0.0 and ic_mode in _SUB_MODES_H:
        cache_key = "bigram_table" if ic_mode in ("bigram_right", "bigram_left") else ic_mode
        cache_attr = f"_ic_lookup_{cache_key}"
        if hasattr(man, cache_attr):
            ic_lookup = getattr(man, cache_attr)
        else:
            # DualHeadModel exposes vocab_ar (same role as BaseModel.num_cls).
            vocab_size = int(getattr(model, "vocab_ar", 0)) or int(getattr(model, "num_cls", 0)) \
                or int(getattr(man.cfgs, "num_cls", 0))
            if vocab_size <= 0:
                raise RuntimeError("input_corruption: cannot infer vocab_size from model")
            ic_smoothing = float(ic_cfg.get("smoothing", 1.0))
            spec_ids = (PAD_ID, BOS_ID, EOS_ID, 0)
            if ic_mode in ("bigram_right", "bigram_left"):
                ic_lookup = _build_bigram_lookup(
                    dataloader, vocab_size, spec_ids,
                    device=man.cfgs.device, smoothing=ic_smoothing,
                )
            elif ic_mode == "self_confusion":
                conf_path_template = ic_cfg.get("confusion_path", None)
                if conf_path_template is None:
                    raise ValueError(
                        "input_corruption mode=self_confusion requires "
                        "input_corruption.confusion_path (with {fold} placeholder)"
                    )
                ic_lookup = _load_confusion_lookup(
                    conf_path_template, fold=int(getattr(man.cfgs, "idx_fold", 0)),
                    vocab_size=vocab_size, spec_ids=spec_ids,
                    device=man.cfgs.device, smoothing=ic_smoothing,
                )
            setattr(man, cache_attr, ic_lookup)
    elif ic_enabled and ic_p > 0.0 and ic_mode not in ("uniform", "adjacent_swap") + _SUB_MODES_H:
        raise ValueError(
            f"input_corruption.mode must be one of "
            f"{{uniform, bigram_right, bigram_left, self_confusion, adjacent_swap}}, "
            f"got: {ic_mode}"
        )

    if ic_enabled and ic_p > 0.0:
        man.log(
            f"[InputCorruption][hybrid] epoch={epoch} | mode={ic_mode} | p_replace={ic_p:.4f} "
            f"(replacement_vocab=[{ic_lo}, {ic_hi}))"
        )
    # ----------------------------------------------------------------------------

    # AMP controllable via config (default true for backward compat)
    use_amp_cfg = bool(getattr(man.cfgs, "use_amp", True)) and torch.cuda.is_available()

    skipped_nonfinite = 0
    skipped_empty_ar = 0

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x = x.to(man.cfgs.device)
        y = y.to(man.cfgs.device)
        len_x = len_x.to(man.cfgs.device)
        len_y = len_y.to(man.cfgs.device)

        # Skip degenerate / non-finite inputs (rare but can happen with corrupt CSVs)
        if not torch.isfinite(x).all():
            skipped_nonfinite += 1
            if skipped_nonfinite <= 5:
                logger.warning(
                    "Non-finite input batch. Skipping. epoch={} iter={} len_x=[{},{}]",
                    epoch,
                    idx,
                    int(len_x.min().item()) if len_x.numel() else -1,
                    int(len_x.max().item()) if len_x.numel() else -1,
                )
            continue

        optimizer.zero_grad(set_to_none=True)

        use_amp = use_amp_cfg and x.is_cuda
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            y_inp, y_tgt = build_ar_batch(
                y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device
            )

            # Guard: if PAD/BOS/EOS IDs are misconfigured (or a truly empty target batch),
            # CrossEntropyLoss can return NaN due to division by zero.
            n_valid_ar = int((y_tgt.reshape(-1) != int(PAD_ID)).sum().item())
            if n_valid_ar <= 0:
                skipped_empty_ar += 1
                if skipped_empty_ar <= 5:
                    logger.warning(
                        "No valid AR target tokens (all PAD). Skipping. epoch={} iter={} PAD_ID={} BOS_ID={} EOS_ID={} len_y=[{},{}]",
                        epoch,
                        idx,
                        int(PAD_ID),
                        int(BOS_ID),
                        int(EOS_ID),
                        int(len_y.min().item()) if len_y.numel() else -1,
                        int(len_y.max().item()) if len_y.numel() else -1,
                    )
                continue

            # ---- Input corruption (hybrid AR head only) ------------------------
            # The AR CE loss is computed against the unmodified y_tgt; the CTC
            # loss is unaffected (encoder output, no teacher-forced input).
            if ic_enabled and ic_p > 0.0:
                real_mask = (y_inp != PAD_ID) & (y_inp != BOS_ID) & (y_inp != EOS_ID)

                if ic_mode == "adjacent_swap":
                    real_next = real_mask.roll(-1, dims=-1)
                    real_next[:, -1] = False
                    swap_cand = (
                        (torch.rand_like(y_inp, dtype=torch.float) < ic_p)
                        & real_mask & real_next
                    )
                    y_next = y_inp.roll(-1, dims=-1)
                    y_prev = y_inp.roll(1, dims=-1)
                    swap_back = swap_cand.roll(1, dims=-1)
                    swap_back[:, 0] = False
                    new_y = torch.where(swap_cand, y_next, y_inp)
                    new_y = torch.where(swap_back, y_prev, new_y)
                    y_inp = new_y
                else:
                    rand_mask = (torch.rand_like(y_inp, dtype=torch.float) < ic_p) & real_mask
                    if ic_mode == "uniform":
                        rand_tokens = torch.randint(
                            ic_lo, ic_hi, y_inp.shape,
                            device=y_inp.device, dtype=y_inp.dtype,
                        )
                    else:
                        if ic_mode == "bigram_left":
                            key = torch.roll(y_inp, shifts=1, dims=-1)
                            key[:, 0] = BOS_ID
                        else:
                            key = y_inp
                        lookup_idx = key.clamp(min=0, max=ic_lookup.size(0) - 1)
                        rows = ic_lookup[lookup_idx]
                        flat = rows.reshape(-1, rows.size(-1))
                        sampled = torch.multinomial(flat, 1).reshape_as(y_inp)
                        rand_tokens = sampled.to(dtype=y_inp.dtype)
                    y_inp = torch.where(rand_mask, rand_tokens, y_inp)
            # --------------------------------------------------------------------

            out = model(x, in_lengths=len_x, y_inp=y_inp, return_ar=True, return_ctc=True)

            ar_logits = out["ar_logits"]
            ctc_logits = out["ctc_logits"].float()  # keep ctc loss in fp32 for stability
            enc_lengths = out["enc_lengths"]

            # AR token loss (optionally with deep supervision on intermediate decoder layers)
            if ds_enabled:
                if "ar_logits_layers" not in out:
                    raise RuntimeError(
                        "dual_head.deep_supervision.enabled=true but model did not return ar_logits_layers"
                    )
                layers = ds_layers_cfg
                if not isinstance(layers, (list, tuple)):
                    layers = [layers]
                layer_idxs = [int(l) - 1 for l in layers]
                layer_idxs = [i for i in layer_idxs if i >= 0]
                if ds_include_final:
                    layer_idxs = list(layer_idxs) + [len(out["ar_logits_layers"]) - 1]
                # unique + stable order
                seen = set()
                layer_idxs = [i for i in layer_idxs if (i not in seen and not seen.add(i))]
                if len(layer_idxs) == 0:
                    raise ValueError("dual_head.deep_supervision.layers must contain valid 1-indexed layer IDs")
                if max(layer_idxs) >= len(out["ar_logits_layers"]):
                    raise ValueError(
                        f"dual_head.deep_supervision.layers out of range. Got {layers}, decoder has {len(out['ar_logits_layers'])} layers"
                    )

                losses = []
                for li in layer_idxs:
                    logits_li = out["ar_logits_layers"][li]
                    losses.append(
                        fn_loss_ar(
                            logits_li.reshape(-1, logits_li.size(-1)),
                            y_tgt.reshape(-1),
                        )
                    )
                if ds_reduce == "sum":
                    loss_ar = torch.stack(losses, dim=0).sum()
                else:
                    loss_ar = torch.stack(losses, dim=0).mean()
            else:
                loss_ar = fn_loss_ar(
                    ar_logits.reshape(-1, ar_logits.size(-1)),
                    y_tgt.reshape(-1),
                )

            loss_ctc = fn_loss_ctc(
                ctc_logits.permute((1, 0, 2)),
                y,
                enc_lengths,
                len_y,
            )

            # Optional decoder-side CTC (CTC on encoder time axis conditioned on decoder layer states)
            loss_dec_ctc = None
            if dec_ctc_enabled and dec_ctc_lambda > 0.0:
                if "dec_ctc_logits_layers" not in out:
                    raise RuntimeError(
                        "dual_head.decoder_side_ctc.enabled=true but model did not return dec_ctc_logits_layers"
                    )
                dec_logits_layers = out["dec_ctc_logits_layers"]
                if len(dec_logits_layers) == 0:
                    raise RuntimeError("dec_ctc_logits_layers is empty")
                dec_losses = []
                for dec_logits in dec_logits_layers:
                    dec_losses.append(
                        fn_loss_ctc(
                            dec_logits.float().permute((1, 0, 2)),
                            y,
                            enc_lengths,
                            len_y,
                        )
                    )
                loss_dec_ctc = torch.stack(dec_losses, dim=0).mean()

            loss = balance_loss(
                float(loss_ar.item()),
                float(loss_ctc.item()),
                lambda_ar,
                lambda_ctc_eff,
                loss_balance_mode,
            )
            loss = torch.tensor(loss, device=ar_logits.device, requires_grad=True)
            # Recompute with actual tensors for backward
            loss = lambda_ar * loss_ar + lambda_ctc_eff * loss_ctc
            if loss_balance_mode == "normalize":
                loss = loss / (lambda_ar + lambda_ctc_eff)
            elif loss_balance_mode == "convex":
                alpha = max(0.0, min(1.0, lambda_ctc_eff))
                loss = (1.0 - alpha) * loss_ar + alpha * loss_ctc

            if loss_dec_ctc is not None:
                loss = loss + dec_ctc_lambda * loss_dec_ctc

        # Skip non-finite losses (prevents corrupting epoch averages and model weights).
        if not (torch.isfinite(loss) and torch.isfinite(loss_ar) and torch.isfinite(loss_ctc)):
            skipped_nonfinite += 1
            if skipped_nonfinite <= 5:
                ar_finite = bool(torch.isfinite(ar_logits).all().detach().cpu().item())
                ctc_finite = bool(torch.isfinite(ctc_logits).all().detach().cpu().item())
                logger.warning(
                    "Non-finite loss. Skipping update. epoch={} iter={} lr={} loss={} loss_ar={} loss_ctc={} ar_logits_finite={} ctc_logits_finite={} len_x=[{},{}] len_y=[{},{}]",
                    epoch,
                    idx,
                    lr_scheduler.get_last_lr()[0],
                    float(loss.detach().cpu().item()),
                    float(loss_ar.detach().cpu().item()),
                    float(loss_ctc.detach().cpu().item()),
                    ar_finite,
                    ctc_finite,
                    int(len_x.min().item()) if len_x.numel() else -1,
                    int(len_x.max().item()) if len_x.numel() else -1,
                    int(len_y.min().item()) if len_y.numel() else -1,
                    int(len_y.max().item()) if len_y.numel() else -1,
                )
            optimizer.zero_grad(set_to_none=True)
            # Don't call scaler.update() here - we haven't called scaler.scale() yet
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if not torch.isfinite(grad_norm):
            skipped_nonfinite += 1
            if skipped_nonfinite <= 5:
                logger.warning(
                    "Non-finite grad norm. Skipping update. epoch={} iter={} grad_norm={}",
                    epoch,
                    idx,
                    float(grad_norm.detach().cpu().item()),
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        man.update_iteration(
            idx,
            float(loss.item()),
            lr_scheduler.get_last_lr()[0],
            loss_ar=float(loss_ar.item()),
            loss_ctc=float(loss_ctc.item()),
            **({"loss_dec_ctc": float(loss_dec_ctc.item())} if loss_dec_ctc is not None else {}),
            lambda_ctc_eff=float(lambda_ctc_eff),
        )

    if (skipped_nonfinite + skipped_empty_ar) > 0:
        man.log(
            f"[HybridTrain] skipped_batches={skipped_nonfinite + skipped_empty_ar} (nonfinite={skipped_nonfinite}, empty_ar={skipped_empty_ar})"
        )

    man.summarize_epoch()

    if not bool(getattr(man.cfgs, "save_best_only", False)) and man.check_step(epoch + 1, 'save'):
        man.save_checkpoint(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())


def test(
    dataloader: DataLoader,
    model: BaseModel,
    fn_loss: nn.Module,
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: Optional[int] = None,
    tokenizer=None,
    force_eval: bool = False,
    qual_cfg: Optional[dict] = None,
) -> None:
    """
    Evaluate CTC/AR model.
    
    Args:
        dataloader: Test dataloader yielding (x, y, len_x, len_y).
        model: BaseModel instance.
        fn_loss: Loss function.
        man: RunManager for logging.
        ctc_decoder: CTC decoder for CTC mode.
        epoch: Current epoch number.
        tokenizer: Optional tokenizer for AR mode decoding.
        force_eval: Whether to force evaluation even if not an eval epoch.
        qual_cfg: Optional qualitative analysis configuration.
    """
    preds = []
    labels = []
    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()
    
    PAD_ID = man.cfgs.PAD_ID
    BOS_ID = man.cfgs.BOS_ID
    EOS_ID = man.cfgs.EOS_ID

    do_eval = force_eval or man.check_step(epoch + 1, 'eval')
    
    # Qualitative analysis setup
    sel_map = None
    q50_thr, q99_thr = None, None
    outdir = None
    use_gradcam = False
    target_layer_name = None

    if qual_cfg is not None and qual_cfg.get("enabled", False) and HAS_PANDAS:
        sel_map = qual_cfg["selection_map"]
        outdir = qual_cfg["outdir"]
        os.makedirs(outdir, exist_ok=True)
        use_gradcam = bool(qual_cfg.get("use_gradcam", False))
        target_layer_name = qual_cfg.get("gradcam_layer", "layers.11.pwconv")

        task_name = getattr(man.cfgs, "qual_task", "word")
        q50_thr, q99_thr = compute_fold_thresholds(man.cfgs.qual_csv, int(man.cfgs.idx_fold), task_name)
        
        # Extract thresholds from selection map
        if sel_map:
            for v in sel_map.values():
                if v.get("target_quantile") == 0.5 and v.get("target_value") is not None:
                    q50_thr = float(v["target_value"])
                if v.get("target_quantile") == 0.99 and v.get("target_value") is not None:
                    q99_thr = float(v["target_value"])

        # Save selection for reproducibility
        pd.DataFrame([
            {"sample_index": k, **v} for k, v in sel_map.items()
        ]).to_csv(os.path.join(outdir, "partB_selected_samples.csv"), index=False)

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
                loss = fn_loss(out.permute((1, 0, 2)), y, len_x // model.ratio_ds, len_y)

            man.update_iteration(idx, loss.item())

            # CTC decoding
            if do_eval and not isinstance(fn_loss, nn.CrossEntropyLoss):
                for pred, len_pred, label in zip(out.cpu(), len_x // model.ratio_ds, y.cpu()):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

            # AR greedy decoding
            if do_eval and isinstance(fn_loss, nn.CrossEntropyLoss):
                B = x.size(0)
                max_len = int(len_y.max().item()) + 2
                device = x.device

                tok = tokenizer
                chars = getattr(man.cfgs, 'categories', None)

                # Autoregressive generation
                y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
                for _ in range(max_len):
                    step_logits = model(x, in_lengths=len_x, y_inp=y_gen)
                    nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)
                    y_gen = torch.cat([y_gen, nxt], dim=1)

                y_gen = y_gen.detach().cpu().tolist()
                y_cpu = y.cpu()
                len_y_cpu = len_y.cpu().tolist()

                for b in range(B):
                    # Decode prediction
                    ids_pred = y_gen[b][1:]  # skip BOS
                    seq = []
                    for t in ids_pred:
                        if t == EOS_ID:
                            break
                        if t == PAD_ID:
                            continue
                        seq.append(int(t))

                    if tok is not None:
                        pred_str = tok.decode(seq)
                    else:
                        pred_str = ''.join(
                            chars[i] for i in seq
                            if chars is not None and 0 <= i < len(chars) and i != 0
                        )
                    preds.append(pred_str)

                    # Decode label
                    if y.dim() == 2:
                        L = int(len_y_cpu[b])
                        lab_ids = y_cpu[b, :L].tolist()
                        if tok is not None:
                            lab_str = tok.decode(lab_ids)
                        else:
                            lab_str = ''.join(
                                chars[i] for i in lab_ids
                                if chars is not None and 0 <= i < len(chars) and i != 0
                            )
                        labels.append(lab_str)

                    # Qualitative analysis for selected samples
                    if sel_map is not None and HAS_PANDAS:
                        _do_qualitative_capture(
                            b, x, len_x, seq, pred_str, lab_str, model,
                            sel_map, outdir, q50_thr, q99_thr,
                            use_gradcam, target_layer_name, man, preds, BOS_ID
                        )

    if sel_map:
        logger.info("Eval preds count = {}", len(preds))
        logger.info("Selected indices min/max = {}/{}", min(sel_map.keys()), max(sel_map.keys()))

    man.summarize_epoch()

    # Export predictions
    export_val_full = bool(getattr(man.cfgs, "export_val_full", False))
    is_test_mode = bool(getattr(man.cfgs, "test", False))
    do_export = is_test_mode or (export_val_full and do_eval)

    if do_export:
        export_dir = os.path.join(man.cfgs.dir_work, "exports")
        os.makedirs(export_dir, exist_ok=True)
        epoch_tag = "best" if epoch is None else f"epoch{epoch}"
        export_path = os.path.join(export_dir, f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}.json")
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds, "labels": labels}, f, ensure_ascii=False)
        logger.info("Exported full validation predictions to {}", export_path)

    # Evaluation and visualization
    if do_eval:
        eval_label = 'AR-greedy' if isinstance(fn_loss, nn.CrossEntropyLoss) else 'CTC-bestpath'
        if not preds or not labels:
            logger.warning(
                "[Eval][{}] skipped: empty preds/labels (epoch={} fold={})",
                eval_label,
                int(epoch) if epoch is not None else -1,
                int(getattr(man.cfgs, "idx_fold", -1)),
            )
            return

        # Make matrix naming explicit across modes.
        # - AR-only:  mat_se_<epoch>_ar.pdf
        # - CTC-only: mat_se_<epoch>_ctc.pdf
        suffix = '_ar' if isinstance(fn_loss, nn.CrossEntropyLoss) else '_ctc'
        visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch, suffix=suffix)

        results_eval = evaluate(preds, labels)
        man.update_evaluation(
            results_eval,
            preds[:20],
            labels[:20],
            key='evaluation',
            label=eval_label,
        )


def test_hybrid(
    dataloader: DataLoader,
    model: DualHeadModel,
    fn_loss_ar: nn.Module,
    fn_loss_ctc: nn.Module,
    *,
    lambda_ar: float,
    lambda_ctc: float,
    lambda_ctc_schedule: dict | None = None,
    loss_balance_mode: str = "sum",
    man: RunManager,
    ctc_decoder: BestPath,
    epoch: int,
    tokenizer=None,
    force_eval: bool = False,
) -> None:
    """Evaluate dual-head (AR+CTC) model.

    - Logs total hybrid loss
    - Computes greedy AR predictions and greedy CTC predictions
    - Writes AR metrics under key='evaluation' (for best checkpoint selection)
    - Writes CTC metrics under key='evaluation_ctc'
    """
    from .lambda_schedule import compute_lambda_ctc, balance_loss

    # Compute effective lambda_ctc for this epoch
    lambda_ctc_eff = compute_lambda_ctc(epoch, lambda_ctc_schedule, lambda_ctc)

    man.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    PAD_ID = man.cfgs.PAD_ID
    BOS_ID = man.cfgs.BOS_ID
    EOS_ID = man.cfgs.EOS_ID

    do_eval = force_eval or man.check_step(epoch + 1, 'eval')

    preds_ar: list[str] = []
    labels_ar: list[str] = []
    preds_ctc: list[str] = []
    labels_ctc: list[str] = []

    rescore_cfg = getattr(man.cfgs, "rescore_ctc", {}) or {}
    rescore_enabled = bool(rescore_cfg.get("enabled", False))
    beam_size = int(rescore_cfg.get("beam_size", 4))
    max_len_override = rescore_cfg.get("max_len", None)
    length_penalty = float(rescore_cfg.get("length_penalty", 0.0))
    beta_ar = float(rescore_cfg.get("beta_ar", 1.0))
    betas_ctc = rescore_cfg.get("betas_ctc", [0.3])
    export_enabled = bool(rescore_cfg.get("export", True))
    export_dirname = str(rescore_cfg.get("export_dirname", "exports_rescore_ctc"))

    try:
        betas_ctc = [float(b) for b in (betas_ctc or [])]
    except Exception:
        betas_ctc = [0.3]

    preds_rescore: dict[float, list[str]] = {b: [] for b in betas_ctc}
    labels_rescore: dict[float, list[str]] = {b: [] for b in betas_ctc}

    skipped_nonfinite = 0

    def _sanitize_beta(x: float) -> str:
        s = f"{x:.4f}"
        return re.sub(r"[^0-9A-Za-z_.-]", "_", s)

    def _decode_ids(ids: list[int]) -> str:
        if tokenizer is not None:
            return tokenizer.decode(ids)
        chars = getattr(man.cfgs, 'categories', None)
        if not chars:
            return ""
        return ''.join(chars[i] for i in ids if 0 <= i < len(chars) and i != 0)

    def _ctc_logprob_for_target(ctc_logits_TV: torch.Tensor, target_ids: list[int], T: int) -> float:
        # target_ids must not include blanks; filter any invalid IDs.
        tgt = [int(t) for t in target_ids if int(t) > 0]
        if len(tgt) == 0:
            return 0.0
        if len(tgt) > int(T):
            return float("-inf")

        log_probs = F.log_softmax(ctc_logits_TV[:T], dim=-1)  # (T, V)
        log_probs = log_probs.unsqueeze(1)  # (T, 1, V)
        targets = torch.tensor(tgt, dtype=torch.long, device=ctc_logits_TV.device)
        input_lengths = torch.tensor([int(T)], dtype=torch.long, device=ctc_logits_TV.device)
        target_lengths = torch.tensor([int(len(tgt))], dtype=torch.long, device=ctc_logits_TV.device)

        # Use sum reduction for a proper total log-prob (up to constant factors).
        nll = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        return float((-nll).detach().cpu().item())

    def _beam_search_one(mem_1TD: torch.Tensor, enc_pad_1T: torch.Tensor, *, max_len: int) -> list[tuple[list[int], float]]:
        # Returns N-best (token_ids_without_BOS/EOS/PAD), ar_logprob
        # Beam state keeps seq WITH BOS for decoding.
        beams: list[tuple[list[int], float, bool]] = [([int(BOS_ID)], 0.0, False)]

        for _step in range(max_len):
            all_cands: list[tuple[list[int], float, bool]] = []
            for seq, score, ended in beams:
                if ended:
                    all_cands.append((seq, score, True))
                    continue

                y_inp = torch.tensor([seq], dtype=torch.long, device=mem_1TD.device)
                logits = model.decoder(y_inp, mem_1TD, enc_pad_1T)
                next_logprobs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                k = min(beam_size, int(next_logprobs.numel()))
                topv, topi = torch.topk(next_logprobs, k=k, dim=-1)
                for lp, tid in zip(topv.tolist(), topi.tolist()):
                    tid = int(tid)
                    if tid == int(PAD_ID):
                        continue
                    new_seq = seq + [tid]
                    new_score = float(score + float(lp))
                    new_ended = (tid == int(EOS_ID))
                    all_cands.append((new_seq, new_score, new_ended))

            # prune
            def _rank_key(item: tuple[list[int], float, bool]) -> float:
                seq, score, _ended = item
                # exclude BOS for length
                L = max(1, len(seq) - 1)
                if length_penalty > 0:
                    return score / (L ** length_penalty)
                return score

            all_cands.sort(key=_rank_key, reverse=True)
            beams = all_cands[:beam_size]
            if all(e for (_s, _sc, e) in beams):
                break

        # finalize
        out: list[tuple[list[int], float]] = []
        for seq, score, _ended in beams:
            # strip BOS, stop at EOS, remove PAD
            ids = []
            for t in seq[1:]:
                if t == int(EOS_ID):
                    break
                if t == int(PAD_ID):
                    continue
                ids.append(int(t))
            out.append((ids, float(score)))
        # sort by same rank key used for pruning
        def _final_rank(item: tuple[list[int], float]) -> float:
            ids, score = item
            L = max(1, len(ids))
            return score / (L ** length_penalty) if length_penalty > 0 else score
        out.sort(key=_final_rank, reverse=True)
        return out

    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x = x.to(man.cfgs.device)
            y = y.to(man.cfgs.device)
            len_x = len_x.to(man.cfgs.device)
            len_y = len_y.to(man.cfgs.device)

            # Encode once; reuse memory for greedy decode/beam search
            use_amp = bool(getattr(man.cfgs, "use_amp", True)) and x.is_cuda
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                feats, enc_pad, enc_lengths = model._encode_with_mask(x, len_x)

                mem = feats
                if getattr(model, "mem_proj", None) is not None:
                    mem = model.mem_proj(mem)

                # teacher-forced AR logits for loss only
                y_inp, y_tgt = build_ar_batch(
                    y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device
                )
                ar_logits = model.decoder(y_inp, mem, enc_pad)

                # CTC logits from the per-timestep head
                ctc_logits = model.compute_ctc_logits(feats).float()

                loss_ar = fn_loss_ar(
                    ar_logits.reshape(-1, ar_logits.size(-1)),
                    y_tgt.reshape(-1),
                )
                loss_ctc = fn_loss_ctc(
                    ctc_logits.permute((1, 0, 2)),
                    y,
                    enc_lengths,
                    len_y,
                )
                loss = lambda_ar * loss_ar + lambda_ctc_eff * loss_ctc
                if loss_balance_mode == "normalize":
                    loss = loss / (lambda_ar + lambda_ctc_eff)
                elif loss_balance_mode == "convex":
                    alpha = max(0.0, min(1.0, lambda_ctc_eff))
                    loss = (1.0 - alpha) * loss_ar + alpha * loss_ctc

            # Skip non-finite loss batches so eval averages don't become NaN.
            if not (torch.isfinite(loss) and torch.isfinite(loss_ar) and torch.isfinite(loss_ctc)):
                skipped_nonfinite += 1
                if skipped_nonfinite <= 5:
                    ar_finite = bool(torch.isfinite(ar_logits).all().detach().cpu().item())
                    ctc_finite = bool(torch.isfinite(ctc_logits).all().detach().cpu().item())
                    logger.warning(
                        "Non-finite eval loss. Skipping batch. epoch={} iter={} loss={} loss_ar={} loss_ctc={} ar_logits_finite={} ctc_logits_finite={} len_x=[{},{}] len_y=[{},{}]",
                        epoch,
                        idx,
                        float(loss.detach().cpu().item()),
                        float(loss_ar.detach().cpu().item()),
                        float(loss_ctc.detach().cpu().item()),
                        ar_finite,
                        ctc_finite,
                        int(len_x.min().item()) if len_x.numel() else -1,
                        int(len_x.max().item()) if len_x.numel() else -1,
                        int(len_y.min().item()) if len_y.numel() else -1,
                        int(len_y.max().item()) if len_y.numel() else -1,
                    )
                continue

            man.update_iteration(
                idx,
                float(loss.item()),
                lr=0.0,
                loss_ar=float(loss_ar.item()),
                loss_ctc=float(loss_ctc.item()),
                lambda_ctc_eff=float(lambda_ctc_eff),
            )

            if not do_eval:
                continue

            # CTC greedy
            for logit, Lx, label, Ly in zip(ctc_logits.detach().cpu(), enc_lengths.detach().cpu(), y.detach().cpu(), len_y.detach().cpu()):
                preds_ctc.append(ctc_decoder.decode(logit[: int(Lx)]))
                labels_ctc.append(ctc_decoder.decode(label[: int(Ly)], True))

            # Convert encoder outputs to fp32 for greedy/beam decoding (avoid dtype mismatch)
            mem = mem.float()
            ctc_logits = ctc_logits.float()

            # AR greedy
            B = x.size(0)
            max_len = int(len_y.max().item()) + 2
            if max_len_override is not None:
                try:
                    max_len = int(max_len_override)
                except Exception:
                    pass

            # AR greedy decode from cached encoder memory
            y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=mem.device)
            for _ in range(max_len):
                step_logits = model.decoder(y_gen, mem, enc_pad)
                nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)
                y_gen = torch.cat([y_gen, nxt], dim=1)

            y_gen = y_gen.detach().cpu().tolist()
            y_cpu = y.detach().cpu()
            len_y_cpu = len_y.detach().cpu().tolist()

            for b in range(B):
                ids_pred = y_gen[b][1:]
                seq = []
                for t in ids_pred:
                    if t == EOS_ID:
                        break
                    if t == PAD_ID:
                        continue
                    seq.append(int(t))
                preds_ar.append(_decode_ids(seq))

                L = int(len_y_cpu[b])
                lab_ids = y_cpu[b, :L].tolist() if y_cpu.dim() == 2 else []
                labels_ar.append(_decode_ids([int(t) for t in lab_ids if int(t) != int(PAD_ID)]))

            # Optional: AR beam + CTC rescoring
            if rescore_enabled and betas_ctc:
                for b in range(B):
                    mem_b = mem[b:b+1]
                    enc_pad_b = enc_pad[b:b+1]
                    T_b = int(enc_lengths[b].detach().cpu().item())
                    ctc_b = ctc_logits[b]

                    nbest = _beam_search_one(mem_b, enc_pad_b, max_len=max_len)
                    if not nbest:
                        # fallback to greedy seq already computed
                        nbest = [([], float("-inf"))]

                    # Precompute CTC logprobs for candidates
                    cand_ctc_lp: list[float] = []
                    for cand_ids, _ar_lp in nbest:
                        # CTC targets are character IDs, no BOS/EOS/PAD, no blanks
                        cand_ids_ctc = [int(t) for t in cand_ids if int(t) != int(PAD_ID) and int(t) != int(EOS_ID) and int(t) != int(BOS_ID) and int(t) > 0]
                        cand_ctc_lp.append(_ctc_logprob_for_target(ctc_b, cand_ids_ctc, T=T_b))

                    # Label once
                    L = int(len_y_cpu[b])
                    lab_ids = y_cpu[b, :L].tolist() if y_cpu.dim() == 2 else []
                    lab_str = _decode_ids([int(t) for t in lab_ids if int(t) != int(PAD_ID)])

                    for beta_ctc in betas_ctc:
                        best_i = 0
                        best_score = float("-inf")
                        for i, ((cand_ids, ar_lp), ctc_lp) in enumerate(zip(nbest, cand_ctc_lp)):
                            Lcand = max(1, len(cand_ids))
                            ar_lp_adj = ar_lp / (Lcand ** length_penalty) if length_penalty > 0 else ar_lp
                            score = beta_ar * ar_lp_adj + float(beta_ctc) * float(ctc_lp)
                            if score > best_score:
                                best_score = score
                                best_i = i

                        best_ids = nbest[best_i][0]
                        preds_rescore[float(beta_ctc)].append(_decode_ids(best_ids))
                        labels_rescore[float(beta_ctc)].append(lab_str)

    if skipped_nonfinite > 0:
        man.log(f"[HybridEval] skipped_batches={skipped_nonfinite} (nonfinite_loss)")

    man.summarize_epoch()

    # Export full-set predictions for analysis.
    # Note: For hybrid we export both AR-greedy and CTC-bestpath.
    export_val_full = bool(getattr(man.cfgs, "export_val_full", False))
    is_test_mode = bool(getattr(man.cfgs, "test", False))
    do_export = is_test_mode or (export_val_full and do_eval)

    if do_export and do_eval:
        export_dir = os.path.join(man.cfgs.dir_work, "exports")
        os.makedirs(export_dir, exist_ok=True)
        epoch_tag = "best" if epoch is None else f"epoch{epoch}"

        export_path_ar = os.path.join(
            export_dir,
            f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}_ar.json",
        )
        with open(export_path_ar, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds_ar, "labels": labels_ar}, f, ensure_ascii=False)
        logger.info("Exported full validation predictions (AR) to {}", export_path_ar)

        export_path_ctc = os.path.join(
            export_dir,
            f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}_ctc.json",
        )
        with open(export_path_ctc, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds_ctc, "labels": labels_ctc}, f, ensure_ascii=False)
        logger.info("Exported full validation predictions (CTC) to {}", export_path_ctc)

    if do_eval:
        cats = man.cfgs.categories[1:]  # exclude blank
        # Make matrix naming explicit across modes.
        visualize(preds_ar, labels_ar, cats, man.dir_vis, epoch, suffix='_ar')
        visualize(preds_ctc, labels_ctc, cats, man.dir_vis, epoch, suffix='_ctc')

        results_ar = evaluate(preds_ar, labels_ar)
        results_ctc = evaluate(preds_ctc, labels_ctc)
        man.update_evaluation(results_ar, preds_ar[:20], labels_ar[:20], key='evaluation', label='AR-greedy')
        man.update_evaluation(results_ctc, preds_ctc[:20], labels_ctc[:20], key='evaluation_ctc', label='CTC-bestpath')

        if rescore_enabled and betas_ctc:
            for beta_ctc in betas_ctc:
                preds_b = preds_rescore.get(float(beta_ctc), [])
                labels_b = labels_rescore.get(float(beta_ctc), [])
                if not preds_b:
                    continue
                res = evaluate(preds_b, labels_b)
                beta_tag = _sanitize_beta(float(beta_ctc))
                key = f"evaluation_rescore_ctc_beta{beta_tag}"
                man.update_evaluation(
                    res,
                    preds_b[:20],
                    labels_b[:20],
                    key=key,
                    label=f"AR-beam+CTC(beta_ctc={beta_ctc})",
                )

                if export_enabled:
                    export_dir = os.path.join(man.cfgs.dir_work, export_dirname)
                    os.makedirs(export_dir, exist_ok=True)
                    export_path = os.path.join(
                        export_dir,
                        f"val_full_fold{man.cfgs.idx_fold}_beta{beta_tag}.json",
                    )
                    with open(export_path, "w", encoding="utf-8") as f:
                        json.dump({"predictions": preds_b, "labels": labels_b}, f, ensure_ascii=False)
                    logger.info("Exported rescored predictions to {}", export_path)


def _do_qualitative_capture(
    b: int,
    x: torch.Tensor,
    len_x: torch.Tensor,
    seq: list,
    pred_str: str,
    lab_str: str,
    model: nn.Module,
    sel_map: dict,
    outdir: str,
    q50_thr: float,
    q99_thr: float,
    use_gradcam: bool,
    target_layer_name: str,
    man: RunManager,
    preds: list,
    BOS_ID: int,
) -> None:
    """Helper to capture qualitative analysis for a selected sample."""
    import numpy as np
    
    lev_rt = lev_dist(pred_str, lab_str)
    d_tilde_rt = lev_rt / max(1, len(lab_str))
    sample_idx = len(preds) - 1

    if sample_idx not in sel_map:
        return

    meta = sel_map[sample_idx]
    regime_csv = meta.get("regime", "unknown")

    # Determine runtime regime
    if lev_rt == 0:
        regime_rt = "correct"
    elif q99_thr is not None and d_tilde_rt >= q99_thr:
        regime_rt = "catastrophic"
    elif q50_thr is not None and d_tilde_rt <= q50_thr:
        regime_rt = "near_miss"
    else:
        regime_rt = "mid_error"

    # Debug log
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

    # Prepare tensors
    xb = x[b:b+1]
    len_xb = len_x[b:b+1]
    pred_ids = seq[:]

    # Attention capture
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
        title = f"fold={man.cfgs.idx_fold} idx={sample_idx} d~={d_tilde_rt:.3f} d={lev_rt}\npred={pred_str} | gt={lab_str}"
        save_attn_heatmap(M, fig_attn, title)

    # Grad-CAM
    fig_cam = None
    if use_gradcam:
        enc_modules = dict(model.encoder.named_modules())
        if target_layer_name not in enc_modules:
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
        model.zero_grad(set_to_none=True)
        y_inp_vis = torch.tensor([[BOS_ID] + pred_ids], dtype=torch.long, device=xb.device)

        with torch.enable_grad():
            logits_vis = model(xb, in_lengths=len_xb, y_inp=y_inp_vis)
            score = seq_logprob_score(logits_vis, pred_ids)
            score.backward()

        cam_vec = cam.cam().detach().cpu().numpy()[0]
        cam.remove()

        fig_cam = os.path.join(outdir, f"fold{man.cfgs.idx_fold}_idx{sample_idx}_{regime_rt}_gradcam1d.png")
        x_cpu = xb.detach().cpu()
        title = f"Grad-CAM1D fold={man.cfgs.idx_fold} idx={sample_idx} d~={d_tilde_rt:.3f} d={lev_rt}\npred={pred_str} | gt={lab_str}"
        save_signal_plus_cam(x_cpu, cam_vec, fig_cam, title, valid_len=int(len_xb.item()))

    # Index file
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
    if not os.path.exists(index_path):
        pd.DataFrame([row]).to_csv(index_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(index_path, mode="a", header=False, index=False)
