"""
Training utilities for model parameter management and debugging.
"""

from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger

from rewi.manager import RunManager


def build_ar_batch(
    y: torch.Tensor,
    len_y: torch.Tensor,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build autoregressive training batch with teacher forcing.
    
    Supports:
      - y shape (B, S_max) with right-padding (common collate)
      - y shape (sum(len_y),) flat concat (rare)
    
    Args:
        y: Label tensor.
        len_y: Lengths of each label.
        pad_id: Padding token ID.
        bos_id: Beginning of sequence token ID.
        eos_id: End of sequence token ID.
        device: Target device.
    
    Returns:
        y_inp: [BOS, y1..yK] padded with PAD -> (B, N)
        y_tgt: [y1..yK, EOS] padded with PAD -> (B, N)
    """
    lengths = len_y.tolist()
    parts = []

    if y.dim() == 2:
        B = y.size(0)
        for b in range(B):
            L = int(lengths[b])
            parts.append(y[b, :L])
    elif y.dim() == 1:
        parts = list(torch.split(y, tuple(int(l) for l in lengths), dim=0))
    else:
        raise ValueError(f"build_ar_batch: unexpected y.dim() = {y.dim()} (expected 1 or 2)")

    N = max(p.size(0) for p in parts) + 1
    B = len(parts)

    y_inp = torch.full((B, N), pad_id, dtype=torch.long, device=device)
    y_tgt = torch.full((B, N), pad_id, dtype=torch.long, device=device)

    for b, lab in enumerate(parts):
        L = lab.size(0)
        y_inp[b, 0] = bos_id
        if L > 0:
            y_inp[b, 1:L+1] = lab.to(torch.long)
            y_tgt[b, 0:L] = lab.to(torch.long)
        y_tgt[b, L] = eos_id

    return y_inp, y_tgt


def count_params(module: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a module.
    
    Args:
        module: PyTorch module.
    
    Returns:
        Tuple of (trainable_params, total_params).
    """
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def maybe_log_trainability(
    man: RunManager,
    model: nn.Module,
    *,
    epoch: int,
    where: str,
) -> None:
    """
    Log encoder/decoder trainability when it changes.

    Stateful on `man` to catch mid-training freeze/unfreeze changes
    without spamming logs.
    
    Args:
        man: RunManager instance.
        model: Model with encoder/decoder attributes.
        epoch: Current epoch.
        where: Description of call location for logging.
    """
    enc = getattr(model, "encoder", None)
    dec = getattr(model, "decoder", None)
    if enc is None and dec is None:
        return

    enc_tr, enc_tot = count_params(enc) if enc is not None else (0, 0)
    dec_tr, dec_tot = count_params(dec) if dec is not None else (0, 0)

    snap = (enc_tr, enc_tot, dec_tr, dec_tot)
    prev = getattr(man, "_dbg_trainability_prev", None)
    if prev != snap:
        logger.info(
            "[Trainability][{}] epoch={} | enc trainable={}/{} | dec trainable={}/{}",
            where, epoch, enc_tr, enc_tot, dec_tr, dec_tot,
        )
        man._dbg_trainability_prev = snap


def set_decoder_frozen(
    man: RunManager,
    model: nn.Module,
    *,
    frozen: bool,
    epoch: int,
    where: str,
) -> None:
    """
    Freeze/unfreeze model.decoder parameters (AR mode ablation).

    Args:
        man: RunManager instance.
        model: Model with decoder attribute.
        frozen: True to freeze, False to unfreeze.
        epoch: Current epoch.
        where: Description of call location for logging.
    """
    dec = getattr(model, "decoder", None)
    if dec is None:
        return

    want_trainable = not bool(frozen)
    cur_trainable = any(p.requires_grad for p in dec.parameters())
    
    if cur_trainable == want_trainable and bool(getattr(man.cfgs, "decoder_frozen", False)) == bool(frozen):
        return

    for p in dec.parameters():
        p.requires_grad = want_trainable
    man.cfgs.decoder_frozen = bool(frozen)

    logger.info(
        "[Freeze] decoder {} at epoch {} (freeze_decoder_epochs={}) | where={}",
        "FROZEN" if frozen else "UNFROZEN",
        int(epoch),
        int(getattr(man.cfgs, "freeze_decoder_epochs", 0) or 0),
        where,
    )

    maybe_log_trainability(man, model, epoch=epoch, where=f"{where}_after_decoder_freeze")


def log_decoder_pretrain_load(
    model: nn.Module,
    *,
    ckpt_path: str,
    state: dict,
) -> None:
    """
    Debug-friendly decoder-only checkpoint load into model.decoder.
    
    Args:
        model: Model with decoder attribute.
        ckpt_path: Path to checkpoint (for logging).
        state: State dict to load.
    """
    dec = getattr(model, "decoder", None)
    if dec is None:
        logger.warning("[DecoderPretrain] no model.decoder found; cannot load {}", ckpt_path)
        return

    tgt_sd = dec.state_dict()
    src_sd = state
    if not isinstance(src_sd, dict):
        raise ValueError("[DecoderPretrain] checkpoint state is not a state_dict dict")

    common = sorted(set(tgt_sd.keys()).intersection(src_sd.keys()))
    missing = sorted(set(tgt_sd.keys()).difference(src_sd.keys()))
    unexpected = sorted(set(src_sd.keys()).difference(tgt_sd.keys()))
    shape_mismatch = [
        k for k in common
        if hasattr(tgt_sd[k], "shape") and hasattr(src_sd[k], "shape")
        and tuple(tgt_sd[k].shape) != tuple(src_sd[k].shape)
    ]

    logger.info(
        "[DecoderPretrain] loading decoder weights | ckpt={} | common_keys={} | missing={} | unexpected={} | shape_mismatch={}",
        ckpt_path, len(common), len(missing), len(unexpected), len(shape_mismatch),
    )

    if shape_mismatch:
        show = shape_mismatch[:20]
        msg = "\n".join([
            f"  {k}: ckpt={tuple(src_sd[k].shape)} vs model={tuple(tgt_sd[k].shape)}"
            for k in show
        ])
        raise ValueError(
            "[DecoderPretrain] shape mismatch when loading pretrained decoder.\n"
            "This usually means vocab size / gating / architecture differs.\n"
            f"First mismatches (up to 20):\n{msg}"
        )

    res = dec.load_state_dict(src_sd, strict=False)
    miss2 = list(getattr(res, "missing_keys", []))
    unexp2 = list(getattr(res, "unexpected_keys", []))
    
    if miss2 or unexp2:
        logger.warning(
            "[DecoderPretrain] loaded with strict=False | missing_keys={} unexpected_keys={} (showing up to 20)",
            miss2[:20], unexp2[:20],
        )
    else:
        logger.info("[DecoderPretrain] loaded cleanly (strict=False, no missing/unexpected keys)")


def maybe_log_optimizer_coverage(
    man: RunManager,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    epoch: int,
    where: str,
) -> None:
    """
    Log whether optimizer param_groups cover encoder/decoder trainable params.

    Catches the common gotcha: you unfreeze something, but the optimizer
    was created only from a subset of params.
    
    Args:
        man: RunManager instance.
        optimizer: Optimizer to check.
        model: Model with encoder/decoder attributes.
        epoch: Current epoch.
        where: Description of call location for logging.
    """
    enc = getattr(model, "encoder", None)
    dec = getattr(model, "decoder", None)
    if enc is None and dec is None:
        return

    enc_all = list(enc.parameters()) if enc is not None else []
    dec_all = list(dec.parameters()) if dec is not None else []

    enc_ids_all = {id(p) for p in enc_all}
    dec_ids_all = {id(p) for p in dec_all}

    enc_ids_train = {id(p) for p in enc_all if p.requires_grad}
    dec_ids_train = {id(p) for p in dec_all if p.requires_grad}

    opt_ids: set = set()
    for g in optimizer.param_groups:
        for p in g.get("params", []):
            opt_ids.add(id(p))

    missing_enc = sorted(enc_ids_train.difference(opt_ids))
    missing_dec = sorted(dec_ids_train.difference(opt_ids))

    # Per-group breakdown
    group_summaries = []
    for i, g in enumerate(optimizer.param_groups):
        params = g.get("params", [])
        gid = g.get("name", str(i))
        lr = g.get("lr", None)
        group_ids = {id(p) for p in params}
        enc_cov = sum(p.numel() for p in params if id(p) in enc_ids_all)
        dec_cov = sum(p.numel() for p in params if id(p) in dec_ids_all)
        tot_cov = sum(p.numel() for p in params)
        group_summaries.append(f"{gid}: lr={lr} total={tot_cov} enc={enc_cov} dec={dec_cov}")

    snap = (
        len(optimizer.param_groups),
        tuple(sorted(float(g.get("lr", 0.0)) for g in optimizer.param_groups)),
        len(missing_enc),
        len(missing_dec),
        tuple(group_summaries),
    )
    prev = getattr(man, "_dbg_opt_prev", None)
    
    if prev != snap:
        logger.info(
            "[OptCoverage][{}] epoch={} | groups={} | missing_trainable enc={} dec={} | {}",
            where, epoch, len(optimizer.param_groups), len(missing_enc), len(missing_dec),
            " | ".join(group_summaries[:8]),
        )
        if len(group_summaries) > 8:
            logger.info("[OptCoverage][{}] (more groups omitted: {})", where, len(group_summaries) - 8)
        if missing_enc or missing_dec:
            logger.warning(
                "[OptCoverage][{}] WARNING: some trainable params are NOT in the optimizer. "
                "missing_enc={} missing_dec={}",
                where, len(missing_enc), len(missing_dec),
            )
        man._dbg_opt_prev = snap
