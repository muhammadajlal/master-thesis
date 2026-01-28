"""
Training and evaluation loop implementations.
"""

import contextlib
import json
import os
from typing import Optional

import torch
import torch.nn as nn
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
from rewi.training.utils import build_ar_batch, maybe_log_trainability
from rewi.visualize import visualize

# Optional pandas import (only needed for qualitative analysis)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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
    amp_dtype = torch.float16  # V100-friendly

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

        optimizer.zero_grad(set_to_none=True)

        # Forward with AMP
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
            if x.is_cuda
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            out = model(x, len_x, labels=labels)
            loss = out.loss

        # Skip non-finite losses
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
                    "Non-finite grad norm. epoch={} iter={} grad_norm={}",
                    epoch, idx, grad_norm
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
                optimizer.zero_grad(set_to_none=True)
                continue
                
            optimizer.step()

        lr_scheduler.step()
        man.update_iteration(idx, float(loss.item()), lr_scheduler.get_last_lr()[0])

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
    
    if man.check_step(epoch + 1, 'eval'):
        results_eval = evaluate(preds, labels_txt)
        man.update_evaluation(results_eval, preds[:20], labels_txt[:20])


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

    for idx, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(man.cfgs.device), y.to(man.cfgs.device)
        optimizer.zero_grad()

        with torch.autocast('cuda', torch.float16):
            if isinstance(fn_loss, nn.CrossEntropyLoss):  # AR mode
                y_inp, y_tgt = build_ar_batch(y, len_y, PAD_ID, BOS_ID, EOS_ID, device=man.cfgs.device)
                logits = model(x, in_lengths=len_x, y_inp=y_inp)
                loss = fn_loss(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
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
            if man.check_step(epoch + 1, 'eval') and not isinstance(fn_loss, nn.CrossEntropyLoss):
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
    do_export = is_test_mode or (export_val_full and man.check_step(epoch + 1, 'eval'))

    if do_export:
        export_dir = os.path.join(man.cfgs.dir_work, "exports")
        os.makedirs(export_dir, exist_ok=True)
        epoch_tag = "best" if epoch is None else f"epoch{epoch}"
        export_path = os.path.join(export_dir, f"val_full_fold{man.cfgs.idx_fold}_{epoch_tag}.json")
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds, "labels": labels}, f, ensure_ascii=False)
        logger.info("Exported full validation predictions to {}", export_path)

    # Evaluation and visualization
    if man.check_step(epoch + 1, 'eval'):
        if not isinstance(fn_loss, nn.CrossEntropyLoss):
            visualize(preds, labels, man.cfgs.categories[1:], man.dir_vis, epoch)
        results_eval = evaluate(preds, labels)
        man.update_evaluation(results_eval, preds[:20], labels[:20])


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
