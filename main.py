"""
Main entry point for IMU Handwriting Recognition (HWR) training and evaluation.

This script provides:
- Training for CTC, Autoregressive (AR), and Language Model (LM) modes
- Evaluation with optional qualitative analysis (attention, Grad-CAM)
- Checkpoint management with best-model saving
- Cross-validation fold support

Usage:
    python main.py -c configs/train.yaml      # Training
    python main.py -c configs/test.yaml       # Evaluation

Configuration:
    See configs/ directory for example YAML configurations.
"""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# Core modules
from rewi.ctc_decoder import BestPath
from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.dataset_concat import ConcatWordDataset, concat_collate
from rewi.loss import CTCLoss
from rewi.manager import RunManager
from rewi.model import BaseModel, build_encoder
from rewi.utils import seed_everything, seed_worker

# LM-specific modules
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig
from rewi.dataset.lm_collate import lm_collate

# Training loops and utilities
from rewi.training import (
    train_one_epoch,
    train_one_epoch_lm,
    test,
    test_lm,
    maybe_log_trainability,
    maybe_log_optimizer_coverage,
    set_decoder_frozen,
    log_decoder_pretrain_load,
)

# Analysis (for qualitative evaluation)
from rewi.analysis import (
    load_partB_selection_unified_quantile,
    load_partB_selection_by_indices,
)

# Tokenizer
from rewi.tokenizer import BPETokenizer

warnings.filterwarnings('ignore', category=UserWarning)


def setup_tokenizer(cfgs: argparse.Namespace):
    """
    Set up tokenizer based on configuration.
    
    Returns:
        tok: Tokenizer instance or None for character mode.
        vocab_dec: Vocabulary size for decoder.
        PAD_ID, BOS_ID, EOS_ID: Special token IDs.
    """
    AR_MODE = cfgs.arch_de in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
    
    tok = None
    if getattr(cfgs, 'use_bpe', False):
        tok = BPETokenizer(cfgs.tokenizer['model'])
        cfgs.tokenizer_model_path = cfgs.tokenizer['model']
    
    if tok is not None:
        vocab_dec = tok.vocab_size
        PAD_ID, BOS_ID, EOS_ID = tok.PAD, tok.BOS, tok.EOS
    else:
        # Character mode: categories + 3 specials
        base = len(cfgs.categories)
        PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
        vocab_dec = base + (3 if AR_MODE else 0)
    
    return tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID


def build_model(cfgs: argparse.Namespace, manager: RunManager):
    """
    Build model based on architecture configuration.
    
    Returns:
        model: Model instance moved to device.
    """
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
            local_files_only=bool(getattr(cfgs, "lm_local_files_only", True)),
        )

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
        model = BaseModel(
            cfgs.arch_en,
            cfgs.arch_de,
            cfgs.num_channel,
            cfgs.vocab_dec,
            cfgs.len_seq,
            use_gated_attention=getattr(cfgs, "use_gated_attention", False),
            gating_type=getattr(cfgs, "gating_type", "elementwise"),
        ).to(cfgs.device)

        # Optional pretrained decoder initialization
        pretrained_dec_ckpt = getattr(cfgs, "pretrained_decoder_checkpoint", None)
        if pretrained_dec_ckpt:
            logger.info(
                "[DecoderPretrain] init requested | arch_de={} gated={} gating_type={} vocab_dec={}",
                cfgs.arch_de, getattr(cfgs, "use_gated_attention", False),
                getattr(cfgs, "gating_type", None), cfgs.vocab_dec,
            )
            ckp = torch.load(pretrained_dec_ckpt, map_location="cpu", weights_only=False)
            state = ckp.get("model", ckp)
            log_decoder_pretrain_load(model, ckpt_path=str(pretrained_dec_ckpt), state=state)
            maybe_log_trainability(manager, model, epoch=0, where="after_pretrained_decoder_load")

    return model


def build_dataloaders(cfgs: argparse.Namespace, model, tok, LM_MODE: bool):
    """
    Build training and test dataloaders.
    
    Returns:
        dataloader_train: Training dataloader (or None in test mode).
        dataloader_test: Test/validation dataloader.
        train_batch_size: Effective training batch size.
    """
    # Test dataset
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
            max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
        )

    dataloader_test = DataLoader(
        dataset_test,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=collate_test,
    )

    # Training dataset (if not test mode)
    dataloader_train = None
    train_batch_size = cfgs.size_batch
    
    if not cfgs.test:
        base_train = HRDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.categories,
            model.ratio_ds,
            cfgs.idx_fold,
            cfgs.len_seq,
            cfgs.aug,
            cfgs.cache,
        )
        base_train.tokenizer = tok

        # Optional concatenation
        concat_cfg = getattr(cfgs, "concat", {}) or {}
        if concat_cfg.get("enabled", False):
            dataset_train = ConcatWordDataset(
                base_ds=base_train,
                items_min=int(concat_cfg.get("items_min", 2)),
                items_max=int(concat_cfg.get("items_max", 4)),
                max_T=int(concat_cfg.get("max_T", 4096)),
                use_separator=False,
                sep_id=None,
                pad_id=cfgs.PAD_ID,
            )
            collate_train = lambda batch: concat_collate(
                batch, pad_value_x=0.0, pad_value_y=cfgs.PAD_ID
            )
            train_batch_size = int(concat_cfg.get("batch_size", cfgs.size_batch))
        else:
            dataset_train = base_train
            collate_train = fn_collate

        # Wrap collate for LM mode
        if LM_MODE:
            base_collate = collate_train
            hf_tok = model.lm.tokenizer
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

        # Log dataloader info
        _log_dataloader_info(cfgs, dataset_train, dataloader_train, base_train)

    return dataloader_train, dataloader_test, train_batch_size


def _log_dataloader_info(cfgs, dataset_train, dataloader_train, base_train):
    """Log information about the training dataloader."""
    cc = getattr(cfgs, "concat", {}) or {}
    logger.info(
        "Train dataset: {} | concat.enabled={} | items_min={} | items_max={} | max_T={}",
        dataset_train.__class__.__name__,
        cc.get("enabled"), cc.get("items_min"), cc.get("items_max"), cc.get("max_T"),
    )

    # Probe first batch
    try:
        _x, _y, _len_x, _len_y = next(iter(dataloader_train))
        _safe_tolist = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
        logger.info(
            "[ConcatProbe] x={} y={} | len_x[:8]={} | len_y[:8]={}",
            tuple(_x.shape), tuple(_y.shape),
            _safe_tolist(_len_x[:8]), _safe_tolist(_len_y[:8]),
        )
        _lx = _len_x.cpu().numpy()
        _ly = _len_y.cpu().numpy()
        logger.info(
            "[ConcatProbe] len_x mean/median/max = {:.1f}/{:.1f}/{} ; len_y mean/median/max = {:.1f}/{:.1f}/{}",
            _lx.mean(), np.median(_lx), int(_lx.max()),
            _ly.mean(), np.median(_ly), int(_ly.max()),
        )
    except (StopIteration, Exception) as e:
        logger.warning("[ConcatProbe] skipped: {}", e)


def build_optimizer_and_scheduler(cfgs, model, dataloader_train, LM_MODE: bool):
    """
    Build optimizer and learning rate scheduler.
    
    Returns:
        optimizer: Optimizer instance.
        scaler: GradScaler for mixed precision.
        lr_scheduler: Learning rate scheduler.
    """
    if LM_MODE:
        return _build_lm_optimizer(cfgs, model, dataloader_train)
    else:
        return _build_standard_optimizer(cfgs, model, dataloader_train)


def _build_lm_optimizer(cfgs, model, dataloader_train):
    """Build discriminative LR optimizer for LM mode."""
    lr_enc = float(getattr(cfgs, "lr_enc", 1e-4))
    lr_proj = float(getattr(cfgs, "lr_proj", 1e-4))
    lr_lm = float(getattr(cfgs, "lr_lm", 1e-5))
    wd = float(getattr(cfgs, "weight_decay", 0.01))

    param_groups = []

    # Encoder
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        param_groups.append({"name": "enc", "params": enc_params, "lr": lr_enc})

    # Projection
    proj_params = [p for p in model.proj.parameters() if p.requires_grad]
    if proj_params:
        param_groups.append({"name": "proj", "params": proj_params, "lr": lr_proj})

    # LM decoder
    hf = model.lm.lm
    dec_params = []
    for name, p in hf.named_parameters():
        if name.startswith("decoder.") or name.startswith("lm_head") or name.startswith("shared"):
            dec_params.append(p)
    if dec_params:
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

    if dataloader_train is not None:
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return optimizer, scaler, lr_scheduler


def _build_standard_optimizer(cfgs, model, dataloader_train):
    """Build standard optimizer for CTC/AR modes."""
    optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
    scaler = GradScaler()

    if dataloader_train is not None:
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return optimizer, scaler, lr_scheduler


def load_checkpoint(cfgs, model, optimizer, lr_scheduler, manager, dataloader_train, epoch_start):
    """
    Load checkpoint and handle resume/freeze/finetune modes.
    
    Returns:
        Updated epoch_start, optimizer, lr_scheduler.
    """
    if not cfgs.checkpoint:
        return epoch_start, optimizer, lr_scheduler

    map_loc = torch.device("cpu") if str(cfgs.device) == "cpu" or not torch.cuda.is_available() else None
    ckp = torch.load(cfgs.checkpoint, map_location=map_loc, weights_only=False)
    res = model.load_state_dict(ckp["model"], strict=False)
    logger.warning("load_state_dict strict=False | missing={} unexpected={}", res.missing_keys, res.unexpected_keys)

    if not cfgs.test:
        if 'epoch' in ckp.keys():  # Resume
            epoch_start = ckp['epoch'] + 1
            optimizer.load_state_dict(ckp['optimizer'])
            lr_scheduler.load_state_dict(ckp['lr_scheduler'])
            logger.info("[Resume] epoch_start={} | checkpoint={}", epoch_start, cfgs.checkpoint)
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_resume")
            
        elif cfgs.freeze:  # Freeze encoder
            for params in model.encoder.parameters():
                params.requires_grad = False
            logger.info("[Freeze] encoder frozen | checkpoint={}", cfgs.checkpoint)
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_freeze")
            
        else:  # Finetune with discriminative LR
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': cfgs.lr * 0.1},
                {'params': model.decoder.parameters(), 'lr': cfgs.lr},
            ])
            lr_scheduler = SequentialLR(
                optimizer,
                [
                    LinearLR(optimizer, 0.01, total_iters=len(dataloader_train) * cfgs.epoch_warmup),
                    CosineAnnealingLR(optimizer, len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup)),
                ],
                [len(dataloader_train) * cfgs.epoch_warmup],
            )
            maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_finetune")

        maybe_log_trainability(manager, model, epoch=epoch_start, where="after_checkpoint_load")

    logger.info(f'Load checkpoint from {cfgs.checkpoint}')
    return epoch_start, optimizer, lr_scheduler


def build_qualitative_config(cfgs) -> dict:
    """Build qualitative analysis configuration from cfgs."""
    if not getattr(cfgs, "qualitative", False):
        return None

    task = getattr(cfgs, "qual_task", "word")
    by_fold = getattr(cfgs, "qual_indices_by_fold", None)
    indices = by_fold.get(int(cfgs.idx_fold), []) if by_fold else None

    if indices:
        selection_map = load_partB_selection_by_indices(
            unified_csv_path=cfgs.qual_csv,
            fold=int(cfgs.idx_fold),
            task_name=task,
            indices=[int(i) for i in indices],
        )
    else:
        selection_map = load_partB_selection_unified_quantile(
            unified_csv_path=cfgs.qual_csv,
            fold=int(cfgs.idx_fold),
            task_name=task,
            n_correct=int(cfgs.qual_n_correct),
            n_nearmiss=int(cfgs.qual_n_nearmiss),
            n_catastrophic=int(cfgs.qual_n_catastrophic),
            seed=int(cfgs.qual_seed),
        )

    return {
        "enabled": True,
        "selection_map": selection_map,
        "outdir": getattr(cfgs, "qual_outdir", "qual_partB"),
        "use_gradcam": getattr(cfgs, "qual_use_gradcam", False),
        "gradcam_layer": getattr(cfgs, "qual_gradcam_layer", "layers.11.pwconv"),
    }


def run_training_loop(
    cfgs, model, optimizer, scaler, lr_scheduler, manager,
    dataloader_train, dataloader_test, ctc_decoder, tok,
    epoch_start, AR_MODE, LM_MODE
):
    """Main training loop."""
    fn_loss = (
        nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        if AR_MODE else CTCLoss()
    )

    # LM state tracking helpers
    if LM_MODE:
        lm_helpers = _setup_lm_helpers(optimizer, lr_scheduler, model)
    else:
        lm_helpers = None

    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            _run_test_epoch(cfgs, model, fn_loss, manager, dataloader_test, ctc_decoder, tok, LM_MODE)
            break

        # Handle decoder freezing (AR ablation)
        if not LM_MODE and hasattr(model, "decoder"):
            freeze_dec_epochs = int(getattr(cfgs, "freeze_decoder_epochs", 0) or 0)
            if freeze_dec_epochs > 0:
                frozen = int(e) < freeze_dec_epochs
                set_decoder_frozen(manager, model, frozen=frozen, epoch=int(e), where="main_loop")
                maybe_log_optimizer_coverage(manager, optimizer, model, epoch=e, where="after_decoder_freeze_toggle")

        # Handle LM unfreezing
        if LM_MODE and not cfgs.test:
            _maybe_unfreeze_lm(cfgs, model, optimizer, lr_scheduler, e, lm_helpers)

        # Train
        if LM_MODE:
            train_one_epoch_lm(dataloader_train, model, optimizer, scaler, lr_scheduler, manager, e)
        else:
            train_one_epoch(dataloader_train, model, fn_loss, optimizer, scaler, lr_scheduler, manager, e)

        # Evaluate
        if LM_MODE:
            test_lm(dataloader_test, model, manager, e)
        else:
            test(dataloader_test, model, fn_loss, manager, ctc_decoder, e, tokenizer=tok)

        # Save checkpoints
        _save_checkpoints(cfgs, model, optimizer, lr_scheduler, manager, e)

    if not cfgs.test:
        manager.summarize_evaluation()


def _run_test_epoch(cfgs, model, fn_loss, manager, dataloader_test, ctc_decoder, tok, LM_MODE):
    """Run a single test/evaluation epoch."""
    qual_cfg = build_qualitative_config(cfgs)

    if LM_MODE:
        test_lm(dataloader_test, model, manager, 0)
    else:
        test(
            dataloader_test, model, fn_loss, manager, ctc_decoder, 0,
            tokenizer=tok, force_eval=True, qual_cfg=qual_cfg,
        )
    manager.summarize_evaluation()


def _setup_lm_helpers(optimizer, lr_scheduler, model):
    """Set up helper functions for LM training."""
    def _iter_schedulers(sched):
        if hasattr(sched, "_schedulers"):
            return list(getattr(sched, "_schedulers"))
        return [sched]

    def _find_group_idx_by_name(opt, name):
        for i, g in enumerate(opt.param_groups):
            if g.get("name") == name:
                return i
        return None

    def _log_lm_state(tag, opt):
        hf = model.lm.lm
        trainable, total = 0, 0
        for n, p in hf.named_parameters():
            if n.startswith("decoder.") or n.startswith("lm_head") or n.startswith("shared"):
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
        idx = _find_group_idx_by_name(opt, "lm_dec")
        lr = opt.param_groups[idx]["lr"] if idx is not None else None
        logger.info("[LM][{}] lm_dec trainable={}/{} params | opt_lr={}", tag, trainable, total, lr)

    def _set_group_lr_and_base(opt, sched, idx, new_lr):
        opt.param_groups[idx]["lr"] = float(new_lr)
        for s in _iter_schedulers(sched):
            if hasattr(s, "base_lrs") and idx < len(s.base_lrs):
                s.base_lrs[idx] = float(new_lr)

    return {
        "find_group_idx": _find_group_idx_by_name,
        "log_state": _log_lm_state,
        "set_lr": _set_group_lr_and_base,
        "iter_schedulers": _iter_schedulers,
    }


def _maybe_unfreeze_lm(cfgs, model, optimizer, lr_scheduler, e, helpers):
    """Handle LM decoder unfreezing at specified epoch."""
    lm_unfreeze_epoch = getattr(cfgs, "lm_unfreeze_epoch", None)
    if lm_unfreeze_epoch is None or int(lm_unfreeze_epoch) < 0:
        return
    if int(e) != int(lm_unfreeze_epoch):
        return

    model.lm.set_decoder_trainable(True)
    helpers["log_state"](f"unfreeze@{e}", optimizer)

    mult = float(getattr(cfgs, "lm_lr_unfreeze_mult", 1.0))
    idx = helpers["find_group_idx"](optimizer, "lm_dec")
    
    if idx is not None and mult > 0 and mult != 1.0:
        old_lr = float(optimizer.param_groups[idx]["lr"])
        new_lr = old_lr * mult
        helpers["set_lr"](optimizer, lr_scheduler, idx, new_lr)
        logger.info("[LM] Unfreeze LR bump: {} -> {} (mult={})", old_lr, new_lr, mult)
    
    logger.info("[LM] Unfroze LM decoder at epoch {}", e)


def _save_checkpoints(cfgs, model, optimizer, lr_scheduler, manager, e):
    """Save best and last checkpoints."""
    # Best checkpoints based on metrics
    if bool(getattr(cfgs, "save_best_only", False)):
        manager.summarize_evaluation()
        best = manager.results.get("best", {})
        if isinstance(best, dict):
            if "character_error_rate" in best and int(best["character_error_rate"][0]) == int(e):
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_cer.pth"
                )
            if "word_error_rate" in best and int(best["word_error_rate"][0]) == int(e):
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_wer.pth"
                )

    # Always save last checkpoint for resuming
    if not cfgs.test:
        manager.save_checkpoint(
            model.state_dict(), optimizer.state_dict(),
            lr_scheduler.state_dict(), filename="last.pth"
        )


def main(cfgs: argparse.Namespace) -> None:
    """
    Main function for training and evaluation.

    Args:
        cfgs: Configuration namespace from YAML file.
    """
    # Determine training regime
    AR_MODE = cfgs.arch_de in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
    LM_MODE = cfgs.arch_de in {"byt5_small", "t5-small"}

    # Set up tokenizer
    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfgs)

    # Attach to cfgs
    cfgs.AR_MODE = AR_MODE
    cfgs.PAD_ID, cfgs.BOS_ID, cfgs.EOS_ID = PAD_ID, BOS_ID, EOS_ID
    cfgs.vocab_dec = vocab_dec
    cfgs.tokenizer_obj = tok

    # Initialize
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    # Build model
    model = build_model(cfgs, manager)

    # Build dataloaders
    dataloader_train, dataloader_test, _ = build_dataloaders(cfgs, model, tok, LM_MODE)

    # Build optimizer and scheduler
    optimizer, scaler, lr_scheduler = build_optimizer_and_scheduler(
        cfgs, model, dataloader_train, LM_MODE
    )

    # Debug: verify optimizer coverage
    epoch_start = 0
    maybe_log_optimizer_coverage(manager, optimizer, model, epoch=epoch_start, where="after_optimizer_init")

    # Load checkpoint
    epoch_start, optimizer, lr_scheduler = load_checkpoint(
        cfgs, model, optimizer, lr_scheduler, manager, dataloader_train, epoch_start
    )

    # Handle LM staged fine-tuning (resume case)
    if LM_MODE and not cfgs.test:
        lm_unfreeze_epoch = getattr(cfgs, "lm_unfreeze_epoch", None)
        if lm_unfreeze_epoch is not None and int(lm_unfreeze_epoch) >= 0 and epoch_start >= int(lm_unfreeze_epoch):
            model.lm.set_decoder_trainable(True)
            logger.info("[LM] Resumed at epoch {} => LM decoder UNFROZEN", epoch_start)

    # Run training/evaluation loop
    run_training_loop(
        cfgs, model, optimizer, scaler, lr_scheduler, manager,
        dataloader_train, dataloader_test, ctc_decoder, tok,
        epoch_start, AR_MODE, LM_MODE
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IMU Handwriting Recognition - Training and Evaluation'
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Path to YAML configuration file.'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
