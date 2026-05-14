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
from rewi.model import DualHeadModel
from rewi.utils import seed_everything, seed_worker

# LM-specific modules
from rewi.model.multimodal_lm_model import MultimodalLMModel
from rewi.model.pretrainedLM import LMConfig
from rewi.dataset.lm_collate import lm_collate, vlm_collate

# VLM-specific modules
from rewi.model.vlm_model import VLMModel

# Training loops and utilities
from rewi.training import (
    train_one_epoch,
    train_one_epoch_hybrid,
    train_one_epoch_lm,
    train_one_epoch_lm_hybrid,
    test,
    test_hybrid,
    test_lm,
    test_lm_hybrid,
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
    AR_MODE = cfgs.arch_de in {"ar_transformer_xs", "ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
    
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
    VLM_MODE = bool(getattr(cfgs, "vlm_enabled", False))

    if VLM_MODE:
        encoder = build_encoder(cfgs.num_channel, cfgs.arch_en, cfgs.len_seq).to(cfgs.device)
        ratio_ds = int(getattr(encoder, "ratio_ds", 1))
        d_cnn = int(getattr(cfgs, "d_cnn", encoder.dim_out))

        vlm_cfg = getattr(cfgs, "vlm", {}) or {}
        lm_name = str(vlm_cfg.get("lm_name", "gpt2"))

        model = VLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_name_or_path=lm_name,
            connector_type=str(vlm_cfg.get("connector_type", "qformer")),
            num_queries=int(vlm_cfg.get("num_queries", 32)),
            qformer_layers=int(vlm_cfg.get("qformer_layers", 4)),
            qformer_nhead=int(vlm_cfg.get("qformer_nhead", 8)),
            qformer_dropout=float(vlm_cfg.get("qformer_dropout", 0.1)),
            prompt_text=vlm_cfg.get("prompt_text", "Transcribe the handwritten text from IMU sensor signals:"),
            refine_prompt_text=vlm_cfg.get("refine_prompt_text", None),
            two_step_decode=bool(vlm_cfg.get("two_step_decode", False)),
            num_soft_tokens=int(vlm_cfg.get("num_soft_tokens", 20)),
            freeze_encoder=bool(getattr(cfgs, "freeze", False)),
            freeze_lm=bool(vlm_cfg.get("freeze_lm", True)),
            random_init_lm=bool(vlm_cfg.get("random_init_lm", False)),
            use_lora=bool(vlm_cfg.get("use_lora", False)),
            lora_r=int(vlm_cfg.get("lora_r", 16)),
            lora_alpha=int(vlm_cfg.get("lora_alpha", 32)),
            lora_dropout=float(vlm_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=vlm_cfg.get("lora_target_modules", None),
            max_new_tokens=int(vlm_cfg.get("max_new_tokens", 64)),
            num_beams=int(vlm_cfg.get("num_beams", 1)),
            repetition_penalty=float(vlm_cfg.get("repetition_penalty", 1.0)),
            local_files_only=bool(vlm_cfg.get("local_files_only", True)),
            z_dropout=float(vlm_cfg.get("z_dropout", 0.1)),
            label_smoothing=float(vlm_cfg.get("label_smoothing", 0.0)),
            vocab_ctc=int(len(cfgs.categories)) if bool(vlm_cfg.get("hybrid_ctc", False)) else None,
            categories=getattr(cfgs, "categories", None),
        ).to(cfgs.device)

        # Attach ratio_ds for dataloader compatibility
        model.ratio_ds = ratio_ds

        # VLM hybrid CTC+LM mode
        vlm_hybrid_enabled = bool(vlm_cfg.get("hybrid_ctc", False))
        cfgs.LM_HYBRID = vlm_hybrid_enabled
        if vlm_hybrid_enabled:
            cfgs.lm_hybrid = {
                "enabled": True,
                "lambda_ctc": float(vlm_cfg.get("hybrid_lambda_ctc", 0.6)),
            }

    elif LM_MODE:
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
            use_lora=bool(getattr(cfgs, "lm_use_lora", False)),
            lora_r=int(getattr(cfgs, "lm_lora_r", 16)),
            lora_alpha=int(getattr(cfgs, "lm_lora_alpha", 32)),
            lora_dropout=float(getattr(cfgs, "lm_lora_dropout", 0.05)),
            lora_target_modules=str(getattr(cfgs, "lm_lora_target_modules", r"decoder\..*\.(q|v)")),
            unfreeze_xattn_k=bool(getattr(cfgs, "lm_unfreeze_xattn_k", False)),
        )

        d_cnn = int(getattr(cfgs, "d_cnn", 0))

        # Check for hybrid CTC+LM mode
        lm_hybrid_cfg = getattr(cfgs, "lm_hybrid", {}) or {}
        lm_hybrid_enabled = bool(lm_hybrid_cfg.get("enabled", False))
        vocab_ctc = int(len(cfgs.categories)) if lm_hybrid_enabled else None

        model = MultimodalLMModel(
            encoder=encoder,
            ratio_ds=ratio_ds,
            d_cnn=d_cnn,
            lm_cfg=lm_cfg,
            proj_dropout=float(getattr(cfgs, "lm_proj_dropout", 0.0)),
            freeze_encoder=bool(getattr(cfgs, "freeze", True)),
            vocab_ctc=vocab_ctc,
            num_soft_tokens=int(getattr(cfgs, "lm_num_soft_tokens", 0)),
            prompt_text=str(getattr(cfgs, "lm_prompt_text", "")),
        ).to(cfgs.device)

        cfgs.LM_HYBRID = lm_hybrid_enabled
    else:
        dual = getattr(cfgs, "dual_head", {}) or {}
        dual_enabled = bool(dual.get("enabled", False))

        if dual_enabled:
            if getattr(cfgs, "use_bpe", False):
                raise ValueError("dual_head.enabled currently supports character mode only (use_bpe must be false)")

            vocab_ctc = int(len(cfgs.categories))
            arch_ctc = str(dual.get("arch_ctc", "linear"))
            tie_cfg = dual.get("tie", {}) or {}

            model = DualHeadModel(
                arch_en=cfgs.arch_en,
                arch_ar=cfgs.arch_de,
                arch_ctc=arch_ctc,
                in_chan=cfgs.num_channel,
                vocab_ar=cfgs.vocab_dec,
                vocab_ctc=vocab_ctc,
                len_seq=cfgs.len_seq,
                use_gated_attention=getattr(cfgs, "use_gated_attention", False),
                gating_type=getattr(cfgs, "gating_type", "elementwise"),
                pad_id=getattr(cfgs, "PAD_ID", None),
                bos_id=getattr(cfgs, "BOS_ID", None),
                eos_id=getattr(cfgs, "EOS_ID", None),
                tie_cfg=tie_cfg,
                dual_cfg=dual,
            ).to(cfgs.device)
            cfgs.DUAL_HEAD = True
            cfgs.vocab_ctc = vocab_ctc
            cfgs.dual_head_arch_ctc = arch_ctc
        else:
            # AR-only mode: optionally enable decoder-side CTC (CTC regularization
            # "between decoder layers" using the AR vocab projection prefix).
            dec_ctc_cfg = getattr(cfgs, "decoder_side_ctc", {}) or {}
            vocab_ctc = int(len(cfgs.categories))  # CTC vocab = base chars + blank

            model = BaseModel(
                cfgs.arch_en,
                cfgs.arch_de,
                cfgs.num_channel,
                cfgs.vocab_dec,
                cfgs.len_seq,
                use_gated_attention=getattr(cfgs, "use_gated_attention", False),
                gating_type=getattr(cfgs, "gating_type", "elementwise"),
                vocab_ctc=vocab_ctc,
                pad_id=getattr(cfgs, "PAD_ID", None),
                decoder_side_ctc_cfg=dec_ctc_cfg,
            ).to(cfgs.device)
            cfgs.DUAL_HEAD = False
            cfgs.vocab_ctc = vocab_ctc

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
    VLM_MODE = bool(getattr(cfgs, "vlm_enabled", False))

    # Test dataset
    dataset_test = HRDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.categories,
        model.ratio_ds,
        cfgs.idx_fold,
        cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataset_test.ignore_unknown_chars = bool(getattr(cfgs, 'ignore_unknown_chars', False))
    dataset_test.return_raw_label = bool(getattr(cfgs, 'return_raw_label', False))

    collate_test = fn_collate
    _vlm_hybrid = bool(getattr(cfgs, "LM_HYBRID", False)) if VLM_MODE else False
    if VLM_MODE:
        hf_tok = model.tokenizer
        collate_test = lambda batch: vlm_collate(
            batch,
            base_collate_fn=fn_collate,
            hf_tokenizer=hf_tok,
            categories=cfgs.categories,
            pad_id=cfgs.PAD_ID,
            max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
            hybrid=_vlm_hybrid,
        )
    elif LM_MODE:
        hf_tok = model.lm.tokenizer
        _lm_hybrid = bool(getattr(cfgs, "LM_HYBRID", False))
        collate_test = lambda batch: lm_collate(
            batch,
            base_collate_fn=fn_collate,
            hf_tokenizer=hf_tok,
            categories=cfgs.categories,
            pad_id=cfgs.PAD_ID,
            max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
            hybrid=_lm_hybrid,
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

        # Wrap collate for LM / VLM mode
        if VLM_MODE:
            base_collate = collate_train
            hf_tok = model.tokenizer
            collate_train = lambda batch: vlm_collate(
                batch,
                base_collate_fn=base_collate,
                hf_tokenizer=hf_tok,
                categories=cfgs.categories,
                pad_id=cfgs.PAD_ID,
                max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
                hybrid=_vlm_hybrid,
            )
        elif LM_MODE:
            base_collate = collate_train
            hf_tok = model.lm.tokenizer
            _lm_hybrid = bool(getattr(cfgs, "LM_HYBRID", False))
            collate_train = lambda batch: lm_collate(
                batch,
                base_collate_fn=base_collate,
                hf_tokenizer=hf_tok,
                categories=cfgs.categories,
                pad_id=cfgs.PAD_ID,
                max_label_len=int(getattr(cfgs, "lm_max_label_len", 128)),
                hybrid=_lm_hybrid,
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
    VLM_MODE = bool(getattr(cfgs, "vlm_enabled", False))
    if VLM_MODE:
        return _build_vlm_optimizer(cfgs, model, dataloader_train)
    elif LM_MODE:
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

    # CTC head (hybrid LM mode)
    if getattr(model, "ctc_head", None) is not None:
        ctc_params = [p for p in model.ctc_head.parameters() if p.requires_grad]
        if ctc_params:
            param_groups.append({"name": "ctc_head", "params": ctc_params, "lr": lr_proj})

    # Soft prompt tokens
    prompt_params = []
    if getattr(model, "prompt", None) is not None:
        prompt_params = [p for p in model.prompt.parameters() if p.requires_grad]
        if prompt_params:
            param_groups.append({"name": "prompt", "params": prompt_params, "lr": lr_proj})

    # LM decoder (LoRA mode: select by requires_grad; full FT: select by name)
    _lm_use_lora = bool(getattr(cfgs, "lm_use_lora", False))
    hf = model.lm.lm
    dec_params = []
    for name, p in hf.named_parameters():
        if _lm_use_lora:
            if p.requires_grad:
                dec_params.append(p)
        else:
            if name.startswith("decoder.") or name.startswith("lm_head") or name.startswith("shared"):
                dec_params.append(p)
    if dec_params:
        param_groups.append({"name": "lm_dec", "params": dec_params, "lr": lr_lm})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    scaler = GradScaler(enabled=bool(getattr(cfgs, "lm_use_amp", False)))

    logger.info(
        "[OptGroups] enc={} proj={} prompt={} lm_dec={} | lr_enc={} lr_proj={} lr_lm={} | lora={}",
        sum(p.numel() for p in enc_params),
        sum(p.numel() for p in proj_params),
        sum(p.numel() for p in prompt_params),
        sum(p.numel() for p in dec_params),
        lr_enc, lr_proj, lr_lm,
        _lm_use_lora,
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


def _initialize_vlm_auxiliary_losses(cfgs, model, optimizer=None):
    """Attach VLM auxiliary-loss modules before checkpoint loading.

    Some VLM experiments create trainable auxiliary modules lazily inside the
    training loop. On resume, their parameters are already present in the
    checkpoint model/optimizer state, so we need to instantiate them before
    loading the checkpoint to keep parameter groups aligned.
    """
    vlm_cfg = getattr(cfgs, "vlm", {}) or {}
    inner_model = getattr(model, "model", model)
    device = getattr(cfgs, "device", "cpu")

    counts = {
        "contrast": 0,
        "embed_mse": 0,
        "ec": 0,
        "sea": 0,
    }

    def _params(module):
        return [p for p in module.parameters() if p.requires_grad]

    def _maybe_add_group(params):
        if optimizer is None or not params:
            return
        opt_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group.get("params", [])
        }
        if any(id(p) not in opt_ids for p in params):
            optimizer.add_param_group({"params": params, "lr": optimizer.defaults["lr"]})

    aux_enabled = False

    lambda_contrast = float(vlm_cfg.get("lambda_contrast", 0.0))
    if lambda_contrast > 0:
        aux_enabled = True
        contrast_loss_fn = getattr(inner_model, "_contrast_loss_fn", None)
        if contrast_loss_fn is None:
            from rewi.training.contrastive import InBatchContrastiveLoss

            init_temp = float(vlm_cfg.get("contrast_temperature", 0.07))
            contrast_loss_fn = InBatchContrastiveLoss(init_temperature=init_temp).to(device)
            inner_model._contrast_loss_fn = contrast_loss_fn
        inner_model._return_text_emb = True
        contrast_params = _params(contrast_loss_fn)
        counts["contrast"] = sum(p.numel() for p in contrast_params)
        _maybe_add_group(contrast_params)

    lambda_embed_mse = float(vlm_cfg.get("lambda_embed_mse", 0.0))
    if lambda_embed_mse > 0:
        aux_enabled = True
        embed_mse_fn = getattr(inner_model, "_embed_mse_fn", None)
        if embed_mse_fn is None:
            from rewi.training.auxiliary_losses import CTCCompressMSELoss

            d_enc = int(getattr(cfgs, "d_cnn", 512))
            d_lm = inner_model.d_lm if hasattr(inner_model, "d_lm") else 768
            embed_mse_fn = CTCCompressMSELoss(d_enc, d_lm).to(device)
            inner_model._embed_mse_fn = embed_mse_fn
        inner_model._return_text_emb = True
        embed_mse_params = _params(embed_mse_fn)
        counts["embed_mse"] = sum(p.numel() for p in embed_mse_params)
        _maybe_add_group(embed_mse_params)

    lambda_ec = float(vlm_cfg.get("lambda_ec", 0.0))
    if lambda_ec > 0:
        aux_enabled = True
        ec_loss_fn = getattr(inner_model, "_ec_loss_fn", None)
        if ec_loss_fn is None:
            from rewi.training.auxiliary_losses import ECContrastiveLoss

            categories = getattr(cfgs, "categories", [])
            vocab_size = len(categories)
            d_lm = inner_model.d_lm if hasattr(inner_model, "d_lm") else 768
            n_neg = int(vlm_cfg.get("ec_n_negatives", 4))
            ec_loss_fn = ECContrastiveLoss(
                vocab_size=vocab_size,
                d_lm=d_lm,
                n_negatives=n_neg,
            ).to(device)
            inner_model._ec_loss_fn = ec_loss_fn
            inner_model._ec_char_to_id = {c: i for i, c in enumerate(categories)}
        ec_params = _params(ec_loss_fn)
        counts["ec"] = sum(p.numel() for p in ec_params)
        _maybe_add_group(ec_params)

    lambda_sea = float(vlm_cfg.get("lambda_sea", 0.0))
    if lambda_sea > 0:
        aux_enabled = True
        sea_loss_fn = getattr(inner_model, "_sea_loss_fn", None)
        if sea_loss_fn is None:
            from rewi.training.auxiliary_losses import SEATokenContrastiveLoss

            sea_loss_fn = SEATokenContrastiveLoss(temperature=0.07).to(device)
            inner_model._sea_loss_fn = sea_loss_fn
        inner_model._return_text_emb = True
        sea_params = _params(sea_loss_fn)
        counts["sea"] = sum(p.numel() for p in sea_params)
        _maybe_add_group(sea_params)

    if aux_enabled:
        inner_model._aux_losses_initialized = True

    return counts


def _build_vlm_optimizer(cfgs, model, dataloader_train):
    """Build discriminative LR optimizer for VLM mode.

    Parameter groups:
        enc       – CNN encoder (if trainable)
        qformer   – Q-Former connector
        prompt    – soft prefix vectors
        lm        – decoder-only LM (LoRA / unfrozen params)
    """
    vlm_cfg = getattr(cfgs, "vlm", {}) or {}
    lr_enc = float(getattr(cfgs, "lr_enc", 1e-4))
    lr_connector = float(vlm_cfg.get("lr_connector", 1e-4))
    lr_prompt = float(vlm_cfg.get("lr_prompt", 1e-4))
    lr_lm = float(vlm_cfg.get("lr_lm", 1e-5))
    wd = float(getattr(cfgs, "weight_decay", 0.01))

    param_groups = []

    # Encoder
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        param_groups.append({"name": "enc", "params": enc_params, "lr": lr_enc})

    # Q-Former / connector
    qf_params = [p for p in model.connector.parameters() if p.requires_grad]
    if qf_params:
        param_groups.append({"name": "connector", "params": qf_params, "lr": lr_connector})

    # CTC head (hybrid VLM mode)
    ctc_params = []
    if getattr(model, "ctc_head", None) is not None:
        ctc_params = [p for p in model.ctc_head.parameters() if p.requires_grad]
        if ctc_params:
            param_groups.append({"name": "ctc_head", "params": ctc_params, "lr": lr_connector})

    # Prompt (soft prefix)
    prompt_params = [p for p in model.prompt_manager.parameters() if p.requires_grad]
    if prompt_params:
        param_groups.append({"name": "prompt", "params": prompt_params, "lr": lr_prompt})

    # LM (LoRA or unfrozen parameters)
    lm_params = [p for p in model.lm.parameters() if p.requires_grad]
    if lm_params:
        param_groups.append({"name": "lm_dec", "params": lm_params, "lr": lr_lm})

    if not param_groups:
        raise ValueError("No trainable parameters — check freeze/LoRA settings")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    aux_counts = _initialize_vlm_auxiliary_losses(cfgs, model, optimizer=optimizer)
    scaler = GradScaler(enabled=bool(getattr(cfgs, "lm_use_amp", False)))

    logger.info(
        "[VLM OptGroups] enc={} connector={} ctc_head={} prompt={} lm={} "
        "contrast={} embed_mse={} ec={} sea={} | "
        "lr_enc={} lr_conn={} lr_prompt={} lr_lm={}",
        sum(p.numel() for p in enc_params),
        sum(p.numel() for p in qf_params),
        sum(p.numel() for p in ctc_params),
        sum(p.numel() for p in prompt_params),
        sum(p.numel() for p in lm_params),
        aux_counts["contrast"],
        aux_counts["embed_mse"],
        aux_counts["ec"],
        aux_counts["sea"],
        lr_enc, lr_connector, lr_prompt, lr_lm,
    )

    if dataloader_train is not None:
        accum = int(getattr(cfgs, "grad_accum_steps", 1))
        steps_per_epoch = max(1, len(dataloader_train) // accum)
        warmup_steps = steps_per_epoch * cfgs.epoch_warmup
        decay_steps = steps_per_epoch * (cfgs.epoch - cfgs.epoch_warmup)
        logger.info(
            "[VLM Scheduler] accum={} steps_per_epoch={} warmup_steps={} decay_steps={}",
            accum, steps_per_epoch, warmup_steps, decay_steps,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, 0.01, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, decay_steps),
            ],
            [warmup_steps],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    return optimizer, scaler, lr_scheduler


def _build_standard_optimizer(cfgs, model, dataloader_train):
    """Build standard optimizer for CTC/AR modes."""
    optimizer = torch.optim.AdamW(model.parameters(), cfgs.lr)
    
    # AMP is configurable via use_amp (default true for backward compat)
    use_amp = bool(getattr(cfgs, "use_amp", True))
    # Use a conservative init_scale to avoid early overflow
    scaler = GradScaler(enabled=use_amp, init_scale=2048.0)
    
    if use_amp:
        logger.info("[AMP] enabled with GradScaler init_scale=2048")
    else:
        logger.info("[AMP] disabled (use_amp=false)")

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


def _migrate_legacy_mha_keys(state_dict: dict) -> dict:
    """Migrate old separate-projection MHA keys to nn.MultiheadAttention format.

    Old format (custom GatedMultiheadAttention with manual projections):
        *.self_attn.q_proj.weight   (D, D)
        *.self_attn.k_proj.weight   (D, D)
        *.self_attn.v_proj.weight   (D, D)
        *.self_attn.q_proj.bias     (D,)
        *.self_attn.k_proj.bias     (D,)
        *.self_attn.v_proj.bias     (D,)
        *.self_attn.out_proj.weight (D, D)
        *.self_attn.out_proj.bias   (D,)

    New format (nn.MultiheadAttention wrapped inside GatedMultiheadAttention):
        *.self_attn.mha.in_proj_weight  (3*D, D)
        *.self_attn.mha.in_proj_bias    (3*D,)
        *.self_attn.mha.out_proj.weight (D, D)
        *.self_attn.mha.out_proj.bias   (D,)

    Applies to both self_attn and multihead_attn (cross-attention).
    """
    import re

    new_sd = {}
    consumed = set()

    # Find all attention module prefixes that have old-style keys
    # Pattern: <prefix>.self_attn.q_proj.weight  or  <prefix>.multihead_attn.q_proj.weight
    old_prefixes = set()
    for k in state_dict:
        m = re.match(r"^(.+\.(self_attn|multihead_attn))\.q_proj\.weight$", k)
        if m:
            old_prefixes.add(m.group(1))

    if old_prefixes:
        logger.info("[MHAMigrate] Found {} attention modules with legacy separate-projection keys", len(old_prefixes))

    for prefix in old_prefixes:
        q_w = state_dict.get(f"{prefix}.q_proj.weight")
        k_w = state_dict.get(f"{prefix}.k_proj.weight")
        v_w = state_dict.get(f"{prefix}.v_proj.weight")
        q_b = state_dict.get(f"{prefix}.q_proj.bias")
        k_b = state_dict.get(f"{prefix}.k_proj.bias")
        v_b = state_dict.get(f"{prefix}.v_proj.bias")
        o_w = state_dict.get(f"{prefix}.out_proj.weight")
        o_b = state_dict.get(f"{prefix}.out_proj.bias")

        if q_w is not None and k_w is not None and v_w is not None:
            # Concatenate Q, K, V into in_proj_weight/bias
            new_sd[f"{prefix}.mha.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update([
                f"{prefix}.q_proj.weight",
                f"{prefix}.k_proj.weight",
                f"{prefix}.v_proj.weight",
            ])

            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{prefix}.mha.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                consumed.update([
                    f"{prefix}.q_proj.bias",
                    f"{prefix}.k_proj.bias",
                    f"{prefix}.v_proj.bias",
                ])

        if o_w is not None:
            new_sd[f"{prefix}.mha.out_proj.weight"] = o_w
            consumed.add(f"{prefix}.out_proj.weight")
        if o_b is not None:
            new_sd[f"{prefix}.mha.out_proj.bias"] = o_b
            consumed.add(f"{prefix}.out_proj.bias")

    # Copy all non-consumed keys unchanged
    for k, v in state_dict.items():
        if k not in consumed:
            new_sd[k] = v

    if old_prefixes:
        logger.info("[MHAMigrate] Migrated {} old keys → {} new keys", len(consumed), len(new_sd) - (len(state_dict) - len(consumed)))

    return new_sd


def _migrate_legacy_vlm_connector_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map older VLM checkpoint prefixes onto the current module names."""
    if not any(k.startswith("qformer.") for k in state_dict):
        return state_dict

    remapped = {}
    migrated = 0
    for key, value in state_dict.items():
        if key.startswith("qformer."):
            remapped[f"connector.{key[len('qformer.'):]}"] = value
            migrated += 1
        else:
            remapped[key] = value

    logger.info("[VLMMigrate] Remapped {} legacy qformer.* keys to connector.*", migrated)
    return remapped


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

    # Migrate legacy separate Q/K/V projection keys to nn.MultiheadAttention format
    ckp["model"] = _migrate_legacy_mha_keys(ckp["model"])
    ckp["model"] = _migrate_legacy_vlm_connector_keys(ckp["model"])

    # Filter out keys with shape mismatches (e.g. CTC head when categories change)
    model_sd = model.state_dict()
    filtered_sd = {}
    shape_skipped = []
    for k, v in ckp["model"].items():
        if k in model_sd and model_sd[k].shape != v.shape:
            shape_skipped.append(k)
        else:
            filtered_sd[k] = v
    if shape_skipped:
        logger.warning("load_state_dict: skipped shape-mismatched keys: {}", shape_skipped)
    res = model.load_state_dict(filtered_sd, strict=False)
    logger.warning("load_state_dict strict=False | missing={} unexpected={}", res.missing_keys, res.unexpected_keys)

    if bool(getattr(cfgs, 'disable_lora_at_test', False)):
        zeroed = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'lora_B' in n:
                    p.zero_()
                    zeroed += 1
        logger.warning("[disable_lora_at_test] zeroed {} lora_B tensors — LoRA deltas now 0, base LM restored", zeroed)

    if not cfgs.test:
        if 'epoch' in ckp.keys():  # Resume
            epoch_start = ckp['epoch'] + 1
            try:
                optimizer.load_state_dict(ckp['optimizer'])
                lr_scheduler.load_state_dict(ckp['lr_scheduler'])
                logger.info("[Resume] epoch_start={} | checkpoint={}", epoch_start, cfgs.checkpoint)
            except (ValueError, KeyError) as e:
                logger.warning(
                    "[Resume] optimizer/scheduler load failed ({}); "
                    "warm-restarting from epoch {} with fresh optimizer",
                    e, epoch_start,
                )
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
    dual = getattr(cfgs, "dual_head", {}) or {}
    dual_enabled = bool(dual.get("enabled", False)) and bool(AR_MODE) and not bool(LM_MODE)

    fn_loss = (
        nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        if AR_MODE else CTCLoss()
    )

    fn_loss_ar = None
    fn_loss_ctc = None
    lambda_ar = 1.0
    lambda_ctc = 0.0
    lambda_ctc_schedule = None
    loss_balance_mode = "sum"
    if dual_enabled:
        fn_loss_ar = nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
        fn_loss_ctc = CTCLoss()
        lambda_ar = float(dual.get("lambda_ar", 1.0))
        lambda_ctc = float(dual.get("lambda_ctc", 0.3))
        lambda_ctc_schedule = dual.get("lambda_ctc_schedule", None)
        loss_balance_mode = str(dual.get("loss_balance", "sum"))

    # LM hybrid (CTC+LM) mode
    lm_hybrid_enabled = bool(getattr(cfgs, "LM_HYBRID", False))
    lm_hybrid_fn_ctc = None
    lm_hybrid_lambda_ctc = 0.0
    if lm_hybrid_enabled:
        lm_hybrid_cfg = getattr(cfgs, "lm_hybrid", {}) or {}
        lm_hybrid_fn_ctc = CTCLoss()
        lm_hybrid_lambda_ctc = float(lm_hybrid_cfg.get("lambda_ctc", 0.6))

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
        if LM_MODE and lm_hybrid_enabled:
            train_one_epoch_lm_hybrid(
                dataloader_train, model, lm_hybrid_fn_ctc, lm_hybrid_lambda_ctc,
                optimizer, scaler, lr_scheduler, manager, e,
            )
        elif LM_MODE:
            train_one_epoch_lm(dataloader_train, model, optimizer, scaler, lr_scheduler, manager, e)
        else:
            if dual_enabled:
                train_one_epoch_hybrid(
                    dataloader_train,
                    model,
                    fn_loss_ar,
                    fn_loss_ctc,
                    lambda_ar=lambda_ar,
                    lambda_ctc=lambda_ctc,
                    lambda_ctc_schedule=lambda_ctc_schedule,
                    loss_balance_mode=loss_balance_mode,
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    man=manager,
                    epoch=e,
                )
            else:
                train_one_epoch(dataloader_train, model, fn_loss, optimizer, scaler, lr_scheduler, manager, e)

        # Evaluate
        if LM_MODE and lm_hybrid_enabled:
            test_lm_hybrid(
                dataloader_test, model, lm_hybrid_fn_ctc, lm_hybrid_lambda_ctc,
                manager, ctc_decoder, e,
            )
        elif LM_MODE:
            test_lm(dataloader_test, model, manager, e)
        else:
            if dual_enabled:
                test_hybrid(
                    dataloader_test,
                    model,
                    fn_loss_ar,
                    fn_loss_ctc,
                    lambda_ar=lambda_ar,
                    lambda_ctc=lambda_ctc,
                    lambda_ctc_schedule=lambda_ctc_schedule,
                    loss_balance_mode=loss_balance_mode,
                    man=manager,
                    ctc_decoder=ctc_decoder,
                    epoch=e,
                    tokenizer=tok,
                )
            else:
                test(dataloader_test, model, fn_loss, manager, ctc_decoder, e, tokenizer=tok)

        # Save checkpoints
        _save_checkpoints(cfgs, model, optimizer, lr_scheduler, manager, e)

    if not cfgs.test:
        manager.summarize_evaluation()


def _run_test_epoch(cfgs, model, fn_loss, manager, dataloader_test, ctc_decoder, tok, LM_MODE):
    """Run a single test/evaluation epoch."""
    qual_cfg = build_qualitative_config(cfgs)

    lm_hybrid_enabled = bool(getattr(cfgs, "LM_HYBRID", False))
    if LM_MODE and lm_hybrid_enabled:
        lm_hybrid_cfg = getattr(cfgs, "lm_hybrid", {}) or {}
        test_lm_hybrid(
            dataloader_test, model, CTCLoss(),
            float(lm_hybrid_cfg.get("lambda_ctc", 0.6)),
            manager, ctc_decoder, 0,
        )
    elif LM_MODE:
        test_lm(dataloader_test, model, manager, 0)
    else:
        dual = getattr(cfgs, "dual_head", {}) or {}
        dual_enabled = bool(getattr(cfgs, "DUAL_HEAD", False)) or bool(dual.get("enabled", False))

        if dual_enabled:
            # In test mode we must call the dual-head evaluator, otherwise the CTC head
            # (and any rescoring logic) is never exercised.
            fn_loss_ar = nn.CrossEntropyLoss(ignore_index=cfgs.PAD_ID, label_smoothing=0.1)
            fn_loss_ctc = CTCLoss()
            lambda_ar = float(dual.get("lambda_ar", 1.0))
            lambda_ctc = float(dual.get("lambda_ctc", 0.3))
            lambda_ctc_schedule = dual.get("lambda_ctc_schedule", None)
            loss_balance_mode = str(dual.get("loss_balance", "sum"))

            test_hybrid(
                dataloader_test,
                model,
                fn_loss_ar,
                fn_loss_ctc,
                lambda_ar=lambda_ar,
                lambda_ctc=lambda_ctc,
                lambda_ctc_schedule=lambda_ctc_schedule,
                loss_balance_mode=loss_balance_mode,
                man=manager,
                ctc_decoder=ctc_decoder,
                epoch=0,
                tokenizer=tok,
                force_eval=True,
            )
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
    def _best_test_loss_epoch(man):
        best_epoch = None
        best_val = None
        for epoch, rec in man.results.items():
            if not isinstance(epoch, int) or not isinstance(rec, dict):
                continue
            test_list = rec.get("test", None)
            if not isinstance(test_list, list) or not test_list:
                continue
            loss_avg = test_list[-1].get("loss_avg", None)
            if loss_avg is None:
                continue
            try:
                loss_val = float(loss_avg)
            except Exception:
                continue
            if best_val is None or loss_val < best_val:
                best_val = loss_val
                best_epoch = int(epoch)
        return best_epoch, best_val

    def _maybe_save_best_from_bestdict(best_dict: dict, *, prefix: str, also_update_primary: bool):
        if not isinstance(best_dict, dict):
            return
        if "character_error_rate" in best_dict and int(best_dict["character_error_rate"][0]) == int(e):
            manager.save_checkpoint(
                model.state_dict(), optimizer.state_dict(),
                lr_scheduler.state_dict(), filename=f"best_{prefix}_cer.pth"
            )
            if also_update_primary:
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_cer.pth"
                )
        if "word_error_rate" in best_dict and int(best_dict["word_error_rate"][0]) == int(e):
            manager.save_checkpoint(
                model.state_dict(), optimizer.state_dict(),
                lr_scheduler.state_dict(), filename=f"best_{prefix}_wer.pth"
            )
            if also_update_primary:
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_wer.pth"
                )

    # Best checkpoints based on metrics/loss
    if bool(getattr(cfgs, "save_best_only", False)):
        dual = getattr(cfgs, "dual_head", {}) or {}
        dual_enabled = bool(getattr(cfgs, "DUAL_HEAD", False)) or bool(dual.get("enabled", False))

        # Always keep backward-compatible best tracking for AR metrics under key='evaluation'.
        # (Used by existing aggregation code that reads results['best']).
        manager.summarize_evaluation(key="evaluation")
        best_ar = manager.results.get("best", {})

        best_ctc = {}
        if dual_enabled:
            manager.summarize_evaluation(key="evaluation_ctc")
            best_ctc = manager.results.get("best_evaluation_ctc", {})

        # Select which head defines best_cer.pth / best_wer.pth.
        # If not explicitly set, infer from loss weights so that the saved
        # "best_*" checkpoints match the intended decoding head.
        primary_raw = dual.get("primary", dual.get("primary_head", None))
        primary = str(primary_raw).lower() if primary_raw is not None else "auto"
        if primary in {"", "none", "null"}:
            primary = "auto"

        if primary == "auto" and dual_enabled:
            try:
                lambda_ar_cfg = float(dual.get("lambda_ar", 1.0))
            except Exception:
                lambda_ar_cfg = 1.0
            try:
                lambda_ctc_cfg = float(dual.get("lambda_ctc", 0.0))
            except Exception:
                lambda_ctc_cfg = 0.0

            sched = dual.get("lambda_ctc_schedule", None) or {}
            sched_enabled = bool(sched.get("enabled", False))
            if sched_enabled and "max" in sched:
                try:
                    lambda_ctc_ref = float(sched.get("max", lambda_ctc_cfg))
                except Exception:
                    lambda_ctc_ref = lambda_ctc_cfg
            else:
                lambda_ctc_ref = lambda_ctc_cfg

            primary = "ctc" if lambda_ctc_ref >= lambda_ar_cfg else "ar"

        if primary not in {"ar", "ctc", "hybrid_loss"}:
            primary = "ar"

        # Save per-head best checkpoints and, for the chosen primary head, also update best_cer/best_wer.
        _maybe_save_best_from_bestdict(best_ar, prefix="ar", also_update_primary=(primary == "ar"))
        if dual_enabled:
            _maybe_save_best_from_bestdict(best_ctc, prefix="ctc", also_update_primary=(primary == "ctc"))

        # Save best-by-loss checkpoints (uses validation loss_avg).
        # In hybrid mode this corresponds to the *hybrid loss* used in test_hybrid.
        save_best_loss = bool(dual.get("save_best_hybrid_loss", False)) if dual_enabled else bool(getattr(cfgs, "save_best_loss", False))
        if save_best_loss:
            be, _ = _best_test_loss_epoch(manager)
            if be is not None and int(be) == int(e):
                # Generic name (works for any regime)
                manager.save_checkpoint(
                    model.state_dict(), optimizer.state_dict(),
                    lr_scheduler.state_dict(), filename="best_loss.pth"
                )
                # Hybrid-specific alias
                if dual_enabled:
                    manager.save_checkpoint(
                        model.state_dict(), optimizer.state_dict(),
                        lr_scheduler.state_dict(), filename="best_hybrid_loss.pth"
                    )
                # If primary selection is loss-based, also update best_cer/best_wer equivalent is ambiguous;
                # we keep best_loss.pth as the primary artifact in that case.

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
    AR_MODE = cfgs.arch_de in {"ar_transformer_xs", "ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}
    LM_MODE = cfgs.arch_de in {"byt5_small", "t5-small"}
    VLM_MODE = bool(getattr(cfgs, "vlm_enabled", False))

    # VLM uses LM training loops — treat it as LM_MODE for loop routing
    LOOP_LM = LM_MODE or VLM_MODE

    # Attach derived flags for debug-friendly logging
    cfgs.AR_MODE = bool(AR_MODE)
    cfgs.LM_MODE = bool(LM_MODE)
    cfgs.VLM_MODE = bool(VLM_MODE)

    # Set up tokenizer
    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfgs)

    # Attach to cfgs
    cfgs.PAD_ID, cfgs.BOS_ID, cfgs.EOS_ID = PAD_ID, BOS_ID, EOS_ID
    cfgs.vocab_dec = vocab_dec
    cfgs.tokenizer_obj = tok

    # Regime string (used by RunManager for summary logs)
    dual = getattr(cfgs, "dual_head", {}) or {}
    dual_enabled_cfg = bool(dual.get("enabled", False))
    if LM_MODE:
        cfgs.TRAINING_REGIME = "lm"
    elif VLM_MODE:
        cfgs.TRAINING_REGIME = "vlm"
    elif AR_MODE and dual_enabled_cfg:
        cfgs.TRAINING_REGIME = "hybrid_ar_ctc"
    elif AR_MODE:
        cfgs.TRAINING_REGIME = "ar_only"
    else:
        cfgs.TRAINING_REGIME = "ctc_only"

    # Initialize
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)

    # Build model
    model = build_model(cfgs, manager)

    # Build dataloaders
    dataloader_train, dataloader_test, _ = build_dataloaders(cfgs, model, tok, LOOP_LM)

    # Build optimizer and scheduler
    optimizer, scaler, lr_scheduler = build_optimizer_and_scheduler(
        cfgs, model, dataloader_train, LOOP_LM
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
        epoch_start, AR_MODE, LOOP_LM
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
