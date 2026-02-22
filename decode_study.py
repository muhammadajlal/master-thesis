#!/usr/bin/env python3
"""
Decoding Study Evaluation Harness
==================================

A single script that runs any decoding setup and logs identical artifacts:
  - Inputs:  checkpoint, split (val), decoder type, decoding config
  - Outputs: predictions + metrics JSON + per-sample edit distance + runtime

Supports:
  - CTC greedy / prefix beam / beam + LM
  - AR greedy / calibrated beam / beam + LM (rescoring or shallow fusion)
  - Hybrid models (AR head + CTC head evaluated simultaneously)

Usage:
    python decode_study.py -c configs/decode_study/base.yaml

    # Override sweep params from CLI:
    python decode_study.py -c configs/decode_study/base.yaml \
        --decoder ar --beam_size 4 --alpha 0.6 --eos_bias 0.0 \
        --lm_path lm/char_5gram.binary --lm_weight 0.2

Output structure:
    <outdir>/<run_tag>/
        config.json          # Full config snapshot
        metrics.json         # CER, WER, runtime, tail-risk stats
        predictions.json     # Per-sample: pred, gt, edit_dist, ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger

# ── Project imports ──────────────────────────────────────────────────
from rewi.ctc_decoder import BestPath
from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.evaluate import evaluate
from rewi.model import BaseModel, build_encoder
from rewi.model.dual_head import DualHeadModel
from rewi.model.ARDecoder import ARDecoder
from rewi.tokenizer import BPETokenizer
from rewi.utils import seed_everything
from rewi.analysis.metrics import lev_dist, normalized_lev_dist, character_error_rate

from rewi.decoding.lm import CharLM, load_lm
from rewi.decoding.neural_lm import NeuralCharLM, load_neural_lm
from rewi.decoding.ctc_beam import ctc_greedy, ctc_prefix_beam_search
from rewi.decoding.ar_beam import (
    ar_greedy,
    ar_calibrated_beam_search,
    ar_nbest_rescore,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════
#  CLI / Config
# ═════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decoding Study Evaluation Harness")
    p.add_argument("-c", "--config", required=True, help="YAML config path")
    # Overrides (for sweep scripts)
    p.add_argument("--decoder", type=str, default=None, choices=["ar", "ctc"])
    p.add_argument("--method", type=str, default=None,
                   choices=["greedy", "beam", "beam_lm", "beam_rescore"])
    p.add_argument("--beam_size", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--eos_bias", type=float, default=None)
    p.add_argument("--min_len", type=int, default=None)
    p.add_argument("--min_len_ratio", type=float, default=None)
    p.add_argument("--lm_path", type=str, default=None)
    p.add_argument("--lm_weight", type=float, default=None)
    p.add_argument("--insertion_bonus", type=float, default=None)
    p.add_argument("--nbest_size", type=int, default=None)
    p.add_argument("--length_bonus", type=float, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--idx_fold", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--size_batch", type=int, default=None)
    p.add_argument("--num_worker", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples (for testing)")
    p.add_argument("--neural_lm_path", type=str, default=None,
                   help="Path to pretrained AR decoder checkpoint for neural LM")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    overrides = {
        "decoder": args.decoder,
        "method": args.method,
        "beam_size": args.beam_size,
        "alpha": args.alpha,
        "eos_bias": args.eos_bias,
        "min_len": args.min_len,
        "min_len_ratio": args.min_len_ratio,
        "lm_path": args.lm_path,
        "lm_weight": args.lm_weight,
        "insertion_bonus": args.insertion_bonus,
        "nbest_size": args.nbest_size,
        "length_bonus": args.length_bonus,
        "outdir": args.outdir,
        "tag": args.tag,
        "idx_fold": args.idx_fold,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "size_batch": args.size_batch,
        "num_worker": args.num_worker,
        "max_samples": args.max_samples,
        "neural_lm_path": args.neural_lm_path,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    # Defaults
    cfg.setdefault("decoder", "ar")
    cfg.setdefault("method", "greedy")
    cfg.setdefault("beam_size", 4)
    cfg.setdefault("alpha", 0.6)
    cfg.setdefault("eos_bias", 0.0)
    cfg.setdefault("min_len", 0)
    cfg.setdefault("min_len_ratio", 0.0)
    cfg.setdefault("lm_path", None)
    cfg.setdefault("neural_lm_path", None)
    cfg.setdefault("lm_weight", 0.0)
    cfg.setdefault("insertion_bonus", 0.0)
    cfg.setdefault("nbest_size", 16)
    cfg.setdefault("length_bonus", 0.0)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.setdefault("seed", 42)
    cfg.setdefault("num_worker", 4)
    cfg.setdefault("size_batch", 64)
    cfg.setdefault("max_len", 200)
    cfg.setdefault("idx_fold", 0)
    cfg.setdefault("use_bpe", False)
    cfg.setdefault("use_gated_attention", False)
    cfg.setdefault("gating_type", "elementwise")
    cfg.setdefault("len_seq", 0)
    cfg.setdefault("num_channel", 13)

    return cfg


# ═════════════════════════════════════════════════════════════════════
#  Model loading
# ═════════════════════════════════════════════════════════════════════

def _migrate_legacy_mha_keys(state_dict: dict) -> dict:
    """Migrate old Q/K/V projection keys to nn.MultiheadAttention format."""
    import re
    new_sd = {}
    consumed = set()
    old_prefixes = set()
    for k in state_dict:
        m = re.match(r"^(.+\.(self_attn|multihead_attn))\.q_proj\.weight$", k)
        if m:
            old_prefixes.add(m.group(1))

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
            new_sd[f"{prefix}.mha.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
            consumed.update([f"{prefix}.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")])
            if q_b is not None and k_b is not None and v_b is not None:
                new_sd[f"{prefix}.mha.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
                consumed.update([f"{prefix}.{p}.bias" for p in ("q_proj", "k_proj", "v_proj")])
        if o_w is not None:
            new_sd[f"{prefix}.mha.out_proj.weight"] = o_w
            consumed.add(f"{prefix}.out_proj.weight")
        if o_b is not None:
            new_sd[f"{prefix}.mha.out_proj.bias"] = o_b
            consumed.add(f"{prefix}.out_proj.bias")

    for k, v in state_dict.items():
        if k not in consumed:
            new_sd[k] = v
    return new_sd


def setup_tokenizer(cfg: dict):
    """Set up tokenizer and special IDs."""
    categories = cfg["categories"]
    AR_MODE = cfg["arch_de"] in {"ar_transformer_s", "ar_transformer_m", "ar_transformer_l"}

    tok = None
    if cfg.get("use_bpe", False):
        tok = BPETokenizer(cfg["tokenizer"]["model"])

    if tok is not None:
        vocab_dec = tok.vocab_size
        PAD_ID, BOS_ID, EOS_ID = tok.PAD, tok.BOS, tok.EOS
    else:
        base = len(categories)
        PAD_ID, BOS_ID, EOS_ID = base, base + 1, base + 2
        vocab_dec = base + (3 if AR_MODE else 0)

    return tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID


def load_model(cfg: dict, device: torch.device):
    """Load model from checkpoint."""
    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfg)
    categories = cfg["categories"]

    dual = cfg.get("dual_head", {}) or {}
    dual_enabled = bool(dual.get("enabled", False))

    if dual_enabled:
        vocab_ctc = len(categories)
        arch_ctc = str(dual.get("arch_ctc", "linear"))
        tie_cfg = dual.get("tie", {}) or {}

        model = DualHeadModel(
            arch_en=cfg["arch_en"],
            arch_ar=cfg["arch_de"],
            arch_ctc=arch_ctc,
            in_chan=cfg["num_channel"],
            vocab_ar=vocab_dec,
            vocab_ctc=vocab_ctc,
            len_seq=cfg.get("len_seq", 0),
            use_gated_attention=cfg.get("use_gated_attention", False),
            gating_type=cfg.get("gating_type", "elementwise"),
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            tie_cfg=tie_cfg,
            dual_cfg=dual,
        )
    else:
        dec_ctc_cfg = cfg.get("decoder_side_ctc", {}) or {}
        vocab_ctc = len(categories)

        model = BaseModel(
            cfg["arch_en"],
            cfg["arch_de"],
            cfg["num_channel"],
            vocab_dec,
            cfg.get("len_seq", 0),
            use_gated_attention=cfg.get("use_gated_attention", False),
            gating_type=cfg.get("gating_type", "elementwise"),
            vocab_ctc=vocab_ctc,
            pad_id=PAD_ID,
            decoder_side_ctc_cfg=dec_ctc_cfg,
        )

    # Load checkpoint
    ckpt_path = cfg["checkpoint"]
    logger.info("Loading checkpoint: {}", ckpt_path)
    ckp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckp.get("model", ckp)
    state = _migrate_legacy_mha_keys(state)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        logger.warning("Missing keys: {}", res.missing_keys)
    if res.unexpected_keys:
        logger.warning("Unexpected keys: {}", res.unexpected_keys)

    model = model.to(device)
    model.eval()

    return model, tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID, dual_enabled


# ═════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════

def load_data(cfg: dict, ratio_ds: int):
    """Load validation dataset."""
    dataset = HRDataset(
        os.path.join(cfg["dir_dataset"], "val.json"),
        cfg["categories"],
        ratio_ds,
        cfg["idx_fold"],
        cfg.get("len_seq", 0),
        cache=cfg.get("cache", True),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["size_batch"],
        num_workers=cfg.get("num_worker", 4),
        collate_fn=fn_collate,
        shuffle=False,
    )
    return dataloader


# ═════════════════════════════════════════════════════════════════════
#  Decoding dispatcher
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_decoding(
    cfg: dict,
    model: nn.Module,
    dataloader,
    *,
    tok,
    PAD_ID: int,
    BOS_ID: int,
    EOS_ID: int,
    dual_enabled: bool,
    lm: Optional[CharLM],
) -> list[dict]:
    """Run decoding over the full dataset.

    Returns a list of per-sample result dicts.
    """
    categories = cfg["categories"]
    device = cfg["device"]
    decoder_type = cfg["decoder"]
    method = cfg["method"]
    max_len = cfg.get("max_len", 200)
    max_samples = cfg.get("max_samples")

    # CTC decoder for CTC greedy (as reference)
    ctc_bp = BestPath(categories)

    results: list[dict] = []
    sample_idx = 0

    for batch in dataloader:
        if max_samples is not None and sample_idx >= max_samples:
            break
        x, y, len_x, len_y = batch
        x = x.to(device)
        y = y.to(device)
        len_x = len_x.to(device)
        len_y = len_y.to(device)

        B = x.size(0)

        # ── Encode ────────────────────────────────────────────
        if dual_enabled:
            feats, enc_pad, enc_lengths = model._encode_with_mask(x, len_x)
            mem = feats
            if getattr(model, "mem_proj", None) is not None:
                mem = model.mem_proj(mem)
            ctc_logits = model.compute_ctc_logits(feats).float()
        elif isinstance(model.decoder, ARDecoder):
            feats, enc_pad = model._encode_with_mask(x, len_x)
            mem = feats
            if getattr(model, "mem_proj", None) is not None:
                mem = model.mem_proj(mem)
            enc_lengths = torch.div(len_x, model.ratio_ds, rounding_mode="floor").clamp(min=1, max=feats.size(1))
            ctc_logits = None
        else:
            # Pure CTC model
            feats = model.encoder(x)
            ctc_logits = model.decoder(feats).float()
            enc_lengths = (len_x // model.ratio_ds).clamp(min=1, max=feats.size(1))
            mem = None
            enc_pad = None

        # Convert to float32 for decoding
        if mem is not None:
            mem = mem.float()
        if ctc_logits is not None:
            ctc_logits = ctc_logits.float()

        # ── Per-sample decoding ───────────────────────────────
        for b in range(B):
            gt_len = int(len_y[b].item())
            gt_ids = y[b, :gt_len].cpu().tolist() if y.dim() == 2 else []
            gt_text = _decode_ids(gt_ids, categories, tok, PAD_ID)

            t_start = time.perf_counter()

            if decoder_type == "ctc":
                result = _decode_ctc_sample(
                    cfg, ctc_logits[b], int(enc_lengths[b].item()),
                    categories, lm, ctc_bp,
                )
            elif decoder_type == "ar":
                result = _decode_ar_sample(
                    cfg, mem[b:b+1], enc_pad[b:b+1],
                    model, categories, tok, lm,
                    PAD_ID, BOS_ID, EOS_ID, max_len,
                )
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")

            t_elapsed = time.perf_counter() - t_start

            # Compute edit distance
            pred_text = result["text"]
            ed = lev_dist(pred_text, gt_text)
            ned = ed / max(1, len(gt_text))
            cer = character_error_rate(pred_text, gt_text)

            results.append({
                "sample_idx": sample_idx,
                "pred": pred_text,
                "gt": gt_text,
                "pred_ids": result.get("ids", []),
                "gt_ids": gt_ids,
                "edit_distance": ed,
                "normalized_edit_distance": ned,
                "cer_sample": cer,
                "pred_length": result.get("length", len(pred_text)),
                "gt_length": len(gt_text),
                "length_ratio": result.get("length", len(pred_text)) / max(1, len(gt_text)),
                "ended_by_eos": result.get("ended_by_eos", None),
                "log_prob": result.get("log_prob", None),
                "lm_score": result.get("lm_score", None),
                "score": result.get("score", None),
                "runtime_s": t_elapsed,
                "decoder": decoder_type,
                "method": method,
            })
            sample_idx += 1

    return results


def _decode_ctc_sample(
    cfg: dict,
    logits_TV: torch.Tensor,  # (T, V)
    T: int,
    categories: list[str],
    lm: Optional[CharLM],
    ctc_bp: BestPath,
) -> dict:
    """Decode a single CTC sample."""
    method = cfg["method"]
    logits = logits_TV[:T]

    if method == "greedy":
        return ctc_greedy(logits, categories)
    elif method in ("beam", "beam_lm"):
        nbest = ctc_prefix_beam_search(
            logits,
            categories,
            beam_size=cfg["beam_size"],
            lm=lm if method == "beam_lm" else None,
            lm_weight=cfg["lm_weight"],
            insertion_bonus=cfg["insertion_bonus"],
        )
        if nbest:
            return nbest[0]
        return {"text": "", "ids": [], "score": -math.inf}
    else:
        raise ValueError(f"Unknown CTC method: {method}")


def _decode_ar_sample(
    cfg: dict,
    mem: torch.Tensor,        # (1, Tm, D)
    enc_pad: torch.Tensor,    # (1, Tm)
    model: nn.Module,
    categories: list[str],
    tok,
    lm: Optional[CharLM],
    PAD_ID: int,
    BOS_ID: int,
    EOS_ID: int,
    max_len: int,
) -> dict:
    """Decode a single AR sample."""
    method = cfg["method"]

    # Decoder function
    decoder = model.decoder if hasattr(model, "decoder") else model
    def decoder_fn(y_inp, memory, enc_pad_mask):
        return decoder(y_inp, memory, enc_pad_mask)

    if method == "greedy":
        return ar_greedy(
            mem, enc_pad, decoder_fn,
            bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            max_len=max_len, categories=categories, tokenizer=tok,
        )
    elif method == "beam":
        nbest = ar_calibrated_beam_search(
            mem, enc_pad, decoder_fn,
            bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            beam_size=cfg["beam_size"],
            max_len=max_len,
            alpha=cfg["alpha"],
            eos_bias=cfg["eos_bias"],
            min_len=cfg["min_len"],
            min_len_ratio=cfg["min_len_ratio"],
            categories=categories, tokenizer=tok,
        )
        return nbest[0] if nbest else {"text": "", "ids": []}
    elif method == "beam_lm":
        # AR beam + LM shallow fusion
        nbest = ar_calibrated_beam_search(
            mem, enc_pad, decoder_fn,
            bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            beam_size=cfg["beam_size"],
            max_len=max_len,
            alpha=cfg["alpha"],
            eos_bias=cfg["eos_bias"],
            min_len=cfg["min_len"],
            min_len_ratio=cfg["min_len_ratio"],
            categories=categories, tokenizer=tok,
            lm=lm,
            lm_weight=cfg["lm_weight"],
        )
        return nbest[0] if nbest else {"text": "", "ids": []}
    elif method == "beam_rescore":
        # AR beam (no LM) → N-best → rescore with LM
        nbest = ar_calibrated_beam_search(
            mem, enc_pad, decoder_fn,
            bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            beam_size=cfg.get("nbest_size", cfg["beam_size"]),
            max_len=max_len,
            alpha=cfg["alpha"],
            eos_bias=cfg["eos_bias"],
            min_len=cfg["min_len"],
            min_len_ratio=cfg["min_len_ratio"],
            categories=categories, tokenizer=tok,
        )
        if lm is not None and cfg["lm_weight"] > 0:
            nbest = ar_nbest_rescore(
                nbest, lm,
                lm_weight=cfg["lm_weight"],
                length_bonus=cfg["length_bonus"],
                alpha=cfg["alpha"],
            )
        return nbest[0] if nbest else {"text": "", "ids": []}
    else:
        raise ValueError(f"Unknown AR method: {method}")


# ═════════════════════════════════════════════════════════════════════
#  Metrics aggregation
# ═════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-sample results."""
    preds = [r["pred"] for r in results]
    gts = [r["gt"] for r in results]

    # jiwer CER/WER
    eval_res = evaluate(preds, gts)

    # Per-sample NED
    neds = [r["normalized_edit_distance"] for r in results]
    neds_arr = np.array(neds)

    # Length statistics
    length_ratios = [r["length_ratio"] for r in results]
    lr_arr = np.array(length_ratios)

    # Early EOS rate (AR only)
    eos_flags = [r["ended_by_eos"] for r in results if r["ended_by_eos"] is not None]
    early_eos_count = sum(
        1 for r in results
        if r["ended_by_eos"] is True and r["pred_length"] < r["gt_length"] * 0.5
    )
    early_eos_rate = early_eos_count / max(1, len(eos_flags))

    # Runtime
    runtimes = [r["runtime_s"] for r in results]

    metrics = {
        "cer": float(eval_res["character_error_rate"]),
        "wer": float(eval_res["word_error_rate"]),
        "levenshtein_distance_avg": float(eval_res["levenshtein_distance"]),
        "n_samples": len(results),
        # Tail-risk
        "ned_mean": float(neds_arr.mean()),
        "ned_median": float(np.median(neds_arr)),
        "ned_p90": float(np.percentile(neds_arr, 90)),
        "ned_p95": float(np.percentile(neds_arr, 95)),
        "ned_p99": float(np.percentile(neds_arr, 99)),
        "ned_max": float(neds_arr.max()),
        # Length stats
        "length_ratio_mean": float(lr_arr.mean()),
        "length_ratio_std": float(lr_arr.std()),
        "length_ratio_median": float(np.median(lr_arr)),
        # EOS (AR only)
        "early_eos_rate": float(early_eos_rate),
        "eos_count": len(eos_flags),
        "early_eos_count": early_eos_count,
        # Runtime
        "runtime_total_s": float(sum(runtimes)),
        "runtime_per_sample_ms": float(np.mean(runtimes) * 1000),
    }
    return metrics


# ═════════════════════════════════════════════════════════════════════
#  Output
# ═════════════════════════════════════════════════════════════════════

def build_run_tag(cfg: dict) -> str:
    """Build a descriptive tag for this decoding run."""
    if cfg.get("tag"):
        return cfg["tag"]
    parts = [
        cfg["decoder"],
        cfg["method"],
        f"B{cfg['beam_size']}",
        f"a{cfg['alpha']}",
    ]
    if cfg["eos_bias"] != 0.0:
        parts.append(f"eos{cfg['eos_bias']}")
    if cfg["lm_weight"] > 0:
        parts.append(f"lm{cfg['lm_weight']}")
    if cfg.get("insertion_bonus", 0) != 0:
        parts.append(f"ib{cfg['insertion_bonus']}")
    if cfg.get("length_bonus", 0) != 0:
        parts.append(f"lb{cfg['length_bonus']}")
    parts.append(f"fold{cfg['idx_fold']}")
    return "__".join(parts)


def save_results(cfg: dict, results: list[dict], metrics: dict, outdir: str):
    """Save all outputs."""
    os.makedirs(outdir, exist_ok=True)

    # Config snapshot
    with open(os.path.join(outdir, "config.json"), "w") as f:
        # Remove non-serialisable items
        cfg_out = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)

    # Metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-sample predictions
    # Strip large fields for compact output
    compact = []
    for r in results:
        compact.append({
            "idx": r["sample_idx"],
            "pred": r["pred"],
            "gt": r["gt"],
            "ed": r["edit_distance"],
            "ned": round(r["normalized_edit_distance"], 4),
            "cer": round(r["cer_sample"], 4),
            "pred_len": r["pred_length"],
            "gt_len": r["gt_length"],
            "lr": round(r["length_ratio"], 3),
            "eos": r["ended_by_eos"],
            "rt_ms": round(r["runtime_s"] * 1000, 2),
        })
    with open(os.path.join(outdir, "predictions.json"), "w") as f:
        json.dump(compact, f, indent=1, ensure_ascii=False)

    logger.info("Results saved to {}", outdir)


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _decode_ids(ids: list[int], categories: list[str], tok, PAD_ID: int) -> str:
    if tok is not None:
        return tok.decode(ids)
    return "".join(
        categories[i] for i in ids
        if 0 <= i < len(categories) and i != 0 and i != PAD_ID
    )


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = load_config(args)

    seed_everything(cfg["seed"])

    # Load model
    device = cfg["device"]
    model, tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID, dual_enabled = load_model(cfg, device)
    logger.info(
        "Model loaded | arch_en={} arch_de={} vocab={} dual={} device={}",
        cfg["arch_en"], cfg["arch_de"], vocab_dec, dual_enabled, device,
    )

    # Load data
    dataloader = load_data(cfg, model.ratio_ds)
    logger.info("Dataset loaded | fold={} samples={}", cfg["idx_fold"], len(dataloader.dataset))

    # Load LM (if requested) — neural LM takes precedence over KenLM
    lm = None
    if cfg.get("neural_lm_path") and cfg["lm_weight"] > 0:
        lm = load_neural_lm(
            checkpoint=cfg["neural_lm_path"],
            categories=cfg["categories"],
            device=device,
        )
        logger.info("Neural LM loaded | path={} d_model={} n_layers={}",
                    cfg["neural_lm_path"], lm.decoder.d_model, lm.decoder.num_layers)
    elif cfg.get("lm_path") and cfg["lm_weight"] > 0:
        lm = load_lm(cfg["lm_path"])
        logger.info("KenLM loaded | path={} order={}", cfg["lm_path"], lm.order)

    # Run decoding
    logger.info(
        "Decoding | decoder={} method={} beam={} alpha={} eos_bias={} lm_weight={}",
        cfg["decoder"], cfg["method"], cfg["beam_size"],
        cfg["alpha"], cfg["eos_bias"], cfg["lm_weight"],
    )
    t_total_start = time.perf_counter()

    results = run_decoding(
        cfg, model, dataloader,
        tok=tok, PAD_ID=PAD_ID, BOS_ID=BOS_ID, EOS_ID=EOS_ID,
        dual_enabled=dual_enabled, lm=lm,
    )

    t_total = time.perf_counter() - t_total_start

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["total_wall_time_s"] = t_total

    # Decoding config in metrics for easy comparison
    metrics["config"] = {
        "decoder": cfg["decoder"],
        "method": cfg["method"],
        "beam_size": cfg["beam_size"],
        "alpha": cfg["alpha"],
        "eos_bias": cfg["eos_bias"],
        "lm_weight": cfg["lm_weight"],
        "insertion_bonus": cfg.get("insertion_bonus", 0),
        "length_bonus": cfg.get("length_bonus", 0),
        "nbest_size": cfg.get("nbest_size", 0),
        "min_len": cfg.get("min_len", 0),
        "min_len_ratio": cfg.get("min_len_ratio", 0),
        "idx_fold": cfg["idx_fold"],
        "checkpoint": cfg["checkpoint"],
        "lm_path": cfg.get("lm_path"),
        "neural_lm_path": cfg.get("neural_lm_path"),
    }

    # Summary
    logger.info(
        "Results | CER={:.4f} WER={:.4f} NED_p95={:.4f} NED_p99={:.4f} early_eos={:.3f} runtime={:.1f}s",
        metrics["cer"], metrics["wer"],
        metrics["ned_p95"], metrics["ned_p99"],
        metrics["early_eos_rate"], t_total,
    )

    # Save
    run_tag = build_run_tag(cfg)
    outdir = os.path.join(cfg.get("outdir", "decode_study_results"), run_tag)
    save_results(cfg, results, metrics, outdir)


if __name__ == "__main__":
    main()
