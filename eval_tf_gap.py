#!/usr/bin/env python3
"""Measure the teacher-forcing (TF) vs free-running (AR) CER gap on a saved
HWRFormer fold.

For one fold's best-CER checkpoint, evaluate the validation set under two
decoding regimes:
  * TF:  feed ground-truth prefix; take per-position argmax from a single
         forward pass; decode the resulting sequence (stop at EOS).
  * AR:  free-running greedy unrolling (the inference path used to produce
         every published number); identical to rewi.training.loops.test().

Output: `eval_tf_gap.json` next to the checkpoint with
    {tf: {cer, wer, n}, ar: {cer, wer, n}, gap: {cer_diff, wer_diff}}

Usage:
    python eval_tf_gap.py -c <path/to/fold/train.yaml> \\
        --checkpoint <path/to/best_cer.pth> \\
        [--out <path/to/eval_tf_gap.json>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from rewi.dataset import HRDataset
from rewi.dataset.utils import fn_collate
from rewi.evaluate import evaluate
from rewi.model import BaseModel, DualHeadModel
from rewi.training.utils import build_ar_batch

# Reuse the same checkpoint-key migration as the decoding-study harness so old
# state dicts (legacy MHA naming) load cleanly.
from decode_study import _migrate_legacy_mha_keys, setup_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True, help="Path to per-fold train.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to best_cer.pth")
    p.add_argument("--out", default=None, help="Output JSON path (default: <ckpt-dir>/../eval_tf_gap.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size_batch", type=int, default=None, help="Override batch size for eval")
    p.add_argument("--max_samples", type=int, default=None, help="Cap samples (for smoke testing)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dual_or_base(cfg: dict, device: torch.device, vocab_dec, PAD_ID, BOS_ID, EOS_ID):
    """Mirror decode_study.load_model — build either BaseModel or DualHeadModel."""
    dual = cfg.get("dual_head", {}) or {}
    dual_enabled = bool(dual.get("enabled", False))

    if dual_enabled:
        vocab_ctc = len(cfg["categories"])
        tie_cfg = dual.get("tie", {}) or {}
        model = DualHeadModel(
            arch_en=cfg["arch_en"],
            arch_ar=cfg["arch_de"],
            arch_ctc=str(dual.get("arch_ctc", "linear")),
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
        vocab_ctc = len(cfg["categories"])
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
            decoder_side_ctc_cfg=cfg.get("decoder_side_ctc", {}) or {},
        )
    return model, dual_enabled


def decode_ids(ids: list[int], categories: list[str], pad_id: int, bos_id: int, eos_id: int) -> str:
    """Decode a list of integer IDs to a string, stopping at EOS, dropping BOS/PAD."""
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t in (pad_id, bos_id):
            continue
        if 0 <= t < len(categories) and t != 0:  # 0 is CTC blank
            out.append(categories[t])
    return "".join(out)


def make_loader(cfg: dict, ratio_ds: int):
    dataset = HRDataset(
        os.path.join(cfg["dir_dataset"], "val.json"),
        cfg["categories"],
        ratio_ds,
        cfg["idx_fold"],
        cfg.get("len_seq", 0),
        cache=cfg.get("cache", True),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["size_batch"]),
        num_workers=int(cfg.get("num_worker", 4)),
        collate_fn=fn_collate,
        shuffle=False,
    )
    return loader, dataset


@torch.no_grad()
def run_tf_eval(model, loader, cfg, device, max_samples: int | None):
    """Teacher-forced argmax-per-position evaluation."""
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    chars = cfg["categories"]

    preds, labels = [], []
    n_seen = 0
    t0 = time.time()
    for x, y, len_x, len_y in loader:
        x = x.to(device)
        y_dev = y.to(device)
        y_inp, _y_tgt = build_ar_batch(y_dev, len_y, PAD_ID, BOS_ID, EOS_ID, device=device)
        out = model(x, in_lengths=len_x, y_inp=y_inp)
        # DualHeadModel returns a dict {"ar_logits": (B,N,V), "ctc_logits": ..., "enc_lengths": ...};
        # BaseModel returns logits directly.
        logits = out["ar_logits"] if isinstance(out, dict) else out
        arg = logits.argmax(-1).cpu().tolist()             # (B, N)
        B = x.size(0)
        len_y_cpu = len_y.cpu().tolist()
        y_cpu = y.cpu()
        for b in range(B):
            # TF position t predicts token at output position t (the model
            # learns logits[t] -> y_tgt[t]); we take argmax across all output
            # positions and decode the resulting sequence as the "what the
            # model would write at every step given the right prefix" string.
            L = int(len_y_cpu[b])
            # Clamp to the valid target span (positions predicting y1..yL plus the
            # EOS slot); decode_ids still stops at an earlier EOS. Prevents missed-EOS
            # argmaxes at padded positions from leaking in as batch-dependent insertions.
            ids = arg[b][:L + 1]
            preds.append(decode_ids(ids, chars, PAD_ID, BOS_ID, EOS_ID))

            if y.dim() == 2:
                lab_ids = y_cpu[b, :L].tolist()
            else:
                # flat-concat fallback: not used by our datasets, but kept safe
                lab_ids = []
            labels.append(decode_ids(lab_ids, chars, PAD_ID, BOS_ID, EOS_ID))
        n_seen += B
        if max_samples and n_seen >= max_samples:
            break
    dt = time.time() - t0
    res = evaluate(preds, labels)
    return {
        "cer": float(res.get("character_error_rate", -1)),
        "wer": float(res.get("word_error_rate", -1)),
        "levenshtein": float(res.get("levenshtein_distance", -1)),
        "n_samples": n_seen,
        "seconds": round(dt, 1),
    }


@torch.no_grad()
def run_ar_eval(model, loader, cfg, device, max_samples: int | None):
    """Free-running greedy AR evaluation (mirrors rewi.training.loops.test)."""
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    chars = cfg["categories"]

    preds, labels = [], []
    n_seen = 0
    t0 = time.time()
    for x, y, len_x, len_y in loader:
        x = x.to(device)
        B = x.size(0)
        max_len = int(len_y.max().item()) + 2

        y_gen = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        for _ in range(max_len):
            step_out = model(x, in_lengths=len_x, y_inp=y_gen)
            step_logits = step_out["ar_logits"] if isinstance(step_out, dict) else step_out
            nxt = step_logits[:, -1, :].argmax(-1, keepdim=True)
            y_gen = torch.cat([y_gen, nxt], dim=1)

        y_gen = y_gen.detach().cpu().tolist()
        y_cpu = y.cpu()
        len_y_cpu = len_y.cpu().tolist()
        for b in range(B):
            ids_pred = y_gen[b][1:]  # skip BOS
            preds.append(decode_ids(ids_pred, chars, PAD_ID, BOS_ID, EOS_ID))

            L = int(len_y_cpu[b])
            lab_ids = y_cpu[b, :L].tolist() if y.dim() == 2 else []
            labels.append(decode_ids(lab_ids, chars, PAD_ID, BOS_ID, EOS_ID))
        n_seen += B
        if max_samples and n_seen >= max_samples:
            break
    dt = time.time() - t0
    res = evaluate(preds, labels)
    return {
        "cer": float(res.get("character_error_rate", -1)),
        "wer": float(res.get("word_error_rate", -1)),
        "levenshtein": float(res.get("levenshtein_distance", -1)),
        "n_samples": n_seen,
        "seconds": round(dt, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.size_batch is not None:
        cfg["size_batch"] = args.size_batch
    device = torch.device(args.device)

    # Tokenizer / vocab setup (mirrors decode_study)
    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfg)
    cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"] = PAD_ID, BOS_ID, EOS_ID
    if tok is not None:
        logger.warning("BPE tokenizer present; eval_tf_gap currently assumes character mode and uses categories[] for decoding.")

    # Build model + load checkpoint
    model, dual_enabled = build_dual_or_base(cfg, device, vocab_dec, PAD_ID, BOS_ID, EOS_ID)
    logger.info("Loading checkpoint: {}", args.checkpoint)
    ckp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckp.get("model", ckp)
    state = _migrate_legacy_mha_keys(state)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        logger.warning("Missing keys: {}", res.missing_keys[:8])
    if res.unexpected_keys:
        logger.warning("Unexpected keys: {}", res.unexpected_keys[:8])
    model = model.to(device).eval()

    # Build dataloader
    ratio_ds = getattr(model, "ratio_ds", 1)
    loader, dataset = make_loader(cfg, ratio_ds)
    logger.info("Validation set: {} samples, batch={}", len(dataset), cfg["size_batch"])

    # Two passes — TF then AR (order is arbitrary, both deterministic)
    tf = run_tf_eval(model, loader, cfg, device, args.max_samples)
    ar = run_ar_eval(model, loader, cfg, device, args.max_samples)

    gap = {
        "cer_diff": round(ar["cer"] - tf["cer"], 6),
        "wer_diff": round(ar["wer"] - tf["wer"], 6),
        "cer_diff_pp": round((ar["cer"] - tf["cer"]) * 100, 4),
        "wer_diff_pp": round((ar["wer"] - tf["wer"]) * 100, 4),
    }

    out_path = args.out
    if out_path is None:
        # default: alongside the fold's train.yaml
        out_path = str(Path(args.config).with_name("eval_tf_gap.json"))

    meta = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "idx_fold": int(cfg.get("idx_fold", -1)),
        "dir_dataset": str(cfg.get("dir_dataset", "")),
        "arch_en": str(cfg.get("arch_en", "")),
        "arch_de": str(cfg.get("arch_de", "")),
        "use_gated_attention": bool(cfg.get("use_gated_attention", False)),
        "gating_type": str(cfg.get("gating_type", "")),
        "dual_head_enabled": dual_enabled,
        "input_corruption_enabled": bool((cfg.get("input_corruption") or {}).get("enabled", False)),
        "noise_injection_enabled": bool((cfg.get("noise_injection") or {}).get("enabled", False)),
    }
    payload = {"_meta": meta, "tf": tf, "ar": ar, "gap": gap}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("TF  CER {:.4f}  WER {:.4f}  ({} samples)", tf["cer"], tf["wer"], tf["n_samples"])
    logger.info("AR  CER {:.4f}  WER {:.4f}  ({} samples)", ar["cer"], ar["wer"], ar["n_samples"])
    logger.info("Gap CER {:+.4f}  WER {:+.4f}", gap["cer_diff"], gap["wer_diff"])
    logger.info("Wrote: {}", out_path)


if __name__ == "__main__":
    main()
