#!/usr/bin/env python3
"""Measure CER under teacher forcing with prefix perturbation.

Extends eval_tf_gap.py to support a configurable prefix-perturbation rate
p_replace. For each batch, the teacher-forced input y_inp is constructed
as usual, then each non-BOS position is replaced with probability
p_replace by a uniform sample drawn from the recognizable character
vocabulary (BOS, EOS, PAD, and the CTC blank are excluded). The model
runs once on the perturbed input, the argmax per position is decoded,
and CER and WER are reported against the unperturbed ground truth.

Output: one JSON file per (checkpoint, fold, p_replace) configuration.
The driver sbatch aggregates the JSON files into a single CSV.

Usage:
    python eval_tf_perturbation.py -c <config.yaml> --checkpoint <pth> \
        --p_replace 0.10 --out <out.json>
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

from decode_study import _migrate_legacy_mha_keys, setup_tokenizer
from eval_tf_gap import (
    build_dual_or_base,
    decode_ids,
    load_config,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--p_replace", type=float, required=True,
                   help="Fraction of non-BOS prefix tokens to replace uniformly.")
    p.add_argument("--seed", type=int, default=0,
                   help="Per-evaluation perturbation seed.")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size_batch", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def vocab_substitution_pool(cfg: dict) -> list[int]:
    """Return the set of token IDs eligible for uniform substitution.

    BOS, EOS, PAD and the CTC blank (index 0) are excluded so that
    perturbation never injects a special token into the prefix.
    """
    n_vocab = len(cfg["categories"])
    excluded = {0, cfg["BOS_ID"], cfg["EOS_ID"], cfg["PAD_ID"]}
    return [i for i in range(n_vocab) if i not in excluded]


def perturb_prefix(
    y_inp: torch.Tensor,
    p_replace: float,
    substitution_pool: torch.Tensor,
    bos_id: int,
    pad_id: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace each non-BOS, non-PAD position with probability p_replace.

    The replacement is a uniform sample from substitution_pool.
    """
    if p_replace <= 0.0:
        return y_inp
    perturbed = y_inp.clone()
    B, N = perturbed.shape
    # Eligible positions are everything except BOS at position 0 and PAD.
    eligibility = (perturbed != bos_id) & (perturbed != pad_id)
    mask = torch.rand(B, N, generator=generator, device=perturbed.device) < p_replace
    mask = mask & eligibility
    n_swap = int(mask.sum().item())
    if n_swap == 0:
        return perturbed
    pool_size = substitution_pool.numel()
    draws = torch.randint(
        low=0, high=pool_size, size=(n_swap,),
        generator=generator, device=perturbed.device,
    )
    sampled = substitution_pool[draws]
    perturbed[mask] = sampled
    return perturbed


@torch.no_grad()
def run_perturbed_tf_eval(
    model,
    loader,
    cfg,
    device,
    p_replace: float,
    seed: int,
    max_samples: int | None,
):
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    chars = cfg["categories"]
    pool_list = vocab_substitution_pool(cfg)
    pool = torch.tensor(pool_list, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    preds, labels = [], []
    n_seen = 0
    t0 = time.time()
    for x, y, len_x, len_y in loader:
        x = x.to(device)
        y_dev = y.to(device)
        y_inp, _ = build_ar_batch(
            y_dev, len_y, PAD_ID, BOS_ID, EOS_ID, device=device,
        )
        y_inp = perturb_prefix(
            y_inp, p_replace, pool, BOS_ID, PAD_ID, generator,
        )
        out = model(x, in_lengths=len_x, y_inp=y_inp)
        logits = out["ar_logits"] if isinstance(out, dict) else out
        arg = logits.argmax(-1).cpu().tolist()
        B = x.size(0)
        len_y_cpu = len_y.cpu().tolist()
        y_cpu = y.cpu()
        for b in range(B):
            L = int(len_y_cpu[b])
            # Clamp to the valid target span (y1..yL + EOS slot); avoids padded-position
            # insertions when EOS is missed (which worsens as p_replace grows).
            ids = arg[b][:L + 1]
            preds.append(decode_ids(ids, chars, PAD_ID, BOS_ID, EOS_ID))
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
        "p_replace": float(p_replace),
        "seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.size_batch is not None:
        cfg["size_batch"] = args.size_batch
    device = torch.device(args.device)

    tok, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfg)
    cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"] = PAD_ID, BOS_ID, EOS_ID

    model, dual_enabled = build_dual_or_base(
        cfg, device, vocab_dec, PAD_ID, BOS_ID, EOS_ID,
    )
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

    ratio_ds = getattr(model, "ratio_ds", 1)
    loader, dataset = make_loader(cfg, ratio_ds)
    logger.info(
        "Validation set: {} samples, batch={}, p_replace={}, seed={}",
        len(dataset), cfg["size_batch"], args.p_replace, args.seed,
    )

    result = run_perturbed_tf_eval(
        model, loader, cfg, device, args.p_replace, args.seed, args.max_samples,
    )

    out_path = args.out
    if out_path is None:
        ckpt_dir = Path(args.checkpoint).parent
        out_path = str(
            ckpt_dir.parent
            / f"eval_tf_perturbation_p{int(round(args.p_replace * 100)):03d}.json"
        )

    meta = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "idx_fold": int(cfg.get("idx_fold", -1)),
        "dir_dataset": str(cfg.get("dir_dataset", "")),
        "arch_de": str(cfg.get("arch_de", "")),
        "dual_head_enabled": dual_enabled,
        "input_corruption_enabled": bool(
            (cfg.get("input_corruption") or {}).get("enabled", False)
        ),
    }
    payload = {"_meta": meta, "tf_perturbation": result}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "TF perturbed p={:.2f} CER {:.4f} WER {:.4f} ({} samples)",
        args.p_replace, result["cer"], result["wer"], result["n_samples"],
    )
    logger.info("Wrote: {}", out_path)


if __name__ == "__main__":
    main()
