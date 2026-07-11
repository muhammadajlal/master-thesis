#!/usr/bin/env python3
"""Single-corruption recovery profile under teacher forcing.

For each sample of label length L >= MIN_L, pick exactly one position t
(uniform over an eligible range) and replace the input character that
holds label[t] with a uniform draw from the recognizable character
vocabulary. The model performs one teacher-forced forward pass; the
argmax at decoder positions t, t+1, ..., t+(N_OFFSETS-1) is compared
against the ground-truth labels. Aggregated per-offset accuracy is the
"recovery profile": offset 0 is the unaffected baseline (predicted from
the uncorrupted prefix label[0..t-1]), offsets 1..N_OFFSETS-1 see the
corruption directly (offset 1) or indirectly through attention.

Indexing convention (from rewi.training.utils.build_ar_batch):
    y_inp = [BOS, label[0], label[1], ..., label[L-1]]            shape (B, L+1)
    y_tgt = [label[0], label[1], ..., label[L-1], EOS]            shape (B, L+1)
    decoder output position k predicts label[k]      (k = 0..L-1)
    decoder input at position k is BOS if k=0 else label[k-1].
So "corruption of label[t]" replaces y_inp[t+1].
    -> output at decoder position t  is UNAFFECTED (offset 0).
    -> output at decoder position t+1 is DIRECTLY AFFECTED (offset 1).

For per-(model, dataset, fold) one JSON file is written:
    {
      "single_corruption": {
        "offsets": [0, 1, 2, 3],
        "n_correct": [...],
        "n_total":   [...],
        "accuracy":  [...],
        ...
      },
      "_meta": {...}
    }

Usage:
    python eval_single_corruption.py -c <config.yaml> --checkpoint <pth> \\
        --out <out.json> [--seed S] [--n_offsets 6] [--n_corruptions_per_sample 1]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

from rewi.training.utils import build_ar_batch

from decode_study import _migrate_legacy_mha_keys, setup_tokenizer
from eval_tf_gap import build_dual_or_base, load_config, make_loader
from eval_tf_perturbation import vocab_substitution_pool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_offsets", type=int, default=4,
                   help="Number of decoder positions starting at t to score "
                        "(fixed cohort: eligibility requires t + n_offsets-1 < L).")
    p.add_argument("--n_corruptions_per_sample", type=int, default=1,
                   help="Independent corruption positions evaluated per sample.")
    p.add_argument("--min_eligibility_offset", type=int, default=1,
                   help="Require sample length L such that t can be placed "
                        "with at least this many offsets fully observable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--size_batch", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def run_single_corruption_eval(
    model,
    loader,
    cfg,
    device,
    n_offsets: int,
    n_per_sample: int,
    min_offset: int,
    seed: int,
    max_samples: int | None,
):
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    pool = torch.tensor(
        vocab_substitution_pool(cfg), dtype=torch.long, device=device,
    )
    pool_size = pool.numel()
    generator = torch.Generator(device=device).manual_seed(int(seed))

    n_correct = [0] * n_offsets
    n_total = [0] * n_offsets
    n_seen = 0
    n_skipped_short = 0
    n_corruptions_applied = 0
    n_identity_skipped = 0
    t0 = time.time()

    for x, y, len_x, len_y in loader:
        x = x.to(device)
        y_dev = y.to(device)
        y_inp_base, y_tgt = build_ar_batch(
            y_dev, len_y, PAD_ID, BOS_ID, EOS_ID, device=device,
        )
        B = y_inp_base.size(0)
        len_y_cpu = len_y.cpu().tolist()

        # Per (sample, repeat) corruption protocol. To keep memory bounded we
        # iterate repeats one at a time and re-corrupt the prefix in place.
        for _rep in range(n_per_sample):
            y_inp = y_inp_base.clone()

            # Eligibility: for a given label length L, valid t satisfies
            #   t in [0, L - 1 - min_offset]   (so t + min_offset < L)
            # We also exclude t == 0 to keep at least one ground-truth char
            # of prefix context before the corruption.
            t_per_sample = torch.full((B,), -1, dtype=torch.long, device=device)
            corrupt_id_per_sample = torch.zeros((B,), dtype=torch.long, device=device)
            for b in range(B):
                L = len_y_cpu[b]
                # Fixed cohort: require t + (n_offsets - 1) < L so every offset
                # 0..n_offsets-1 is scored for the SAME samples (no population
                # drift across offsets). Equivalent to t in [1, L - n_offsets].
                hi = L - 1 - (n_offsets - 1)
                lo = 1
                if hi < lo:
                    n_skipped_short += 1
                    continue
                t = int(torch.randint(lo, hi + 1, (1,), generator=generator, device=device).item())
                # Exclude identity replacements: redraw until the replacement
                # differs from the original character label[t] (= y_tgt[b, t]).
                orig_id = int(y_tgt[b, t].item())
                corrupt_id = orig_id
                while corrupt_id == orig_id:
                    draw = int(torch.randint(0, pool_size, (1,), generator=generator, device=device).item())
                    corrupt_id = int(pool[draw].item())
                    if corrupt_id == orig_id:
                        n_identity_skipped += 1
                t_per_sample[b] = t
                corrupt_id_per_sample[b] = corrupt_id

            # Apply corruption: y_inp[b, t+1] holds label[t] in standard layout.
            for b in range(B):
                t = int(t_per_sample[b].item())
                if t < 0:
                    continue
                y_inp[b, t + 1] = corrupt_id_per_sample[b]
                n_corruptions_applied += 1

            # One forward pass per (batch, repeat).
            out = model(x, in_lengths=len_x, y_inp=y_inp)
            logits = out["ar_logits"] if isinstance(out, dict) else out
            argmax = logits.argmax(-1)  # (B, N) where N = L_max + 1

            # Tally per-offset accuracy. Output at decoder position k predicts
            # label[k]; ground truth is y_tgt[b, k].
            for b in range(B):
                t = int(t_per_sample[b].item())
                if t < 0:
                    continue
                L = len_y_cpu[b]
                for off in range(n_offsets):
                    pos = t + off
                    if pos >= L:
                        break
                    pred_id = int(argmax[b, pos].item())
                    true_id = int(y_tgt[b, pos].item())
                    if true_id == PAD_ID or true_id == EOS_ID:
                        break
                    n_total[off] += 1
                    if pred_id == true_id:
                        n_correct[off] += 1

        n_seen += B
        if max_samples and n_seen >= max_samples:
            break

    dt = time.time() - t0
    accuracy = [
        (n_correct[off] / n_total[off]) if n_total[off] > 0 else float("nan")
        for off in range(n_offsets)
    ]
    return {
        "offsets": list(range(n_offsets)),
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
        "n_eligible": n_total[0] if n_offsets > 0 else 0,
        "n_samples_seen": n_seen,
        "n_corruptions_applied": n_corruptions_applied,
        "n_identity_skipped": n_identity_skipped,
        "n_skipped_too_short": n_skipped_short,
        "n_per_sample": n_per_sample,
        "min_offset": min_offset,
        "seed": int(seed),
        "seconds": round(dt, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.size_batch is not None:
        cfg["size_batch"] = args.size_batch
    device = torch.device(args.device)

    _, vocab_dec, PAD_ID, BOS_ID, EOS_ID = setup_tokenizer(cfg)
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
        "Validation set: {} samples, batch={}, n_offsets={}, n_per_sample={}, seed={}",
        len(dataset), cfg["size_batch"], args.n_offsets,
        args.n_corruptions_per_sample, args.seed,
    )

    result = run_single_corruption_eval(
        model, loader, cfg, device,
        n_offsets=args.n_offsets,
        n_per_sample=args.n_corruptions_per_sample,
        min_offset=args.min_eligibility_offset,
        seed=args.seed,
        max_samples=args.max_samples,
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
    payload = {"_meta": meta, "single_corruption": result}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Single-corruption recovery: accuracy per offset = {}",
        [f"{a:.4f}" if a == a else "nan" for a in result["accuracy"]],
    )
    logger.info("Wrote: {}", args.out)


if __name__ == "__main__":
    main()
