#!/usr/bin/env python3
"""Extract encoder weights from a VLM checkpoint for use in AR baseline.

Creates a checkpoint containing only the encoder state dict, formatted
so that main.py's load_checkpoint() enters the 'freeze' branch (no 'epoch' key).

Usage:
    # Extract all folds for a given VLM experiment + dataset
    python scripts/extract_vlm_encoder.py \
        --vlm_dir results/hwr2/Ablations-MMLM/GPT-2/AR-only/vlm_Qformer_gpt2_word_v2/vlm__onhw_wi_word_rh \
        --out_dir work/REWI_work/assets/pretrained_encoders/vlm_qformer_v2_onhw \
        --ckpt_name best_cer.pth

    # Extract a single fold
    python scripts/extract_vlm_encoder.py \
        --vlm_dir results/hwr2/Ablations-MMLM/GPT-2/AR-only/vlm_Qformer_gpt2_word_v2/vlm__onhw_wi_word_rh \
        --out_dir work/REWI_work/assets/pretrained_encoders/vlm_qformer_v2_onhw \
        --fold 0
"""
import argparse
import os
from pathlib import Path

import torch


def extract_encoder(ckpt_path: str, out_path: str) -> dict:
    """Load VLM checkpoint, keep only encoder.* keys, save without 'epoch'."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    full_sd = ckpt["model"]
    encoder_sd = {k: v for k, v in full_sd.items() if k.startswith("encoder.")}

    if not encoder_sd:
        raise ValueError(f"No encoder.* keys found in {ckpt_path}")

    # Save without 'epoch' so load_checkpoint enters freeze branch
    out_ckpt = {"model": encoder_sd}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out_ckpt, out_path)

    total_params = sum(v.numel() for v in encoder_sd.values())
    print(f"  Extracted {len(encoder_sd)} encoder keys ({total_params:,} params)")
    print(f"  Saved to {out_path}")
    return encoder_sd


def main():
    parser = argparse.ArgumentParser(description="Extract encoder from VLM checkpoint")
    parser.add_argument("--vlm_dir", required=True, help="VLM experiment dataset dir (contains fold_*/)")
    parser.add_argument("--out_dir", required=True, help="Output directory for extracted checkpoints")
    parser.add_argument("--fold", type=int, default=None, help="Single fold to extract (default: all)")
    parser.add_argument("--ckpt_name", default="best_cer.pth", help="Checkpoint filename")
    args = parser.parse_args()

    vlm_dir = Path(args.vlm_dir)
    out_dir = Path(args.out_dir)

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = sorted(
            int(d.name.split("_")[1])
            for d in vlm_dir.iterdir()
            if d.is_dir() and d.name.startswith("fold_")
        )

    print(f"Extracting encoder from: {vlm_dir}")
    print(f"Folds: {folds}")
    print(f"Checkpoint: {args.ckpt_name}")
    print()

    for fold in folds:
        # Subdirectory can be "0" (fold_0) or match the fold index
        ckpt_path = vlm_dir / f"fold_{fold}" / "0" / "checkpoints" / args.ckpt_name
        if not ckpt_path.exists():
            ckpt_path = vlm_dir / f"fold_{fold}" / str(fold) / "checkpoints" / args.ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] fold_{fold}: no checkpoint found")
            continue

        out_path = out_dir / f"encoder_fold{fold}.pth"
        print(f"[fold_{fold}] {ckpt_path}")
        extract_encoder(str(ckpt_path), str(out_path))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
