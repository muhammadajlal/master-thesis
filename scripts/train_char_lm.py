#!/usr/bin/env python3
"""
Train a character-level KenLM n-gram language model from dataset labels.

Reads training-fold annotations from the OnHW500 dataset JSON and produces
a character-level ARPA model that can be compiled to binary format.

Usage:
    # From work/REWI_work/:
    python scripts/train_char_lm.py \
        --dataset ../../data/onhw_wi_word_rh \
        --folds 0 1 2 3 4 \
        --order 5 \
        --outdir lm

    # Build binary for faster loading:
    build_binary lm/char_5gram.arpa lm/char_5gram.binary

Output:
    lm/char_5gram.arpa   — ARPA n-gram model
    lm/corpus.txt         — training corpus (one line per word, chars space-separated)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def extract_labels(dataset_dir: str, folds: list[int], split: str = "train") -> list[str]:
    """Extract text labels from the dataset JSON."""
    json_path = os.path.join(dataset_dir, f"{split}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    annotations = data.get("annotations", {})
    for fold_key, fold_annots in annotations.items():
        if int(fold_key) in folds:
            for annot in fold_annots:
                label = annot.get("label", "").strip()
                if label:
                    labels.append(label)

    return labels


def chars_to_kenlm_line(text: str) -> str:
    """Convert a text string to space-separated characters for KenLM.

    Spaces become <space> tokens.
    """
    tokens = []
    for ch in text:
        if ch == " ":
            tokens.append("<space>")
        else:
            tokens.append(ch)
    return " ".join(tokens)


def write_corpus(labels: list[str], outpath: str) -> None:
    """Write corpus file for KenLM training."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for label in labels:
            line = chars_to_kenlm_line(label)
            f.write(line + "\n")
    print(f"Wrote {len(labels)} lines to {outpath}")


def train_kenlm(
    corpus_path: str,
    arpa_path: str,
    order: int = 5,
    memory: str = "4G",
    temp_prefix: str | None = None,
) -> None:
    """Train KenLM n-gram model."""
    # Try to find lmplz: first in vendor/kenlm/bin, then in PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_lmplz = os.path.join(script_dir, "..", "vendor", "kenlm", "bin", "lmplz")
    if os.path.isfile(vendor_lmplz):
        lmplz_bin = vendor_lmplz
    else:
        lmplz_bin = "lmplz"

    cmd = [
        lmplz_bin,
        "-o", str(order),
        "-S", memory,
        "--text", corpus_path,
        "--arpa", arpa_path,
        "--discount_fallback",  # handle sparse data gracefully
    ]
    if temp_prefix:
        cmd.extend(["-T", temp_prefix])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"lmplz stderr:\n{result.stderr}")
        sys.exit(1)
    print(f"ARPA model written to {arpa_path}")


def build_binary(arpa_path: str, binary_path: str) -> None:
    """Convert ARPA to binary format for faster loading."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_bb = os.path.join(script_dir, "..", "vendor", "kenlm", "bin", "build_binary")
    bb_bin = vendor_bb if os.path.isfile(vendor_bb) else "build_binary"

    cmd = [bb_bin, arpa_path, binary_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"build_binary stderr:\n{result.stderr}")
        print("Warning: binary conversion failed. Use ARPA file directly.")
        return
    print(f"Binary model written to {binary_path}")


def train_one(dataset: str, folds: list[int], order: int, outdir: str,
              split: str = "train", skip_binary: bool = False, suffix: str = "",
              memory: str = "4G", temp_prefix: str | None = None):
    """Train a single LM from specified folds."""
    labels = extract_labels(dataset, folds, split)
    print(f"Extracted {len(labels)} labels from folds {folds}")

    if not labels:
        print("ERROR: No labels found!")
        sys.exit(1)

    all_chars = set()
    for label in labels:
        all_chars.update(label)
    print(f"Unique characters: {len(all_chars)}")
    print(f"Characters: {sorted(all_chars)}")
    print(f"Avg label length: {sum(len(l) for l in labels) / len(labels):.1f}")

    stem = f"char_{order}gram{suffix}"
    corpus_path = os.path.join(outdir, f"corpus{suffix}.txt")
    write_corpus(labels, corpus_path)

    arpa_path = os.path.join(outdir, f"{stem}.arpa")
    train_kenlm(corpus_path, arpa_path, order, memory=memory, temp_prefix=temp_prefix)

    if not skip_binary:
        binary_path = os.path.join(outdir, f"{stem}.binary")
        build_binary(arpa_path, binary_path)


def main():
    parser = argparse.ArgumentParser(description="Train character-level KenLM")
    parser.add_argument(
        "--dataset", type=str,
        default="../../data/onhw_wi_word_rh",
        help="Path to dataset directory",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--order", type=int, default=5, help="N-gram order")
    parser.add_argument("--outdir", type=str, default="lm", help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--skip_binary", action="store_true", help="Skip binary conversion")
    parser.add_argument(
        "--memory", type=str,
        default=os.environ.get("KENLM_MEM", "4G"),
        help="KenLM lmplz memory for -S (e.g., 4G, 16G, 50%).",
    )
    parser.add_argument(
        "--temp_prefix", type=str,
        default=os.environ.get("KENLM_TMP", None),
        help="Optional temp directory prefix for lmplz -T (e.g., $SLURM_TMPDIR).",
    )
    parser.add_argument(
        "--per_fold", action="store_true",
        help="Train a separate LM per CV fold (excludes the test fold). "
             "Produces lm/fold_0/char_5gram.binary etc.",
    )
    args = parser.parse_args()

    all_folds = [0, 1, 2, 3, 4]

    if args.per_fold:
        # Train one LM per fold, each excluding the held-out fold.
        #
        # The corpus is built from val.json, which is the disjoint
        # writer-independent fold partition (each of the 25,199 samples appears
        # under exactly one key). Taking val.json folds != test_fold therefore
        # yields the exact training portion for that fold, with every held-out
        # writer excluded and no sample duplicated.
        #
        # NOTE: train.json[k] is the per-fold *complement* (all samples except
        # val fold k), so concatenating several train.json keys — as an earlier
        # version did — re-introduces the held-out fold's writers (4x) and
        # inflates their word frequencies. Building from the val.json partition
        # avoids that leak.
        for test_fold in all_folds:
            train_folds = [f for f in all_folds if f != test_fold]
            fold_outdir = os.path.join(args.outdir, f"fold_{test_fold}")
            print(f"\n{'═' * 60}")
            print(f"  Training LM for fold {test_fold} (val.json partition folds {train_folds})")
            print(f"{'═' * 60}")
            train_one(
                args.dataset, train_folds, args.order, fold_outdir,
                split="val", skip_binary=args.skip_binary,
                memory=args.memory, temp_prefix=args.temp_prefix,
            )
        print(f"\nPer-fold LMs saved under {args.outdir}/fold_*/")
    else:
        # Train a single LM on specified folds
        train_one(
            args.dataset, args.folds, args.order, args.outdir,
            args.split, args.skip_binary,
            memory=args.memory, temp_prefix=args.temp_prefix,
        )


if __name__ == "__main__":
    main()
