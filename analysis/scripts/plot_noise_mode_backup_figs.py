#!/usr/bin/env python3
"""Presentation backup figures explaining the structured noise modes.

Outputs (presentation/figures/):
  - confusion_matrix_priv_word_f0.pdf : self-confusion substitution counts,
    private word fold 0, lowercase block heatmap (row = true char X,
    column = predicted char Y). Source: results/hwr2/confusion_matrices/
    wi_word_hw6_meta/fold_0.npy (extract_confusion_matrix.py output).
  - bigram_rows_priv_word_f0.pdf : the two corpus-statistics rows the bigram
    modes consult for the slide example "mach" (replaced char c, left
    neighbor a), counted from the fold-0 TRAINING labels of the private
    word dataset, mirroring _build_bigram_lookup in rewi/training/loops.py.

Run from anywhere:
    python plot_noise_mode_backup_figs.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
OUT_DIR = REPO / "presentation" / "figures"

CONF_NPY = (REPO / "results" / "hwr2" / "confusion_matrices"
            / "wi_word_hw6_meta" / "fold_0.npy")
TRAIN_JSON = REPO / "data" / "wi_word_hw6_meta" / "train.json"
FOLD = "0"

WORD_CATS = ["", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
             "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
             "Z", "Ä", "Ö", "Ü", "a", "b", "c", "d", "e", "f", "g", "h", "i",
             "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
             "w", "x", "y", "z", "ä", "ö", "ü", "ß"]

CER_COLOR = "#1f77b4"


def plot_confusion() -> None:
    M = np.load(CONF_NPY)
    labels = list(WORD_CATS) + ["BOS", "EOS"]
    labels = labels[: M.shape[0]]
    lo, hi = labels.index("a"), labels.index("z") + 1
    block = M[lo:hi, lo:hi]

    fig, ax = plt.subplots(figsize=(7.6, 6.9))
    im = ax.imshow(np.log1p(block), cmap="Blues")
    ax.set_xticks(range(hi - lo), labels[lo:hi], fontsize=9)
    ax.set_yticks(range(hi - lo), labels[lo:hi], fontsize=9)
    ax.set_xlabel("predicted character $Y$", fontsize=11)
    ax.set_ylabel("true character $X$", fontsize=11)
    for i in range(hi - lo):
        for j in range(hi - lo):
            v = int(block[i, j])
            if v >= 15:
                dark = np.log1p(v) > 0.7 * np.log1p(block.max())
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + substitution count)")
    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix_priv_word_f0.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def plot_bigram_rows() -> None:
    with open(TRAIN_JSON) as f:
        ann = json.load(f)["annotations"][FOLD]
    succ_a: Counter = Counter()
    succ_c: Counter = Counter()
    for rec in ann:
        lab = rec["label"]
        for x, y in zip(lab, lab[1:]):
            if x == "a":
                succ_a[y] += 1
            elif x == "c":
                succ_c[y] += 1

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
    for ax, cnt, ch, mode in [
        (axes[0], succ_a, "a", "bigram-left row for the example\n(left neighbor of the hit position)"),
        (axes[1], succ_c, "c", "bigram-right row for the example\n(the replaced character itself)"),
    ]:
        top = cnt.most_common(8)
        total = sum(cnt.values())
        chars = [c for c, _ in top]
        probs = [100.0 * n / total for _, n in top]
        ax.bar(range(len(top)), probs, color=CER_COLOR, alpha=0.85)
        ax.set_xticks(range(len(top)), chars, fontsize=12)
        ax.set_ylabel("P(next char) (%)", fontsize=10)
        ax.set_title(f"What follows `{ch}' in the training labels?\n{mode}",
                     fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.4, f"{p:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "bigram_rows_priv_word_f0.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion()
    plot_bigram_rows()
