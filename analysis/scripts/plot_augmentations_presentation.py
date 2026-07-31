#!/usr/bin/env python3
"""Presentation figure: the four training-time augmentations on a real signal.

Applies the pipeline's own transforms (rewi.dataset.transforms, with the exact
parameters used in HRDataset) to one channel of one public OnHW-words500
training sample, so the panels show what the augmenter actually does. Public
data only.

Output: presentation/figures/augmentations.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work")
from rewi.dataset.transforms import AddNoise, Drift, Dropout, TimeWarp  # noqa: E402

SAMPLE = ("/home/woody/iwso/iwso214h/imu-hwr/data/onhw_wi_word_rh/"
          "data/0/train/00000000.csv")
OUT = ("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
       "augmentations.pdf")
CHANNEL = 12         # smooth channel: augmentation effects stay legible on a projector
WINDOW = 110         # zoom, so per-sample effects are visible on a projector
SEED = 7

BASE = "#9aa5b1"     # original trace
AUG = "#1f77b4"      # deck blue

PANELS = [
    ("Multiplicative noise", AddNoise(scale=0.05, kind="multiplicative"),
     r"i.i.d. per sample, $\sigma=5\,\%$"),
    ("Smooth drift", Drift(0.1, 40, "multiplicative"),
     r"slow gain wander, $\pm10\,\%$"),
    ("Segment dropout", Dropout(size=(5, 10), per_channel=True),
     "5\u201310 samples held constant"),
    ("Time warping", TimeWarp(5, 4),
     "local speed changes, up to $4\\times$"),
]


def main() -> None:
    x = np.loadtxt(SAMPLE, delimiter=";")[:WINDOW]
    np.random.seed(SEED)

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.05), sharey=True)
    base = x[:, CHANNEL]
    scale = base.std()
    for ax, (name, aug, sub) in zip(axes, PANELS):
        # these augmenters are stochastic; show the first draw whose effect is
        # visible at slide size rather than an arbitrary near-identity draw
        for seed in range(SEED, SEED + 400):
            np.random.seed(seed)
            y = aug(x.copy())[:, CHANNEL]
            if np.abs(y - base).mean() / scale > 0.03:
                break
        ax.plot(x[:, CHANNEL], color=BASE, lw=1.6, label="original")
        ax.plot(y, color=AUG, lw=1.4, label="augmented")
        ax.set_title(name, fontsize=12.5, pad=18)
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color="#444444")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    axes[0].legend(fontsize=9.5, frameon=False, loc="lower left",
                   bbox_to_anchor=(-0.02, -0.04), ncol=2,
                   handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(top=0.74, bottom=0.04, left=0.01, right=0.99,
                        wspace=0.09)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
