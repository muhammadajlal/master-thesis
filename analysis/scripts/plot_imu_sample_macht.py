#!/usr/bin/env python3
"""Regenerate thesis/figures/imu_sample_macht.pdf.

Representative OnHW500 WI raw-signal figure (fold 0, val sample id 377,
writer 5, label "macht", T=160). Recreates the original 5-panel layout
(front acc / front gyro / front mag / back acc / tip force, all channels
z-normalized per channel) with shortened y-axis labels so adjacent
subplot labels no longer collide.

Run:
    python analysis/scripts/plot_imu_sample_macht.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_CSV = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/data/onhw_wi_word_rh/data/0/val/00000377.csv"
)
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/imu_sample_macht.pdf"
)

# 13-channel layout: front IMU (acc, gyro, mag), back acc, tip force
GROUPS = [
    ("Front Acc.", slice(0, 3)),
    ("Front Gyro", slice(3, 6)),
    ("Front Mag.", slice(6, 9)),
    ("Back Acc.", slice(9, 12)),
    ("Tip Force", slice(12, 13)),
]
AXIS_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # x, y, z
AXIS_NAMES = ["x", "y", "z"]
FORCE_COLOR = "0.45"


def main() -> None:
    raw = np.loadtxt(DATA_CSV, delimiter=";")
    T = raw.shape[0]
    # z-normalize per channel (as in the model input pipeline)
    z = (raw - raw.mean(axis=0)) / np.clip(raw.std(axis=0), 1e-8, None)

    fig, axes = plt.subplots(5, 1, figsize=(7.2, 5.8), sharex=True)
    fig.suptitle(
        'IMU sample: ground truth "macht" (OnHW500 WI, fold 0, T=160)',
        fontsize=11,
    )

    for ax, (name, sl) in zip(axes, GROUPS):
        chans = z[:, sl]
        if chans.shape[1] == 3:
            for j in range(3):
                ax.plot(chans[:, j], color=AXIS_COLORS[j], linewidth=1.0,
                        label=AXIS_NAMES[j])
            ax.legend(loc="upper right", ncol=3, fontsize=7,
                      framealpha=0.85, borderpad=0.25,
                      columnspacing=1.0, handlelength=1.4)
        else:
            ax.plot(chans[:, 0], color=FORCE_COLOR, linewidth=1.2)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel(r"Sample Index (time, $f_s$ = 100 Hz)", fontsize=9.5)
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PDF}  (T={T})")


if __name__ == "__main__":
    main()
