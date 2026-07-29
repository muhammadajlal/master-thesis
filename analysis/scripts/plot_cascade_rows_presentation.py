#!/usr/bin/env python3
"""Presentation-only variants of the cascade/noise-injection figure.

Splits the 2x4 grid of cascade_analysis.py (thesis fig) into two standalone
1x4 rows so each can fill a 16:9 slide at full width with its own dataset
titles and legend:

  - presentation/figures/cascade_ecdf_row.pdf      (per-model eCDF of e_tilde)
  - presentation/figures/cascade_position_row.pdf  (aligned per-ref-position error)

Reads the cached CSVs written by cascade_analysis.py (no realignment):
  analysis/aligned_edit_per_sample.csv
  analysis/aligned_edit_position_curves.csv
  analysis/cascade_noise_injection_summary.csv

Run from anywhere:
    python plot_cascade_rows_presentation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
ANALYSIS = REPO / "work" / "REWI_work" / "analysis"
OUT_DIR = REPO / "presentation" / "figures"

# Constants mirrored from cascade_analysis.py
TASK_AR = "AR-only"
TASK_NOISE = "Noise (uniform p=0.15)"
LABEL_AR = "HWRFormer"
LABEL_NOISE = "HWRFormer + noise injection"
AR_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"
POS_CAP_WORD = 15
POS_CAP_SENT = 40
MIN_SAMPLES_AT_POS = 50
DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word", "OnHW WI word"),
    ("onhw_wd_word", "OnHW WD word"),
    ("priv_word", "Private word"),
    ("priv_sent", "Private sentence"),
]

LEGEND_HANDLES_ECDF = [
    Line2D([0], [0], color=AR_COLOR, lw=2.2, label=LABEL_AR),
    Line2D([0], [0], color=NOISE_COLOR, lw=2.2, label=LABEL_NOISE),
    Line2D([0], [0], color="0.4", linestyle="--", lw=1.0,
           label=r"per-model mean $\tilde{e}$"),
]
LEGEND_HANDLES_POS = [
    Line2D([0], [0], color=AR_COLOR, marker="o", markersize=4, lw=2,
           label=LABEL_AR),
    Line2D([0], [0], color=NOISE_COLOR, marker="s", markersize=4, lw=2,
           label=LABEL_NOISE),
]


def plot_ecdf_row(per_sample: pd.DataFrame, summary: pd.DataFrame,
                  out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14.4, 3.3))
    for col, (dataset_task, label) in enumerate(DATASETS):
        ax = axes[col]
        ds_samples = per_sample[per_sample["dataset_task"] == dataset_task]
        for task, color in [(TASK_AR, AR_COLOR), (TASK_NOISE, NOISE_COLOR)]:
            evals = ds_samples[ds_samples["task"] == task]["edit_dist_norm"].to_numpy()
            if evals.size == 0:
                continue
            e_sorted = np.sort(evals)
            yvals = np.arange(1, e_sorted.size + 1) / e_sorted.size
            ax.plot(e_sorted, yvals, color=color, linewidth=2.2)
            ax.axvline(float(np.mean(evals)), color=color, linestyle="--",
                       linewidth=1.0, alpha=0.7)
        ax.set_xlim(0.0, 1.5)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(r"$\tilde{e} = d(\hat{y}, y) / \max(1, |y|)$", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"$P(\tilde{e}_i \leq x)$", fontsize=11)

        res = summary[summary["dataset_task"] == dataset_task].iloc[0]
        text = (
            r"$\bar{\Delta\tilde{e}} = "
            f"{res['delta_e_norm_mean_pp']:+.2f}$ pp"
            "\n95% CI = ["
            f"{res['delta_e_norm_ci_lo_pp']:+.2f}, "
            f"{res['delta_e_norm_ci_hi_pp']:+.2f}"
            "] pp"
        )
        ax.text(0.97, 0.03, text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.7", alpha=0.92))

    fig.legend(handles=LEGEND_HANDLES_ECDF, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=11,
               frameon=False, columnspacing=2.0, handlelength=2.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def plot_position_row(curves: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14.4, 3.3))
    for col, (dataset_task, label) in enumerate(DATASETS):
        ax = axes[col]
        pos_cap = POS_CAP_SENT if "sent" in dataset_task else POS_CAP_WORD
        for task, color, marker in [(TASK_AR, AR_COLOR, "o"),
                                    (TASK_NOISE, NOISE_COLOR, "s")]:
            sub = curves[
                (curves["dataset_task"] == dataset_task)
                & (curves["task"] == task)
                & (curves["ref_pos"] < pos_cap)
                & (curves["n_samples_at_pos"] >= MIN_SAMPLES_AT_POS)
            ].sort_values("ref_pos")
            ax.plot(sub["ref_pos"].to_numpy(),
                    sub["err_rate"].to_numpy() * 100.0,
                    marker=marker, color=color, linewidth=2, markersize=4)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"Reference position $k$", fontsize=10)
        if col == 0:
            ax.set_ylabel("Aligned per-ref-position error (%)", fontsize=10.5)

    fig.legend(handles=LEGEND_HANDLES_POS, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=11,
               frameon=False, columnspacing=2.0, handlelength=2.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main() -> None:
    per_sample = pd.read_csv(ANALYSIS / "aligned_edit_per_sample.csv")
    curves = pd.read_csv(ANALYSIS / "aligned_edit_position_curves.csv")
    summary = pd.read_csv(ANALYSIS / "cascade_noise_injection_summary.csv")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ecdf_row(per_sample, summary, OUT_DIR / "cascade_ecdf_row.pdf")
    plot_position_row(curves, OUT_DIR / "cascade_position_row.pdf")


if __name__ == "__main__":
    main()
