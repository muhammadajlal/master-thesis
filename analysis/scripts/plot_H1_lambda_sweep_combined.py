#!/usr/bin/env python3
"""Presentation figure: HWR-GPT (MLP + CTC) lambda_ctc sweep, both datasets.

One panel per dataset (OnHW WI word, Private word), CER and WER together in
each panel on twin y-axes (CER left/blue, WER right/orange). Reads the
metrics CSVs written by plot_H1_lambda_sweep_onhw.py / _word.py — no
re-aggregation of results.json.

Output: presentation/figures/h1_lambda_sweep_combined.pdf

Run from anywhere:
    python plot_H1_lambda_sweep_combined.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
SWEEP = REPO / "results" / "hwr2" / "H1_LambdaSweep"
OUT_PDF = REPO / "presentation" / "figures" / "h1_lambda_sweep_combined.pdf"

CER_COLOR = "#1f77b4"
WER_COLOR = "#ff7f0e"
SELECTED_COLOR = "#9467bd"

PANELS = [
    ("OnHW WI word", SWEEP / "h1_lambda_sweep_onhw_metrics.csv", 0.2),
    ("Private word", SWEEP / "h1_lambda_sweep_word_metrics.csv", 0.6),
]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax_cer, (title, csv_path, selected_lambda) in zip(axes, PANELS):
        df = pd.read_csv(csv_path)
        ref = df[df["lambda_ctc"] == "mlp_noCTC_ref"].iloc[0]
        pts = df[df["lambda_ctc"] != "mlp_noCTC_ref"].astype(float)

        ax_wer = ax_cer.twinx()

        ax_cer.axvspan(selected_lambda - 0.015, selected_lambda + 0.015,
                       color=SELECTED_COLOR, alpha=0.22, zorder=1)
        ax_cer.axvline(selected_lambda, color=SELECTED_COLOR, linestyle="-",
                       linewidth=1.8, alpha=0.95, zorder=2)

        lam = pts["lambda_ctc"].to_numpy()
        for ax, metric, color, marker in [
            (ax_cer, "cer", CER_COLOR, "o"),
            (ax_wer, "wer", WER_COLOR, "s"),
        ]:
            mean = pts[f"{metric}_mean"].to_numpy()
            sem = pts[f"{metric}_sem"].to_numpy()
            ax.plot(lam, mean, color=color, linewidth=2, marker=marker,
                    markersize=5, zorder=5)
            ax.fill_between(lam, mean - sem, mean + sem, color=color,
                            alpha=0.15, linewidth=0, zorder=4)
            ax.axhline(float(ref[f"{metric}_mean"]), color=color,
                       linestyle=":", linewidth=1.4, alpha=0.9, zorder=3)
            ax.tick_params(axis="y", labelcolor=color)

        ax_cer.set_title(title, fontsize=12)
        ax_cer.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax_cer.set_ylabel("CER (%)", fontsize=11, color=CER_COLOR)
        ax_wer.set_ylabel("WER (%)", fontsize=11, color=WER_COLOR)
        ax_cer.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], color=CER_COLOR, marker="o", markersize=5,
               linewidth=2, label=r"CER, 5-fold mean $\pm$ SEM (left axis)"),
        Line2D([0], [0], color=WER_COLOR, marker="s", markersize=5,
               linewidth=2, label=r"WER, 5-fold mean $\pm$ SEM (right axis)"),
        Line2D([0], [0], color="0.2", linestyle=":", linewidth=1.4,
               label="MLP (no CTC) reference"),
        Line2D([0], [0], color=SELECTED_COLOR, linestyle="-", linewidth=1.8,
               label=r"best-observed $\lambda_{\mathrm{ctc}}$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=2, fontsize=10, frameon=False,
               handlelength=2.2, columnspacing=1.8)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
