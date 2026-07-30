#!/usr/bin/env python3
"""Presentation figure: noise-injection mode comparison, HWRFormer row only.

Single-axes grouped bar chart (4 dataset groups x 6 series) of five-fold
mean CER at p_ni = 0.15. Data: publications/paper2_lncs_overleaf/
xs_numbers.json, key table4_corruption_modes -- the same source as the
thesis figure noise_modes_hwrformer_vs_l.pdf (thesis "HWRFormer" row), so
the bars match the thesis mode-ablation table by construction. Error bars
are intentionally omitted for slide legibility (the OnHW-WI fold
bimodality has its own backup slide).

Output: presentation/figures/noise_modes_hwrformer.pdf

Run from anywhere:
    python plot_noise_modes_presentation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

XS_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/xs_numbers.json"
)
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/noise_modes_hwrformer.pdf"
)

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word_rh", "OnHW-WI"),
    ("onhw_wd_word_rh", "OnHW-WD"),
    ("wi_word_hw6_meta", "Priv-words"),
    ("wi_sent_hw6_meta", "Priv-sent"),
]

SERIES: list[tuple[str, str, str]] = [
    ("__baseline__", "no-noise baseline", "#7f7f7f"),
    ("uniform", "uniform", "#1f77b4"),
    ("bigramright", "bigram-right", "#2ca02c"),
    ("bigramleft", "bigram-left", "#ff7f0e"),
    ("selfconf", "self-confusion", "#9467bd"),
    ("adjacentswap", "adjacent-swap", "#d62728"),
]


def main() -> None:
    with open(XS_JSON) as f:
        table = json.load(f)["table4_corruption_modes"]

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    n_series = len(SERIES)
    group_w = n_series + 1.5
    bar_w = 0.92

    for j, (key, _label, color) in enumerate(SERIES):
        xs, ys = [], []
        for g, (ds_key, _ds_label) in enumerate(DATASETS):
            cell = table[ds_key][key]
            xs.append(g * group_w + j)
            ys.append(cell["cer_mean"])
        ax.bar(xs, ys, bar_w, color=color, edgecolor="black", linewidth=0.45)

    centers = [g * group_w + (n_series - 1) / 2 for g in range(len(DATASETS))]
    ax.set_xticks(centers, [lab for _, lab in DATASETS], fontsize=12)
    ax.set_ylabel("CER (%)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=c, edgecolor="black", label=lab) for _, lab, c in SERIES]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10.5, frameon=False, handlelength=1.6)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")

    # Print the plotted values for verification against the thesis table.
    for ds_key, ds_label in DATASETS:
        row = {k: table[ds_key][k]["cer_mean"] for k, _, _ in SERIES}
        print(ds_label, row)


if __name__ == "__main__":
    main()
