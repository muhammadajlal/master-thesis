#!/usr/bin/env python3
"""Generate thesis/figures/noise_modes_hwrformer_vs_l.pdf.

Dual-variant variant of plot_corruption_modes_bars.py. Rows = HWRFormer
configuration (HWRFormer / HWRFormer-L), columns = dataset (OnHW WI word,
OnHW WD word, private word, private sentence). Each panel is a 6-bar
group with error bars: no-noise baseline, uniform, bigram-right,
bigram-left, self-confusion, adjacent-swap at p_replace = 0.15.

Data sources:
    xs row -> publications/paper2_lncs_overleaf/xs_numbers.json
              key: table4_corruption_modes
    s  row -> publications/paper2_lncs_overleaf/s_numbers.json
              key: table4_corruption_modes_s, baseline_no_noise_s

Missing/null cells are silently skipped (bar omitted).

Run from anywhere:
    python plot_corruption_modes_bars_dual.py
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
S_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/s_numbers.json"
)
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/noise_modes_hwrformer_vs_l.pdf"
)

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word_rh", "OnHW WI word"),
    ("onhw_wd_word_rh", "OnHW WD word"),
    ("wi_word_hw6_meta", "Private word"),
    ("wi_sent_hw6_meta", "Private sentence"),
]

# (key, display label, colour) -- matches plot_corruption_modes_bars.py palette.
MODES: list[tuple[str, str, str]] = [
    ("uniform", "uniform", "#1f77b4"),
    ("bigramright", "bigram-right", "#2ca02c"),
    ("bigramleft", "bigram-left", "#ff7f0e"),
    ("selfconf", "self-confusion", "#9467bd"),
    ("adjacentswap", "adjacent-swap", "#d62728"),
]

SERIES: list[tuple[str, str, str]] = [
    ("__baseline__", "no-noise baseline", "#7f7f7f"),
    *MODES,
]

ROW_TITLES = ["HWRFormer", "HWRFormer-L"]


def _get(d, *keys, default=None):
    """Safe nested-dict access."""
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def xs_cell(xs_json: dict, dataset: str, series_key: str) -> tuple[float | None, float | None]:
    """Extract (cer_mean, cer_std) for the xs (baseline) variant."""
    cell = _get(xs_json, "table4_corruption_modes", dataset, series_key)
    if cell is None:
        return None, None
    if cell.get("status") and cell.get("status") != "ready":
        return None, None
    return cell.get("cer_mean"), cell.get("cer_std")


def s_cell(s_json: dict, dataset: str, series_key: str) -> tuple[float | None, float | None]:
    """Extract (cer_mean, cer_std) for the s (scaled) variant."""
    if series_key == "__baseline__":
        cell = _get(s_json, "baseline_no_noise_s", dataset)
    else:
        cell = _get(s_json, "table4_corruption_modes_s", series_key, dataset)
    if cell is None:
        return None, None
    return cell.get("cer_mean"), cell.get("cer_std")


def main() -> None:
    with open(XS_JSON) as f:
        xs = json.load(f)
    with open(S_JSON) as f:
        s = json.load(f)

    # row 0 = xs, row 1 = s
    cells: list[list[dict[str, tuple[float | None, float | None]]]] = []
    for row_idx, getter in enumerate((xs_cell, s_cell)):
        row_data = []
        src = xs if row_idx == 0 else s
        for ds_key, _ in DATASETS:
            row: dict[str, tuple[float | None, float | None]] = {}
            for series_key, _label, _color in SERIES:
                row[series_key] = getter(src, ds_key, series_key)
            row_data.append(row)
        cells.append(row_data)

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0), sharex=False)

    # Compute per-column ymax so each dataset shares a y-scale across the two rows.
    col_ymax = []
    for col in range(len(DATASETS)):
        vals: list[float] = []
        for row in range(2):
            for series_key, _, _ in SERIES:
                m, sd = cells[row][col][series_key]
                if m is None:
                    continue
                vals.append(float(m) + (float(sd) if sd is not None else 0.0))
        col_ymax.append(max(vals) * 1.18 if vals else 1.0)

    bar_xs = np.arange(len(SERIES))
    bar_w = 0.75

    for row in range(2):
        for col, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[row, col]
            row_cells = cells[row][col]
            any_drawn = False
            for j, (series_key, _label, color) in enumerate(SERIES):
                m, sd = row_cells[series_key]
                if m is None:
                    continue
                yerr = float(sd) if sd is not None else 0.0
                ax.bar(
                    bar_xs[j],
                    float(m),
                    bar_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    yerr=yerr,
                    error_kw={"elinewidth": 0.9, "capsize": 2.2, "capthick": 0.9, "ecolor": "black"},
                )
                any_drawn = True

            if not any_drawn:
                ax.text(
                    0.5, 0.5, "no data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="0.45", style="italic",
                )

            ax.set_xticks(bar_xs)
            ax.set_xticklabels([])  # bar identity carried by the legend
            ax.set_ylim(0, col_ymax[col])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=9.5)
            ax.grid(axis="y", alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)

            if row == 0:
                ax.set_title(ds_label, fontsize=11.5)
            if col == 0:
                ax.set_ylabel(f"{ROW_TITLES[row]}\nCER (%)", fontsize=11.0)
            else:
                ax.set_ylabel("CER (%)", fontsize=10.0)

    handles = [Patch(facecolor=c, edgecolor="black", label=lab) for _, lab, c in SERIES]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=len(SERIES),
        fontsize=10.0,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
