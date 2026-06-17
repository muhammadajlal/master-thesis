#!/usr/bin/env python3
"""Generate thesis/figures/p_replace_sweep_baseline_vs_scaled.pdf.

Dual-variant version of plot_corruption_p_sweep.py.
Rows = HWRFormer configuration (baseline xs / scaled s),
columns = 4 datasets (OnHW WI word, OnHW WD word, private word,
private sentence). Each panel: CER (left axis, solid line, circle
markers) and WER (right axis, dashed line, square markers) vs
p_replace in {0.00, 0.05, 0.10, 0.15, 0.20, 0.30} with shaded SEM
bands. Dotted vertical at p=0.15.

Data sources:
    xs row -> publications/paper2_lncs_overleaf/xs_numbers.json
              keys: figure_pic_sweep (uniform sweep) +
                    table4_corruption_modes (p=0.00 no-noise baseline)
    s  row -> publications/paper2_lncs_overleaf/s_numbers.json
              key: figure_pic_sweep_s (already includes p=0.00)

Missing cells render as faded markers (alpha=0.25, no error band) or
omit the point if neither row has data; panels with no data at all
print 'no data'.

Run from anywhere:
    python plot_corruption_p_sweep_dual.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

XS_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/xs_numbers.json"
)
S_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/s_numbers.json"
)
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/p_replace_sweep_baseline_vs_scaled.pdf"
)

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word_rh", "OnHW WI word"),
    ("onhw_wd_word_rh", "OnHW WD word"),
    ("wi_word_hw6_meta", "Private word"),
    ("wi_sent_hw6_meta", "Private sentence"),
]

P_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_P = 0.15

ROW_TITLES = ["HWRFormer (baseline)", "HWRFormer (scaled)"]

CER_COLOR = "#1f77b4"
WER_COLOR = "#ff7f0e"


def _std_to_sem(std: float | None, n: int | None) -> float:
    if std is None or n is None or n <= 1:
        return 0.0
    try:
        return float(std) / math.sqrt(int(n))
    except (TypeError, ValueError):
        return 0.0


def xs_point(xs_json: dict, dataset: str, p: float) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (cer_mean, cer_sem, wer_mean, wer_sem) for HWRFormer (baseline)."""
    if p == 0.00:
        cell = (
            xs_json.get("table4_corruption_modes", {})
            .get(dataset, {})
            .get("__baseline__")
        )
    else:
        key = f"p_{p:.2f}"
        cell = xs_json.get("figure_pic_sweep", {}).get(dataset, {}).get(key)
    if cell is None or (cell.get("status") and cell.get("status") != "ready"):
        return None, None, None, None
    cer_mean = cell.get("cer_mean")
    wer_mean = cell.get("wer_mean")
    n = cell.get("n_folds")
    cer_sem = _std_to_sem(cell.get("cer_std"), n)
    wer_sem = _std_to_sem(cell.get("wer_std"), n)
    return cer_mean, cer_sem, wer_mean, wer_sem


def s_point(s_json: dict, dataset: str, p: float) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (cer_mean, cer_sem, wer_mean, wer_sem) for HWRFormer (scaled)."""
    key = f"{p:.2f}"
    cell = s_json.get("figure_pic_sweep_s", {}).get(dataset, {}).get(key)
    if cell is None:
        return None, None, None, None
    cer_mean = cell.get("cer_mean")
    wer_mean = cell.get("wer_mean")
    n = cell.get("n_folds")
    cer_sem = _std_to_sem(cell.get("cer_std"), n)
    wer_sem = _std_to_sem(cell.get("wer_std"), n)
    return cer_mean, cer_sem, wer_mean, wer_sem


def _draw_metric(ax, xs, means, sems, color, marker, linestyle):
    """Draw curve + SEM band; skip None points but connect across them."""
    pts = [(x, m, s) for x, m, s in zip(xs, means, sems) if m is not None]
    if not pts:
        return
    xs_p = [p[0] for p in pts]
    means_p = [float(p[1]) for p in pts]
    sems_p = [float(p[2]) if p[2] is not None else 0.0 for p in pts]
    ax.plot(
        xs_p, means_p,
        color=color, marker=marker, linestyle=linestyle,
        linewidth=2.0, markersize=5.0, zorder=5,
    )
    ax.fill_between(
        xs_p,
        [m - s for m, s in zip(means_p, sems_p)],
        [m + s for m, s in zip(means_p, sems_p)],
        color=color, alpha=0.18, linewidth=0, zorder=3,
    )


def main() -> None:
    with open(XS_JSON) as f:
        xs = json.load(f)
    with open(S_JSON) as f:
        s = json.load(f)

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0), sharex=True)

    for row_idx, getter, source in ((0, xs_point, xs), (1, s_point, s)):
        for col, (ds_key, ds_label) in enumerate(DATASETS):
            ax = axes[row_idx, col]
            cer_means, cer_sems, wer_means, wer_sems = [], [], [], []
            for p in P_VALUES:
                cm, cs, wm, ws = getter(source, ds_key, p)
                cer_means.append(cm)
                cer_sems.append(cs)
                wer_means.append(wm)
                wer_sems.append(ws)

            any_data = any(v is not None for v in cer_means + wer_means)
            if not any_data:
                ax.text(
                    0.5, 0.5, "no data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="0.45", style="italic",
                )
                ax.set_xticks(P_VALUES)
                continue

            _draw_metric(ax, P_VALUES, cer_means, cer_sems, CER_COLOR, "o", "-")
            ax.tick_params(axis="y", labelcolor=CER_COLOR, labelsize=9.0)
            ax.axvline(DEFAULT_P, linestyle=":", color="gray", linewidth=1.1, zorder=1)
            ax.grid(True, alpha=0.3, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_xticks(P_VALUES)
            ax.tick_params(axis="x", labelsize=9.0)

            ax2 = ax.twinx()
            _draw_metric(ax2, P_VALUES, wer_means, wer_sems, WER_COLOR, "s", "--")
            ax2.tick_params(axis="y", labelcolor=WER_COLOR, labelsize=9.0)

            if row_idx == 0:
                ax.set_title(ds_label, fontsize=11.5)
            if row_idx == 1:
                ax.set_xlabel(r"$p_{\mathrm{replace}}$", fontsize=10.5)
            if col == 0:
                ax.set_ylabel(f"{ROW_TITLES[row_idx]}\nCER (%)", fontsize=10.5, color=CER_COLOR)
            else:
                ax.set_ylabel("CER (%)", fontsize=9.5, color=CER_COLOR)
            if col == len(DATASETS) - 1:
                ax2.set_ylabel("WER (%)", fontsize=10.0, color=WER_COLOR)
            else:
                ax2.set_ylabel("")

    legend_handles = [
        Line2D([0], [0], color=CER_COLOR, marker="o", linewidth=2,
               label="CER (left axis, 5-fold mean $\\pm$ SEM)"),
        Line2D([0], [0], color=WER_COLOR, marker="s", linewidth=2, linestyle="--",
               label="WER (right axis, 5-fold mean $\\pm$ SEM)"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.1,
               label=r"adopted $p_{\mathrm{replace}}{=}0.15$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        fontsize=10,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
