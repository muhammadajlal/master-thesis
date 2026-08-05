#!/usr/bin/env python3
"""Grouped-column charts replacing every result table in the main deck.

One figure per slide, two panels each (CER left, WER right): x = datasets,
one column per model, per-column value labels, legend on top. Numbers are
copied verbatim from the deck tables (verified five-fold means). Colors are
entity-fixed across all charts and CVD-validated (adjacent-pair OKLab dE:
worst simulated deficiency >= 17 for every neighboring pair).

Outputs (presentation/figures/):
  bars_migration.pdf   REWI vs HWRFormer
  bars_noise.pdf       + HWRFormer + noise injection
  bars_hybrid.pdf      + Hybrid HWRFormer
  bars_decoding.pdf    greedy / beam / + KenLM fusion / + KenLM rescoring
  bars_classical.pdf   five-model cumulative comparison
  bars_gptinit.pdf     GPT-2 init: random vs pretrained (2 datasets)
  bars_connectors.pdf  HWRFormer + four connectors (2 datasets)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT_DIR = "/home/woody/iwso/iwso214h/imu-hwr/presentation/figures"
DS4 = ["OnHW-WI", "OnHW-WD", "Priv. words", "Priv. sent."]
DS2 = ["OnHW-WI", "Priv. words"]
INK = "#222222"

# entity-fixed colors (CVD-validated adjacency in each chart's bar order)
C_REWI = "#A6A6A6"
C_HWRF = "#1F77B4"
C_NOISE = "#D62728"
C_HYBRID = "#17BECF"
C_FUSION = "#E8871E"
C_GREEDY = "#AECDE8"
C_RESCORE = "#6B4C9A"
C_RANDOM = "#AECDE8"
C_PRETRAIN = "#6B4C9A"
C_MLP = "#E7BA52"
C_POOLMLP = "#6B4C9A"
C_QFORMER = "#D95F02"
C_GATEDMV = "#17BECF"


def panel(ax, datasets, series, ylabel, label_fs, ymax=None):
    """series: list of (name, color, values)."""
    n = len(series)
    x = np.arange(len(datasets))
    gw = 0.82
    bw = gw / n
    top = ymax or max(max(v) for _, _, v in series) * 1.16
    stagger = top * 0.045 if n >= 4 else 0.0
    for j, (_, color, vals) in enumerate(series):
        xs = x - gw / 2 + bw * (j + 0.5)
        ax.bar(xs, vals, bw * 0.94, color=color, edgecolor="white",
               linewidth=0.5)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + top * 0.012 + stagger * (j % 2), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=label_fs, color=INK)
    ax.set_ylim(0, top)
    ax.set_xticks(x, datasets, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)


def chart(name, datasets, models, figsize, label_fs=8.0, legend_ncol=None,
          annotate=None, legend_fs=10):
    """models: list of (name, color, cer_list, wer_list)."""
    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=figsize)
    panel(ax_c, datasets, [(n, c, cer) for n, c, cer, _ in models],
          "CER (%)", label_fs)
    panel(ax_w, datasets, [(n, c, wer) for n, c, _, wer in models],
          "WER (%)", label_fs)
    ax_c.set_title("CER", fontsize=12, pad=6)
    ax_w.set_title("WER", fontsize=12, pad=6)
    if annotate:
        annotate(ax_c, ax_w)
    handles = [Patch(facecolor=c, label=n) for n, c, _, _ in models]
    fig.legend(handles=handles, loc="upper center",
               ncol=legend_ncol or len(models), frameon=False,
               fontsize=legend_fs, bbox_to_anchor=(0.5, 1.03),
               columnspacing=1.3, handlelength=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = f"{OUT_DIR}/{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


REWI = ("REWI", C_REWI,
        [7.30, 14.81, 9.39, 6.55], [15.16, 44.77, 31.82, 23.52])
HWRF = ("HWRFormer", C_HWRF,
        [6.94, 16.31, 9.96, 9.28], [10.51, 31.92, 19.10, 15.00])
NOISE = ("HWRFormer + noise injection", C_NOISE,
         [6.86, 13.52, 7.79, 7.09], [13.18, 36.05, 23.69, 18.87])
HYBRID = ("Hybrid HWRFormer", C_HYBRID,
          [6.83, 13.39, 9.37, 9.38], [10.17, 27.80, 19.76, 17.08])
FUSION = ("HWRFormer + character 5-gram (KenLM)", C_FUSION,
          [6.80, 16.63, 8.96, 7.47], [10.36, 32.48, 17.16, 11.65])


def main() -> None:
    chart("bars_migration", DS4, [REWI, HWRF], (12.6, 3.6), label_fs=8.5)
    chart("bars_noise", DS4, [REWI, HWRF, NOISE], (12.9, 3.6), label_fs=8.0)
    chart("bars_hybrid", DS4, [REWI, HWRF, NOISE, HYBRID], (13.2, 3.6),
          label_fs=7.4, legend_ncol=4)
    chart("bars_classical", DS4, [REWI, HWRF, NOISE, HYBRID, FUSION],
          (13.6, 3.7), label_fs=6.8, legend_ncol=5, legend_fs=9.0)

    decoding = [
        ("Greedy", C_GREEDY,
         [6.95, 16.33, 9.96, 9.33], [10.51, 31.92, 19.09, 15.04]),
        ("Calibrated beam", C_HWRF,
         [6.79, 16.19, 8.99, 7.45], [10.38, 31.77, 17.48, 11.75]),
        ("+ KenLM fusion", C_FUSION,
         [6.80, 16.63, 8.96, 7.47], [10.36, 32.48, 17.16, 11.65]),
        ("+ KenLM rescoring", C_RESCORE,
         [6.79, 18.06, 9.04, 8.14], [10.34, 34.08, 16.69, 12.17]),
    ]
    chart("bars_decoding", DS4, decoding, (13.2, 3.6), label_fs=7.4,
          legend_ncol=4)

    gptinit = [
        ("GPT-2 random init", C_RANDOM, [7.69, 57.21], [12.13, 80.69]),
        ("GPT-2 pretrained", C_PRETRAIN, [7.45, 27.93], [11.45, 43.93]),
    ]

    def ann_init(ax_c, ax_w):
        for ax, pairs in ((ax_c, [(0, 7.69, "$-3.1\\,\\%$"),
                                  (1, 57.21, "$-51.2\\,\\%$")]),):
            top = ax.get_ylim()[1]
            for xi, v, lab in pairs:
                ax.text(xi, v + top * 0.055, lab, ha="center", va="bottom",
                        fontsize=9.5, color=INK)

    chart("bars_gptinit", DS2, gptinit, (9.6, 3.5), label_fs=9.0,
          annotate=ann_init)

    connectors = [
        ("HWRFormer", C_HWRF, [6.94, 9.96], [10.51, 19.10]),
        ("MLP", C_MLP, [7.45, 27.93], [11.45, 43.93]),
        ("Pool-MLP", C_POOLMLP, [7.71, 24.87], [11.74, 40.24]),
        ("Lightweight Q-Former", C_QFORMER, [7.32, 24.89], [11.33, 35.25]),
        ("Gated Multi-View", C_GATEDMV, [8.01, 27.67], [12.03, 44.29]),
    ]
    chart("bars_connectors", DS2, connectors, (12.4, 3.5), label_fs=8.0,
          legend_ncol=5)


if __name__ == "__main__":
    main()
