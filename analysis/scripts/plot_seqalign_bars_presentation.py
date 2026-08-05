#!/usr/bin/env python3
"""Presentation clustered-column chart for the sequence-alignment results.

Replaces the slope-chart (plot_seqalign_paradox_presentation.py) with the
same visual grammar as aux_ctc_bars.pdf: light bar = without alignment,
dark bar = with alignment, value labels on top, relative change above the
pair, colored panel titles for direction. Values are the thesis numbers
(tab:contrastive-alignment and tab:contrastive-results), private words.

Output: presentation/figures/seqalign_bars.pdf
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = "/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/seqalign_bars.pdf"

CONNECTORS = ["MLP", "Pool-MLP"]
COS_WO, COS_W = [0.03, 0.06], [0.27, 0.28]
COS_REL = [r"$9.0\times$", r"$4.7\times$"]
GEOM_WO, GEOM_W = [16.1, 20.5], [3.6, 4.1]
GEOM_REL = ["$-78\\,\\%$", "$-80\\,\\%$"]
RECOG_WO, RECOG_W = [17.20, 16.87], [18.04, 17.52]
RECOG_REL = ["$+4.9\\,\\%$", "$+3.9\\,\\%$"]

C_WO = "#AECDE8"   # light step of the deck blue (without alignment)
C_W = "#1F77B4"    # deck blue (with alignment)
INK = "#222222"
GOOD = "#2e7d32"
BAD = "#c62828"


def panel(ax, wo, w, rel, title, tcolor, ylabel, fmt, ymax):
    x = np.arange(len(CONNECTORS))
    bw = 0.34
    ax.bar(x - bw / 2 - 0.01, wo, bw, color=C_WO)
    ax.bar(x + bw / 2 + 0.01, w, bw, color=C_W)
    for xi, v in zip(x - bw / 2 - 0.01, wo):
        ax.text(xi, v + ymax * 0.015, fmt.format(v), ha="center",
                va="bottom", fontsize=9, color=INK)
    for xi, v in zip(x + bw / 2 + 0.01, w):
        ax.text(xi, v + ymax * 0.015, fmt.format(v), ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="bold")
    for xi, lab in zip(x, rel):
        top = max(wo[xi], w[xi])
        ax.text(xi, top + ymax * 0.10, lab, ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(CONNECTORS) - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, CONNECTORS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11.5, color=tcolor, pad=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig, (ax_c, ax_g, ax_r) = plt.subplots(1, 3, figsize=(11.4, 3.3))
    panel(ax_c, COS_WO, COS_W, COS_REL,
          r"Private-word cosine similarity $\uparrow$ (gap closes)", GOOD,
          "sensor--text cosine similarity", "{:.2f}", 0.36)
    panel(ax_g, GEOM_WO, GEOM_W, GEOM_REL,
          r"Private-word $\ell_2$ distance $\downarrow$ (gap closes)", GOOD,
          r"sensor--text $\ell_2$ distance", "{:.1f}", 25.5)
    panel(ax_r, RECOG_WO, RECOG_W, RECOG_REL,
          r"Private-word CER $\uparrow$ (worse)", BAD,
          "private-word CER (%)", "{:.2f}", 21.5)
    handles = [Patch(color=C_WO, label="without contrastive alignment"),
               Patch(color=C_W, label="with contrastive alignment")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
