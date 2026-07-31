#!/usr/bin/env python3
"""Presentation bar chart for the auxiliary-CTC results table (slide: Aux CTC).

Paired no-CTC vs + aux. CTC CER per connector, two panels (private words,
OnHW-WI), dashed HWRFormer reference line per panel. Values are the verified
five-fold means from results.json of the canonical runs (identical to the
slide/thesis table tab:vlm-h1).

Output: presentation/figures/aux_ctc_bars.pdf
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/aux_ctc_bars.pdf"

CONNECTORS = ["MLP", "Pool-MLP", "Lightweight\nQ-Former", "Gated\nMulti-View"]
PRIV_NOCTC = [27.93, 24.87, 24.89, 27.67]
PRIV_CTC = [17.20, 16.87, 16.33, 17.99]
PRIV_REL = ["$-38.4\\,\\%$", "$-32.2\\,\\%$", "$-34.4\\,\\%$", "$-34.9\\,\\%$"]
ONHW_NOCTC = [7.45, 7.71, 7.32, 8.01]
ONHW_CTC = [7.11, 7.34, 6.86, 7.39]
HWRFORMER = {"priv": 9.96, "onhw": 6.94}

C_NOCTC = "#AECDE8"   # light step of the deck blue (no aux. CTC)
C_CTC = "#1F77B4"     # deck blue (+ aux. CTC)
C_REF = "#444444"
INK = "#222222"


def panel(ax, noctc, ctc, ref, title, rel=None, ymax=None):
    x = np.arange(len(CONNECTORS))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, noctc, w, color=C_NOCTC)
    ax.bar(x + w / 2 + 0.01, ctc, w, color=C_CTC)
    for xi, v in zip(x - w / 2 - 0.01, noctc):
        ax.text(xi, v + (ymax or max(noctc)) * 0.012, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    for xi, v in zip(x + w / 2 + 0.01, ctc):
        ax.text(xi, v + (ymax or max(noctc)) * 0.012, f"{v:.2f}", ha="center",
                va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    if rel:
        for xi, v_no, lab in zip(x, noctc, rel):
            ax.text(xi, v_no + (ymax or max(noctc)) * 0.075, lab, ha="center",
                    va="bottom", fontsize=9.5, color=INK)
    ax.axhline(ref, color=C_REF, ls="--", lw=1.4)
    ax.set_xticks(x, CONNECTORS, fontsize=9.5)
    ax.set_ylabel("CER (%)", fontsize=10)
    ax.set_title(title, fontsize=11.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ymax:
        ax.set_ylim(0, ymax)


def main() -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, (ax_p, ax_o) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    panel(ax_p, PRIV_NOCTC, PRIV_CTC, HWRFORMER["priv"],
          "Private words", rel=PRIV_REL, ymax=33.0)
    panel(ax_o, ONHW_NOCTC, ONHW_CTC, HWRFORMER["onhw"],
          "OnHW-WI", ymax=9.6)
    handles = [Patch(color=C_NOCTC, label="no aux. CTC"),
               Patch(color=C_CTC, label="+ aux. CTC"),
               Line2D([], [], color=C_REF, ls="--", lw=1.4,
                      label="HWRFormer reference (9.96 / 6.94)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
