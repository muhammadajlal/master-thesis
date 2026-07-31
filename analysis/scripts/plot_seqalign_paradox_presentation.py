#!/usr/bin/env python3
"""Presentation figure: sequence alignment moves geometry, not recognition.

Two slope panels on the private-word dataset, both read left-to-right as
"without alignment -> with alignment": embedding geometry collapses, while
recognition error rises slightly. Values are the thesis numbers
(tab:contrastive-alignment and tab:contrastive-results).

Output: presentation/figures/seqalign_paradox.pdf
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
       "seqalign_paradox.pdf")

CONNECTORS = [
    ("MLP", "#1f77b4"),
    ("Pool-MLP", "#d95f02"),
]
# private words: (without alignment, with alignment)
GEOM = {"MLP": (16.1, 3.6), "Pool-MLP": (20.5, 4.1)}       # l2 distance
RECOG = {"MLP": (17.20, 18.04), "Pool-MLP": (16.87, 17.52)}  # CER %
GEOM_REL = {"MLP": "$-78\\,\\%$", "Pool-MLP": "$-80\\,\\%$"}
RECOG_REL = {"MLP": "$+4.9\\,\\%$", "Pool-MLP": "$+3.9\\,\\%$"}

XT = ["without\nalignment", "with\nalignment"]


def panel(ax, data, rel, title, ylabel, fmt, ypad, good, dys=(0, 0)):
    for (name, color), dy in zip(CONNECTORS, dys):
        a, b = data[name]
        ax.plot([0, 1], [a, b], color=color, lw=2.4, marker="o", ms=8,
                zorder=3, label=name)
        ax.annotate(fmt.format(a), (0, a), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center", fontsize=11,
                    color=color)
        ax.annotate(fmt.format(b), (1, b), textcoords="offset points",
                    xytext=(10, dy + 6), ha="left", va="center", fontsize=11,
                    fontweight="bold", color=color)
        ax.annotate(rel[name], (1, b), textcoords="offset points",
                    xytext=(10, dy - 8), ha="left", va="center", fontsize=10.5,
                    color=color)
    ax.set_xlim(-0.55, 1.62)
    lo = min(min(v) for v in data.values())
    hi = max(max(v) for v in data.values())
    ax.set_ylim(lo - ypad, hi + ypad)
    ax.set_xticks([0, 1], XT, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10,
                 color=("#2e7d32" if good else "#c62828"))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    panel(ax_g, GEOM, GEOM_REL,
          "Embedding geometry: gap closes",
          r"sensor--text $\ell_2$ distance", "{:.1f}", 3.0, good=True,
          dys=(-16, 14))
    panel(ax_r, RECOG, RECOG_REL,
          "Recognition: error rises",
          "private-word CER (%)", "{:.2f}", 0.55, good=False)
    ax_g.legend(fontsize=10.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
