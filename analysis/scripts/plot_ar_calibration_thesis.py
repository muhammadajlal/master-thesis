#!/usr/bin/env python3
"""Thesis calibration figure (fig:ar-calibration) with three recognizers.

Reproduces the committed thesis figure style (two panels: beam-size sweep and
length-normalization sweep) from the canonical XS decode-study runs, with
corpus-level CER recomputed from predictions.json (sum ed / sum gt_len, the
thesis convention -- reproduces tab:app-ar-cal-* exactly). Adds the
noise-trained HWRFormer alongside HWRFormer and hybrid HWRFormer, and aligns
legend labels with the thesis terminology.

Output: thesis/figures/ar_calibration.pdf
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work")
OUT = Path("/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/ar_calibration.pdf")

MODELS = [
    ("ar", "HWRFormer", "#FF5722", "s"),
    ("noise", "HWRFormer + noise injection", "#9C27B0", "^"),
    ("hybrid", "Hybrid HWRFormer", "#2196F3", "o"),
]
DIRS = {"ar": BASE / "results/hwr2/decode_study_xs_full_ar",
        "hybrid": BASE / "results/hwr2/decode_study_xs_full_hybrid",
        "noise": BASE / "results/hwr2/decode_study_xs_full_noise"}


def load_all(results_dir: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(glob.glob(str(results_dir / "*" / "predictions.json"))):
        tag = os.path.basename(os.path.dirname(path))
        groups[re.sub(r"__fold\d+$", "", tag)].append(path)
    return dict(groups)


def mean_cer(paths: list[str]) -> float:
    fold_cers = []
    for p in paths:
        recs = json.load(open(p))
        fold_cers.append(100 * sum(r["ed"] for r in recs)
                         / sum(r["gt_len"] for r in recs))
    return float(np.mean(fold_cers))


def main() -> None:
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(10.0, 3.5))
    for key, label, color, marker in MODELS:
        groups = load_all(DIRS[key])
        beams = [(B, mean_cer(groups[f"stageA1_ar_beam_B{B}"]))
                 for B in [1, 2, 4, 8, 16]
                 if len(groups.get(f"stageA1_ar_beam_B{B}", [])) >= 5]
        ax_b.plot([b for b, _ in beams], [c for _, c in beams],
                  marker=marker, color=color, lw=1.5, ms=6, label=label)
        alphas = [(a, mean_cer(groups[f"stageA2_ar_lenorm_a{a:.1f}"]))
                  for a in [0.0, 0.2, 0.4, 0.6, 0.8]
                  if len(groups.get(f"stageA2_ar_lenorm_a{a:.1f}", [])) >= 5]
        ax_a.plot([a for a, _ in alphas], [c for _, c in alphas],
                  marker=marker, color=color, lw=1.5, ms=6, label=label)

    ax_b.set_title("Beam size sweep", fontsize=12)
    ax_b.set_xlabel(r"Beam size $B$")
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks([1, 2, 4, 8, 16], ["$2^0$", "$2^1$", "$2^2$", "$2^3$", "$2^4$"])
    ax_b.axvline(4, color="0.4", ls=":", lw=1.2)
    ax_a.set_title("Length normalization sweep", fontsize=12)
    ax_a.set_xlabel(r"Length normalization $\alpha$")
    ax_a.axvline(0.0, color="0.4", ls=":", lw=1.2)
    for ax in (ax_b, ax_a):
        ax.set_ylabel("CER (%)")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
