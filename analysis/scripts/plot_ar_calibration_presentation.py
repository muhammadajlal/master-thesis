#!/usr/bin/env python3
"""Presentation variant of the decoding calibration figure.

Canonical XS decode-study runs (results/hwr2/decode_study_xs_full_{ar,hybrid});
corpus-level CER recomputed from predictions.json (sum ed / sum gt_len, the
thesis convention -- reproduces the decode-study table exactly). Colors:
HWRFormer = blue circles, Hybrid HWRFormer = green squares.

Output: presentation/figures/ar_calibration_presentation.pdf
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
OUT = Path("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
           "ar_calibration_presentation.pdf")

MODELS = [
    ("ar", "HWRFormer", "#1f77b4", "o"),
    ("hybrid", "Hybrid HWRFormer", "#2ca02c", "s"),
]
DIRS = {"ar": BASE / "results/hwr2/decode_study_xs_full_ar",
        "hybrid": BASE / "results/hwr2/decode_study_xs_full_hybrid"}


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
    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for key, label, color, marker in MODELS:
        groups = load_all(DIRS[key])
        beams = [(B, mean_cer(groups[f"stageA1_ar_beam_B{B}"]))
                 for B in [1, 2, 4, 8, 16]
                 if f"stageA1_ar_beam_B{B}" in groups
                 and len(groups[f"stageA1_ar_beam_B{B}"]) >= 5]
        ax_b.plot([b for b, _ in beams], [c for _, c in beams],
                  marker=marker, color=color, lw=2, ms=6, label=label)
        alphas = [(a, mean_cer(groups[f"stageA2_ar_lenorm_a{a:.1f}"]))
                  for a in [0.0, 0.2, 0.4, 0.6, 0.8]
                  if f"stageA2_ar_lenorm_a{a:.1f}" in groups
                  and len(groups[f"stageA2_ar_lenorm_a{a:.1f}"]) >= 5]
        ax_a.plot([a for a, _ in alphas], [c for _, c in alphas],
                  marker=marker, color=color, lw=2, ms=6, label=label)

    ax_b.set_title("Beam size sweep", fontsize=12)
    ax_b.set_xlabel(r"Beam size $B$", fontsize=11)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks([1, 2, 4, 8, 16], ["$2^0$", "$2^1$", "$2^2$", "$2^3$", "$2^4$"])
    ax_b.axvline(4, color="0.4", ls=":", lw=1.2)
    ax_a.set_title("Length normalization sweep", fontsize=12)
    ax_a.set_xlabel(r"Length normalization $\alpha$", fontsize=11)
    ax_a.axvline(0.0, color="0.4", ls=":", lw=1.2)
    for ax in (ax_b, ax_a):
        ax.set_ylabel("CER (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
