#!/usr/bin/env python3
"""Generate corruption_modes_bars.pdf for paper 2.

Grouped bar chart: one cluster per dataset (6 datasets), within each
cluster the 5 corruption modes are ordered by CER descending so the
worst mode is leftmost and the best mode rightmost. The no-corruption
baseline is overlaid as a dashed horizontal line per cluster so the
reader can read "did corruption help on this dataset?" at a glance.

Data source: results/hwr2/Baseline-AR-InputCorruption-* and
Baseline-AR-ElementwiseGating[-WD|-Equations-{WI,WD}], 5-fold mean.

Run from anywhere:
    python plot_corruption_modes_bars.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS_ROOT = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/"
    "figures/corruption_modes_bars.pdf"
)

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word_rh",           "OnHW-WI"),
    ("onhw_wd_word_rh",           "OnHW-WD"),
    ("onhw_equations_wi_word_rh", "Eq-WI"),
    ("onhw_equations_wd_word_rh", "Eq-WD"),
    ("wi_word_hw6_meta",          "Priv-words"),
    ("wi_sent_hw6_meta",          "Priv-sent"),
]

# (key, display label, colour)
MODES: list[tuple[str, str, str]] = [
    ("uniform",      "uniform",        "#1f77b4"),
    ("bigramright",  "bigram-right",   "#2ca02c"),
    ("bigramleft",   "bigram-left",    "#ff7f0e"),
    ("selfconf",     "self-confusion", "#9467bd"),
    ("adjacentswap", "adjacent-swap",  "#d62728"),
]

# XS layout: one corruption-mode directory per mode, with per-dataset
# subdirs inside. No -WD / -Equations variants needed (XS dirs cover
# all 6 datasets uniformly). Baseline (no corruption) is the
# elementwise-gating run under Baseline-AR-XS-blconv_b/.
def mode_dir(mode_key: str, dataset_key: str) -> Path:
    return (
        RESULTS_ROOT
        / f"Baseline-AR-XS-InputCorruption-{mode_key}"
        / f"ar_transformer_xs__{dataset_key}"
    )


def baseline_dir(dataset_key: str) -> Path:
    return RESULTS_ROOT / "Baseline-AR-XS-blconv_b" / f"ar_transformer_xs__{dataset_key}"


def read_5fold_cer(model_dir: Path) -> float | None:
    cers: list[float] = []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"fold_{k}/{k}/train_*.json")))
        if not files:
            return None
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
    return float(np.mean(cers))


def main() -> None:
    data: dict[str, dict[str, float | None]] = {}
    for ds_key, _ in DATASETS:
        row: dict[str, float | None] = {}
        row["__baseline__"] = read_5fold_cer(baseline_dir(ds_key))
        for mode_key, _, _ in MODES:
            row[mode_key] = read_5fold_cer(mode_dir(mode_key, ds_key))
        data[ds_key] = row
        missing = [m for m, v in row.items() if v is None]
        if missing:
            print(f"WARN: dataset {ds_key} missing modes: {missing}")

    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    color_by_mode = {k: c for k, _, c in MODES}
    n_modes = len(MODES)
    group_w = 0.84
    bar_w = group_w / n_modes

    # Headroom is set wide enough that vertical (rotated) bar labels
    # fit above the bars without being clipped by the top spine.
    all_cers: list[float] = []
    for row in data.values():
        for v in row.values():
            if v is not None:
                all_cers.append(v)
    ymax = max(all_cers) * 1.32 if all_cers else 1.0
    label_offset = ymax * 0.012

    x = np.arange(len(DATASETS))
    for ds_idx, (ds_key, _ds_label) in enumerate(DATASETS):
        row = data[ds_key]
        mode_keys = [k for k, _, _ in MODES]
        present = [m for m in mode_keys if row[m] is not None]
        # Descending by CER (worst leftmost, best rightmost)
        present.sort(key=lambda m: -row[m])  # type: ignore[operator]
        bcer = row["__baseline__"]
        for j, m in enumerate(present):
            cer = row[m]
            pos = x[ds_idx] - group_w / 2 + (j + 0.5) * bar_w
            ax.bar(
                pos,
                cer,
                bar_w * 0.94,
                color=color_by_mode[m],
                edgecolor="black",
                linewidth=0.4,
            )
            # Anchor each label above max(bar_top, baseline) and
            # rotate vertical so adjacent labels in a tight cluster do
            # not overlap horizontally. Vertical orientation also means
            # the label cannot collide with the dashed baseline line.
            anchor = max(cer, bcer) if bcer is not None else cer
            ax.text(
                pos,
                anchor + label_offset,
                f"{cer:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="black",
                rotation=90,
            )
        if bcer is not None:
            ax.hlines(
                bcer,
                x[ds_idx] - group_w / 2,
                x[ds_idx] + group_w / 2,
                colors="black",
                linestyles="dashed",
                linewidth=1.1,
                zorder=4,
            )

    handles = [Patch(facecolor=c, edgecolor="black", label=lab) for _, lab, c in MODES]
    handles.append(
        Line2D([], [], color="black", linestyle="dashed", linewidth=1.1, label="no corruption")
    )
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9.0,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DATASETS], fontsize=10.0)
    ax.set_ylabel("CER (%)", fontsize=10.5)
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    # Reserve right margin for the external legend.
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
