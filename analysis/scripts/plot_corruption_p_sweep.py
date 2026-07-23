#!/usr/bin/env python3
"""Generate corruption_p_sweep.pdf for paper 2.

Per-dataset panel: CER and WER vs corruption rate p_ic, 5-fold mean.
CER on left axis (blue), WER on right axis (orange). Dotted vertical line
marks the recommended default p_ic=0.15. Baseline (no corruption) is the
left-most x=0.00 point.

Three panels (left to right): OnHW-WI (closed-vocab short), OnHW-WD
(closed-vocab short, writer-dependent), private sentences (open-vocab,
long). XS data only.

Run from anywhere:
    python plot_corruption_p_sweep.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/"
    "figures/corruption_p_sweep.pdf"
)

PANELS = [
    ("onhw_wi_word_rh", "OnHW-Words500 (WI)"),
    ("onhw_wd_word_rh", "OnHW-Words500 (WD)"),
    ("wi_word_hw6_meta", "Private (words)"),
    ("wi_sent_hw6_meta", "Private (sentences)"),
]

P_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_P = 0.15


def read_5fold(model_dir: Path) -> tuple[float | None, float | None]:
    cers, wers = [], []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"fold_{k}/{k}/train_*.json")))
        if not files:
            return None, None
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
            wers.append(float(d["best"]["word_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            return None, None
    if len(cers) != 5:
        return None, None
    return statistics.mean(cers), statistics.mean(wers)


def cell_path(dataset_key: str, p: float) -> Path:
    if p == 0.00:
        return RESULTS / "Baseline-AR-XS-blconv_b" / f"ar_transformer_xs__{dataset_key}"
    if abs(p - 0.15) < 1e-9:
        return RESULTS / "Baseline-AR-XS-InputCorruption-uniform" / f"ar_transformer_xs__{dataset_key}"
    pstr = f"p0p{int(round(p * 100)):02d}"
    return (
        RESULTS
        / "Baseline-AR-XS-InputCorruption-Sweep-blconv_b"
        / f"ar_transformer_xs__{dataset_key}__{pstr}"
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex=False)
    legend_handles = None

    for ax, (dataset_key, label) in zip(axes.flat, PANELS):
        cers, wers = [], []
        for p in P_VALUES:
            cer, wer = read_5fold(cell_path(dataset_key, p))
            cers.append(cer)
            wers.append(wer)

        cer_color = "#1f77b4"  # blue
        wer_color = "#ff7f0e"  # orange

        l1 = ax.plot(P_VALUES, cers, marker="o", color=cer_color, linewidth=2, label="CER")
        ax.set_ylabel("CER (%)", color=cer_color)
        ax.tick_params(axis="y", labelcolor=cer_color)
        ax.set_xlabel(r"$p_{\mathrm{ni}}$")
        ax.set_title(label, fontsize=11)
        ax.axvline(DEFAULT_P, linestyle=":", color="gray", linewidth=1.2)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        l2 = ax2.plot(P_VALUES, wers, marker="s", color=wer_color, linewidth=2, linestyle="--", label="WER")
        ax2.set_ylabel("WER (%)", color=wer_color)
        ax2.tick_params(axis="y", labelcolor=wer_color)

        if legend_handles is None:
            legend_handles = l1 + l2

    fig.legend(
        legend_handles,
        ["CER (left axis)", "WER (right axis)"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
