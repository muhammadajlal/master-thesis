#!/usr/bin/env python3
"""Generate ctc_diagnostics_compact.pdf for paper 2 (XS data).

Two-panel figure: downstream AR CER (left) and WER (right) vs lambda_ctc,
shown as 5-fold mean curves with translucent SEM bands. Dashed vertical
line marks the selected lambda_ctc=0.1. A horizontal AR-only reference is
drawn as a dotted line.

XS lambda sweep data lives under
  results/hwr2/train_element_word_hybrid_{01..10}_xs_onhw_wi/
  ar_transformer_xs__onhw_wi_word_rh/fold_{0..4}/{0..4}/train_*.json
AR-only baseline is under
  results/hwr2/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/

Run from anywhere:
    python plot_lambda_sweep.py
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type3 (Springer)
import matplotlib.pyplot as plt

RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/"
    "figures/ctc_diagnostics_compact.pdf"
)

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTED = 0.1


def read_per_fold(model_dir: Path) -> tuple[list[float], list[float]]:
    cers, wers = [], []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"fold_{k}/{k}/train_*.json")))
        if not files:
            continue
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
            wers.append(float(d["best"]["word_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return cers, wers


def lambda_dir(k: int) -> Path:
    return (
        RESULTS
        / f"train_element_word_hybrid_{k:02d}_xs_onhw_wi"
        / "ar_transformer_xs__onhw_wi_word_rh"
    )


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def main() -> None:
    cer_per_lambda: list[list[float]] = []
    wer_per_lambda: list[list[float]] = []
    for lam in LAMBDAS:
        idx = int(round(lam * 10))
        c, w = read_per_fold(lambda_dir(idx))
        cer_per_lambda.append(c)
        wer_per_lambda.append(w)

    ar_only_dir = RESULTS / "Baseline-AR-XS-blconv_b" / "ar_transformer_xs__onhw_wi_word_rh"
    ar_cers, ar_wers = read_per_fold(ar_only_dir)
    ar_cer_mean = statistics.mean(ar_cers)
    ar_wer_mean = statistics.mean(ar_wers)

    fig, (ax_cer, ax_wer) = plt.subplots(1, 2, figsize=(9.0, 3.2), sharex=True)

    cer_stats = [mean_sem(c) for c in cer_per_lambda]
    wer_stats = [mean_sem(w) for w in wer_per_lambda]
    cer_means = [m for m, _ in cer_stats]
    cer_sems = [s for _, s in cer_stats]
    wer_means = [m for m, _ in wer_stats]
    wer_sems = [s for _, s in wer_stats]

    cer_color = "#1f77b4"
    wer_color = "#ff7f0e"
    ax_cer.plot(LAMBDAS, cer_means, color=cer_color, linewidth=2, marker="o", markersize=6, zorder=5)
    ax_cer.fill_between(
        LAMBDAS,
        [m - s for m, s in zip(cer_means, cer_sems)],
        [m + s for m, s in zip(cer_means, cer_sems)],
        color=cer_color,
        alpha=0.18,
        linewidth=0,
        zorder=4,
    )
    ax_wer.plot(LAMBDAS, wer_means, color=wer_color, linewidth=2, marker="s", markersize=6, zorder=5)
    ax_wer.fill_between(
        LAMBDAS,
        [m - s for m, s in zip(wer_means, wer_sems)],
        [m + s for m, s in zip(wer_means, wer_sems)],
        color=wer_color,
        alpha=0.18,
        linewidth=0,
        zorder=4,
    )

    from matplotlib.lines import Line2D

    ax_cer.axhline(ar_cer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)
    ax_wer.axhline(ar_wer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)

    HIGHLIGHT = "#9467bd"
    for ax in (ax_cer, ax_wer):
        ax.axvspan(SELECTED - 0.015, SELECTED + 0.015, color=HIGHLIGHT, alpha=0.22, zorder=1)
        ax.axvline(SELECTED, color=HIGHLIGHT, linestyle="--", linewidth=1.6, alpha=0.9, zorder=2)
        ax.annotate(
            "", xy=(SELECTED, 1.0), xytext=(SELECTED, 1.06),
            xycoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=1.8),
        )
        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.grid(True, alpha=0.3)

    ax_cer.set_ylabel("CER (%)", fontsize=12)
    ax_wer.set_ylabel("WER (%)", fontsize=12)
    ax_cer.set_title(r"Downstream AR CER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)
    ax_wer.set_title(r"Downstream AR WER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)

    cer_envelope = [(m - s, m + s) for m, s in zip(cer_means, cer_sems)]
    wer_envelope = [(m - s, m + s) for m, s in zip(wer_means, wer_sems)]
    cer_lo = min(lo for lo, _ in cer_envelope + [(ar_cer_mean, ar_cer_mean)]) - 0.4
    cer_hi = max(hi for _, hi in cer_envelope + [(ar_cer_mean, ar_cer_mean)]) + 0.4
    wer_lo = min(lo for lo, _ in wer_envelope + [(ar_wer_mean, ar_wer_mean)]) - 0.4
    wer_hi = max(hi for _, hi in wer_envelope + [(ar_wer_mean, ar_wer_mean)]) + 0.4
    ax_cer.set_ylim(cer_lo, cer_hi)
    ax_wer.set_ylim(wer_lo, wer_hi)

    bbox_small = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.95)
    ax_cer.text(0.99, ar_cer_mean, f"AR-only {ar_cer_mean:.2f}",
                transform=ax_cer.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, bbox=bbox_small, zorder=6)
    ax_wer.text(0.99, ar_wer_mean, f"AR-only {ar_wer_mean:.2f}",
                transform=ax_wer.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, bbox=bbox_small, zorder=6)

    fig.legend(
        handles=[
            Line2D([0], [0], color=cer_color, marker="o", markersize=6,
                   linewidth=2, label="5-fold mean $\\pm$ SEM (CER)"),
            Line2D([0], [0], color=wer_color, marker="s", markersize=6,
                   linewidth=2, label="5-fold mean $\\pm$ SEM (WER)"),
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.5,
                   label="AR-only baseline"),
            Line2D([0], [0], color=HIGHLIGHT, linestyle="--", linewidth=1.6,
                   label=f"selected $\\lambda_{{\\mathrm{{ctc}}}}={SELECTED}$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
