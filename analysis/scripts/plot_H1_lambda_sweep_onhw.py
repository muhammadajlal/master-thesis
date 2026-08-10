#!/usr/bin/env python3
"""Aggregate and plot the multimodal lambda_ctc sweep on OnHW WI word (thesis fig 6.1).

Companion to plot_H1_lambda_sweep_word.py (private word). For each lambda in
{0.1..1.0}, read CER/WER from the canonical results.json (WER at the
best-CER epoch). The lambda=0.6 point lives under
Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__onhw_wi_word_rh; the other
nine points live under H1_LambdaSweep/H1_hybrid_mlp_lamNN__onhw_wi_word_rh.
Marks lambda=0.2 (solid) as the best-observed OnHW loss weight.

Outputs:
  - results/hwr2/H1_LambdaSweep/h1_lambda_sweep_onhw_metrics.csv
  - thesis/figures/h1_lambda_sweep.pdf  (paper2 keeps its own copy via
    the original plot_H1_lambda_sweep.py)

Run from anywhere:
    python plot_H1_lambda_sweep_onhw.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
RESULTS = REPO / "results" / "hwr2"

LAMBDA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HYBRID_BASE = RESULTS / "Ablations-MMLM" / "GPT-2" / "Hybrid" / "H1_hybrid_mlp"
LAMBDA_DIRS: dict[float, Path] = {}
for lam in LAMBDA_VALUES:
    if lam == 0.6:
        LAMBDA_DIRS[lam] = HYBRID_BASE / "vlm__onhw_wi_word_rh"
    else:
        LAMBDA_DIRS[lam] = (
            RESULTS / "H1_LambdaSweep"
            / f"H1_hybrid_mlp_lam{int(round(lam * 10)):02d}__onhw_wi_word_rh"
        )

# MLP-without-CTC reference (effectively lambda=0) on private word.
NOCTC_DIR = (
    RESULTS / "Ablations-MMLM" / "GPT-2" / "AR-only" / "vlm_ablation_A1_mlp_pretrained"
    / "vlm__onhw_wi_word_rh"
)

OUT_CSV = RESULTS / "H1_LambdaSweep" / "h1_lambda_sweep_onhw_metrics.csv"
OUT_PDF_THESIS = REPO / "thesis" / "figures" / "h1_lambda_sweep.pdf"


def read_metrics(target_dir: Path) -> tuple[list[float], list[float]]:
    rj = target_dir / "results.json"
    if not rj.is_file():
        raise FileNotFoundError(f"missing canonical results.json: {rj}")
    with open(rj, "r") as f:
        d = json.load(f)
    cers = [float(v) * 100.0 for v in d["cer"]["raw"].values()]
    wers = [float(v) * 100.0 for v in d["wer"]["raw"].values()]
    return cers, wers


def mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


# Two-sided 95% Student-t half-width factor for df=4 (five folds). The folds are
# fixed and dependent, so the resulting interval is descriptive, not confirmatory.
T95_DF4 = 2.776


def main() -> None:
    cer_per_lambda: list[list[float]] = []
    wer_per_lambda: list[list[float]] = []
    for lam in LAMBDA_VALUES:
        d = LAMBDA_DIRS[lam]
        cers, wers = read_metrics(d)
        print(f"[lambda={lam}] {len(cers)} folds; CER mean={statistics.mean(cers):.2f}  dir={d.name}")
        cer_per_lambda.append(cers)
        wer_per_lambda.append(wers)

    ref_cers, ref_wers = read_metrics(NOCTC_DIR)
    ref_cer_mean = statistics.mean(ref_cers)
    ref_wer_mean = statistics.mean(ref_wers)
    print(f"[no-CTC ref] CER={ref_cer_mean:.2f} WER={ref_wer_mean:.2f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_ctc", "n_folds", "cer_mean", "cer_sem", "wer_mean", "wer_sem",
                    "cer_ci95_half", "wer_ci95_half",
                    "cer_pdiff_vs_lam01", "cer_pdiff_ci95_half"])
        base_c = cer_per_lambda[0]  # lambda = 0.1 (paired per-fold reference)
        for lam, c, wv in zip(LAMBDA_VALUES, cer_per_lambda, wer_per_lambda):
            cm, cs = mean_sem(c)
            wm, ws = mean_sem(wv)
            pd = [x - y for x, y in zip(c, base_c)]
            pm, ps = mean_sem(pd)
            w.writerow([lam, len(c), f"{cm:.3f}", f"{cs:.3f}", f"{wm:.3f}", f"{ws:.3f}",
                        f"{T95_DF4 * cs:.3f}", f"{T95_DF4 * ws:.3f}",
                        f"{pm:.3f}", f"{T95_DF4 * ps:.3f}"])
        w.writerow(["mlp_noCTC_ref", len(ref_cers), f"{ref_cer_mean:.3f}", "", f"{ref_wer_mean:.3f}", "", "", "", "", ""])
    print(f"wrote {OUT_CSV}")

    cer_stats = [mean_sem(c) for c in cer_per_lambda]
    wer_stats = [mean_sem(w) for w in wer_per_lambda]
    cer_means = [m for m, _ in cer_stats]
    cer_sems = [s for _, s in cer_stats]
    wer_means = [m for m, _ in wer_stats]
    wer_sems = [s for _, s in wer_stats]

    SELECTED_LAMBDA = 0.2
    empirical_min = LAMBDA_VALUES[0]
    best_cer = float("inf")
    for lam, m in zip(LAMBDA_VALUES, cer_means):
        if not math.isnan(m) and m < best_cer:
            best_cer = m
            empirical_min = lam

    fig, (ax_cer, ax_wer) = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    cer_color = "#1f77b4"
    wer_color = "#ff7f0e"
    ax_cer.plot(LAMBDA_VALUES, cer_means, color=cer_color, linewidth=2,
                marker="o", markersize=6, zorder=5)
    ax_cer.fill_between(
        LAMBDA_VALUES,
        [m - s for m, s in zip(cer_means, cer_sems)],
        [m + s for m, s in zip(cer_means, cer_sems)],
        color=cer_color, alpha=0.18, linewidth=0, zorder=4,
    )
    ax_wer.plot(LAMBDA_VALUES, wer_means, color=wer_color, linewidth=2,
                marker="s", markersize=6, zorder=5)
    ax_wer.fill_between(
        LAMBDA_VALUES,
        [m - s for m, s in zip(wer_means, wer_sems)],
        [m + s for m, s in zip(wer_means, wer_sems)],
        color=wer_color, alpha=0.18, linewidth=0, zorder=4,
    )

    ax_cer.axhline(ref_cer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)
    ax_wer.axhline(ref_wer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)

    SELECTED_COLOR = "#9467bd"
    for ax in (ax_cer, ax_wer):
        ax.axvspan(SELECTED_LAMBDA - 0.015, SELECTED_LAMBDA + 0.015,
                   color=SELECTED_COLOR, alpha=0.22, zorder=1)
        ax.axvline(SELECTED_LAMBDA, color=SELECTED_COLOR, linestyle="-",
                   linewidth=1.8, alpha=0.95, zorder=2)
        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=14.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=13)

    ax_cer.set_ylabel("CER (%)", fontsize=14.5)
    ax_wer.set_ylabel("WER (%)", fontsize=14.5)
    ax_cer.set_title(r"HWR-GPT (MLP + CTC): CER vs $\lambda_{\mathrm{ctc}}$ (OnHW WI word)", fontsize=12)
    ax_wer.set_title(r"HWR-GPT (MLP + CTC): WER vs $\lambda_{\mathrm{ctc}}$ (OnHW WI word)", fontsize=12)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=cer_color, marker="o", markersize=6,
               linewidth=2, label=r"5-fold mean $\pm$ SEM (CER)"),
        Line2D([0], [0], color=wer_color, marker="s", markersize=6,
               linewidth=2, label=r"5-fold mean $\pm$ SEM (WER)"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.5,
               label="MLP (no CTC)"),
        Line2D([0], [0], color=SELECTED_COLOR, linestyle="-", linewidth=1.8,
               label=rf"best-observed OnHW weight $\lambda_{{\mathrm{{ctc}}}}={SELECTED_LAMBDA}$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, fontsize=12.5, frameon=False,
               handlelength=2.2, columnspacing=1.8)

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    OUT_PDF_THESIS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF_THESIS, bbox_inches="tight")
    print(f"saved: {OUT_PDF_THESIS}")


if __name__ == "__main__":
    main()
