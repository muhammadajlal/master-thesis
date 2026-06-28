#!/usr/bin/env python3
"""Aggregate and plot the H1 multimodal lambda_ctc sweep on OnHW WI word.

For each lambda in {0.1..1.0}, point evaluate.py at the per-fold output dir
and read CER/WER from the resulting results.json (canonical convention,
WER read at the best-CER epoch). lambda=0.6 is the reference point and
lives under Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/...; the other nine
points live under H1_LambdaSweep/H1_hybrid_mlp_lamNN__...

Outputs:
  - results/hwr2/H1_LambdaSweep/h1_lambda_sweep_metrics.csv
  - thesis/figures/h1_lambda_sweep.pdf  (also dropped under
    paper2_lncs_overleaf/figures/ if that directory exists)

Run from anywhere:
    python plot_H1_lambda_sweep.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
RESULTS = REPO / "results" / "hwr2"
REWI = REPO / "work" / "REWI_work"

LAMBDA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDA_DIRS: dict[float, Path] = {
    lam: (
        RESULTS / "Ablations-MMLM" / "GPT-2" / "Hybrid" / "H1_hybrid_mlp"
        / "vlm__onhw_wi_word_rh"
        if lam == 0.6
        else RESULTS / "H1_LambdaSweep"
        / f"H1_hybrid_mlp_lam{int(round(lam * 10)):02d}__onhw_wi_word_rh"
    )
    for lam in LAMBDA_VALUES
}

# MLP-without-CTC reference (effectively lambda=0): pretrained-LM MLP variant.
PHASE3_DIR = (
    RESULTS / "Ablations-MMLM" / "GPT-2" / "AR-only" / "vlm_ablation_A1_mlp_pretrained"
    / "vlm__onhw_wi_word_rh"
)

OUT_CSV = RESULTS / "H1_LambdaSweep" / "h1_lambda_sweep_metrics.csv"
OUT_PDF_THESIS = REPO / "thesis" / "figures" / "h1_lambda_sweep.pdf"
OUT_PDF_PAPER = (
    REPO / "publications" / "paper2_lncs_overleaf" / "figures" / "h1_lambda_sweep.pdf"
)

EVAL_TEMPLATE = REWI / "configs" / "H1_hybrid_ctc_vlm" / "lambda_sweep" / "train-H1-mlp-onhw-lam06.yaml"
EVAL_TEMPLATE_FALLBACK = REWI / "configs" / "H1_hybrid_ctc_vlm" / "train-H1-mlp-onhw.yaml"


def _eval_template() -> Path:
    return EVAL_TEMPLATE if EVAL_TEMPLATE.is_file() else EVAL_TEMPLATE_FALLBACK


def run_evaluate(target_dir: Path) -> Path | None:
    """Run evaluate.py with dir_work=target_dir; return results.json path."""
    results_json = target_dir / "results.json"
    train_jsons = list(target_dir.glob("**/train_*.json"))
    if not train_jsons:
        return None
    if results_json.is_file():
        return results_json
    with open(_eval_template(), "r") as f:
        cfg = yaml.safe_load(f)
    cfg["dir_work"] = str(target_dir)
    cfg["idx_fold"] = -1
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=str(target_dir)
    ) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = tmp.name
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REWI}:{env.get('PYTHONPATH', '')}"
    try:
        subprocess.check_call(
            [sys.executable, str(REWI / "evaluate.py"), "-c", tmp_path],
            env=env,
            cwd=str(REWI),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return results_json if results_json.is_file() else None


def read_metrics(target_dir: Path) -> tuple[list[float], list[float]]:
    rj = run_evaluate(target_dir)
    if rj is None or not rj.is_file():
        return [], []
    with open(rj, "r") as f:
        d = json.load(f)
    cer_raw = d.get("cer", {}).get("raw", {}) or {}
    wer_raw = d.get("wer", {}).get("raw", {}) or {}
    cers = [float(v) * 100.0 for v in cer_raw.values()]
    wers = [float(v) * 100.0 for v in wer_raw.values()]
    return cers, wers


def mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def main() -> None:
    cer_per_lambda: list[list[float]] = []
    wer_per_lambda: list[list[float]] = []
    for lam in LAMBDA_VALUES:
        d = LAMBDA_DIRS[lam]
        print(f"[lambda={lam}] dir={d}")
        cers, wers = read_metrics(d)
        print(f"  -> {len(cers)} folds; CER={cers}")
        cer_per_lambda.append(cers)
        wer_per_lambda.append(wers)

    # Phase-3 (no-CTC MLP) reference
    p3_cers, p3_wers = read_metrics(PHASE3_DIR) if PHASE3_DIR.is_dir() else ([], [])
    p3_cer_mean = statistics.mean(p3_cers) if p3_cers else float("nan")
    p3_wer_mean = statistics.mean(p3_wers) if p3_wers else float("nan")

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_ctc", "n_folds", "cer_mean", "cer_sem", "wer_mean", "wer_sem"])
        for lam, c, wv in zip(LAMBDA_VALUES, cer_per_lambda, wer_per_lambda):
            cm, cs = mean_sem(c)
            wm, ws = mean_sem(wv)
            w.writerow([lam, len(c), f"{cm:.3f}", f"{cs:.3f}", f"{wm:.3f}", f"{ws:.3f}"])
        if p3_cers:
            w.writerow(
                ["phase3_mlp_noCTC", len(p3_cers), f"{p3_cer_mean:.3f}", "", f"{p3_wer_mean:.3f}", ""]
            )
    print(f"wrote {OUT_CSV}")

    # Plot
    cer_stats = [mean_sem(c) for c in cer_per_lambda]
    wer_stats = [mean_sem(w) for w in wer_per_lambda]
    cer_means = [m for m, _ in cer_stats]
    cer_sems = [s for _, s in cer_stats]
    wer_means = [m for m, _ in wer_stats]
    wer_sems = [s for _, s in wer_stats]

    # SELECTED = the lambda actually used in Chapter 6 (inherited from the
    # HWRFormer-L sweep of Chapter 5 for continuity, not a preference
    # statement). EMPIRICAL_MIN = argmin CER among finite means on this
    # sweep, kept on the figure as a secondary marker.
    SELECTED_LAMBDA = 0.6
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

    if not math.isnan(p3_cer_mean):
        ax_cer.axhline(p3_cer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)
        ax_wer.axhline(p3_wer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)

    SELECTED_COLOR = "#9467bd"  # purple — the chapter's operating point
    EMPIRICAL_COLOR = "#7f7f7f"  # grey  — empirical minimum (secondary)
    for ax in (ax_cer, ax_wer):
        # Selected (inherited) operating point: solid emphasis.
        ax.axvspan(SELECTED_LAMBDA - 0.015, SELECTED_LAMBDA + 0.015,
                   color=SELECTED_COLOR, alpha=0.22, zorder=1)
        ax.axvline(SELECTED_LAMBDA, color=SELECTED_COLOR, linestyle="-",
                   linewidth=1.8, alpha=0.95, zorder=2)
        # Empirical minimum: secondary thin dotted line.
        if empirical_min != SELECTED_LAMBDA:
            ax.axvline(empirical_min, color=EMPIRICAL_COLOR, linestyle=":",
                       linewidth=1.2, alpha=0.8, zorder=2)
        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.grid(True, alpha=0.3)

    ax_cer.set_ylabel("CER (\\%)", fontsize=12)
    ax_wer.set_ylabel("WER (\\%)", fontsize=12)
    ax_cer.set_title(r"H1 multimodal CER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)
    ax_wer.set_title(r"H1 multimodal WER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=cer_color, marker="o", markersize=6,
               linewidth=2, label=r"5-fold mean $\pm$ SEM (CER)"),
        Line2D([0], [0], color=wer_color, marker="s", markersize=6,
               linewidth=2, label=r"5-fold mean $\pm$ SEM (WER)"),
        Line2D([0], [0], color=SELECTED_COLOR, linestyle="-", linewidth=1.8,
               label=rf"selected $\lambda_{{\mathrm{{ctc}}}}={SELECTED_LAMBDA}$"),
        Line2D([0], [0], color=EMPIRICAL_COLOR, linestyle=":", linewidth=1.2,
               label=rf"empirical min $\lambda_{{\mathrm{{ctc}}}}={empirical_min}$"),
    ]
    if not math.isnan(p3_cer_mean):
        handles.insert(
            2,
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.5,
                   label="MLP without CTC"),
        )

    # 2-row legend (ceil(N/3) rows): avoids a single very wide row that
    # forces the subplots to shrink horizontally.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, fontsize=10, frameon=False,
               handlelength=2.2, columnspacing=1.8)

    # Reserve ~16% of vertical for the 2-row legend below the subplots.
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    for out_pdf in (OUT_PDF_THESIS, OUT_PDF_PAPER):
        if out_pdf.parent.is_dir() or out_pdf == OUT_PDF_THESIS:
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"saved: {out_pdf}")


if __name__ == "__main__":
    main()
