#!/usr/bin/env python3
"""Aggregate and plot the single-corruption recovery profile.

Reads the per-(model, dataset, fold) JSON files written by
eval_single_corruption.py and renders a 1x4 grid: one panel per dataset,
x-axis = corruption offset (0..N-1), y-axis = P(predict label[t+offset]
correctly | label[t] corrupted). Two curves per panel (AR-only vs
noise-injection) with shaded +-1 SEM bands over the 5 folds.

Offset 0 is the unaffected baseline (the model predicts label[t] from the
uncorrupted prefix label[0..t-1]). Offsets 1..N-1 see the corruption
directly (offset 1, immediate next-step prediction) or indirectly through
the causal attention chain.

Outputs:
    thesis/figures/single_corruption_recovery.pdf
    analysis/single_corruption_recovery.csv

Run from work/REWI_work:
    python analysis/scripts/plot_single_corruption_recovery.py
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
WORK = REPO / "work" / "REWI_work"
OUT_FIG = REPO / "thesis" / "figures" / "single_corruption_recovery.pdf"
OUT_CSV = WORK / "analysis" / "single_corruption_recovery.csv"

MODELS = [
    ("HWRFormer", "Baseline-AR-XS-blconv_b", "#1f77b4", "o"),
    ("HWRFormer + noise", "Baseline-AR-XS-InputCorruption-uniform", "#d62728", "s"),
]
DATASETS = [
    ("ar_transformer_xs__onhw_wi_word_rh", "onhw_wi_word", "OnHW WI word"),
    ("ar_transformer_xs__onhw_wd_word_rh", "onhw_wd_word", "OnHW WD word"),
    ("ar_transformer_xs__wi_word_hw6_meta", "priv_word", "Private word"),
    ("ar_transformer_xs__wi_sent_hw6_meta", "priv_sent", "Private sentence"),
]


def load_fold(model_dir: str, arch_dataset: str, fold: int) -> dict | None:
    p = RESULTS / model_dir / arch_dataset / f"fold_{fold}" / "eval_single_corruption.json"
    if not p.is_file():
        return None
    with open(p, "r") as f:
        payload = json.load(f)
    rec = payload.get("single_corruption") or {}
    return rec


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = [v for v in values if v == v]  # drop nan
    if not arr:
        return float("nan"), float("nan")
    m = statistics.mean(arr)
    if len(arr) > 1:
        s = statistics.stdev(arr) / math.sqrt(len(arr))
    else:
        s = 0.0
    return m, s


def main() -> None:
    # Collect: results[dataset][model] = list of 5 per-fold accuracy lists
    n_offsets_seen = 0
    table_rows = []
    panel_data = {}  # (dataset_key) -> {(model_label) -> per-fold accuracies}

    for arch_dataset, ds_key, _ds_label in DATASETS:
        panel_data[ds_key] = {}
        for model_label, model_dir, _color, _marker in MODELS:
            per_fold: list[list[float]] = []
            n_total_per_fold: list[list[int]] = []
            for fold in range(5):
                rec = load_fold(model_dir, arch_dataset, fold)
                if rec is None:
                    print(f"  missing: {model_label} {ds_key} fold{fold}")
                    continue
                per_fold.append(rec["accuracy"])
                n_total_per_fold.append(rec["n_total"])
                n_offsets_seen = max(n_offsets_seen, len(rec["accuracy"]))
            panel_data[ds_key][model_label] = (per_fold, n_total_per_fold)

    if n_offsets_seen == 0:
        print("no JSON files found; nothing to plot")
        return

    # CSV write: per (dataset, model, offset) mean/sem and pooled n_total
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset_task", "model", "offset", "n_folds",
            "mean_accuracy", "sem_accuracy", "n_total_pooled",
        ])
        for _arch_ds, ds_key, _ in DATASETS:
            for model_label, _, _, _ in MODELS:
                per_fold, n_tot = panel_data[ds_key].get(model_label, ([], []))
                if not per_fold:
                    continue
                for off in range(n_offsets_seen):
                    accs = [pf[off] for pf in per_fold if off < len(pf)]
                    nts = [n[off] for n in n_tot if off < len(n)]
                    m, s = mean_sem(accs)
                    w.writerow([
                        ds_key, model_label, off, len(accs),
                        f"{m:.6f}", f"{s:.6f}", sum(nts),
                    ])
    print(f"wrote csv: {OUT_CSV}")

    # Plot 1x4 grid
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.8 * n_ds, 3.8),
                             sharey=True)
    if n_ds == 1:
        axes = [axes]
    for col, (_arch_ds, ds_key, ds_label) in enumerate(DATASETS):
        ax = axes[col]
        for model_label, _, color, marker in MODELS:
            per_fold, n_tot = panel_data[ds_key].get(model_label, ([], []))
            if not per_fold:
                continue
            offsets = list(range(n_offsets_seen))
            means = []
            sems = []
            for off in offsets:
                accs = [pf[off] for pf in per_fold if off < len(pf)]
                m, s = mean_sem(accs)
                means.append(m)
                sems.append(s)
            ax.plot(offsets, [m * 100 for m in means],
                    color=color, marker=marker, linewidth=2.2,
                    markersize=7, label=model_label)
            ax.fill_between(
                offsets,
                [(m - s) * 100 for m, s in zip(means, sems)],
                [(m + s) * 100 for m, s in zip(means, sems)],
                color=color, alpha=0.18, linewidth=0,
            )
        ax.set_title(ds_label, fontsize=14)
        ax.set_xlabel(r"Offset from corrupted position $t$", fontsize=14)
        if col == 0:
            ax.set_ylabel(r"$\Pr$(predict label$[t{+}\delta]$ correctly) (%)",
                          fontsize=14)
        ax.set_xticks(range(n_offsets_seen))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    # Single figure-level legend below the panels (no per-panel legend).
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", marker="o", lw=2.2, markersize=7,
                   label="HWRFormer"),
            Line2D([0], [0], color="#d62728", marker="s", lw=2.2, markersize=7,
                   label=r"HWRFormer + noise ($p{=}0.15$)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
        fontsize=13, frameon=False, columnspacing=2.0, handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
