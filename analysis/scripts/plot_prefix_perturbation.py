#!/usr/bin/env python3
"""Aggregate eval_tf_perturbation_p{NNN}.json files and plot prefix-perturbation
robustness curves on all four datasets.

Inputs (per dataset, per checkpoint, per fold, per p in {0.00, 0.05, 0.10,
0.15, 0.20}):
    results/hwr2/<dir_root>/ar_transformer_xs__<dataset>/fold_{k}/
        eval_tf_perturbation_p{NNN}.json

Outputs:
    analysis/prefix_perturbation_sweep.csv         (long-format aggregate)
    thesis/figures/prefix_perturbation_robustness.pdf  (1 row x 4 cols)

Run from work/REWI_work:
    python analysis/scripts/plot_prefix_perturbation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_CSV = WORK_DIR / "analysis" / "prefix_perturbation_sweep.csv"
THESIS_DIR = WORK_DIR.parent.parent / "thesis"
OUT_FIG = THESIS_DIR / "figures" / "prefix_perturbation_robustness.pdf"

P_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20]

# (display_label, arch_suffix)
DATASETS: list[tuple[str, str]] = [
    ("OnHW WI word", "ar_transformer_xs__onhw_wi_word_rh"),
    ("OnHW WD word", "ar_transformer_xs__onhw_wd_word_rh"),
    ("Private word", "ar_transformer_xs__wi_word_hw6_meta"),
    ("Private sentence", "ar_transformer_xs__wi_sent_hw6_meta"),
]

CHECKPOINTS: list[tuple[str, str]] = [
    ("AR-only", "Baseline-AR-XS-blconv_b"),
    ("AR + noise (uniform p=0.15)", "Baseline-AR-XS-InputCorruption-uniform"),
]

AR_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"


def load_results() -> list[dict]:
    rows: list[dict] = []
    for task, dir_root in CHECKPOINTS:
        for display, arch in DATASETS:
            for fold in range(5):
                for p in P_VALUES:
                    p_tag = f"{int(round(p * 100)):03d}"
                    path = (
                        RESULTS / dir_root / arch / f"fold_{fold}"
                        / f"eval_tf_perturbation_p{p_tag}.json"
                    )
                    if not path.exists():
                        print(f"missing: {path}")
                        continue
                    with open(path) as fp:
                        d = json.load(fp)
                    r = d.get("tf_perturbation", {})
                    rows.append({
                        "task": task,
                        "dir_root": dir_root,
                        "arch": arch,
                        "dataset_label": display,
                        "fold": fold,
                        "p_replace": float(p),
                        "cer": float(r.get("cer", -1)),
                        "wer": float(r.get("wer", -1)),
                        "n_samples": int(r.get("n_samples", 0)),
                        "json_path": str(path),
                    })
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no rows to write")
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


def _curve(rows: list[dict], task: str, arch: str) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    sems: list[float] = []
    for p in P_VALUES:
        vals = [
            r["cer"] * 100.0
            for r in rows
            if r["task"] == task and r["arch"] == arch
            and abs(r["p_replace"] - p) < 1e-9
        ]
        if not vals:
            means.append(float("nan"))
            sems.append(0.0)
            continue
        arr = np.array(vals, dtype=float)
        means.append(float(arr.mean()))
        sems.append(
            float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        )
    return np.array(means), np.array(sems)


def plot(rows: list[dict]) -> None:
    # 1x4 layout matching single_corruption_recovery.pdf for chapter
    # consistency. Independent y-axes (no sharey) because the four datasets
    # have substantially different CER scales.
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.8 * n_ds, 3.8))
    if n_ds == 1:
        axes = [axes]
    for idx, (display, arch) in enumerate(DATASETS):
        ax = axes[idx]
        ar_m, ar_s = _curve(rows, "AR-only", arch)
        no_m, no_s = _curve(rows, "AR + noise (uniform p=0.15)", arch)
        ax.errorbar(
            P_VALUES, ar_m, yerr=ar_s,
            marker="o", color=AR_COLOR, linewidth=2, markersize=7, capsize=3,
            label="HWRFormer",
        )
        ax.errorbar(
            P_VALUES, no_m, yerr=no_s,
            marker="s", color=NOISE_COLOR, linewidth=2, markersize=7, capsize=3,
            label="HWRFormer + noise injection",
        )
        ax.set_title(display, fontsize=14)
        ax.set_xlabel(r"Prefix perturbation rate $p_{\mathrm{replace}}$",
                      fontsize=14)
        ax.set_xticks(P_VALUES)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel(r"Oracle-prefix CER (%)", fontsize=14)
    # Single figure-level legend below the panels (no per-panel legend).
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], color=AR_COLOR, marker="o", lw=2, markersize=7,
                   label="HWRFormer"),
            Line2D([0], [0], color=NOISE_COLOR, marker="s", lw=2, markersize=7,
                   label="HWRFormer + noise injection"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
        fontsize=13, frameon=False, columnspacing=2.0, handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {OUT_FIG}")


def main() -> None:
    rows = load_results()
    write_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()
