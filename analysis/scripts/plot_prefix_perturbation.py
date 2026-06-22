#!/usr/bin/env python3
"""Aggregate eval_tf_perturbation_p{NNN}.json files and plot the
prefix-perturbation robustness curves for the AR-only and AR + noise
injection HWRFormer xs checkpoints on OnHW WI word.

Inputs:
    results/hwr2/Baseline-AR-XS-blconv_b/.../fold_{k}/eval_tf_perturbation_p{NNN}.json
    results/hwr2/Baseline-AR-XS-InputCorruption-uniform/.../fold_{k}/eval_tf_perturbation_p{NNN}.json

Outputs:
    analysis/prefix_perturbation_sweep.csv      (long-format aggregate)
    thesis/figures/prefix_perturbation_robustness.pdf

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
ARCH = "ar_transformer_xs__onhw_wi_word_rh"

SOURCES = [
    ("AR-only", "Baseline-AR-XS-blconv_b"),
    ("AR + noise (uniform p=0.15)", "Baseline-AR-XS-InputCorruption-uniform"),
]

AR_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"


def load_results() -> list[dict]:
    rows: list[dict] = []
    for task, dir_root in SOURCES:
        for fold in range(5):
            for p in P_VALUES:
                p_tag = f"{int(round(p * 100)):03d}"
                path = (
                    RESULTS / dir_root / ARCH / f"fold_{fold}"
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


def plot(rows: list[dict]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    for task, color, marker in [
        ("AR-only", AR_COLOR, "o"),
        ("AR + noise (uniform p=0.15)", NOISE_COLOR, "s"),
    ]:
        task_rows = [r for r in rows if r["task"] == task]
        means: list[float] = []
        sems: list[float] = []
        for p in P_VALUES:
            vals = [r["cer"] * 100.0 for r in task_rows if abs(r["p_replace"] - p) < 1e-9]
            if not vals:
                means.append(float("nan"))
                sems.append(0.0)
                continue
            arr = np.array(vals, dtype=float)
            means.append(float(arr.mean()))
            sems.append(float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
        means_arr = np.array(means)
        sems_arr = np.array(sems)
        ax.errorbar(
            P_VALUES, means_arr, yerr=sems_arr,
            marker=marker, color=color, linewidth=2, capsize=3,
            label=task,
        )
    ax.set_xlabel(r"Prefix perturbation rate $p_{\mathrm{replace}}$", fontsize=12)
    ax.set_ylabel(r"Teacher-forced \gls{cer} (\%)", fontsize=12)
    ax.set_title(
        r"Prefix-perturbation robustness, HWRFormer on OnHW WI word",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
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
