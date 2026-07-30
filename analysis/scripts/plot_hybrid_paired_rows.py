#!/usr/bin/env python3
"""Presentation figure: paired HWRFormer vs hybrid CTC-AR analysis, OnHW-WI.

Mirrors the noise-injection analysis rows (per-reference-position error +
eCDF of normalized edit distance) for the hybrid comparison, using the
curated per-sample predictions of analysis/quant_all_val_predictions_
ar_vs_hybrid.csv (OnHW-words500 WI, all five validation folds; tasks
'AR-only' and 'Hybrid (AR Decoding)').

Output: presentation/figures/hybrid_paired_onhw_wi.pdf
Also prints exact-match counts and the paired five-fold Delta-e interval.

Run from anywhere:
    python plot_hybrid_paired_rows.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import Levenshtein
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/"
           "quant_all_val_predictions_ar_vs_hybrid.csv")
OUT = Path("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
           "hybrid_paired_onhw_wi.pdf")

TASK_AR = "AR-only"
TASK_HY = "Hybrid (AR Decoding)"
AR_COLOR = "#1f77b4"
HY_COLOR = "#2ca02c"
POS_CAP = 15
MIN_AT_POS = 50
T95_DF4 = 2.776


def main() -> None:
    rows = list(csv.DictReader(open(CSV), delimiter=";"))
    data: dict[str, list[dict]] = {TASK_AR: [], TASK_HY: []}
    for r in rows:
        data[r["task"]].append(r)

    # ----- per-sample normalized edit distance and per-position error -----
    enorm: dict[str, np.ndarray] = {}
    fold_of: dict[str, np.ndarray] = {}
    pos_err: dict[str, np.ndarray] = {}
    pos_n = np.zeros(POS_CAP)
    for task in (TASK_AR, TASK_HY):
        es, fs = [], []
        err = np.zeros(POS_CAP)
        n_at = np.zeros(POS_CAP)
        for r in data[task]:
            lab, pred = r["label"], r["prediction"]
            d = int(r["levenshtein_distance"])
            es.append(d / max(1, len(lab)))
            fs.append(int(r["fold"]))
            for k in range(min(len(lab), POS_CAP)):
                n_at[k] += 1
            for op, i, _j in Levenshtein.editops(lab, pred):
                if op in ("replace", "delete") and i < POS_CAP:
                    err[i] += 1
        enorm[task] = np.array(es)
        fold_of[task] = np.array(fs)
        pos_err[task] = err / np.maximum(n_at, 1)
        pos_n = n_at

    # paired per-fold Delta e (hybrid minus AR); rows are sample-aligned per task
    d_folds = []
    for f in range(5):
        m = fold_of[TASK_AR] == f
        d_folds.append(float(np.mean(enorm[TASK_HY][fold_of[TASK_HY] == f]))
                       - float(np.mean(enorm[TASK_AR][m])))
    dm = float(np.mean(d_folds))
    dh = T95_DF4 * float(np.std(d_folds, ddof=1)) / np.sqrt(5)

    em_ar = int(np.sum(enorm[TASK_AR] == 0))
    em_hy = int(np.sum(enorm[TASK_HY] == 0))
    print(f"exact match: AR {em_ar}  hybrid {em_hy}  (of {len(enorm[TASK_AR])})")
    print(f"Delta e-norm mean {100*dm:+.2f} pp, 95% CI "
          f"[{100*(dm-dh):+.2f}, {100*(dm+dh):+.2f}] pp")

    # ----- figure: position row + eCDF, single row, two panels -----
    fig, (ax_pos, ax_cdf) = plt.subplots(1, 2, figsize=(10.8, 3.4))
    ks = np.arange(POS_CAP)
    keep = pos_n >= MIN_AT_POS
    for task, color, marker, lab in [(TASK_AR, AR_COLOR, "o", "HWRFormer"),
                                     (TASK_HY, HY_COLOR, "s", "Hybrid CTC--AR")]:
        ax_pos.plot(ks[keep], 100 * pos_err[task][keep], marker=marker,
                    color=color, lw=2, ms=4, label=lab.replace("--", "–"))
        e = np.sort(enorm[task])
        ax_cdf.plot(e, np.arange(1, e.size + 1) / e.size, color=color, lw=2.2)
        ax_cdf.axvline(float(np.mean(enorm[task])), color=color, ls="--",
                       lw=1.0, alpha=0.7)
    ax_pos.set_xlabel(r"Reference position $k$", fontsize=10)
    ax_pos.set_ylabel("Aligned per-ref-position error (%)", fontsize=10)
    ax_pos.set_title("OnHW-WI word", fontsize=11)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(fontsize=9, frameon=False)

    ax_cdf.set_xlim(0, 1.5)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_xlabel(r"$\tilde{e} = d(\hat{y}, y)/\max(1, |y|)$", fontsize=10)
    ax_cdf.set_ylabel(r"$P(\tilde{e}_i \leq x)$", fontsize=10)
    ax_cdf.set_title("OnHW-WI word", fontsize=11)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.text(0.97, 0.03,
                (r"$\bar{\Delta\tilde{e}} = " + f"{100*dm:+.2f}$ pp\n"
                 f"95% CI = [{100*(dm-dh):+.2f}, {100*(dm+dh):+.2f}] pp"),
                transform=ax_cdf.transAxes, ha="right", va="bottom",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="0.7", alpha=0.92))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
