#!/usr/bin/env python3
"""Paired HWRFormer vs hybrid CTC-AR analysis across all four datasets.

Mirrors the noise-injection analysis figure (top: aligned per-reference-
position error; bottom: eCDF of normalized edit distance) for hybrid
training at lambda_ctc = 0.1 (XS scale = the thesis "HWRFormer").

Canonical sources (best-checkpoint epoch0 re-export decodes):
  OnHW WI/WD : analysis/quant_all_val_predictions_ar_vs_hybrid_xs.csv
  Private    : Baseline-AR-XS-blconv_b vs train_element_word_hybrid_01_xs_
               {stabilo, stabilo_sent} val_full_fold<k>_epoch0[_ar].json

Output: presentation/figures/hybrid_paired_all.pdf
Prints per-dataset exact-match deltas and paired five-fold Delta-e CIs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import Levenshtein
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

R = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
CSV = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/"
           "quant_all_val_predictions_ar_vs_hybrid_xs.csv")
OUT = Path("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
           "hybrid_paired_all.pdf")

AR_COLOR = "#1f77b4"
HY_COLOR = "#2ca02c"
MIN_AT_POS = 50
T95_DF4 = 2.776

DATASETS = ["OnHW-WI word", "OnHW-WD word", "Private word", "Private sentence"]
POS_CAPS = {"OnHW-WI word": 15, "OnHW-WD word": 15,
            "Private word": 15, "Private sentence": 40}


def load_pairs() -> dict[str, dict[str, list[tuple[int, str, str]]]]:
    """-> data[dataset][model] = list of (fold, label, prediction)."""
    data = {d: {"AR": [], "HY": []} for d in DATASETS}
    for r in csv.DictReader(open(CSV), delimiter=";"):
        ds = "OnHW-WD word" if "_wd" in r["json_path"] else "OnHW-WI word"
        m = "AR" if r["task"] == "AR-only" else "HY"
        data[ds][m].append((int(r["fold"]), r["label"], r["prediction"]))
    priv = {
        "Private word": ("wi_word_hw6_meta", "train_element_word_hybrid_01_xs_stabilo"),
        "Private sentence": ("wi_sent_hw6_meta", "train_element_word_hybrid_01_xs_stabilo_sent"),
    }
    for ds, (dsdir, hybgrp) in priv.items():
        for f in range(5):
            ar = json.load(open(R / "Baseline-AR-XS-blconv_b" / f"ar_transformer_xs__{dsdir}"
                                / f"fold_{f}" / "exports" / f"val_full_fold{f}_epoch0.json"))
            hy = json.load(open(R / hybgrp / f"ar_transformer_xs__{dsdir}"
                                / f"fold_{f}" / "exports" / f"val_full_fold{f}_epoch0_ar.json"))
            for lab, pred in zip(ar["labels"], ar["predictions"]):
                data[ds]["AR"].append((f, lab, pred))
            for lab, pred in zip(hy["labels"], hy["predictions"]):
                data[ds]["HY"].append((f, lab, pred))
    return data


def main() -> None:
    data = load_pairs()
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.4))

    for col, ds in enumerate(DATASETS):
        cap = POS_CAPS[ds]
        ax_top, ax_bot = axes[0, col], axes[1, col]
        stats = {}
        for m, color, marker in [("AR", AR_COLOR, "o"), ("HY", HY_COLOR, "s")]:
            recs = data[ds][m]
            es = np.array([Levenshtein.distance(l, p) / max(1, len(l))
                           for _, l, p in recs])
            fs = np.array([f for f, _, _ in recs])
            err = np.zeros(cap); n_at = np.zeros(cap)
            for _, lab, pred in recs:
                for k in range(min(len(lab), cap)):
                    n_at[k] += 1
                for op, i, _j in Levenshtein.editops(lab, pred):
                    if op in ("replace", "delete") and i < cap:
                        err[i] += 1
            stats[m] = (es, fs)
            keep = n_at >= MIN_AT_POS
            ks = np.arange(cap)
            ax_top.plot(ks[keep], 100 * err[keep] / n_at[keep], marker=marker,
                        color=color, lw=2, ms=4)
            e = np.sort(es)
            ax_bot.plot(e, np.arange(1, e.size + 1) / e.size, color=color, lw=2.2)
            ax_bot.axvline(float(es.mean()), color=color, ls="--", lw=1.0, alpha=0.7)

        d_folds = [float(stats["HY"][0][stats["HY"][1] == f].mean()
                         - stats["AR"][0][stats["AR"][1] == f].mean()) for f in range(5)]
        dm = float(np.mean(d_folds))
        dh = T95_DF4 * float(np.std(d_folds, ddof=1)) / np.sqrt(5)
        n = stats["AR"][0].size
        em_d = 100 * (np.sum(stats["HY"][0] == 0) - np.sum(stats["AR"][0] == 0)) / n
        print(f"{ds:18s} n={n:6d}  exact-match delta {em_d:+.2f} pp   "
              f"D-e {100*dm:+.2f} pp CI [{100*(dm-dh):+.2f}, {100*(dm+dh):+.2f}]")

        ax_top.set_title(ds, fontsize=11)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xlabel(r"Reference position $k$", fontsize=10)
        if col == 0:
            ax_top.set_ylabel("Aligned per-ref-position error (%)", fontsize=10)
        ax_bot.set_xlim(0, 1.5); ax_bot.set_ylim(0, 1.02)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_xlabel(r"$\tilde{e} = d(\hat{y}, y)/\max(1, |y|)$", fontsize=10)
        if col == 0:
            ax_bot.set_ylabel(r"$P(\tilde{e}_i \leq x)$", fontsize=10.5)
        ax_bot.text(0.97, 0.03,
                    (r"$\bar{\Delta\tilde{e}} = " + f"{100*dm:+.2f}$ pp\n"
                     f"95% CI = [{100*(dm-dh):+.2f}, {100*(dm+dh):+.2f}] pp"),
                    transform=ax_bot.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="0.7", alpha=0.92))

    fig.legend(handles=[
        Line2D([0], [0], color=AR_COLOR, lw=2.2, label="HWRFormer"),
        Line2D([0], [0], color=HY_COLOR, lw=2.2,
               label="Hybrid CTC–AR ($\\lambda_{\\mathrm{ctc}}=0.1$)")],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2,
        fontsize=11, frameon=False, columnspacing=2.0, handlelength=2.4)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
