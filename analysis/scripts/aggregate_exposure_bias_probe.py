#!/usr/bin/env python3
"""Aggregate the teacher-forcing-gap probe (Table 5.7 / tab:exposure-bias-probe).

Reads the per-fold JSONs written by eval_tf_gap.py for the 16 cells
(4 conditions x 4 datasets) and reports, per (condition, dataset):

  - greedy (free-running) CER            : mean +/- SEM over folds
  - teacher-forced / oracle-prefix CER   : mean +/- SEM over folds
  - absolute gap (greedy - TF) in pp     : mean +/- SEM over folds   <-- the stable quantity
  - relative gap narrowing %             : (ref_gap - cell_gap)/ref_gap * 100,
                                           ref = HWRFormer-without-noise (AR-only) on the
                                           same dataset; computed from the fold-mean gaps.
  - per-fold narrowing % mean +/- SEM    : guarded (folds with ref_gap > 0.5 pp only),
                                           reported for reference -- the ratio is unstable
                                           when the reference gap is small.

This replaces the previous hand-assembly of the table: it adds the absolute gap and
fold-level uncertainty that were missing, and reports the oracle (teacher-forced) CER
explicitly so the gap can be read jointly with free-running CER (per the reviewer note).

Usage:
    envs/rewi26/bin/python analysis/scripts/aggregate_exposure_bias_probe.py
Output: results/exposure_bias_probe_aggregated.csv  (+ printed summary)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_CSV = Path("/home/woody/iwso/iwso214h/imu-hwr/results/exposure_bias_probe_aggregated.csv")
N_FOLDS = 5

# dataset key -> (arch-dataset suffix, hybrid dir suffix)
DATASETS = {
    "OnHW-WI": ("onhw_wi_word_rh", "onhw_wi"),
    "OnHW-WD": ("onhw_wd_word_rh", "onhw_wd"),
    "Priv-W":  ("wi_word_hw6_meta", "stabilo"),
    "Priv-S":  ("wi_sent_hw6_meta", "stabilo_sent"),
}
# condition -> function(ds_suffix, hyb_suffix) -> cell subdir under results/hwr2
CONDITIONS = {
    "HWRFormer":       lambda ds, hy: f"Baseline-AR-XS-blconv_b/ar_transformer_xs__{ds}",
    "+Noise":          lambda ds, hy: f"Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__{ds}",
    "+Hybrid":         lambda ds, hy: f"train_element_word_hybrid_01_xs_{hy}/ar_transformer_xs__{ds}",
    "+Hybrid+Noise":   lambda ds, hy: f"HybridInputCorruption-XS-L01_uniform/ar_transformer_xs__{ds}",
}
REF_CONDITION = "HWRFormer"  # HWRFormer without noise injection = the normalization reference


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sem(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)   # sample variance
    return math.sqrt(var) / math.sqrt(n)


def read_cell(subdir: str):
    """Return per-fold dicts {greedy, tf, gap_pp} (CER in %), or [] if missing."""
    cell = RESULTS / subdir / "eval_tf_gap"
    rows = []
    for k in range(N_FOLDS):
        f = cell / f"fold_{k}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        greedy = float(r["ar"]["cer"]) * 100.0
        tf = float(r["tf"]["cer"]) * 100.0
        rows.append({"fold": k, "greedy": greedy, "tf": tf, "gap_pp": greedy - tf})
    return rows


def main():
    # 1) read every cell
    data = {}  # (cond, ds) -> per-fold rows
    for ds_key, (ds, hy) in DATASETS.items():
        for cond, tmpl in CONDITIONS.items():
            data[(cond, ds_key)] = read_cell(tmpl(ds, hy))

    # 2) per-dataset reference (AR-only) fold gaps, keyed by fold
    out_rows = []
    for ds_key in DATASETS:
        ref_rows = data[(REF_CONDITION, ds_key)]
        ref_by_fold = {r["fold"]: r["gap_pp"] for r in ref_rows}
        ref_gap_mean = mean([r["gap_pp"] for r in ref_rows]) if ref_rows else float("nan")
        for cond in CONDITIONS:
            rows = data[(cond, ds_key)]
            if not rows:
                out_rows.append({"dataset": ds_key, "condition": cond, "n_folds": 0})
                continue
            greedy = [r["greedy"] for r in rows]
            tf = [r["tf"] for r in rows]
            gap = [r["gap_pp"] for r in rows]
            # relative narrowing from fold-mean gaps (matches how the thesis reported it)
            cell_gap_mean = mean(gap)
            narrowing_from_means = (
                (ref_gap_mean - cell_gap_mean) / ref_gap_mean * 100.0
                if ref_gap_mean and ref_gap_mean == ref_gap_mean and abs(ref_gap_mean) > 1e-9
                else float("nan")
            )
            # per-fold narrowing, guarded to folds with a non-trivial reference gap
            per_fold_narrowing = [
                (ref_by_fold[r["fold"]] - r["gap_pp"]) / ref_by_fold[r["fold"]] * 100.0
                for r in rows
                if r["fold"] in ref_by_fold and ref_by_fold[r["fold"]] > 0.5
            ]
            out_rows.append({
                "dataset": ds_key, "condition": cond, "n_folds": len(rows),
                "greedy_cer_mean": mean(greedy), "greedy_cer_sem": sem(greedy),
                "tf_cer_mean": mean(tf), "tf_cer_sem": sem(tf),
                "abs_gap_pp_mean": mean(gap), "abs_gap_pp_sem": sem(gap),
                "narrowing_pct_from_means": narrowing_from_means,
                "narrowing_pct_perfold_mean": mean(per_fold_narrowing) if per_fold_narrowing else float("nan"),
                "narrowing_pct_perfold_sem": sem(per_fold_narrowing) if len(per_fold_narrowing) > 1 else float("nan"),
                "narrowing_perfold_n": len(per_fold_narrowing),
            })

    # 3) write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "condition", "n_folds",
            "greedy_cer_mean", "greedy_cer_sem", "tf_cer_mean", "tf_cer_sem",
            "abs_gap_pp_mean", "abs_gap_pp_sem",
            "narrowing_pct_from_means", "narrowing_pct_perfold_mean",
            "narrowing_pct_perfold_sem", "narrowing_perfold_n"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # 4) printed summary
    def fmt(v):
        return f"{v:5.2f}" if isinstance(v, float) and v == v else "  -- "
    print(f"{'dataset':8s} {'condition':14s} {'n':>2s}  "
          f"{'greedy CER':>16s}  {'TF/oracle CER':>16s}  {'abs gap pp':>14s}  {'narrow%':>8s}")
    for r in out_rows:
        if r["n_folds"] == 0:
            print(f"{r['dataset']:8s} {r['condition']:14s}  0   (no fold JSONs found)")
            continue
        print(f"{r['dataset']:8s} {r['condition']:14s} {r['n_folds']:>2d}  "
              f"{fmt(r['greedy_cer_mean'])}+/-{fmt(r['greedy_cer_sem'])}  "
              f"{fmt(r['tf_cer_mean'])}+/-{fmt(r['tf_cer_sem'])}  "
              f"{fmt(r['abs_gap_pp_mean'])}+/-{fmt(r['abs_gap_pp_sem'])}  "
              f"{fmt(r['narrowing_pct_from_means'])}")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
