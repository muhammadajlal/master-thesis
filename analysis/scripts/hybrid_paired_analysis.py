#!/usr/bin/env python3
"""Paired significance analysis: HWRFormer vs hybrid HWRFormer on OnHW.

Committed, reproducible source for the hybrid rows of the paired-error-analysis
table. For each writer split (WI, WD) it reports the paired exact-match effect
with BOTH fold-level and writer-clustered uncertainty, the pooled McNemar test
(secondary evidence only), a writer-level Wilcoxon signed-rank test, and the
paired all-sample normalized-edit-distance (NED) effect. It also emits each
model's conditional error severity on the correct xs data, replacing the stale
values that were carried over by hand from the old ``s``-variant CSV.

Model naming: HWRFormer = shared CNN encoder + ar_transformer_xs decoder (the
``xs`` variant analysed here); HWRFormer-L would be the ar_transformer_s decoder.
"AR-only" and "hybrid" are training conditions of the same recognizer.

McNemar and the fold-t-interval mirror ``cascade_analysis.py`` so the hybrid and
noise comparisons use identical methodology. lambda_ctc = 0.1 and the best epoch
were selected on these same validation folds, so the pooled p-value is
exploratory/post-selection, not confirmatory -- hence the fold- and
writer-clustered intervals are the primary uncertainty statements.

Run:
    envs/rewi26/bin/python analysis/scripts/hybrid_paired_analysis.py
Outputs: analysis/tables/hybrid_paired_analysis_xs.{json,csv}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WORK = Path(__file__).resolve().parent.parent.parent            # REWI_work
DATA = WORK.parent.parent / "data"                              # imu-hwr/data
CSV = WORK / "analysis" / "quant_all_val_predictions_ar_vs_hybrid_xs.csv"
OUT_DIR = WORK / "analysis" / "tables"
OUT_JSON = OUT_DIR / "hybrid_paired_analysis_xs.json"
OUT_CSV = OUT_DIR / "hybrid_paired_analysis_xs.csv"

AR = "AR-only"
HYB = "Hybrid (AR Decoding)"
SPLIT_DATASET = {"wi": "onhw_wi_word_rh", "wd": "onhw_wd_word_rh"}
N_BOOT = 20000
SEED = 0


def fold_t_interval(vals: list[float]) -> tuple[float, tuple[float, float]]:
    """95% paired t-interval over per-fold means (df = n-1), as in cascade_analysis."""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, (float("nan"), float("nan"))
    sem = float(arr.std(ddof=1)) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, (float(lo), float(hi))


def writer_map(split: str) -> dict[tuple[int, int], object]:
    """(fold, sample_index) -> id_writer, read from the split's val.json."""
    vj = json.load(open(DATA / SPLIT_DATASET[split] / "val.json"))
    out = {}
    for fold_str, items in vj["annotations"].items():
        f = int(fold_str)
        for idx, it in enumerate(items):
            out[(f, idx)] = it.get("id_writer", it.get("writer_id"))
    return out


def load_pairs(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Inner-join AR-only vs hybrid on (fold, sample_index) for one split."""
    d = df[df["split"] == split]
    ar = d[d["task"] == AR].set_index(["fold", "sample_index"])
    hy = d[d["task"] == HYB].set_index(["fold", "sample_index"])
    keep = ["levenshtein_distance", "label"]
    j = ar[keep].join(hy[keep], lsuffix="_ar", rsuffix="_hyb", how="inner").reset_index()
    # Sanity: the paired rows must share the same reference label.
    if not (j["label_ar"].astype(str) == j["label_hyb"].astype(str)).all():
        raise SystemExit(f"[{split}] AR/hybrid label mismatch on paired rows -- keys misaligned.")
    j["exact_ar"] = j["levenshtein_distance_ar"] == 0
    j["exact_hyb"] = j["levenshtein_distance_hyb"] == 0
    j["len_ref"] = j["label_ar"].astype(str).str.len().clip(lower=1)
    j["ned_ar"] = j["levenshtein_distance_ar"] / j["len_ref"]
    j["ned_hyb"] = j["levenshtein_distance_hyb"] / j["len_ref"]
    return j


def analyse(df: pd.DataFrame, split: str) -> dict:
    j = load_pairs(df, split)
    n = len(j)
    ar_c = j["exact_ar"].to_numpy(bool)
    hy_c = j["exact_hyb"].to_numpy(bool)

    # --- exact-match McNemar (pooled; secondary evidence) ---
    b = int(np.sum(ar_c & ~hy_c))          # AR correct, hybrid wrong
    c = int(np.sum(~ar_c & hy_c))          # hybrid correct, AR wrong
    n_disc = b + c
    mcnemar_p = 1.0 if n_disc == 0 else float(
        stats.binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided").pvalue)

    # --- per-fold exact-match effect (hybrid - AR), pp, with fold t-CI ---
    fold_eff = [100.0 * (g["exact_hyb"].mean() - g["exact_ar"].mean())
                for _, g in j.groupby("fold")]
    fold_mean_eff, (fold_lo, fold_hi) = fold_t_interval(fold_eff)

    # --- paired all-sample NED effect (hybrid - AR), pp, with fold t-CI (F3) ---
    ned_fold = [100.0 * (g["ned_hyb"].mean() - g["ned_ar"].mean())
                for _, g in j.groupby("fold")]
    ned_mean, (ned_lo, ned_hi) = fold_t_interval(ned_fold)

    # --- conditional severity on each model's OWN error subset (descriptive) ---
    # Reported on the correct xs data. NOTE: the two means condition on disjoint
    # sample sets, so they are NOT a paired quantity -- the paired NED above is
    # the honest all-sample effect.
    ar_err = j[j["levenshtein_distance_ar"] > 0]
    hy_err = j[j["levenshtein_distance_hyb"] > 0]
    severity = {
        "ar_n_errors": int(len(ar_err)),
        "hyb_n_errors": int(len(hy_err)),
        "ar_cond_mean_d": float(ar_err["levenshtein_distance_ar"].mean()),
        "hyb_cond_mean_d": float(hy_err["levenshtein_distance_hyb"].mean()),
        "ar_cond_mean_dnorm": float(ar_err["ned_ar"].mean()),
        "hyb_cond_mean_dnorm": float(hy_err["ned_hyb"].mean()),
    }

    # --- writer clustering ---
    wmap = writer_map(split)
    writers_col = [wmap.get((int(f), int(i))) for f, i in zip(j["fold"], j["sample_index"])]
    j = j.assign(writer=writers_col)
    n_missing = int(pd.isna(j["writer"]).sum())
    if n_missing:
        raise SystemExit(f"[{split}] {n_missing} paired rows have no writer id (val.json misaligned).")
    writers = j["writer"].unique()

    # Per-writer correct-counts and sizes for a fast writer-clustered bootstrap.
    g = j.groupby("writer")
    w_ar = g["exact_ar"].sum().to_numpy(float)
    w_hy = g["exact_hyb"].sum().to_numpy(float)
    w_n = g.size().to_numpy(float)
    rng = np.random.default_rng(SEED)
    nW = len(writers)
    boot = np.empty(N_BOOT)
    for t in range(N_BOOT):
        pick = rng.integers(0, nW, nW)
        tot = w_n[pick].sum()
        boot[t] = 100.0 * (w_hy[pick].sum() - w_ar[pick].sum()) / tot
    boot_lo, boot_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Writer-level Wilcoxon signed-rank on per-writer exact-match effect.
    per_w = 100.0 * (w_hy / w_n - w_ar / w_n)
    try:
        wilcoxon_p = float(stats.wilcoxon(per_w).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")

    return {
        "split": split,
        "n_paired": n,
        "n_writers": int(nW),
        "ar_exact_pct": 100.0 * float(ar_c.mean()),
        "hyb_exact_pct": 100.0 * float(hy_c.mean()),
        "exact_effect_pp": 100.0 * float(hy_c.mean() - ar_c.mean()),
        "b_ar_correct_hyb_wrong": b,
        "c_hyb_correct_ar_wrong": c,
        "net_hyb_gain": c - b,
        "mcnemar_p_pooled": mcnemar_p,
        "fold_effect_pp": [round(float(x), 3) for x in fold_eff],
        "fold_mean_effect_pp": fold_mean_eff,
        "fold_ci_lo_pp": fold_lo,
        "fold_ci_hi_pp": fold_hi,
        "writer_boot_ci_lo_pp": boot_lo,
        "writer_boot_ci_hi_pp": boot_hi,
        "writer_wilcoxon_p": wilcoxon_p,
        "paired_ned_effect_pp": ned_mean,
        "paired_ned_ci_lo_pp": ned_lo,
        "paired_ned_ci_hi_pp": ned_hi,
        **severity,
    }


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing input CSV: {CSV} (run _build_xs_aggregated_csv.py first)")
    df = pd.read_csv(CSV, sep=";")
    df["label"] = df["label"].fillna("").astype(str)
    if "split" not in df.columns:
        raise SystemExit("input CSV lacks a 'split' column; rebuild with _build_xs_aggregated_csv.py")

    results = [analyse(df, split) for split in ("wi", "wd")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    # Printed summary.
    for r in results:
        tag = f"OnHW-{r['split'].upper()}"
        print(f"\n=== {tag}: HWRFormer vs hybrid HWRFormer (N={r['n_paired']}, "
              f"{r['n_writers']} writers) ===")
        print(f"  exact-match:   AR {r['ar_exact_pct']:.2f}%  hybrid {r['hyb_exact_pct']:.2f}%  "
              f"effect {r['exact_effect_pp']:+.2f} pp")
        print(f"  per-fold eff:  {r['fold_effect_pp']}  mean {r['fold_mean_effect_pp']:+.3f} pp  "
              f"fold-CI [{r['fold_ci_lo_pp']:+.2f}, {r['fold_ci_hi_pp']:+.2f}]")
        print(f"  writer boot:   95% CI [{r['writer_boot_ci_lo_pp']:+.2f}, "
              f"{r['writer_boot_ci_hi_pp']:+.2f}] pp   Wilcoxon p={r['writer_wilcoxon_p']:.3f}")
        print(f"  McNemar pooled: b={r['b_ar_correct_hyb_wrong']} c={r['c_hyb_correct_ar_wrong']} "
              f"net {r['net_hyb_gain']:+d}  p={r['mcnemar_p_pooled']:.4g}  (secondary)")
        print(f"  paired NED eff: {r['paired_ned_effect_pp']:+.3f} pp  "
              f"fold-CI [{r['paired_ned_ci_lo_pp']:+.2f}, {r['paired_ned_ci_hi_pp']:+.2f}]")
        print(f"  cond. severity (own errors): AR d={r['ar_cond_mean_d']:.2f}/"
              f"{r['ar_cond_mean_dnorm']:.4f}  hybrid d={r['hyb_cond_mean_d']:.2f}/"
              f"{r['hyb_cond_mean_dnorm']:.4f}")
    print(f"\nWrote {OUT_JSON}\n      {OUT_CSV}")


if __name__ == "__main__":
    main()
