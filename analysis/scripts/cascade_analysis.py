#!/usr/bin/env python3
"""Cascade analysis for noise injection on the HWRFormer (xs), all four datasets.

Reads the per-sample CSV produced by _build_xs_aggregated_csv_noise.py
and produces, separately for each of the four datasets (OnHW WI word,
OnHW WD word, private word, private sentence):

1. Per-position error rate as a function of word position k for AR-only
   and AR + noise injection.
2. Conditional cascade probability P(err_{k+1} | err_k), both pooled
   over k and as a function of k.
3. Paired McNemar exact binomial test on per-sample exact-match.

Outputs:
    thesis/figures/cascade_noise_injection.pdf     (2 rows x 4 cols)
    analysis/cascade_noise_injection_summary.csv   (4 rows, one per dataset)

Run from work/REWI_work:
    python analysis/scripts/cascade_analysis.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORK_DIR = Path(__file__).resolve().parent.parent.parent
THESIS_DIR = WORK_DIR.parent.parent / "thesis"
IN_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_noise_xs.csv"
OUT_FIG = THESIS_DIR / "figures" / "cascade_noise_injection.pdf"
OUT_SUMMARY = WORK_DIR / "analysis" / "cascade_noise_injection_summary.csv"

TASK_AR = "AR-only"
TASK_NOISE = "Noise (uniform p=0.15)"

AR_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word", "OnHW WI word"),
    ("onhw_wd_word", "OnHW WD word"),
    ("priv_word", "Priv.\\ word"),
    ("priv_sent", "Priv.\\ sent."),
]


def per_position_errors(pred: str, label: str) -> list[bool]:
    """Return a per-position error vector of length max(len(pred), len(label))."""
    n = max(len(pred), len(label))
    errs: list[bool] = []
    for k in range(n):
        if k >= len(pred) or k >= len(label):
            errs.append(True)
        else:
            errs.append(pred[k] != label[k])
    return errs


def aggregate_position_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_at_pos: dict[int, int] = defaultdict(int)
    err_at_pos: dict[int, int] = defaultdict(int)
    for _, row in df.iterrows():
        pred = "" if pd.isna(row["prediction"]) else str(row["prediction"])
        label = "" if pd.isna(row["label"]) else str(row["label"])
        errs = per_position_errors(pred, label)
        for k, e in enumerate(errs):
            n_at_pos[k] += 1
            if e:
                err_at_pos[k] += 1
    if not n_at_pos:
        return np.array([]), np.array([])
    k_max = max(n_at_pos.keys())
    positions = np.arange(k_max + 1)
    rates = np.array(
        [err_at_pos[k] / max(1, n_at_pos[k]) for k in positions], dtype=float
    )
    return positions, rates


def conditional_cascade(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    pooled_num = 0
    pooled_den = 0
    num_at_k: dict[int, int] = defaultdict(int)
    den_at_k: dict[int, int] = defaultdict(int)
    for _, row in df.iterrows():
        pred = "" if pd.isna(row["prediction"]) else str(row["prediction"])
        label = "" if pd.isna(row["label"]) else str(row["label"])
        errs = per_position_errors(pred, label)
        for k in range(len(errs) - 1):
            if errs[k]:
                den_at_k[k] += 1
                pooled_den += 1
                if errs[k + 1]:
                    num_at_k[k] += 1
                    pooled_num += 1
    pooled = pooled_num / pooled_den if pooled_den > 0 else float("nan")
    if not den_at_k:
        return pooled, np.array([])
    k_max = max(den_at_k.keys())
    rates = np.array(
        [
            (num_at_k[k] / den_at_k[k]) if den_at_k[k] > 0 else np.nan
            for k in range(k_max + 1)
        ],
        dtype=float,
    )
    return pooled, rates


def paired_mcnemar(ar_df: pd.DataFrame, noise_df: pd.DataFrame) -> dict[str, float]:
    ar_idx = ar_df.set_index(["fold", "sample_index"])["levenshtein_distance"]
    noise_idx = noise_df.set_index(["fold", "sample_index"])["levenshtein_distance"]
    joined = pd.concat({"ar": ar_idx, "noise": noise_idx}, axis=1).dropna()
    ar_correct = (joined["ar"] == 0).to_numpy()
    noise_correct = (joined["noise"] == 0).to_numpy()
    n_total = len(joined)
    b = int(np.sum(ar_correct & ~noise_correct))
    c = int(np.sum(~ar_correct & noise_correct))
    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(b, c), n_disc, p=0.5,
                                        alternative="two-sided").pvalue)
    return {
        "n_total": n_total,
        "ar_correct_noise_wrong": b,
        "ar_wrong_noise_correct": c,
        "n_discordant": n_disc,
        "p_value": p_value,
        "ar_exact_match_pct": 100.0 * float(np.mean(ar_correct)),
        "noise_exact_match_pct": 100.0 * float(np.mean(noise_correct)),
    }


def analyse_dataset(df_all: pd.DataFrame, dataset_task: str) -> dict:
    df_ds = df_all[df_all["dataset_task"] == dataset_task]
    ar_df = df_ds[df_ds["task"] == TASK_AR].copy()
    noise_df = df_ds[df_ds["task"] == TASK_NOISE].copy()

    ar_positions, ar_rates = aggregate_position_stats(ar_df)
    noise_positions, noise_rates = aggregate_position_stats(noise_df)
    ar_cascade_pooled, ar_cascade_k = conditional_cascade(ar_df)
    noise_cascade_pooled, noise_cascade_k = conditional_cascade(noise_df)
    mcnemar = paired_mcnemar(ar_df, noise_df)

    return {
        "dataset_task": dataset_task,
        "ar_positions": ar_positions,
        "ar_rates": ar_rates,
        "noise_positions": noise_positions,
        "noise_rates": noise_rates,
        "ar_cascade_k": ar_cascade_k,
        "noise_cascade_k": noise_cascade_k,
        "ar_cascade_pooled": ar_cascade_pooled,
        "noise_cascade_pooled": noise_cascade_pooled,
        "cascade_delta_pp": 100.0 * (ar_cascade_pooled - noise_cascade_pooled),
        **mcnemar,
    }


def plot_grid(results: list[dict], out_path: Path) -> None:
    n_ds = len(results)
    fig, axes = plt.subplots(2, n_ds, figsize=(3.6 * n_ds, 6.2), sharey="row")
    if n_ds == 1:
        axes = axes.reshape(2, 1)
    for col, (res, (_, label)) in enumerate(zip(results, DATASETS)):
        # Position cap: trim the tail where samples become sparse. Words rarely
        # exceed 12 characters on OnHW; sentences can be longer.
        is_sentence = "sent" in res["dataset_task"]
        k_cap_left = min(len(res["ar_rates"]), len(res["noise_rates"]),
                         40 if is_sentence else 15)
        ax_top = axes[0, col]
        ax_top.plot(
            res["ar_positions"][:k_cap_left], res["ar_rates"][:k_cap_left] * 100.0,
            marker="o", color=AR_COLOR, linewidth=2, markersize=4, label="AR-only",
        )
        ax_top.plot(
            res["noise_positions"][:k_cap_left], res["noise_rates"][:k_cap_left] * 100.0,
            marker="s", color=NOISE_COLOR, linewidth=2, markersize=4,
            label=r"AR + noise (uniform $p{=}0.15$)",
        )
        ax_top.set_title(label, fontsize=11)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel(r"Per-position error rate (\%)", fontsize=10.5)
        ax_top.set_xlabel(r"Character position $k$", fontsize=10)
        if col == n_ds - 1:
            ax_top.legend(loc="lower right", fontsize=8)

        k_cap_right = min(len(res["ar_cascade_k"]), len(res["noise_cascade_k"]),
                          40 if is_sentence else 12)
        ax_bot = axes[1, col]
        ax_bot.plot(
            np.arange(k_cap_right), res["ar_cascade_k"][:k_cap_right] * 100.0,
            marker="o", color=AR_COLOR, linewidth=2, markersize=4, label="AR-only",
        )
        ax_bot.plot(
            np.arange(k_cap_right), res["noise_cascade_k"][:k_cap_right] * 100.0,
            marker="s", color=NOISE_COLOR, linewidth=2, markersize=4,
            label=r"AR + noise (uniform $p{=}0.15$)",
        )
        ax_bot.grid(True, alpha=0.3)
        if col == 0:
            ax_bot.set_ylabel(
                r"$\Pr[\mathrm{err}_{k+1} \mid \mathrm{err}_{k}]$ (\%)",
                fontsize=10.5,
            )
        ax_bot.set_xlabel(r"Character position $k$", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def write_summary(results: list[dict], out_path: Path) -> None:
    fieldnames = [
        "dataset_task",
        "n_total",
        "ar_exact_match_pct",
        "noise_exact_match_pct",
        "ar_correct_noise_wrong",
        "ar_wrong_noise_correct",
        "n_discordant",
        "p_value",
        "ar_cascade_pooled",
        "noise_cascade_pooled",
        "cascade_delta_pp",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            w.writerow({k: res[k] for k in fieldnames})
    print(f"wrote summary: {out_path}")


def main() -> None:
    print(f"reading {IN_CSV}")
    df = pd.read_csv(IN_CSV, sep=";")
    available = set(df["dataset_task"].unique())
    print(f"datasets in CSV: {sorted(available)}")

    results: list[dict] = []
    for dataset_task, label in DATASETS:
        if dataset_task not in available:
            print(f"skipping {dataset_task}: not in CSV")
            continue
        res = analyse_dataset(df, dataset_task)
        results.append(res)
        print(
            f"  {label:<18s}  N={res['n_total']:>6d}"
            f"  AR={res['ar_exact_match_pct']:5.2f}%"
            f"  Noise={res['noise_exact_match_pct']:5.2f}%"
            f"  b={res['ar_correct_noise_wrong']:>5d}"
            f"  c={res['ar_wrong_noise_correct']:>5d}"
            f"  p={res['p_value']:.2e}"
            f"  cascade {res['ar_cascade_pooled']:.3f}->{res['noise_cascade_pooled']:.3f}"
        )

    if not results:
        print("no datasets available; nothing to plot")
        return

    write_summary(results, OUT_SUMMARY)
    plot_grid(results, OUT_FIG)


if __name__ == "__main__":
    main()
