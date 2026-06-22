#!/usr/bin/env python3
"""Cascade analysis for noise injection on the HWRFormer (xs).

Reads the per-sample CSV produced by _build_xs_aggregated_csv_noise.py
and produces:

1. Per-position error rate as a function of word position k for AR-only
   and AR + noise injection.
2. Conditional cascade probability P(err_{k+1} | err_k) for the two
   models, both as a scalar (pooled over k) and as a function of k.
3. Paired McNemar exact binomial test on per-sample exact-match.

Outputs:
    thesis/figures/cascade_noise_injection.pdf      (two-panel figure)
    analysis/cascade_noise_injection_summary.csv    (one-row summary)

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


def per_position_errors(pred: str, label: str) -> list[bool]:
    """Return a per-position error vector of length max(len(pred), len(label)).

    Positions inside both strings compare character by character.
    Positions inside only one string are counted as errors (extra
    prediction or missed character).
    """
    n = max(len(pred), len(label))
    errs: list[bool] = []
    for k in range(n):
        if k >= len(pred) or k >= len(label):
            errs.append(True)
        else:
            errs.append(pred[k] != label[k])
    return errs


def aggregate_position_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-position error counts across samples.

    Returns (position_axis, error_rate_per_position) up to a length
    cap that covers all observed positions.
    """
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
    """Compute P(err_{k+1} | err_k) pooled over k and as a function of k.

    Returns (pooled_probability, per_position_conditional_rates).
    """
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
    """Paired McNemar exact binomial test on per-sample exact-match correctness.

    The two dataframes must share the (fold, sample_index) keys.
    """
    ar_idx = ar_df.set_index(["fold", "sample_index"]) ["levenshtein_distance"]
    noise_idx = noise_df.set_index(["fold", "sample_index"]) ["levenshtein_distance"]
    joined = pd.concat({"ar": ar_idx, "noise": noise_idx}, axis=1).dropna()
    ar_correct = (joined["ar"] == 0).to_numpy()
    noise_correct = (joined["noise"] == 0).to_numpy()
    n_total = len(joined)
    b = int(np.sum(ar_correct & ~noise_correct))  # AR correct, Noise wrong
    c = int(np.sum(~ar_correct & noise_correct))  # AR wrong, Noise correct
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


def plot_results(
    ar_positions: np.ndarray,
    ar_rates: np.ndarray,
    noise_positions: np.ndarray,
    noise_rates: np.ndarray,
    ar_cascade_k: np.ndarray,
    noise_cascade_k: np.ndarray,
    summary: dict[str, float],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    k_cap = min(
        len(ar_rates), len(noise_rates),
        max(int(np.sum(np.array([n >= 50 for n in [10] * 20]))), 12),
    )
    # Truncate the position axis to a meaningful range. OnHW words rarely
    # exceed ~12 characters, so positions beyond k=12 are sparse.
    k_cap_left = min(len(ar_rates), len(noise_rates), 15)
    ax1.plot(
        ar_positions[:k_cap_left], ar_rates[:k_cap_left] * 100.0,
        marker="o", color=AR_COLOR, linewidth=2, label="AR-only",
    )
    ax1.plot(
        noise_positions[:k_cap_left], noise_rates[:k_cap_left] * 100.0,
        marker="s", color=NOISE_COLOR, linewidth=2, label="AR + noise (uniform $p{=}0.15$)",
    )
    ax1.set_xlabel("Character position $k$ (0-indexed)", fontsize=11)
    ax1.set_ylabel("Per-position error rate (\\%)", fontsize=11)
    ax1.set_title("Per-position error rate", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)

    k_cap_right = min(len(ar_cascade_k), len(noise_cascade_k), 12)
    ax2.plot(
        np.arange(k_cap_right), ar_cascade_k[:k_cap_right] * 100.0,
        marker="o", color=AR_COLOR, linewidth=2, label="AR-only",
    )
    ax2.plot(
        np.arange(k_cap_right), noise_cascade_k[:k_cap_right] * 100.0,
        marker="s", color=NOISE_COLOR, linewidth=2, label="AR + noise (uniform $p{=}0.15$)",
    )
    ax2.set_xlabel("Character position $k$ (0-indexed)", fontsize=11)
    ax2.set_ylabel(r"$\Pr[\mathrm{err}_{k+1} \mid \mathrm{err}_{k}]$ (\\%)", fontsize=11)
    ax2.set_title("Conditional cascade probability", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main() -> None:
    print(f"reading {IN_CSV}")
    df = pd.read_csv(IN_CSV, sep=";")
    ar_df = df[df["task"] == TASK_AR].copy()
    noise_df = df[df["task"] == TASK_NOISE].copy()
    print(f"AR-only rows: {len(ar_df)}  Noise rows: {len(noise_df)}")

    ar_positions, ar_rates = aggregate_position_stats(ar_df)
    noise_positions, noise_rates = aggregate_position_stats(noise_df)
    ar_cascade_pooled, ar_cascade_k = conditional_cascade(ar_df)
    noise_cascade_pooled, noise_cascade_k = conditional_cascade(noise_df)
    mcnemar = paired_mcnemar(ar_df, noise_df)

    summary = {
        "n_total": mcnemar["n_total"],
        "ar_exact_match_pct": mcnemar["ar_exact_match_pct"],
        "noise_exact_match_pct": mcnemar["noise_exact_match_pct"],
        "ar_correct_noise_wrong": mcnemar["ar_correct_noise_wrong"],
        "ar_wrong_noise_correct": mcnemar["ar_wrong_noise_correct"],
        "n_discordant": mcnemar["n_discordant"],
        "mcnemar_p_value": mcnemar["p_value"],
        "ar_cascade_pooled": ar_cascade_pooled,
        "noise_cascade_pooled": noise_cascade_pooled,
        "cascade_delta_pp": 100.0 * (ar_cascade_pooled - noise_cascade_pooled),
    }

    print("=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")
        else:
            print(f"  {k:30s} = {v}")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"wrote summary: {OUT_SUMMARY}")

    plot_results(
        ar_positions, ar_rates,
        noise_positions, noise_rates,
        ar_cascade_k, noise_cascade_k,
        summary, OUT_FIG,
    )


if __name__ == "__main__":
    main()
