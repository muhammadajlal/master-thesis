#!/usr/bin/env python3
"""Aligned-edit analysis for noise injection on the HWRFormer (xs), all four datasets.

Replaces the prior raw-index cascade analysis. For each sample, computes
the Levenshtein alignment of prediction against label and decomposes the
edit script into substitution, insertion, and deletion counts. A position
is counted as an error at reference index k iff the alignment assigns a
replace or delete operation to k (insertions do not count: the reference
character at k is still produced after the insertion). Reports:

  1. Aligned per-reference-position error rate per model and dataset.
  2. Paired change in length-normalized edit distance,
     Delta_tilde_e = e_noise - e_AR, per sample.
  3. Fold-level mean Delta_tilde_e and a paired t-interval over the 5 folds.
  4. Fractions improved / unchanged / worsened on raw integer edit_dist.
  5. Per-character-of-reference sub/ins/del rates for the noise model
     (AR-only decomposition is written as an appendix companion).
  6. Paired McNemar exact binomial test on per-sample exact-match (unchanged
     from the prior analysis to preserve table continuity).

Outputs:
    thesis/figures/cascade_noise_injection.pdf         (per-position err + Delta_e histogram)
    thesis/figures/cascade_noise_injection_ecdf.pdf    (per-position err + per-model eCDF)
    thesis/tables/cascade_noise_ar_decomp.tex          (HWRFormer sub/ins/del appendix)
    analysis/cascade_noise_injection_summary.csv       (one row per dataset)
    analysis/aligned_edit_per_sample.csv               (per-sample diagnostic)
    analysis/aligned_edit_position_curves.csv          (per-(model, dataset, pos) curves)

Run from work/REWI_work:
    python analysis/scripts/cascade_analysis.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import Levenshtein
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
OUT_FIG_ECDF = THESIS_DIR / "figures" / "cascade_noise_injection_ecdf.pdf"
OUT_SUMMARY = WORK_DIR / "analysis" / "cascade_noise_injection_summary.csv"
OUT_PER_SAMPLE = WORK_DIR / "analysis" / "aligned_edit_per_sample.csv"
OUT_POSITION_CURVES = WORK_DIR / "analysis" / "aligned_edit_position_curves.csv"
OUT_APPENDIX_TABLE = THESIS_DIR / "tables" / "cascade_noise_ar_decomp.tex"
OUT_APPENDIX_MAGNITUDE = THESIS_DIR / "tables" / "cascade_noise_magnitude.tex"

# CSV task labels (legacy; do not change without re-exporting the paired CSV)
TASK_AR = "AR-only"
TASK_NOISE = "Noise (uniform p=0.15)"

# Display labels used in figures and tables. Align with thesis terminology:
# the XS-architecture recognizer is HWRFormer, and the noise-injection
# variant is the same architecture trained with uniform input corruption.
LABEL_AR = "HWRFormer"
LABEL_NOISE = r"HWRFormer + noise (uniform $p{=}0.15$)"
LABEL_NOISE_SHORT = "HWRFormer + noise"

AR_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"

# Position cap for the aligned-error-rate panels: words rarely exceed 12 chars
# on OnHW; private sentences run longer.
POS_CAP_WORD = 15
POS_CAP_SENT = 40
MIN_SAMPLES_AT_POS = 50

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word", "OnHW WI word"),
    ("onhw_wd_word", "OnHW WD word"),
    ("priv_word", "Priv.\\ word"),
    ("priv_sent", "Priv.\\ sent."),
]


def _safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def aligned_edit_decomp(label: str, pred: str) -> dict:
    """Decompose Levenshtein alignment of (label -> pred) into op counts and
    the set of reference positions touched by a replace or delete op.

    Convention: Levenshtein.editops(source, dest) returns tuples (op, i, j)
    with op in {replace, delete, insert}, i indexing source, j indexing dest.
    Passing (label, pred):
        replace (i, j): label[i] is replaced by pred[j]            -> ref pos i counts
        delete  (i, j): label[i] is removed from the prediction    -> ref pos i counts
        insert  (i, j): pred[j] is inserted before label[i]        -> ref pos i is NOT
                                                                       errored (label[i]
                                                                       is still produced
                                                                       after the insert)
    """
    ops = Levenshtein.editops(label, pred)
    sub = ins = dele = 0
    err_ref = set()
    for op, i, _ in ops:
        if op == "replace":
            sub += 1
            err_ref.add(i)
        elif op == "delete":
            dele += 1
            err_ref.add(i)
        else:  # insert
            ins += 1
    first_err = min(err_ref) if err_ref else -1
    return {
        "sub": sub,
        "ins": ins,
        "del": dele,
        "edit_dist": sub + ins + dele,
        "first_err_ref_pos": first_err,
        "err_ref_positions": err_ref,
    }


def build_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose every (task, fold, sample) into aligned edit counts.

    Returns a per-sample DataFrame with columns suitable for the per_sample
    audit CSV and for downstream aggregation. The 'err_ref_positions' column
    holds the Python set object and is dropped before writing the CSV.
    """
    rows = []
    for row in df.itertuples(index=False):
        label = _safe_str(row.label)
        pred = _safe_str(row.prediction)
        decomp = aligned_edit_decomp(label, pred)
        # Continuity check: our decomposed edit distance must match the
        # precomputed levenshtein_distance column (catches editops convention
        # mismatch). nan in label/pred would inflate edit_dist, which is fine
        # as long as the input CSV is consistent.
        precomputed = int(row.levenshtein_distance)
        if decomp["edit_dist"] != precomputed:
            raise AssertionError(
                f"editops decomposition mismatch at "
                f"({row.dataset_task}, fold={row.fold}, sample={row.sample_index}, "
                f"task={row.task}): sub+ins+del={decomp['edit_dist']} vs "
                f"levenshtein_distance={precomputed}"
            )
        len_ref = len(label)
        rows.append({
            "dataset_task": row.dataset_task,
            "task": row.task,
            "fold": int(row.fold),
            "sample_index": int(row.sample_index),
            "len_ref": len_ref,
            "len_hyp": len(pred),
            "sub": decomp["sub"],
            "ins": decomp["ins"],
            "del": decomp["del"],
            "edit_dist": decomp["edit_dist"],
            "edit_dist_norm": decomp["edit_dist"] / max(1, len_ref),
            "exact_match": int(decomp["edit_dist"] == 0),
            "first_err_ref_pos": decomp["first_err_ref_pos"],
            "err_ref_positions": decomp["err_ref_positions"],
        })
    return pd.DataFrame(rows)


def position_curves(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Aggregate aligned per-reference-position error rate per (dataset, task).

    Denominator at position k: count of samples with len_ref > k.
    Numerator at k: count of samples whose err_ref_positions contains k.
    """
    rows = []
    grouped = per_sample.groupby(["dataset_task", "task"], sort=False)
    for (dataset_task, task), grp in grouped:
        n_at = defaultdict(int)
        n_err = defaultdict(int)
        for r in grp.itertuples(index=False):
            for k in range(r.len_ref):
                n_at[k] += 1
            for k in r.err_ref_positions:
                n_err[k] += 1
        if not n_at:
            continue
        k_max = max(n_at.keys())
        for k in range(k_max + 1):
            denom = n_at[k]
            if denom == 0:
                continue
            rows.append({
                "dataset_task": dataset_task,
                "task": task,
                "ref_pos": k,
                "n_samples_at_pos": denom,
                "n_err_at_pos": n_err[k],
                "err_rate": n_err[k] / denom,
            })
    return pd.DataFrame(rows)


def paired_join(per_sample: pd.DataFrame, dataset_task: str) -> pd.DataFrame:
    """Inner join AR vs noise per-sample rows on (fold, sample_index).

    Returns a DataFrame with columns suffixed _ar and _noise, including
    edit_dist_norm, edit_dist, len_ref. Used for the Delta_tilde_e
    statistics and the improved/unchanged/worsened triple.
    """
    ds = per_sample[per_sample["dataset_task"] == dataset_task]
    ar = ds[ds["task"] == TASK_AR].set_index(["fold", "sample_index"])
    no = ds[ds["task"] == TASK_NOISE].set_index(["fold", "sample_index"])
    keep = ["len_ref", "edit_dist", "edit_dist_norm", "sub", "ins", "del", "exact_match"]
    joined = ar[keep].join(no[keep], lsuffix="_ar", rsuffix="_noise", how="inner")
    return joined.reset_index()


def fold_t_interval(per_fold_means: list[float]) -> tuple[float, tuple[float, float]]:
    """Paired t-interval (95%) over 5 per-fold means.

    With df = n - 1 = 4, scipy.stats.t.interval gives the symmetric
    confidence interval around the grand mean using the sample standard
    deviation across folds. SEM = stdev / sqrt(n).
    """
    arr = np.asarray(per_fold_means, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, (float("nan"), float("nan"))
    sem = float(np.std(arr, ddof=1)) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return mean, (float(lo), float(hi))


def analyse_dataset(per_sample: pd.DataFrame, dataset_task: str) -> dict:
    paired = paired_join(per_sample, dataset_task)
    n_paired = len(paired)

    # Per-fold mean of length-normalized Delta_tilde_e
    paired["delta_e_norm"] = paired["edit_dist_norm_noise"] - paired["edit_dist_norm_ar"]
    fold_means = (
        paired.groupby("fold")["delta_e_norm"].mean().sort_index().tolist()
    )
    mean_delta, (ci_lo, ci_hi) = fold_t_interval(fold_means)

    # Improved / unchanged / worsened on raw integer edit_dist
    improved = int((paired["edit_dist_noise"] < paired["edit_dist_ar"]).sum())
    worsened = int((paired["edit_dist_noise"] > paired["edit_dist_ar"]).sum())
    unchanged = n_paired - improved - worsened

    # Per-character-of-reference rates for the noise model
    total_len_ref = int(paired["len_ref_noise"].sum())
    sub_per_ref_noise = (
        float(paired["sub_noise"].sum()) / max(1, total_len_ref)
    )
    ins_per_ref_noise = (
        float(paired["ins_noise"].sum()) / max(1, total_len_ref)
    )
    del_per_ref_noise = (
        float(paired["del_noise"].sum()) / max(1, total_len_ref)
    )
    sub_per_ref_ar = (
        float(paired["sub_ar"].sum()) / max(1, total_len_ref)
    )
    ins_per_ref_ar = (
        float(paired["ins_ar"].sum()) / max(1, total_len_ref)
    )
    del_per_ref_ar = (
        float(paired["del_ar"].sum()) / max(1, total_len_ref)
    )

    # Exact-match McNemar (unchanged formula)
    ar_correct = paired["exact_match_ar"].to_numpy().astype(bool)
    noise_correct = paired["exact_match_noise"].to_numpy().astype(bool)
    b = int(np.sum(ar_correct & ~noise_correct))
    c = int(np.sum(~ar_correct & noise_correct))
    n_disc = b + c
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(
            min(b, c), n_disc, p=0.5, alternative="two-sided"
        ).pvalue)

    # Mean absolute Delta_e_norm among the improved subset and the worsened
    # subset, supporting the "improvements are larger in magnitude" claim
    # in sec:cascade-noise closing prose.
    impr_mask = paired["edit_dist_noise"] < paired["edit_dist_ar"]
    wors_mask = paired["edit_dist_noise"] > paired["edit_dist_ar"]
    mean_abs_delta_improved = (
        float(paired.loc[impr_mask, "delta_e_norm"].abs().mean())
        if impr_mask.any() else float("nan")
    )
    mean_abs_delta_worsened = (
        float(paired.loc[wors_mask, "delta_e_norm"].abs().mean())
        if wors_mask.any() else float("nan")
    )

    return {
        "dataset_task": dataset_task,
        "n_total": n_paired,
        "ar_exact_match_pct": 100.0 * float(np.mean(ar_correct)),
        "noise_exact_match_pct": 100.0 * float(np.mean(noise_correct)),
        "ar_correct_noise_wrong": b,
        "ar_wrong_noise_correct": c,
        "n_discordant": n_disc,
        "p_value": p_value,
        "sub_per_ref_noise": sub_per_ref_noise,
        "ins_per_ref_noise": ins_per_ref_noise,
        "del_per_ref_noise": del_per_ref_noise,
        "sub_per_ref_ar": sub_per_ref_ar,
        "ins_per_ref_ar": ins_per_ref_ar,
        "del_per_ref_ar": del_per_ref_ar,
        # Per-character-of-reference deltas, noise minus HWRFormer, in pp.
        # Negative favours noise.
        "delta_sub_per_ref_pp": 100.0 * (sub_per_ref_noise - sub_per_ref_ar),
        "delta_ins_per_ref_pp": 100.0 * (ins_per_ref_noise - ins_per_ref_ar),
        "delta_del_per_ref_pp": 100.0 * (del_per_ref_noise - del_per_ref_ar),
        "delta_e_norm_mean_pp": 100.0 * mean_delta,
        "delta_e_norm_ci_lo_pp": 100.0 * ci_lo,
        "delta_e_norm_ci_hi_pp": 100.0 * ci_hi,
        "fold_mean_delta_e_norm": fold_means,
        "frac_improved_pct": 100.0 * improved / max(1, n_paired),
        "frac_unchanged_pct": 100.0 * unchanged / max(1, n_paired),
        "frac_worsened_pct": 100.0 * worsened / max(1, n_paired),
        "mean_abs_delta_e_norm_improved_pp": 100.0 * mean_abs_delta_improved,
        "mean_abs_delta_e_norm_worsened_pp": 100.0 * mean_abs_delta_worsened,
        "delta_e_norm_samples": paired["delta_e_norm"].to_numpy(),
    }


def plot_grid(results: list[dict], curves: pd.DataFrame, out_path: Path) -> None:
    n_ds = len(results)
    fig, axes = plt.subplots(2, n_ds, figsize=(3.6 * n_ds, 6.4),
                             sharey=False)
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for col, (res, (dataset_task, label)) in enumerate(zip(results, DATASETS)):
        is_sentence = "sent" in dataset_task
        pos_cap = POS_CAP_SENT if is_sentence else POS_CAP_WORD

        # ----- Top row: aligned per-reference-position error rate -----
        ax_top = axes[0, col]
        for task, color, marker, leg in [
            (TASK_AR, AR_COLOR, "o", LABEL_AR),
            (TASK_NOISE, NOISE_COLOR, "s", LABEL_NOISE),
        ]:
            sub = curves[
                (curves["dataset_task"] == dataset_task)
                & (curves["task"] == task)
                & (curves["ref_pos"] < pos_cap)
                & (curves["n_samples_at_pos"] >= MIN_SAMPLES_AT_POS)
            ].sort_values("ref_pos")
            ax_top.plot(
                sub["ref_pos"].to_numpy(),
                sub["err_rate"].to_numpy() * 100.0,
                marker=marker, color=color, linewidth=2, markersize=4, label=leg,
            )
        ax_top.set_title(label, fontsize=11)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel(
                r"Aligned per-ref-position error (\%)", fontsize=10.5
            )
        ax_top.set_xlabel(r"Reference position $k$", fontsize=10)
        if col == n_ds - 1:
            ax_top.legend(loc="lower right", fontsize=8)

        # ----- Bottom row: paired Delta_tilde_e histogram -----
        ax_bot = axes[1, col]
        deltas = res["delta_e_norm_samples"]
        deltas_clipped = np.clip(deltas, -1.0, 1.0)
        # Exact-zero mass plotted as a separate bar (centered at 0)
        zero_mask = deltas == 0.0
        nonzero = deltas_clipped[~zero_mask]

        bins = np.linspace(-1.0, 1.0, 41)  # 40 bins
        colors = [AR_COLOR if (bins[i] + bins[i + 1]) / 2 < 0 else NOISE_COLOR
                  for i in range(len(bins) - 1)]
        hist, edges = np.histogram(nonzero, bins=bins)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2
        ax_bot.bar(centers, hist, width=widths * 0.95, color=colors,
                   edgecolor="none", linewidth=0)
        # Unchanged mass as a separate bar slightly wider for visibility
        if int(zero_mask.sum()) > 0:
            ax_bot.bar(
                [0.0], [int(zero_mask.sum())], width=widths[0] * 1.05,
                color="0.4", edgecolor="black", linewidth=0.5,
                label=r"same $\tilde{e}$",
            )

        # Vertical line at the per-sample mean
        ax_bot.axvline(float(np.mean(deltas)), color="black",
                       linestyle="--", linewidth=1.2)
        # Horizontal whisker above the histogram showing the 95% t-CI of
        # the per-fold mean (the headline statistic in the table)
        ax_bot.set_yscale("symlog", linthresh=1)
        ymax_log = max(10, int(hist.max())) if hist.size else 10
        whisker_y = ymax_log * 1.8
        ci_lo = res["delta_e_norm_ci_lo_pp"] / 100.0
        ci_hi = res["delta_e_norm_ci_hi_pp"] / 100.0
        mean_delta = res["delta_e_norm_mean_pp"] / 100.0
        ax_bot.hlines(whisker_y, ci_lo, ci_hi, color="black", linewidth=1.8)
        ax_bot.plot([ci_lo, ci_hi], [whisker_y, whisker_y],
                    marker="|", color="black", linewidth=0,
                    markersize=8, markeredgewidth=1.6)
        ax_bot.plot([mean_delta], [whisker_y], marker="D", color="black",
                    markersize=4.5, linewidth=0)

        ax_bot.grid(True, alpha=0.3, axis="y")
        ax_bot.set_xlim(-1.02, 1.02)
        ax_bot.set_xlabel(
            r"$\Delta\tilde{e} = \tilde{e}_{\mathrm{noise}} - \tilde{e}_{\mathrm{AR}}$",
            fontsize=10,
        )
        if col == 0:
            ax_bot.set_ylabel(r"Samples (symlog)", fontsize=10.5)
        if col == n_ds - 1:
            from matplotlib.lines import Line2D
            ax_bot.legend(
                handles=[
                    Line2D([0], [0], color=AR_COLOR, lw=4,
                           label=r"noise wins ($\Delta\tilde{e}<0$)"),
                    Line2D([0], [0], color=NOISE_COLOR, lw=4,
                           label=r"HWRFormer wins ($\Delta\tilde{e}>0$)"),
                    Line2D([0], [0], color="0.4", lw=4,
                           label=r"same $\tilde{e}$ ($\Delta\tilde{e}=0$)"),
                    Line2D([0], [0], color="black", marker="D", lw=0,
                           label=r"mean + 95\% t-CI"),
                ],
                loc="upper right", fontsize=7, framealpha=0.92,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def plot_ecdf_grid(
    per_sample: pd.DataFrame,
    results: list[dict],
    curves: pd.DataFrame,
    out_path: Path,
) -> None:
    """Alternative bottom row: per-model empirical CDF of length-normalized
    edit distance e_tilde. Two curves per panel: HWRFormer and HWRFormer +
    noise. The crossing point (if any) shows where one model overtakes the
    other in the cumulative sense. Per-model mean is overlaid as a dashed
    vertical line; the paired t-CI of the per-fold mean Delta_e_norm from
    the table is inset as a text box. The top row is unchanged from
    plot_grid (aligned per-reference-position error rate). Legend is
    figure-level at the bottom so it does not overlap per-panel insets."""
    n_ds = len(results)
    fig, axes = plt.subplots(2, n_ds, figsize=(3.6 * n_ds, 6.4), sharey=False)
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for col, (res, (dataset_task, label)) in enumerate(zip(results, DATASETS)):
        is_sentence = "sent" in dataset_task
        pos_cap = POS_CAP_SENT if is_sentence else POS_CAP_WORD

        # ----- Top row: aligned per-reference-position error rate (same as histogram fig) -----
        ax_top = axes[0, col]
        for task, color, marker, leg in [
            (TASK_AR, AR_COLOR, "o", LABEL_AR),
            (TASK_NOISE, NOISE_COLOR, "s", LABEL_NOISE),
        ]:
            sub = curves[
                (curves["dataset_task"] == dataset_task)
                & (curves["task"] == task)
                & (curves["ref_pos"] < pos_cap)
                & (curves["n_samples_at_pos"] >= MIN_SAMPLES_AT_POS)
            ].sort_values("ref_pos")
            ax_top.plot(
                sub["ref_pos"].to_numpy(),
                sub["err_rate"].to_numpy() * 100.0,
                marker=marker, color=color, linewidth=2, markersize=4, label=leg,
            )
        ax_top.set_title(label, fontsize=11)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel(
                r"Aligned per-ref-position error (\%)", fontsize=10.5
            )
        ax_top.set_xlabel(r"Reference position $k$", fontsize=10)
        if col == n_ds - 1:
            ax_top.legend(loc="lower right", fontsize=8)

        # ----- Bottom row: per-model eCDF of e_tilde (HWRFormer vs HWRFormer+noise) -----
        ax_bot = axes[1, col]
        ds_samples = per_sample[per_sample["dataset_task"] == dataset_task]
        for task, color, leg in [
            (TASK_AR, AR_COLOR, LABEL_AR),
            (TASK_NOISE, NOISE_COLOR, LABEL_NOISE_SHORT),
        ]:
            evals = ds_samples[ds_samples["task"] == task]["edit_dist_norm"].to_numpy()
            if evals.size == 0:
                continue
            e_sorted = np.sort(evals)
            yvals = np.arange(1, e_sorted.size + 1) / e_sorted.size
            ax_bot.plot(e_sorted, yvals, color=color, linewidth=2.2, label=leg)
            mean_e = float(np.mean(evals))
            ax_bot.axvline(mean_e, color=color, linestyle="--",
                           linewidth=1.0, alpha=0.7)

        ax_bot.set_xlim(0.0, 1.5)
        ax_bot.set_ylim(0.0, 1.02)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.set_xlabel(
            r"$\tilde{e} = d(\hat{y}, y) / \max(1, |y|)$", fontsize=10,
        )
        if col == 0:
            ax_bot.set_ylabel(r"$P(\tilde{e}_i \leq x)$", fontsize=10.5)

        # Inset (bottom-left: free space below the exact-match plateau the
        # curves jump to at x=0; avoids the top-left where the curves sit).
        text = (
            r"$\bar{\Delta\tilde{e}} = "
            f"{res['delta_e_norm_mean_pp']:+.2f}$ pp"
            "\n95\\% CI = ["
            f"{res['delta_e_norm_ci_lo_pp']:+.2f}, "
            f"{res['delta_e_norm_ci_hi_pp']:+.2f}"
            "] pp"
        )
        ax_bot.text(
            0.97, 0.03, text, transform=ax_bot.transAxes,
            ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.7", alpha=0.92),
        )

    # Figure-level legend at the very bottom, outside any panel, to avoid
    # overlapping the per-panel inset on the rightmost column.
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], color=AR_COLOR, lw=2.2, label=LABEL_AR),
            Line2D([0], [0], color=NOISE_COLOR, lw=2.2,
                   label=LABEL_NOISE_SHORT),
            Line2D([0], [0], color="0.4", linestyle="--", lw=1.0,
                   label=r"per-model mean $\tilde{e}$"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3,
        fontsize=9, frameon=False, columnspacing=2.0, handlelength=2.4,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_path}")


def write_summary(results: list[dict], out_path: Path) -> None:
    fieldnames = [
        "dataset_task",
        "n_total",
        "ar_exact_match_pct", "noise_exact_match_pct",
        "ar_correct_noise_wrong", "ar_wrong_noise_correct",
        "n_discordant", "p_value",
        "sub_per_ref_noise", "ins_per_ref_noise", "del_per_ref_noise",
        "sub_per_ref_ar", "ins_per_ref_ar", "del_per_ref_ar",
        "delta_sub_per_ref_pp", "delta_ins_per_ref_pp", "delta_del_per_ref_pp",
        "delta_e_norm_mean_pp",
        "delta_e_norm_ci_lo_pp", "delta_e_norm_ci_hi_pp",
        "frac_improved_pct", "frac_unchanged_pct", "frac_worsened_pct",
        "mean_abs_delta_e_norm_improved_pp", "mean_abs_delta_e_norm_worsened_pp",
        "fold_mean_delta_e_norm",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for res in results:
            row = {k: res[k] for k in fieldnames if k != "fold_mean_delta_e_norm"}
            row["fold_mean_delta_e_norm"] = ";".join(
                f"{x:.6f}" for x in res["fold_mean_delta_e_norm"]
            )
            w.writerow(row)
    print(f"wrote summary: {out_path}")


def write_per_sample(per_sample: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset_task", "task", "fold", "sample_index",
        "len_ref", "len_hyp",
        "sub", "ins", "del", "edit_dist", "edit_dist_norm",
        "exact_match", "first_err_ref_pos",
    ]
    per_sample[cols].to_csv(out_path, index=False)
    print(f"wrote per-sample: {out_path}")


def write_position_curves(curves: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_path, index=False)
    print(f"wrote position curves: {out_path}")


def write_appendix_ar_decomp(results: list[dict], out_path: Path) -> None:
    """Auto-generate the appendix companion to tab:cascade-noise-mcnemar.

    The main table reports DELTA sub/ins/del (noise minus HWRFormer). This
    appendix table reports the two absolute decompositions side by side so
    a reader can verify the deltas and read the underlying per-model rates.
    """
    lines = []
    lines.append("% Auto-generated by analysis/scripts/cascade_analysis.py")
    lines.append("% Side-by-side HWRFormer and HWRFormer + noise decompositions.")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-character-of-reference sub/ins/del rates for "
        r"HWRFormer (baseline training, no noise injection) and HWRFormer + "
        r"noise injection side by side, reported as the companion to "
        r"\cref{tab:cascade-noise-mcnemar}. Rates are pooled over the five "
        r"folds and the same paired sample set used in the main table. The "
        r"main table reports the corresponding deltas (noise minus "
        r"HWRFormer) in percentage points.}"
    )
    lines.append(r"\label{tab:cascade-noise-ar-decomp}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lSSSSSS}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{Dataset} "
        r"& \multicolumn{3}{c}{HWRFormer (\%)} "
        r"& \multicolumn{3}{c}{HWRFormer + noise (\%)} \\"
    )
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(
        r"& {sub/ref} & {ins/ref} & {del/ref} "
        r"& {sub/ref} & {ins/ref} & {del/ref} \\"
    )
    lines.append(r"\midrule")
    label_map = dict(DATASETS)
    for res in results:
        ds_label = label_map.get(res["dataset_task"], res["dataset_task"])
        lines.append(
            f"{ds_label} & "
            f"{100.0 * res['sub_per_ref_ar']:.2f} & "
            f"{100.0 * res['ins_per_ref_ar']:.2f} & "
            f"{100.0 * res['del_per_ref_ar']:.2f} & "
            f"{100.0 * res['sub_per_ref_noise']:.2f} & "
            f"{100.0 * res['ins_per_ref_noise']:.2f} & "
            f"{100.0 * res['del_per_ref_noise']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote appendix table: {out_path}")


def write_appendix_magnitude(results: list[dict], out_path: Path) -> None:
    """Per-improvement-subset magnitude breakdown that supports the
    'improvements are larger in magnitude' claim. Reports mean absolute
    Delta_e_norm (pp) in the improved and worsened subsets per dataset,
    plus the magnitude ratio (improved over worsened). Companion to
    \\cref{tab:cascade-noise-mcnemar}, moved out of the main table to keep
    that table at ten columns."""
    lines = []
    lines.append("% Auto-generated by analysis/scripts/cascade_analysis.py")
    lines.append("% Magnitude breakdown of paired Delta_e_norm by improvement subset.")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Mean absolute paired length-normalized edit-distance "
        r"change within the improved and worsened subsets, in percentage "
        r"points, alongside the magnitude ratio impr.\,/\,wors. Companion "
        r"to \cref{tab:cascade-noise-mcnemar}. On every dataset the "
        r"improvements are larger in magnitude than the degradations, "
        r"which is what allows the per-fold mean $\bar{\Delta\tilde{e}}$ "
        r"to remain negative on private sentences even though more samples "
        r"worsen than improve.}"
    )
    lines.append(r"\label{tab:cascade-noise-magnitude}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lSSS}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & {$|\Delta\tilde{e}|$ impr.\ (pp)} "
        r"& {$|\Delta\tilde{e}|$ wors.\ (pp)} & {impr.\,/\,wors.} \\"
    )
    lines.append(r"\midrule")
    label_map = dict(DATASETS)
    for res in results:
        ds_label = label_map.get(res["dataset_task"], res["dataset_task"])
        impr = res["mean_abs_delta_e_norm_improved_pp"]
        wors = res["mean_abs_delta_e_norm_worsened_pp"]
        ratio = impr / wors if wors > 0 else float("nan")
        lines.append(
            f"{ds_label} & "
            f"{impr:.2f} & "
            f"{wors:.2f} & "
            f"{ratio:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote appendix magnitude table: {out_path}")


def main() -> None:
    print(f"reading {IN_CSV}")
    df = pd.read_csv(IN_CSV, sep=";")
    available = set(df["dataset_task"].unique())
    print(f"datasets in CSV: {sorted(available)}")

    print("decomposing edits...")
    per_sample = build_per_sample(df)
    print(f"  per_sample rows: {len(per_sample)}")

    print("aggregating position curves...")
    curves = position_curves(per_sample)

    results: list[dict] = []
    for dataset_task, label in DATASETS:
        if dataset_task not in available:
            print(f"skipping {dataset_task}: not in CSV")
            continue
        res = analyse_dataset(per_sample, dataset_task)
        results.append(res)
        print(
            f"  {label:<18s}  N={res['n_total']:>6d}"
            f"  AR={res['ar_exact_match_pct']:5.2f}%"
            f"  Noise={res['noise_exact_match_pct']:5.2f}%"
            f"  b={res['ar_correct_noise_wrong']:>5d}"
            f"  c={res['ar_wrong_noise_correct']:>5d}"
            f"  p={res['p_value']:.2e}"
            f"  Delta_e_norm={res['delta_e_norm_mean_pp']:+.3f}pp"
            f"  CI=[{res['delta_e_norm_ci_lo_pp']:+.3f},"
            f" {res['delta_e_norm_ci_hi_pp']:+.3f}]"
            f"  improved={res['frac_improved_pct']:5.2f}%"
            f"  worsened={res['frac_worsened_pct']:5.2f}%"
        )

    if not results:
        print("no datasets available; nothing to plot")
        return

    write_per_sample(per_sample, OUT_PER_SAMPLE)
    write_position_curves(curves, OUT_POSITION_CURVES)
    write_summary(results, OUT_SUMMARY)
    write_appendix_ar_decomp(results, OUT_APPENDIX_TABLE)
    write_appendix_magnitude(results, OUT_APPENDIX_MAGNITUDE)
    plot_grid(results, curves, OUT_FIG)
    plot_ecdf_grid(per_sample, results, curves, OUT_FIG_ECDF)


if __name__ == "__main__":
    main()
