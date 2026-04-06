#!/usr/bin/env python3
"""
Dual-dataset analysis for the thesis: OnHW (AR-only vs Hybrid) and Stabilo (word + sentence).

Generates all missing figures and tables:
  1. Levenshtein distance distribution (histogram + CDF)
  2. Normalized LD distribution
  3. Per-character error rate bar charts
  4. Length dependence (Pearson correlations)
  5. Collision analysis (label-frequency, writer-stratified)
  6. Representative error examples (p50/p90/p99)
  7. Fold-wise summary tables
  8. Overall summary tables

All outputs saved to thesis/figures/ and printed as LaTeX tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ============================================================
# Paths
# ============================================================
WORK_DIR = Path(__file__).resolve().parent.parent.parent  # REWI_work
THESIS_DIR = WORK_DIR.parent.parent / "thesis"
FIG_DIR = THESIS_DIR / "figures"
TABLE_DIR = WORK_DIR / "analysis" / "tables" / "thesis_dual_dataset"

# Input CSVs
ONHW_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_hybrid.csv"
STABILO_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_new.csv"

# OnHW validation exports for collision analysis
ONHW_AR_EXPORT_PAT = str(WORK_DIR.parent.parent / "results/hwr2/Baseline-AR/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0.json")
ONHW_HYB_EXPORT_PAT = str(WORK_DIR.parent.parent / "results/hwr2/Baseline-Hybrid/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0_ar.json")


# ============================================================
# Helpers
# ============================================================
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def load_csv(path: Path, sep: str = ";") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # Normalize column names
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)
    # Ensure strings
    df["prediction"] = df["prediction"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)
    # Compute derived columns
    df["label_chars"] = df["label"].str.len()
    df["lev_norm"] = df["levenshtein_distance"] / df["label_chars"].clip(lower=1)
    df["exact"] = (df["levenshtein_distance"] == 0).astype(int)
    return df


def load_export_json(path: str) -> Tuple[List[str], List[str]]:
    with open(path) as f:
        d = json.load(f)
    preds = [str(p) if p is not None else "" for p in d["predictions"]]
    labels = [str(l) if l is not None else "" for l in d["labels"]]
    return preds, labels


# ============================================================
# 1. Summary Tables
# ============================================================
def compute_summary(df: pd.DataFrame, group_col: str = "task") -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col):
        n = len(g)
        exact_pct = 100.0 * g["exact"].mean()
        mean_chars = g["label_chars"].mean()

        lev_sum = g["levenshtein_distance"].sum()
        char_sum = g["label_chars"].sum()
        micro = lev_sum / max(1, char_sum)
        macro = g["lev_norm"].mean()

        vals = g["lev_norm"].to_numpy()
        rows.append({
            "Dataset": name,
            "N": n,
            "Exact (\\%)": f"{exact_pct:.2f}",
            r"$|\bar{y}|$": f"{mean_chars:.1f}",
            "Micro": f"{micro:.4f}",
            "Macro": f"{macro:.4f}",
            "p50": f"{np.quantile(vals, 0.5):.2f}",
            "p90": f"{np.quantile(vals, 0.9):.2f}",
            "p95": f"{np.quantile(vals, 0.95):.2f}",
            "p99": f"{np.quantile(vals, 0.99):.2f}",
            r"P($e>0.5$) (\%)": f"{100.0 * (vals > 0.5).mean():.2f}",
            r"P($e>1$) (\%)": f"{100.0 * (vals > 1.0).mean():.2f}",
        })
    return pd.DataFrame(rows)


def compute_foldwise(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for fold, g in df.groupby("fold"):
        n = len(g)
        exact_pct = 100.0 * g["exact"].mean()
        lev_sum = g["levenshtein_distance"].sum()
        char_sum = g["label_chars"].sum()
        micro = lev_sum / max(1, char_sum)
        vals = g["lev_norm"].to_numpy()
        rows.append({
            "Dataset": dataset_name,
            "Fold": int(fold),
            "N": n,
            "Exact (\\%)": f"{exact_pct:.2f}",
            "Micro": f"{micro:.4f}",
            "p90": f"{np.quantile(vals, 0.9):.2f}",
            "p99": f"{np.quantile(vals, 0.99):.2f}",
            r"P($e>0.5$) (\%)": f"{100.0 * (vals > 0.5).mean():.2f}",
            r"P($e>1$) (\%)": f"{100.0 * (vals > 1.0).mean():.2f}",
        })
    return pd.DataFrame(rows)


# ============================================================
# 2. LD Distribution Figures
# ============================================================
def plot_ld_distribution(
    datasets: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
    normalized: bool = False,
):
    """Plot histogram + CDF for multiple datasets on same axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    colors = ["C0", "C1", "C2", "C3"]
    x_label = "Normalized Levenshtein Distance ($d / |y|$)" if normalized else "Levenshtein Distance"

    all_max = max(float(v.max()) for v in datasets.values())
    if normalized:
        bins = np.linspace(0.0, min(all_max, 2.0), 50)
    else:
        bins = np.arange(0, min(int(all_max) + 2, 20)) - 0.5

    for i, (name, dists) in enumerate(datasets.items()):
        c = colors[i % len(colors)]
        ax1.hist(dists, bins=bins, alpha=0.5, label=f"{name} (mean={np.mean(dists):.3f})",
                 density=True, color=c)
        sorted_d = np.sort(dists)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, label=name, color=c)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title} — Histogram")
    ax1.legend(fontsize=9)

    if normalized:
        ax2.set_xlim(-0.05, 2.0)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Cumulative Proportion")
    ax2.set_title(f"{title} — CDF")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 3. Per-Character Error Rate
# ============================================================
def compute_per_char_error_simple(preds: List[str], labels: List[str]) -> Dict[str, Dict]:
    """Compute per-character error rates without jiwer dependency."""
    try:
        import jiwer
        stats = {}
        cats = sorted(set("".join(labels)))
        cats = [c for c in cats if c.strip()]
        for c in cats:
            stats[c] = {"total": 0, "correct": 0, "substituted": 0, "deleted": 0}

        out = jiwer.process_characters(labels, preds)
        for alignment, ref_str, hyp_str in zip(out.alignments, out.references, out.hypotheses):
            for event in alignment:
                if event.type == "equal":
                    for i in range(event.ref_start_idx, event.ref_end_idx):
                        c = ref_str[i]
                        if c in stats:
                            stats[c]["total"] += 1
                            stats[c]["correct"] += 1
                elif event.type == "substitute":
                    for i in range(event.ref_start_idx, event.ref_end_idx):
                        c = ref_str[i]
                        if c in stats:
                            stats[c]["total"] += 1
                            stats[c]["substituted"] += 1
                elif event.type == "delete":
                    for i in range(event.ref_start_idx, event.ref_end_idx):
                        c = ref_str[i]
                        if c in stats:
                            stats[c]["total"] += 1
                            stats[c]["deleted"] += 1

        for c, s in stats.items():
            tot = max(s["total"], 1)
            s["error_rate"] = (s["substituted"] + s["deleted"]) / tot
        return stats

    except ImportError:
        # Fallback: use alignment-free character counting
        print("  Warning: jiwer not available, using fallback per-char error")
        char_counts = {}
        char_errors = {}
        for pred, label in zip(preds, labels):
            for c in label:
                char_counts[c] = char_counts.get(c, 0) + 1
            d = levenshtein(pred, label)
            if d > 0:
                # Distribute errors proportionally across label chars
                for c in set(label):
                    n_c = label.count(c)
                    char_errors[c] = char_errors.get(c, 0) + d * n_c / max(1, len(label))
        stats = {}
        for c in sorted(char_counts.keys()):
            if c.strip():
                stats[c] = {"total": char_counts[c],
                            "error_rate": char_errors.get(c, 0) / max(1, char_counts[c])}
        return stats


def plot_per_char_error(
    datasets: Dict[str, Dict[str, Dict]],
    save_path: Path,
    title: str,
):
    """Bar chart of per-character error rates for multiple datasets."""
    all_chars = sorted(set().union(*(set(s.keys()) for s in datasets.values())))
    all_chars = [c for c in all_chars if c.strip()]

    n_datasets = len(datasets)
    x = np.arange(len(all_chars))
    w = 0.8 / n_datasets

    fig, ax = plt.subplots(figsize=(max(14, len(all_chars) * 0.45), 6), dpi=150)
    colors = ["C0", "C1", "C2", "C3"]

    for i, (name, stats) in enumerate(datasets.items()):
        er = [stats.get(c, {}).get("error_rate", 0) for c in all_chars]
        offset = (i - (n_datasets - 1) / 2) * w
        ax.bar(x + offset, er, w, label=name, alpha=0.8, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(all_chars, fontsize=7)
    ax.set_ylabel("Error Rate (sub + del)")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 4. Length Dependence
# ============================================================
def compute_length_dependence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("task"):
        y_len = g["label_chars"].to_numpy().astype(float)
        d = g["levenshtein_distance"].to_numpy().astype(float)
        e = g["lev_norm"].to_numpy()

        rho_d_all, _ = scipy_stats.pearsonr(d, y_len) if len(d) > 2 else (np.nan, np.nan)
        rho_e_all, _ = scipy_stats.pearsonr(e, y_len) if len(e) > 2 else (np.nan, np.nan)

        mask = d > 0
        if mask.sum() > 2:
            rho_d_err, _ = scipy_stats.pearsonr(d[mask], y_len[mask])
            rho_e_err, _ = scipy_stats.pearsonr(e[mask], y_len[mask])
        else:
            rho_d_err, rho_e_err = np.nan, np.nan

        rows.append({
            "Dataset": name,
            r"$\rho(d, |y|)$ (All)": f"{rho_d_all:.3f}",
            r"$\rho(e, |y|)$ (All)": f"{rho_e_all:.3f}",
            r"$\rho(d, |y|)$ ($e>0$)": f"{rho_d_err:.3f}",
            r"$\rho(e, |y|)$ ($e>0$)": f"{rho_e_err:.3f}",
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. Collision Analysis
# ============================================================
def compute_collisions(df: pd.DataFrame, task_name: str) -> Dict:
    """Find collisions: incorrect predictions matching other ground-truth labels."""
    errors = df[df["levenshtein_distance"] > 0].copy()
    gt_set = set(df["label"].unique())

    # Find collisions
    errors["is_collision"] = errors["prediction"].isin(gt_set)
    collisions = errors[errors["is_collision"]].copy()

    n_errors = len(errors)
    n_collisions = len(collisions)

    # Label frequency of collided predictions
    gt_counts = df["label"].value_counts().to_dict()
    collisions["gt_label_count"] = collisions["prediction"].map(gt_counts).fillna(0).astype(int)

    # Summary stats
    uid_weighted = collisions.drop_duplicates(subset=["fold", "sample_index"])
    gt_freq = uid_weighted["gt_label_count"]

    result = {
        "task": task_name,
        "n_total": len(df),
        "n_errors": n_errors,
        "n_collisions_uid": len(uid_weighted),
        "n_collisions_event": n_collisions,
        "collision_rate": n_collisions / max(1, n_errors),
    }

    if len(gt_freq) > 0:
        result.update({
            "gt_freq_median": float(gt_freq.median()),
            "gt_freq_p90": float(gt_freq.quantile(0.9)),
            "gt_freq_p95": float(gt_freq.quantile(0.95)),
        })

    # Top collided predictions
    top_collided = uid_weighted["prediction"].value_counts().head(10)
    result["top_collided"] = top_collided.to_dict()

    return result


# ============================================================
# 6. Representative Examples
# ============================================================
def find_representative_examples(df: pd.DataFrame, task_name: str, n_per_quantile: int = 2) -> pd.DataFrame:
    """Find examples nearest to p50, p90, p99 of nonzero error distribution."""
    errors = df[df["levenshtein_distance"] > 0].copy()
    if len(errors) == 0:
        return pd.DataFrame()

    e = errors["lev_norm"].to_numpy()
    quantiles = {"p50": np.quantile(e, 0.5), "p90": np.quantile(e, 0.9), "p99": np.quantile(e, 0.99)}

    rows = []
    for qname, qval in quantiles.items():
        errors["dist_to_q"] = (errors["lev_norm"] - qval).abs()
        nearest = errors.nsmallest(n_per_quantile, "dist_to_q")
        for _, row in nearest.iterrows():
            rows.append({
                "Dataset": task_name,
                "Quantile": qname,
                "Fold": int(row["fold"]),
                "Idx": int(row["sample_index"]),
                "d": int(row["levenshtein_distance"]),
                "e": round(row["lev_norm"], 2),
                "Prediction": row["prediction"],
                "Ground Truth": row["label"],
            })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading OnHW data...")
    onhw = load_csv(ONHW_CSV)
    onhw_ar = onhw[onhw["task"] == "AR-only"].copy()
    onhw_hyb = onhw[onhw["task"] == "Hybrid (AR Decoding)"].copy()

    print("Loading Stabilo data...")
    stabilo = load_csv(STABILO_CSV)
    stabilo_word = stabilo[stabilo["task"] == "word"].copy()
    stabilo_sent = stabilo[stabilo["task"] == "sent"].copy()

    print(f"  OnHW AR-only: {len(onhw_ar)}, Hybrid: {len(onhw_hyb)}")
    print(f"  Stabilo Word: {len(stabilo_word)}, Sent: {len(stabilo_sent)}")

    # ------------------------------------------------------------------
    # 1. Overall Summary Tables
    # ------------------------------------------------------------------
    print("\n=== Overall Summary Tables ===")

    # OnHW: AR-only vs Hybrid
    onhw_summary = compute_summary(onhw, group_col="task")
    print("\nOnHW Summary:")
    print(onhw_summary.to_string(index=False))
    onhw_summary.to_csv(TABLE_DIR / "onhw_overall_summary.csv", index=False)

    # Stabilo: word vs sent
    stabilo_summary = compute_summary(stabilo, group_col="task")
    print("\nStabilo Summary:")
    print(stabilo_summary.to_string(index=False))
    stabilo_summary.to_csv(TABLE_DIR / "stabilo_overall_summary.csv", index=False)

    # Combined dual-dataset table
    # For dual-dataset: use AR-only OnHW and Stabilo word (both word-level, comparable)
    combined = pd.DataFrame()
    onhw_ar_copy = onhw_ar.copy()
    onhw_ar_copy["task"] = "OnHW (AR-only)"
    onhw_hyb_copy = onhw_hyb.copy()
    onhw_hyb_copy["task"] = "OnHW (Hybrid)"
    stabilo_word_copy = stabilo_word.copy()
    stabilo_word_copy["task"] = "Stabilo Word"
    stabilo_sent_copy = stabilo_sent.copy()
    stabilo_sent_copy["task"] = "Stabilo Sent"
    combined = pd.concat([onhw_ar_copy, onhw_hyb_copy, stabilo_word_copy, stabilo_sent_copy])
    combined_summary = compute_summary(combined, group_col="task")
    print("\nCombined Summary:")
    print(combined_summary.to_string(index=False))
    combined_summary.to_csv(TABLE_DIR / "combined_overall_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Fold-Wise Summary
    # ------------------------------------------------------------------
    print("\n=== Fold-Wise Summary ===")
    foldwise_parts = []
    for name, sub in [("OnHW (AR-only)", onhw_ar), ("OnHW (Hybrid)", onhw_hyb),
                       ("Stabilo Word", stabilo_word), ("Stabilo Sent", stabilo_sent)]:
        fw = compute_foldwise(sub, name)
        foldwise_parts.append(fw)
    foldwise = pd.concat(foldwise_parts, ignore_index=True)
    print(foldwise.to_string(index=False))
    foldwise.to_csv(TABLE_DIR / "combined_foldwise_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 3. LD Distribution Figures
    # ------------------------------------------------------------------
    print("\n=== LD Distribution Figures ===")

    # OnHW: AR vs Hybrid
    plot_ld_distribution(
        {"AR-only": onhw_ar["levenshtein_distance"].to_numpy(),
         "Hybrid": onhw_hyb["levenshtein_distance"].to_numpy()},
        FIG_DIR / "ld_distribution_onhw.pdf",
        "OnHW Word-Level: Levenshtein Distance Distribution",
        normalized=False,
    )
    plot_ld_distribution(
        {"AR-only": onhw_ar["lev_norm"].to_numpy(),
         "Hybrid": onhw_hyb["lev_norm"].to_numpy()},
        FIG_DIR / "ld_distribution_norm_onhw.pdf",
        "OnHW Word-Level: Normalized LD Distribution",
        normalized=True,
    )

    # Stabilo: Word vs Sent
    plot_ld_distribution(
        {"Word": stabilo_word["lev_norm"].to_numpy(),
         "Sentence": stabilo_sent["lev_norm"].to_numpy()},
        FIG_DIR / "ld_distribution_norm_stabilo.pdf",
        "Stabilo: Normalized LD Distribution",
        normalized=True,
    )

    # Cross-dataset comparison (word-level only, normalized)
    plot_ld_distribution(
        {"OnHW AR-only": onhw_ar["lev_norm"].to_numpy(),
         "OnHW Hybrid": onhw_hyb["lev_norm"].to_numpy(),
         "Stabilo Word": stabilo_word["lev_norm"].to_numpy(),
         "Stabilo Sent": stabilo_sent["lev_norm"].to_numpy()},
        FIG_DIR / "ld_distribution_norm_cross_dataset.pdf",
        "Cross-Dataset: Normalized LD Distribution",
        normalized=True,
    )

    # ------------------------------------------------------------------
    # 4. Per-Character Error Rate
    # ------------------------------------------------------------------
    print("\n=== Per-Character Error Rate ===")

    onhw_ar_preds = onhw_ar["prediction"].tolist()
    onhw_ar_labels = onhw_ar["label"].tolist()
    onhw_hyb_preds = onhw_hyb["prediction"].tolist()
    onhw_hyb_labels = onhw_hyb["label"].tolist()

    stats_ar = compute_per_char_error_simple(onhw_ar_preds, onhw_ar_labels)
    stats_hyb = compute_per_char_error_simple(onhw_hyb_preds, onhw_hyb_labels)

    plot_per_char_error(
        {"AR-only": stats_ar, "Hybrid": stats_hyb},
        FIG_DIR / "per_char_error_onhw.pdf",
        "OnHW Word-Level: Per-Character Error Rate (AR-only vs Hybrid)",
    )

    # Stabilo word
    stab_word_stats = compute_per_char_error_simple(
        stabilo_word["prediction"].tolist(), stabilo_word["label"].tolist()
    )
    # Stabilo sent
    stab_sent_stats = compute_per_char_error_simple(
        stabilo_sent["prediction"].tolist(), stabilo_sent["label"].tolist()
    )
    plot_per_char_error(
        {"Stabilo Word": stab_word_stats, "Stabilo Sent": stab_sent_stats},
        FIG_DIR / "per_char_error_stabilo.pdf",
        "Stabilo: Per-Character Error Rate (Word vs Sentence)",
    )

    # ------------------------------------------------------------------
    # 5. Length Dependence
    # ------------------------------------------------------------------
    print("\n=== Length Dependence ===")
    len_dep = compute_length_dependence(combined)
    print(len_dep.to_string(index=False))
    len_dep.to_csv(TABLE_DIR / "length_dependence.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Collision Analysis
    # ------------------------------------------------------------------
    print("\n=== Collision Analysis ===")
    collision_results = []
    for name, sub in [("OnHW (AR-only)", onhw_ar), ("OnHW (Hybrid)", onhw_hyb),
                       ("Stabilo Word", stabilo_word), ("Stabilo Sent", stabilo_sent)]:
        result = compute_collisions(sub, name)
        collision_results.append(result)
        print(f"\n{name}:")
        print(f"  Total: {result['n_total']}, Errors: {result['n_errors']}, "
              f"Collisions (UID): {result['n_collisions_uid']}, "
              f"Collision rate: {result['collision_rate']:.3f}")
        if "gt_freq_median" in result:
            print(f"  GT freq: median={result['gt_freq_median']:.0f}, "
                  f"p90={result['gt_freq_p90']:.0f}, p95={result['gt_freq_p95']:.0f}")
        print(f"  Top collided: {list(result['top_collided'].items())[:5]}")

    # Save collision summary table
    collision_df = pd.DataFrame([{
        "Dataset": r["task"],
        "N": r["n_total"],
        "Errors": r["n_errors"],
        "Collisions (UID)": r["n_collisions_uid"],
        "Collision Rate": f"{r['collision_rate']:.3f}",
        "GT Freq Median": f"{r.get('gt_freq_median', 0):.0f}",
        "GT Freq p90": f"{r.get('gt_freq_p90', 0):.0f}",
        "GT Freq p95": f"{r.get('gt_freq_p95', 0):.0f}",
    } for r in collision_results])
    print("\nCollision Summary:")
    print(collision_df.to_string(index=False))
    collision_df.to_csv(TABLE_DIR / "collision_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Representative Examples
    # ------------------------------------------------------------------
    print("\n=== Representative Error Examples ===")
    examples_parts = []
    for name, sub in [("OnHW (AR-only)", onhw_ar), ("OnHW (Hybrid)", onhw_hyb),
                       ("Stabilo Word", stabilo_word), ("Stabilo Sent", stabilo_sent)]:
        ex = find_representative_examples(sub, name)
        if len(ex) > 0:
            examples_parts.append(ex)
    examples = pd.concat(examples_parts, ignore_index=True)
    print(examples.to_string(index=False))
    examples.to_csv(TABLE_DIR / "representative_examples.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Generate LaTeX tables
    # ------------------------------------------------------------------
    print("\n=== Generating LaTeX Tables ===")
    generate_latex_tables(combined_summary, foldwise, len_dep, collision_df, examples)

    print("\nDone! All figures saved to:", FIG_DIR)
    print("All tables saved to:", TABLE_DIR)


def generate_latex_tables(summary, foldwise, len_dep, collision, examples):
    """Generate LaTeX table files."""

    # Overall summary
    with open(TABLE_DIR / "table_overall_summary.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Overall quantitative summary using normalized Levenshtein distance $e_i = d_i / |y_i|$ across all datasets.
Micro denotes $\sum_i d_i / \sum_i |y_i|$; Macro denotes $\frac{1}{N}\sum_i e_i$.
Quantiles are computed over per-sample $e_i$.}
\label{tab:overall-dual-dataset}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
""")
        f.write(" & ".join(summary.columns) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, row in summary.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""")

    # Length dependence
    with open(TABLE_DIR / "table_length_dependence.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Pearson correlation between ground-truth length $|y|$ and error: raw Levenshtein distance $d$ vs.\ normalized $e = d/|y|$. Results shown for the full set (All) and for incorrect predictions only ($e > 0$).}
\label{tab:length-dependence}
\begin{tabular}{lrrrr}
\toprule
""")
        f.write(" & ".join(len_dep.columns) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, row in len_dep.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Collision summary
    with open(TABLE_DIR / "table_collision_summary.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Collision analysis across datasets. A collision occurs when an incorrect prediction exactly matches the ground-truth label of a different sample. GT Freq shows the distribution of \texttt{gt\_label\_count} for collided predictions (UID-weighted).}
\label{tab:collision-summary}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrr}
\toprule
""")
        f.write(" & ".join(collision.columns) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        for _, row in collision.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""")

    # Representative examples
    with open(TABLE_DIR / "table_representative_examples.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Representative examples near quantiles of the nonzero normalized Levenshtein error distribution ($e > 0$), two samples per quantile and dataset.}
\label{tab:representative-examples}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrrl l}
\toprule
Dataset & Quantile & Fold & Idx & $d$ & $e$ & Prediction & Ground Truth \\
\midrule
""")
        for _, row in examples.iterrows():
            pred = str(row["Prediction"]).replace("_", r"\_").replace("&", r"\&")
            gt = str(row["Ground Truth"]).replace("_", r"\_").replace("&", r"\&")
            # Truncate long strings
            if len(pred) > 25:
                pred = pred[:22] + "..."
            if len(gt) > 25:
                gt = gt[:22] + "..."
            f.write(f"{row['Dataset']} & {row['Quantile']} & {row['Fold']} & {row['Idx']} & "
                    f"{row['d']} & {row['e']:.2f} & {pred} & {gt}" + r" \\" + "\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""")

    print(f"  LaTeX tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    main()
