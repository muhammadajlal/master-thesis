#!/usr/bin/env python3
"""
Dual-dataset descriptive analysis for the thesis.

Reports how HWRFormer recognition errors are distributed across the public
OnHW word benchmark and the private datasets. OnHW is always reported per
writer split -- writer-independent (WI) and writer-dependent (WD) are NEVER
pooled -- and the private data by target type (word, sentence).

Model naming (fixed across the thesis): HWRFormer is the complete end-to-end
recognizer (shared CNN encoder + AR-transformer decoder). The decoder size sets
the name:
  * HWRFormer   = CNN encoder + ar_transformer_xs decoder (primary model).
  * HWRFormer-L = CNN encoder + ar_transformer_s  decoder (scaled model).
"AR-only" and "hybrid" are training conditions of the same recognizer.

This script produces ONLY descriptive artifacts (distributions, per-character
rates, within-task length correlations, collision rates, representative
examples). It makes no causal or cross-task-comparability claim; the paired
inferential statistics (McNemar, fold/writer uncertainty, paired NED effect)
live in ``hybrid_paired_analysis.py``.

Outputs: figures under thesis/figures/<variant>/ and tables/CSVs under
analysis/tables/thesis_dual_dataset_<variant>/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ============================================================
# Paths (variant-aware: HWRFormer = xs, HWRFormer-L = s)
# ============================================================
WORK_DIR = Path(__file__).resolve().parent.parent.parent  # REWI_work
THESIS_DIR = WORK_DIR.parent.parent / "thesis"

# Each variant selects a decoder arch and its result groups. Input CSVs follow
# the same naming convention (``_xs`` suffix for HWRFormer, no suffix for the
# scaled HWRFormer-L run).
VARIANT_SPECS = {
    "xs": {  # HWRFormer
        "arch_de": "ar_transformer_xs",
        "csv_suffix": "_xs",
    },
    "s": {   # HWRFormer-L
        "arch_de": "ar_transformer_s",
        "csv_suffix": "",
    },
}

# Expected per-(condition, split) paired-sample counts for OnHW; the loader
# asserts these so WI/WD can never be silently pooled or truncated.
ONHW_EXPECTED = {("AR-only", "wi"): 25199, ("AR-only", "wd"): 25193,
                 ("Hybrid (AR Decoding)", "wi"): 25199, ("Hybrid (AR Decoding)", "wd"): 25193}


def _resolve_paths(variant: str):
    spec = VARIANT_SPECS[variant]
    suffix = spec["csv_suffix"]
    return {
        "fig_dir": THESIS_DIR / "figures" / variant,
        "table_dir": WORK_DIR / "analysis" / "tables" / f"thesis_dual_dataset_{variant}",
        "onhw_csv": WORK_DIR / "analysis" / f"quant_all_val_predictions_ar_vs_hybrid{suffix}.csv",
        "stabilo_csv": WORK_DIR / "analysis" / f"quant_all_val_predictions_new{suffix}.csv",
    }


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
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df["prediction"] = df["prediction"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)
    df["label_chars"] = df["label"].str.len()
    df["lev_norm"] = df["levenshtein_distance"] / df["label_chars"].clip(lower=1)
    df["exact"] = (df["levenshtein_distance"] == 0).astype(int)
    return df


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

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    x_label = "Normalized Levenshtein Distance ($d / |y|$)" if normalized else "Levenshtein Distance"

    datasets = {k: v for k, v in datasets.items() if np.asarray(v).size > 0}
    if not datasets:
        plt.close(fig)
        print(f"  Skipped (all subsets empty): {save_path}")
        return
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
    """Per-character error rates via jiwer alignment (fallback: proportional)."""
    try:
        import jiwer
        stats = {}
        cats = [c for c in sorted(set("".join(labels))) if c.strip()]
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
        print("  Warning: jiwer not available, using fallback per-char error")
        char_counts: Dict[str, int] = {}
        char_errors: Dict[str, float] = {}
        for pred, label in zip(preds, labels):
            for c in label:
                char_counts[c] = char_counts.get(c, 0) + 1
            d = levenshtein(pred, label)
            if d > 0:
                for c in set(label):
                    n_c = label.count(c)
                    char_errors[c] = char_errors.get(c, 0) + d * n_c / max(1, len(label))
        stats = {}
        for c in sorted(char_counts.keys()):
            if c.strip():
                stats[c] = {"total": char_counts[c],
                            "error_rate": char_errors.get(c, 0) / max(1, char_counts[c])}
        return stats


def plot_per_char_error(datasets: Dict[str, Dict[str, Dict]], save_path: Path, title: str):
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
# 4. Length Dependence (within-task, descriptive)
# ============================================================
def compute_length_dependence(df: pd.DataFrame) -> pd.DataFrame:
    """Within-task Pearson correlation of error with target length.

    This is a within-task near-invariance check for the normalized metric e;
    it does NOT license any cross-task comparison (target length, vocabulary,
    and character frequencies all differ between tasks).
    """
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
# 5. Collision Analysis (descriptive)
# ============================================================
def compute_collisions(df: pd.DataFrame, task_name: str) -> Dict:
    """Rate at which incorrect predictions coincide with some ground-truth label.

    Reported descriptively. ``corpus_median_label_freq`` gives the base rate of
    label frequencies so the collided-prediction frequency can be read against
    it -- a high collision rate primarily reflects vocabulary-constrained
    decoder outputs, not a tested frequency-bias mechanism.
    """
    errors = df[df["levenshtein_distance"] > 0].copy()
    gt_set = set(df["label"].unique())

    errors["is_collision"] = errors["prediction"].isin(gt_set)
    collisions = errors[errors["is_collision"]].copy()

    n_errors = len(errors)
    n_collisions = len(collisions)

    gt_counts = df["label"].value_counts().to_dict()
    collisions["gt_label_count"] = collisions["prediction"].map(gt_counts).fillna(0).astype(int)

    uid_weighted = collisions.drop_duplicates(subset=["fold", "sample_index"])
    gt_freq = uid_weighted["gt_label_count"]

    # Base rate: median frequency of a label across the corpus (per-sample rows).
    corpus_label_freq = df["label"].map(gt_counts)

    result = {
        "task": task_name,
        "n_total": len(df),
        "n_errors": n_errors,
        "n_collisions_uid": len(uid_weighted),
        "n_collisions_event": n_collisions,
        "collision_rate": n_collisions / max(1, n_errors),
        "corpus_median_label_freq": float(corpus_label_freq.median()),
    }

    if len(gt_freq) > 0:
        result.update({
            "gt_freq_median": float(gt_freq.median()),
            "gt_freq_p90": float(gt_freq.quantile(0.9)),
            "gt_freq_p95": float(gt_freq.quantile(0.95)),
        })

    result["top_collided"] = uid_weighted["prediction"].value_counts().head(10).to_dict()
    return result


# ============================================================
# 6. Representative Examples
# ============================================================
def find_representative_examples(df: pd.DataFrame, task_name: str, n_per_quantile: int = 2) -> pd.DataFrame:
    """Examples nearest p50/p90/p99 of the nonzero error distribution.

    Emits explicit ``Ground Truth`` and ``Prediction`` columns; any arrow prose
    must be written as ``Ground Truth -> Prediction`` to match this ordering.
    """
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
                "Ground Truth": row["label"],
                "Prediction": row["prediction"],
            })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Dual-dataset descriptive analysis (variant-aware).")
    p.add_argument("--variant", choices=sorted(VARIANT_SPECS.keys()), default="xs",
                   help="'xs' = HWRFormer (ar_transformer_xs); 's' = HWRFormer-L (ar_transformer_s).")
    p.add_argument("--onhw-csv", type=Path, default=None, help="Override OnHW predictions CSV.")
    p.add_argument("--stabilo-csv", type=Path, default=None, help="Override Stabilo predictions CSV.")
    return p.parse_args()


def _split_onhw(onhw: pd.DataFrame, condition: str, split: str) -> pd.DataFrame:
    """Return the OnHW rows for one (condition, writer-split), asserting the count."""
    if "split" not in onhw.columns:
        sys.exit("OnHW CSV lacks a 'split' column; rebuild it with _build_xs_aggregated_csv.py "
                 "(WI and WD must be distinguishable, never pooled).")
    sub = onhw[(onhw["task"] == condition) & (onhw["split"] == split)].copy()
    exp = ONHW_EXPECTED.get((condition, split))
    if exp is not None and len(sub) != exp:
        sys.exit(f"OnHW {condition}/{split}: got {len(sub)} rows, expected {exp} (WI/WD pooling?).")
    return sub


def main():
    args = _parse_args()
    paths = _resolve_paths(args.variant)
    fig_dir = paths["fig_dir"]
    table_dir = paths["table_dir"]
    onhw_csv = Path(args.onhw_csv) if args.onhw_csv else paths["onhw_csv"]
    stabilo_csv = Path(args.stabilo_csv) if args.stabilo_csv else paths["stabilo_csv"]

    print(f"=== Variant: {args.variant} "
          f"({'HWRFormer' if args.variant == 'xs' else 'HWRFormer-L'}, "
          f"{VARIANT_SPECS[args.variant]['arch_de']}) ===")
    print(f"  ONHW_CSV:    {onhw_csv}  exists={onhw_csv.exists()}")
    print(f"  STABILO_CSV: {stabilo_csv}  exists={stabilo_csv.exists()}\n")

    missing = [(p, n) for p, n in [(onhw_csv, "ONHW_CSV"), (stabilo_csv, "STABILO_CSV")] if not p.exists()]
    if missing:
        sys.exit("ERROR: input CSV(s) not found:\n" + "\n".join(f"  {n}: {p}" for p, n in missing))

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data. OnHW is split per writer condition (WI/WD), never pooled.
    # ------------------------------------------------------------------
    print("Loading OnHW data (per writer split)...")
    onhw = load_csv(onhw_csv)
    onhw_ar_wi = _split_onhw(onhw, "AR-only", "wi")
    onhw_ar_wd = _split_onhw(onhw, "AR-only", "wd")
    onhw_hyb_wi = _split_onhw(onhw, "Hybrid (AR Decoding)", "wi")
    onhw_hyb_wd = _split_onhw(onhw, "Hybrid (AR Decoding)", "wd")

    print("Loading Stabilo data...")
    stabilo = load_csv(stabilo_csv)
    stabilo_word = stabilo[stabilo["task"] == "word"].copy()
    stabilo_sent = stabilo[stabilo["task"] == "sent"].copy()

    # Canonical labelled frames. OnHW appears per (writer split, training
    # condition); nothing is pooled across WI and WD.
    frames = [
        ("OnHW WI (HWRFormer)", onhw_ar_wi),
        ("OnHW WI (hybrid)", onhw_hyb_wi),
        ("OnHW WD (HWRFormer)", onhw_ar_wd),
        ("OnHW WD (hybrid)", onhw_hyb_wd),
        ("Private word", stabilo_word),
        ("Private sent.", stabilo_sent),
    ]
    for label, sub in frames:
        print(f"  {label:22s} N={len(sub)}")

    # One combined frame with the split-aware task labels above.
    labeled = []
    for label, sub in frames:
        s = sub.copy()
        s["task"] = label
        labeled.append(s)
    combined = pd.concat(labeled, ignore_index=True)

    # ------------------------------------------------------------------
    # 1. Overall + fold-wise summary (per-split)
    # ------------------------------------------------------------------
    print("\n=== Overall Summary (per split) ===")
    combined_summary = compute_summary(combined, group_col="task")
    print(combined_summary.to_string(index=False))
    combined_summary.to_csv(table_dir / "combined_overall_summary.csv", index=False)

    print("\n=== Fold-Wise Summary ===")
    foldwise = pd.concat([compute_foldwise(sub, label) for label, sub in frames], ignore_index=True)
    foldwise.to_csv(table_dir / "combined_foldwise_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 2. LD distribution figures (descriptive, per split)
    # ------------------------------------------------------------------
    print("\n=== LD Distribution Figures ===")
    plot_ld_distribution(
        {"HWRFormer": onhw_ar_wi["lev_norm"].to_numpy(),
         "hybrid": onhw_hyb_wi["lev_norm"].to_numpy()},
        fig_dir / "ld_distribution_norm_onhw_wi.pdf",
        "OnHW WI Word: Normalized LD Distribution", normalized=True)
    plot_ld_distribution(
        {"Word": stabilo_word["lev_norm"].to_numpy(),
         "Sentence": stabilo_sent["lev_norm"].to_numpy()},
        fig_dir / "ld_distribution_norm_stabilo.pdf",
        "Private: Normalized LD Distribution", normalized=True)
    plot_ld_distribution(
        {label: sub["lev_norm"].to_numpy() for label, sub in frames},
        fig_dir / "ld_distribution_norm_cross_dataset.pdf",
        "Cross-Dataset (descriptive): Normalized LD Distribution", normalized=True)

    # ------------------------------------------------------------------
    # 3. Per-character error rate (OnHW WI AR vs hybrid; private word vs sent)
    # ------------------------------------------------------------------
    print("\n=== Per-Character Error Rate ===")
    plot_per_char_error(
        {"HWRFormer": compute_per_char_error_simple(onhw_ar_wi["prediction"].tolist(),
                                                    onhw_ar_wi["label"].tolist()),
         "hybrid": compute_per_char_error_simple(onhw_hyb_wi["prediction"].tolist(),
                                                 onhw_hyb_wi["label"].tolist())},
        fig_dir / "per_char_error_onhw_wi.pdf",
        "OnHW WI Word: Per-Character Error Rate (HWRFormer vs hybrid)")
    plot_per_char_error(
        {"Private word": compute_per_char_error_simple(stabilo_word["prediction"].tolist(),
                                                       stabilo_word["label"].tolist()),
         "Private sent.": compute_per_char_error_simple(stabilo_sent["prediction"].tolist(),
                                                        stabilo_sent["label"].tolist())},
        fig_dir / "per_char_error_stabilo.pdf",
        "Private: Per-Character Error Rate (word vs sentence)")

    # ------------------------------------------------------------------
    # 4. Length dependence (within-task) + 5. Collisions + 6. Examples
    # ------------------------------------------------------------------
    print("\n=== Length Dependence (within-task) ===")
    len_dep = compute_length_dependence(combined)
    print(len_dep.to_string(index=False))
    len_dep.to_csv(table_dir / "length_dependence.csv", index=False)

    print("\n=== Collision Analysis (descriptive) ===")
    collision_results = [compute_collisions(sub, label) for label, sub in frames]
    for r in collision_results:
        print(f"  {r['task']:22s} rate={r['collision_rate']:.3f} "
              f"collided-freq med={r.get('gt_freq_median', 0):.0f} "
              f"(corpus base rate={r['corpus_median_label_freq']:.0f})")
    collision_df = pd.DataFrame([{
        "Dataset": r["task"],
        "N": r["n_total"],
        "Errors": r["n_errors"],
        "Collisions (UID)": r["n_collisions_uid"],
        "Collision Rate": f"{r['collision_rate']:.3f}",
        "Collided Freq Median": f"{r.get('gt_freq_median', 0):.0f}",
        "Corpus Freq Median": f"{r['corpus_median_label_freq']:.0f}",
        "GT Freq p95": f"{r.get('gt_freq_p95', 0):.0f}",
    } for r in collision_results])
    collision_df.to_csv(table_dir / "collision_summary.csv", index=False)

    print("\n=== Representative Error Examples ===")
    examples = pd.concat([find_representative_examples(sub, label) for label, sub in frames
                          if len(find_representative_examples(sub, label)) > 0], ignore_index=True)
    examples.to_csv(table_dir / "representative_examples.csv", index=False)

    # ------------------------------------------------------------------
    # 7. LaTeX tables
    # ------------------------------------------------------------------
    print("\n=== Generating LaTeX Tables ===")
    generate_latex_tables(table_dir, combined_summary, len_dep, collision_df, examples)

    print("\nDone. Figures:", fig_dir, "\nTables:", table_dir)


def generate_latex_tables(table_dir, summary, len_dep, collision, examples):
    """Emit LaTeX table files from the computed frames."""
    with open(table_dir / "table_overall_summary.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Descriptive per-split summary using normalized Levenshtein distance $e_i = d_i / |y_i|$.
Micro denotes $\sum_i d_i / \sum_i |y_i|$; Macro denotes $\frac{1}{N}\sum_i e_i$.
OnHW WI and WD are reported separately and never pooled.}
\label{tab:overall-dual-dataset}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
""")
        f.write(" & ".join(summary.columns) + r" \\" + "\n\\midrule\n")
        for _, row in summary.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")

    with open(table_dir / "table_length_dependence.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Within-task Pearson correlation between ground-truth length $|y|$ and error (raw $d$ vs.\ normalized $e = d/|y|$), for the full set (All) and incorrect predictions only ($e>0$). A within-task value near zero does not license cross-task comparison.}
\label{tab:length-dependence}
\begin{tabular}{lrrrr}
\toprule
""")
        f.write(" & ".join(len_dep.columns) + r" \\" + "\n\\midrule\n")
        for _, row in len_dep.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(table_dir / "table_collision_summary.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Collision analysis (descriptive). A collision is an incorrect prediction that exactly matches some ground-truth label. ``Collided Freq'' is the median corpus frequency of collided predictions; ``Corpus Freq'' is the median label frequency overall, given as the base rate for comparison.}
\label{tab:collision-summary}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrr}
\toprule
""")
        f.write(" & ".join(collision.columns) + r" \\" + "\n\\midrule\n")
        for _, row in collision.iterrows():
            f.write(" & ".join(str(v) for v in row.values) + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")

    with open(table_dir / "table_representative_examples.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Representative examples near quantiles of the nonzero normalized error distribution ($e>0$), two per quantile and dataset. Columns are ordered Ground Truth then Prediction.}
\label{tab:representative-examples}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrrll}
\toprule
Dataset & Quantile & Fold & Idx & $d$ & $e$ & Ground Truth & Prediction \\
\midrule
""")
        for _, row in examples.iterrows():
            gt = str(row["Ground Truth"]).replace("_", r"\_").replace("&", r"\&")
            pred = str(row["Prediction"]).replace("_", r"\_").replace("&", r"\&")
            if len(gt) > 25:
                gt = gt[:22] + "..."
            if len(pred) > 25:
                pred = pred[:22] + "..."
            f.write(f"{row['Dataset']} & {row['Quantile']} & {row['Fold']} & {row['Idx']} & "
                    f"{row['d']} & {row['e']:.2f} & {gt} & {pred}" + r" \\" + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")

    print(f"  LaTeX tables saved to {table_dir}")


if __name__ == "__main__":
    main()
