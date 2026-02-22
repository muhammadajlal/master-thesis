#!/usr/bin/env python3
"""
Decoding Study — Results Aggregation & Thesis Table Generator
=============================================================

Reads all decode_study results and produces:
1. Main ablation table (Table X in thesis)
2. Performance vs LM weight λ data (for plots)
3. Tail-risk statistics
4. LaTeX table source

Usage:
    python scripts/aggregate_decode_study.py \
        --results_dir /path/to/decode_study/ \
        --outdir results/hwr2/decode_study/tables
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_all_results(results_dir: str) -> list[dict]:
    """Load all metrics.json files from the results directory."""
    results = []
    pattern = os.path.join(results_dir, "**", "metrics.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path) as f:
                metrics = json.load(f)
            # Add run tag from directory name
            run_dir = os.path.dirname(path)
            metrics["_run_tag"] = os.path.basename(run_dir)
            metrics["_run_dir"] = run_dir

            # Load config if available
            config_path = os.path.join(run_dir, "config.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    metrics["_config"] = json.load(f)

            results.append(metrics)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    return results


def extract_fold(tag: str) -> int:
    """Extract fold number from run tag."""
    m = re.search(r"fold(\d+)", tag)
    return int(m.group(1)) if m else -1


def extract_stage(tag: str) -> str:
    """Extract stage identifier from tag."""
    m = re.match(r"(stage[A-D]\d+)", tag)
    return m.group(1) if m else "unknown"


def group_by_experiment(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by experiment (removing fold suffix)."""
    groups = defaultdict(list)
    for r in results:
        tag = r["_run_tag"]
        # Remove __foldN suffix for grouping
        key = re.sub(r"__fold\d+$", "", tag)
        groups[key].append(r)
    return dict(groups)


def compute_cv_stats(group: list[dict]) -> dict:
    """Compute mean ± std across folds for a group of results."""
    cers = [r["cer"] for r in group]
    wers = [r["wer"] for r in group]
    neds_p95 = [r.get("ned_p95", float("nan")) for r in group]
    neds_p99 = [r.get("ned_p99", float("nan")) for r in group]
    runtimes = [r.get("runtime_per_sample_ms", 0) for r in group]
    eos_rates = [r.get("early_eos_rate", float("nan")) for r in group]
    lr_means = [r.get("length_ratio_mean", float("nan")) for r in group]

    n_folds = len(group)
    return {
        "n_folds": n_folds,
        "cer_mean": float(np.mean(cers)),
        "cer_std": float(np.std(cers)) if n_folds > 1 else 0.0,
        "wer_mean": float(np.mean(wers)),
        "wer_std": float(np.std(wers)) if n_folds > 1 else 0.0,
        "ned_p95_mean": float(np.nanmean(neds_p95)),
        "ned_p99_mean": float(np.nanmean(neds_p99)),
        "runtime_ms": float(np.mean(runtimes)),
        "early_eos_rate": float(np.nanmean(eos_rates)),
        "length_ratio_mean": float(np.nanmean(lr_means)),
    }


def print_main_table(groups: dict[str, list[dict]], outdir: str):
    """Print the main ablation table."""
    # Define row order for the final table (Stage D)
    final_rows = [
        ("AR greedy", "stageA0_ar_greedy"),
        ("AR calibrated beam", None),  # Will be filled with best from A
        ("AR beam + LM (rescore)", None),  # Best from C1
        ("AR beam + LM (shallow)", None),  # Best from C2
        ("CTC greedy", "stageB1_ctc_greedy"),
        ("CTC beam", None),  # Best from B2
        ("CTC beam + LM", None),  # Best from B3
    ]

    # Find best in each stage by CER
    def find_best(prefix: str) -> tuple[str | None, dict | None]:
        candidates = {k: v for k, v in groups.items() if k.startswith(prefix)}
        if not candidates:
            return None, None
        best_key = min(candidates, key=lambda k: compute_cv_stats(candidates[k])["cer_mean"])
        return best_key, candidates[best_key]

    best_a2_key, best_a2 = find_best("stageA2_")
    best_a3_key, best_a3 = find_best("stageA3_")
    best_b2_key, best_b2 = find_best("stageB2_")
    best_b3_key, best_b3 = find_best("stageB3_")
    best_c1_key, best_c1 = find_best("stageC1_")
    best_c2_key, best_c2 = find_best("stageC2_")

    # Use best AR calibrated as whichever is better between A2 and A3
    if best_a2 and best_a3:
        s_a2 = compute_cv_stats(best_a2)["cer_mean"]
        s_a3 = compute_cv_stats(best_a3)["cer_mean"]
        ar_cal_key = best_a2_key if s_a2 <= s_a3 else best_a3_key
        ar_cal = best_a2 if s_a2 <= s_a3 else best_a3
    elif best_a2:
        ar_cal_key, ar_cal = best_a2_key, best_a2
    elif best_a3:
        ar_cal_key, ar_cal = best_a3_key, best_a3
    else:
        ar_cal_key, ar_cal = None, None

    print("\n" + "=" * 100)
    print("MAIN ABLATION TABLE (Thesis Table)")
    print("=" * 100)
    header = f"{'Row':<30s} {'CER↓':>12s} {'WER↓':>12s} {'NED P95':>10s} {'NED P99':>10s} {'EOS%':>8s} {'LenRat':>8s} {'ms/smp':>8s} {'Config':>30s}"
    print(header)
    print("-" * 100)

    def print_row(label: str, key: str | None, group: list[dict] | None):
        if group is None:
            print(f"{label:<30s}  {'N/A':>12s}")
            return
        stats = compute_cv_stats(group)
        n = stats["n_folds"]
        cer_s = f"{stats['cer_mean']:.4f}±{stats['cer_std']:.4f}" if n > 1 else f"{stats['cer_mean']:.4f}"
        wer_s = f"{stats['wer_mean']:.4f}±{stats['wer_std']:.4f}" if n > 1 else f"{stats['wer_mean']:.4f}"
        print(
            f"{label:<30s} {cer_s:>12s} {wer_s:>12s} "
            f"{stats['ned_p95_mean']:>10.4f} {stats['ned_p99_mean']:>10.4f} "
            f"{stats['early_eos_rate']:>8.3f} {stats['length_ratio_mean']:>8.3f} "
            f"{stats['runtime_ms']:>8.1f} {(key or ''):>30s}"
        )

    # AR rows
    ar_greedy = groups.get("stageA0_ar_greedy")
    print_row("AR greedy", "stageA0_ar_greedy", ar_greedy)
    print_row("AR calibrated beam", ar_cal_key, ar_cal)
    print_row("AR + LM (rescore)", best_c1_key, best_c1)
    print_row("AR + LM (shallow)", best_c2_key, best_c2)

    print("-" * 100)

    # CTC rows
    ctc_greedy = groups.get("stageB1_ctc_greedy")
    print_row("CTC greedy", "stageB1_ctc_greedy", ctc_greedy)
    print_row("CTC beam", best_b2_key, best_b2)
    print_row("CTC beam + LM", best_b3_key, best_b3)

    print("=" * 100)

    # Save as JSON
    table_data = {}
    for label, key, grp in [
        ("ar_greedy", "stageA0_ar_greedy", ar_greedy),
        ("ar_calibrated_beam", ar_cal_key, ar_cal),
        ("ar_lm_rescore", best_c1_key, best_c1),
        ("ar_lm_shallow", best_c2_key, best_c2),
        ("ctc_greedy", "stageB1_ctc_greedy", ctc_greedy),
        ("ctc_beam", best_b2_key, best_b2),
        ("ctc_beam_lm", best_b3_key, best_b3),
    ]:
        if grp:
            table_data[label] = {**compute_cv_stats(grp), "config_key": key}

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "main_table.json"), "w") as f:
        json.dump(table_data, f, indent=2)


def print_sweep_tables(groups: dict[str, list[dict]], outdir: str):
    """Print detailed sweep tables for each stage."""
    stage_order = ["stageA1", "stageA2", "stageA3", "stageB2", "stageB3", "stageC1", "stageC2"]
    stage_names = {
        "stageA1": "A1: AR Beam Size Sweep",
        "stageA2": "A2: AR Length Normalisation Sweep",
        "stageA3": "A3: AR EOS Control Sweep",
        "stageB2": "B2: CTC Beam Size Sweep",
        "stageB3": "B3: CTC Beam + LM Sweep",
        "stageC1": "C1: AR N-best Rescoring Sweep",
        "stageC2": "C2: AR Shallow Fusion Sweep",
    }

    os.makedirs(outdir, exist_ok=True)

    for stage in stage_order:
        stage_groups = {k: v for k, v in groups.items() if k.startswith(stage)}
        if not stage_groups:
            continue

        print(f"\n{'─' * 80}")
        print(f"  {stage_names.get(stage, stage)}")
        print(f"{'─' * 80}")
        print(f"  {'Config':<50s} {'CER↓':>10s} {'WER↓':>10s} {'NED P95':>8s} {'ms/smp':>8s}")

        sweep_data = []
        for key in sorted(stage_groups.keys()):
            grp = stage_groups[key]
            stats = compute_cv_stats(grp)
            short_key = key.replace(f"{stage}_", "")
            print(f"  {short_key:<50s} {stats['cer_mean']:>10.4f} {stats['wer_mean']:>10.4f} {stats['ned_p95_mean']:>8.4f} {stats['runtime_ms']:>8.1f}")
            sweep_data.append({"key": key, **stats})

        with open(os.path.join(outdir, f"sweep_{stage}.json"), "w") as f:
            json.dump(sweep_data, f, indent=2)


def print_lm_weight_data(groups: dict[str, list[dict]], outdir: str):
    """Extract performance vs LM weight λ data for plots."""
    os.makedirs(outdir, exist_ok=True)

    # CTC + LM
    ctc_lm_data = []
    for key, grp in groups.items():
        if not key.startswith("stageB3_"):
            continue
        m = re.search(r"lw([\d.]+)", key)
        if m:
            lw = float(m.group(1))
            stats = compute_cv_stats(grp)
            ctc_lm_data.append({"lm_weight": lw, **stats, "config": key})

    # AR + LM (rescoring)
    ar_rescore_data = []
    for key, grp in groups.items():
        if not key.startswith("stageC1_"):
            continue
        m = re.search(r"lw([\d.]+)", key)
        if m:
            lw = float(m.group(1))
            stats = compute_cv_stats(grp)
            ar_rescore_data.append({"lm_weight": lw, **stats, "config": key})

    # AR + LM (shallow fusion)
    ar_shallow_data = []
    for key, grp in groups.items():
        if not key.startswith("stageC2_"):
            continue
        m = re.search(r"lw([\d.]+)", key)
        if m:
            lw = float(m.group(1))
            stats = compute_cv_stats(grp)
            ar_shallow_data.append({"lm_weight": lw, **stats, "config": key})

    plot_data = {
        "ctc_lm": sorted(ctc_lm_data, key=lambda x: x["lm_weight"]),
        "ar_rescore": sorted(ar_rescore_data, key=lambda x: x["lm_weight"]),
        "ar_shallow": sorted(ar_shallow_data, key=lambda x: x["lm_weight"]),
    }
    with open(os.path.join(outdir, "lm_weight_plot_data.json"), "w") as f:
        json.dump(plot_data, f, indent=2)

    print("\n── LM Weight vs CER (for plotting) ──")
    for name, data in plot_data.items():
        if data:
            print(f"\n  {name}:")
            for d in data:
                print(f"    λ={d['lm_weight']:.2f}  CER={d['cer_mean']:.4f}  WER={d['wer_mean']:.4f}")


def generate_latex(groups: dict[str, list[dict]], outdir: str):
    """Generate LaTeX table source."""
    os.makedirs(outdir, exist_ok=True)

    def find_best(prefix):
        candidates = {k: v for k, v in groups.items() if k.startswith(prefix)}
        if not candidates:
            return None
        best_key = min(candidates, key=lambda k: compute_cv_stats(candidates[k])["cer_mean"])
        return compute_cv_stats(candidates[best_key])

    rows = [
        ("AR greedy", groups.get("stageA0_ar_greedy")),
        ("AR calibrated beam", None),
        ("AR + LM (rescore)", None),
        ("AR + LM (shallow)", None),
        ("CTC greedy", groups.get("stageB1_ctc_greedy")),
        ("CTC beam", None),
        ("CTC beam + LM", None),
    ]

    # Fill best rows
    best_configs = {
        "AR calibrated beam": find_best("stageA2_") or find_best("stageA3_"),
        "AR + LM (rescore)": find_best("stageC1_"),
        "AR + LM (shallow)": find_best("stageC2_"),
        "CTC beam": find_best("stageB2_"),
        "CTC beam + LM": find_best("stageB3_"),
    }

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"  \centering")
    latex.append(r"  \caption{Decoding study ablation on OnHW500 WI (word-level, 5-fold CV).}")
    latex.append(r"  \label{tab:decode_study}")
    latex.append(r"  \begin{tabular}{l c c c c c}")
    latex.append(r"    \toprule")
    latex.append(r"    Method & CER$\downarrow$ & WER$\downarrow$ & NED P95 & NED P99 & ms/sample \\")
    latex.append(r"    \midrule")

    for label, grp in rows:
        if grp is not None:
            stats = compute_cv_stats(grp)
        elif label in best_configs and best_configs[label] is not None:
            stats = best_configs[label]
        else:
            latex.append(f"    {label} & --- & --- & --- & --- & --- \\\\")
            continue

        n = stats["n_folds"]
        if n > 1:
            cer_s = f"${stats['cer_mean']:.3f} \\pm {stats['cer_std']:.3f}$"
            wer_s = f"${stats['wer_mean']:.3f} \\pm {stats['wer_std']:.3f}$"
        else:
            cer_s = f"${stats['cer_mean']:.3f}$"
            wer_s = f"${stats['wer_mean']:.3f}$"

        latex.append(
            f"    {label} & {cer_s} & {wer_s} "
            f"& ${stats['ned_p95_mean']:.3f}$ & ${stats['ned_p99_mean']:.3f}$ "
            f"& ${stats['runtime_ms']:.1f}$ \\\\"
        )

        if label == "AR + LM (shallow)":
            latex.append(r"    \midrule")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    with open(os.path.join(outdir, "decode_study_table.tex"), "w") as f:
        f.write(latex_str + "\n")
    print(f"\nLaTeX table written to {os.path.join(outdir, 'decode_study_table.tex')}")
    print(latex_str)


def main():
    parser = argparse.ArgumentParser(description="Aggregate decoding study results")
    parser.add_argument(
        "--results_dir", type=str,
        default="/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/decode_study",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/decode_study/tables",
    )
    args = parser.parse_args()

    # Load results
    all_results = load_all_results(args.results_dir)
    print(f"Loaded {len(all_results)} result files")

    if not all_results:
        print("No results found! Run the decode study first.")
        sys.exit(1)

    # Group by experiment (across folds)
    groups = group_by_experiment(all_results)
    print(f"Found {len(groups)} unique experiments across folds")

    # Print detailed sweep tables
    print_sweep_tables(groups, args.outdir)

    # Print main ablation table
    print_main_table(groups, args.outdir)

    # LM weight data for plots
    print_lm_weight_data(groups, args.outdir)

    # Generate LaTeX
    generate_latex(groups, args.outdir)

    # Save full summary
    summary = {}
    for key, grp in sorted(groups.items()):
        summary[key] = compute_cv_stats(grp)
    with open(os.path.join(args.outdir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {os.path.join(args.outdir, 'full_summary.json')}")


if __name__ == "__main__":
    main()
