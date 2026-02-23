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


def load_baseline(baseline_path: str | None) -> dict | None:
    """Load baseline CER/WER stats from a results.json file."""
    if not baseline_path:
        return None
    if not os.path.isfile(baseline_path):
        print(f"Warning: baseline file not found: {baseline_path}")
        return None
    try:
        with open(baseline_path) as f:
            data = json.load(f)
        return {
            "cer_mean": float(data["cer"]["mean"]),
            "cer_std": float(data["cer"].get("std", 0.0)),
            "wer_mean": float(data["wer"]["mean"]),
            "wer_std": float(data["wer"].get("std", 0.0)),
        }
    except Exception as e:
        print(f"Warning: failed to load baseline {baseline_path}: {e}")
        return None


def baseline_for_stage(stage_or_key: str, baseline_ar: dict | None, baseline_ctc: dict | None) -> dict | None:
    """Select baseline based on stage family.

    AR-family: stageA, stageC, stageD1, stageD2
    CTC-family: stageB, stageD3
    """
    if stage_or_key.startswith(("stageA", "stageC", "stageD1", "stageD2")):
        return baseline_ar
    if stage_or_key.startswith(("stageB", "stageD3")):
        return baseline_ctc
    return baseline_ar or baseline_ctc


def compute_baseline_from_group(groups: dict[str, list[dict]], group_key: str) -> dict | None:
    """Compute a baseline dict from an existing grouped experiment key."""
    grp = groups.get(group_key)
    if not grp:
        return None
    stats = compute_cv_stats(grp)
    return {
        "cer_mean": stats["cer_mean"],
        "cer_std": stats["cer_std"],
        "wer_mean": stats["wer_mean"],
        "wer_std": stats["wer_std"],
    }


def print_main_table(groups: dict[str, list[dict]], outdir: str, baseline_ar: dict | None, baseline_ctc: dict | None):
    """Print the main ablation table."""
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
    if baseline_ar:
        print(f"Baseline[AR]  CER={baseline_ar['cer_mean']:.4f}  WER={baseline_ar['wer_mean']:.4f}")
    if baseline_ctc:
        print(f"Baseline[CTC] CER={baseline_ctc['cer_mean']:.4f}  WER={baseline_ctc['wer_mean']:.4f}")
    show_delta = bool(baseline_ar or baseline_ctc)
    if show_delta:
        header = (
            f"{'Row':<30s} {'CER↓':>12s} {'ΔCER':>10s} {'WER↓':>12s} {'ΔWER':>10s} "
            f"{'NED P95':>10s} {'NED P99':>10s} {'EOS%':>8s} {'LenRat':>8s} {'ms/smp':>8s} {'Config':>30s}"
        )
    else:
        header = (
            f"{'Row':<30s} {'CER↓':>12s} {'WER↓':>12s} {'NED P95':>10s} {'NED P99':>10s} "
            f"{'EOS%':>8s} {'LenRat':>8s} {'ms/smp':>8s} {'Config':>30s}"
        )
    print(header)
    print("-" * 100)

    def print_row(label: str, key: str | None, group: list[dict] | None):
        if group is None:
            print(f"{label:<30s}  {'N/A':>12s}")
            return
        stats = compute_cv_stats(group)
        n = stats["n_folds"]
        cer_s = f"{stats['cer_mean']:.4f}" if n <= 1 else f"{stats['cer_mean']:.4f}±{stats['cer_std']:.4f}"
        wer_s = f"{stats['wer_mean']:.4f}" if n <= 1 else f"{stats['wer_mean']:.4f}±{stats['wer_std']:.4f}"
        baseline = baseline_for_stage(key or "", baseline_ar, baseline_ctc)
        if show_delta:
            if baseline:
                dcer = stats["cer_mean"] - baseline["cer_mean"]
                dwer = stats["wer_mean"] - baseline["wer_mean"]
                dcer_s = f"{dcer:>10.4f}"
                dwer_s = f"{dwer:>10.4f}"
            else:
                dcer_s = f"{'N/A':>10s}"
                dwer_s = f"{'N/A':>10s}"
            print(
                f"{label:<30s} {cer_s:>12s} {dcer_s} {wer_s:>12s} {dwer_s} "
                f"{stats['ned_p95_mean']:>10.4f} {stats['ned_p99_mean']:>10.4f} "
                f"{stats['early_eos_rate']:>8.3f} {stats['length_ratio_mean']:>8.3f} "
                f"{stats['runtime_ms']:>8.1f} {(key or ''):>30s}"
            )
        else:
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
            stats = compute_cv_stats(grp)
            baseline = baseline_for_stage(key or "", baseline_ar, baseline_ctc)
            if baseline:
                stats["delta_cer_mean"] = stats["cer_mean"] - baseline["cer_mean"]
                stats["delta_wer_mean"] = stats["wer_mean"] - baseline["wer_mean"]
            table_data[label] = {**stats, "config_key": key}

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "main_table.json"), "w") as f:
        json.dump(table_data, f, indent=2)


def print_sweep_tables(groups: dict[str, list[dict]], outdir: str, baseline_ar: dict | None, baseline_ctc: dict | None):
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

        print(f"\n{'-' * 80}")
        print(f"  {stage_names.get(stage, stage)}")
        print(f"{'-' * 80}")
        stage_baseline = baseline_for_stage(stage, baseline_ar, baseline_ctc)
        if stage_baseline is baseline_ctc:
            print(f"  Baseline: CTC  CER={stage_baseline['cer_mean']:.4f}  WER={stage_baseline['wer_mean']:.4f}")
        elif stage_baseline is baseline_ar:
            print(f"  Baseline: AR   CER={stage_baseline['cer_mean']:.4f}  WER={stage_baseline['wer_mean']:.4f}")
        else:
            print("  Baseline: N/A")
        if baseline_ar or baseline_ctc:
            print(f"  {'Config':<50s} {'CER↓':>10s} {'ΔCER':>8s} {'WER↓':>10s} {'ΔWER':>8s} {'NED P95':>8s} {'ms/smp':>8s}")
        else:
            print(f"  {'Config':<50s} {'CER↓':>10s} {'WER↓':>10s} {'NED P95':>8s} {'ms/smp':>8s}")

        sweep_data = []
        for key in sorted(stage_groups.keys()):
            grp = stage_groups[key]
            stats = compute_cv_stats(grp)
            short_key = key.replace(f"{stage}_", "")
            if baseline_ar or baseline_ctc:
                if stage_baseline:
                    dcer = stats["cer_mean"] - stage_baseline["cer_mean"]
                    dwer = stats["wer_mean"] - stage_baseline["wer_mean"]
                    dcer_s = f"{dcer:>8.4f}"
                    dwer_s = f"{dwer:>8.4f}"
                else:
                    dcer_s = f"{'N/A':>8s}"
                    dwer_s = f"{'N/A':>8s}"
                print(
                    f"  {short_key:<50s} {stats['cer_mean']:>10.4f} {dcer_s} "
                    f"{stats['wer_mean']:>10.4f} {dwer_s} {stats['ned_p95_mean']:>8.4f} {stats['runtime_ms']:>8.1f}"
                )
                if stage_baseline:
                    sweep_data.append({"key": key, **stats, "delta_cer_mean": dcer, "delta_wer_mean": dwer})
                else:
                    sweep_data.append({"key": key, **stats})
            else:
                print(
                    f"  {short_key:<50s} {stats['cer_mean']:>10.4f} {stats['wer_mean']:>10.4f} "
                    f"{stats['ned_p95_mean']:>8.4f} {stats['runtime_ms']:>8.1f}"
                )
                sweep_data.append({"key": key, **stats})

        with open(os.path.join(outdir, f"sweep_{stage}.json"), "w") as f:
            json.dump(sweep_data, f, indent=2)


def print_lm_weight_data(groups: dict[str, list[dict]], outdir: str, baseline_ar: dict | None, baseline_ctc: dict | None):
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
            if baseline_ctc:
                stats["delta_cer_mean"] = stats["cer_mean"] - baseline_ctc["cer_mean"]
                stats["delta_wer_mean"] = stats["wer_mean"] - baseline_ctc["wer_mean"]
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
            if baseline_ar:
                stats["delta_cer_mean"] = stats["cer_mean"] - baseline_ar["cer_mean"]
                stats["delta_wer_mean"] = stats["wer_mean"] - baseline_ar["wer_mean"]
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
            if baseline_ar:
                stats["delta_cer_mean"] = stats["cer_mean"] - baseline_ar["cer_mean"]
                stats["delta_wer_mean"] = stats["wer_mean"] - baseline_ar["wer_mean"]
            ar_shallow_data.append({"lm_weight": lw, **stats, "config": key})

    plot_data = {
        "ctc_lm": sorted(ctc_lm_data, key=lambda x: x["lm_weight"]),
        "ar_rescore": sorted(ar_rescore_data, key=lambda x: x["lm_weight"]),
        "ar_shallow": sorted(ar_shallow_data, key=lambda x: x["lm_weight"]),
    }
    with open(os.path.join(outdir, "lm_weight_plot_data.json"), "w") as f:
        json.dump(plot_data, f, indent=2)

    print("\n-- LM Weight vs CER (for plotting) --")
    for name, data in plot_data.items():
        if data:
            print(f"\n  {name}:")
            for d in data:
                print(f"    λ={d['lm_weight']:.2f}  CER={d['cer_mean']:.4f}  WER={d['wer_mean']:.4f}")


def generate_latex(groups: dict[str, list[dict]], outdir: str, baseline_ar: dict | None, baseline_ctc: dict | None):
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
    show_delta = bool(baseline_ar or baseline_ctc)
    if show_delta:
        latex.append(r"  \begin{tabular}{l c c c c c c}")
    else:
        latex.append(r"  \begin{tabular}{l c c c c c}")
    latex.append(r"    \toprule")
    if show_delta:
        latex.append(r"    Method & CER$\downarrow$ & $\Delta$CER & WER$\downarrow$ & $\Delta$WER & NED P95 & ms/sample \\")
    else:
        latex.append(r"    Method & CER$\downarrow$ & WER$\downarrow$ & NED P95 & NED P99 & ms/sample \\")
    latex.append(r"    \midrule")

    for label, grp in rows:
        if grp is not None:
            stats = compute_cv_stats(grp)
        elif label in best_configs and best_configs[label] is not None:
            stats = best_configs[label]
        else:
            latex.append(f"    {label} & --- & --- & --- & --- & --- \\")
            continue

        n = stats["n_folds"]
        if n > 1:
            cer_s = f"${stats['cer_mean']:.3f} \\pm {stats['cer_std']:.3f}$"
            wer_s = f"${stats['wer_mean']:.3f} \\pm {stats['wer_std']:.3f}$"
        else:
            cer_s = f"${stats['cer_mean']:.3f}$"
            wer_s = f"${stats['wer_mean']:.3f}$"

        row_baseline = baseline_ctc if label.startswith("CTC") else baseline_ar
        if show_delta and row_baseline:
            dcer = stats["cer_mean"] - row_baseline["cer_mean"]
            dwer = stats["wer_mean"] - row_baseline["wer_mean"]
            latex.append(
                f"    {label} & {cer_s} & ${dcer:.3f}$ & {wer_s} & ${dwer:.3f}$ "
                f"& ${stats['ned_p95_mean']:.3f}$ & ${stats['runtime_ms']:.1f}$ \\\\"
            )
        elif show_delta:
            latex.append(
                f"    {label} & {cer_s} & --- & {wer_s} & --- "
                f"& ${stats['ned_p95_mean']:.3f}$ & ${stats['runtime_ms']:.1f}$ \\\\"
            )
        else:
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
    parser.add_argument(
        "--baseline_json", type=str, default=None,
        help="Deprecated fallback baseline path (applied to both AR/CTC if specific baselines are not given)",
    )
    parser.add_argument(
        "--baseline_ar_json", type=str, default=None,
        help="AR baseline results.json (used for stages A/C/D1/D2)",
    )
    parser.add_argument(
        "--baseline_ctc_json", type=str, default=None,
        help="CTC baseline results.json (used for stages B/D3)",
    )
    args = parser.parse_args()

    baseline_fallback = load_baseline(args.baseline_json)
    baseline_ar = load_baseline(args.baseline_ar_json) or baseline_fallback
    baseline_ctc = load_baseline(args.baseline_ctc_json) or baseline_fallback

    # Load results
    all_results = load_all_results(args.results_dir)
    print(f"Loaded {len(all_results)} result files")

    if not all_results:
        print("No results found! Run the decode study first.")
        sys.exit(1)

    # Group by experiment (across folds)
    groups = group_by_experiment(all_results)
    print(f"Found {len(groups)} unique experiments across folds")

    if baseline_ar is None:
        baseline_ar = compute_baseline_from_group(groups, "stageA0_ar_greedy")
        if baseline_ar:
            print("Note: Using stageA0_ar_greedy as AR baseline.")
    if baseline_ctc is None:
        baseline_ctc = compute_baseline_from_group(groups, "stageB1_ctc_greedy")
        if baseline_ctc:
            print("Note: Using stageB1_ctc_greedy as CTC baseline.")

    # Print detailed sweep tables
    print_sweep_tables(groups, args.outdir, baseline_ar, baseline_ctc)

    # Print main ablation table
    print_main_table(groups, args.outdir, baseline_ar, baseline_ctc)

    # LM weight data for plots
    print_lm_weight_data(groups, args.outdir, baseline_ar, baseline_ctc)

    # Generate LaTeX
    generate_latex(groups, args.outdir, baseline_ar, baseline_ctc)

    # Save full summary
    summary = {}
    for key, grp in sorted(groups.items()):
        stats = compute_cv_stats(grp)
        row_baseline = baseline_for_stage(key, baseline_ar, baseline_ctc)
        if row_baseline:
            stats["delta_cer_mean"] = stats["cer_mean"] - row_baseline["cer_mean"]
            stats["delta_wer_mean"] = stats["wer_mean"] - row_baseline["wer_mean"]
        summary[key] = stats
    with open(os.path.join(args.outdir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {os.path.join(args.outdir, 'full_summary.json')}")


if __name__ == "__main__":
    main()