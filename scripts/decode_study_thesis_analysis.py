#!/usr/bin/env python3
"""
Decoding Study — Comprehensive Analysis for Thesis
====================================================
Produces:
  1. Main ablation table (Table: cross-model comparison)
  2. Per-stage sweep heatmaps / bar charts (Figures)
  3. CER vs LM weight λ sensitivity plot
  4. Runtime vs CER Pareto plot
  5. LaTeX tables ready for thesis insertion
  6. JSON summaries for programmatic access

Usage:
    python scripts/decode_study_thesis_analysis.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Try matplotlib; fall back gracefully on headless systems ──
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available; skipping figure generation")


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work")
RESULT_DIRS = {
    "hybrid": BASE / "results/hwr2/decode_study_hybrid",
    "rewi_ctc": BASE / "results/hwr2/decode_study_rewi_ctc",
    "ar": BASE / "results/hwr2/decode_study_ar",
}
OUTDIR = BASE / "results/hwr2/decode_study_thesis_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTDIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_all(results_dir: Path) -> dict[str, list[dict]]:
    """Load metrics.json files grouped by experiment (fold removed)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    pattern = str(results_dir / "**" / "metrics.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            m = json.load(f)
        tag = os.path.basename(os.path.dirname(path))
        key = re.sub(r"__fold\d+$", "", tag)
        fold_m = re.search(r"fold(\d+)", tag)
        m["_fold"] = int(fold_m.group(1)) if fold_m else -1
        m["_key"] = key
        groups[key].append(m)
    return dict(groups)


def cv_stats(group: list[dict]) -> dict:
    """Mean ± std across folds."""
    n = len(group)
    cers = [r["cer"] for r in group]
    wers = [r["wer"] for r in group]
    p95 = [r.get("ned_p95", np.nan) for r in group]
    p99 = [r.get("ned_p99", np.nan) for r in group]
    rt = [r.get("runtime_per_sample_ms", 0) for r in group]
    eos = [r.get("early_eos_rate", np.nan) for r in group]
    lr = [r.get("length_ratio_mean", np.nan) for r in group]
    return {
        "n": n,
        "cer": float(np.mean(cers)),
        "cer_std": float(np.std(cers)) if n > 1 else 0.0,
        "wer": float(np.mean(wers)),
        "wer_std": float(np.std(wers)) if n > 1 else 0.0,
        "p95": float(np.nanmean(p95)),
        "p99": float(np.nanmean(p99)),
        "rt_ms": float(np.mean(rt)),
        "eos": float(np.nanmean(eos)),
        "lr": float(np.nanmean(lr)),
    }


def best_in(groups: dict[str, list[dict]], prefix: str, require_5: bool = True) -> tuple[str | None, dict | None]:
    """Find the config with lowest mean CER for a stage prefix."""
    cands = {k: v for k, v in groups.items() if k.startswith(prefix)}
    if require_5:
        cands = {k: v for k, v in cands.items() if len(v) == 5}
    if not cands:
        return None, None
    best_key = min(cands, key=lambda k: np.mean([r["cer"] for r in cands[k]]))
    return best_key, cv_stats(cands[best_key])


# ═══════════════════════════════════════════════════════════════════════
# 1. Cross-Model Main Ablation Table
# ═══════════════════════════════════════════════════════════════════════

def build_main_table() -> dict:
    """Build the main ablation table across all three models."""
    table = {}
    for model_name, rdir in RESULT_DIRS.items():
        groups = load_all(rdir)
        if not groups:
            continue

        # Baselines
        ar_grdy = cv_stats(groups["stageA0_ar_greedy"]) if "stageA0_ar_greedy" in groups else None
        ctc_grdy = cv_stats(groups["stageB1_ctc_greedy"]) if "stageB1_ctc_greedy" in groups else None

        # Best per stage
        _, ar_cal = best_in(groups, "stageA2_")
        # Also check A3 (EOS sweep without min-len which is catastrophic)
        a3_cands = {k: v for k, v in groups.items()
                    if k.startswith("stageA3_ar_eos") and len(v) == 5}
        if a3_cands:
            a3_best_key = min(a3_cands, key=lambda k: np.mean([r["cer"] for r in a3_cands[k]]))
            a3_stats = cv_stats(a3_cands[a3_best_key])
            if ar_cal is None or a3_stats["cer"] < ar_cal["cer"]:
                ar_cal = a3_stats

        _, ctc_beam = best_in(groups, "stageB2_")
        _, ctc_lm = best_in(groups, "stageB3_")
        _, ar_rescore_kenlm = best_in(groups, "stageC1_")
        _, ar_shallow_kenlm = best_in(groups, "stageC2_")
        _, ar_rescore_neural = best_in(groups, "stageD1_")
        _, ar_shallow_neural = best_in(groups, "stageD2_")
        _, ctc_neural = best_in(groups, "stageD3_")

        table[model_name] = {
            "ar_greedy": ar_grdy,
            "ar_calibrated": ar_cal,
            "ctc_greedy": ctc_grdy,
            "ctc_beam": ctc_beam,
            "ctc_kenlm": ctc_lm,
            "ar_rescore_kenlm": ar_rescore_kenlm,
            "ar_shallow_kenlm": ar_shallow_kenlm,
            "ar_rescore_neural": ar_rescore_neural,
            "ar_shallow_neural": ar_shallow_neural,
            "ctc_neural": ctc_neural,
        }
    return table


def print_main_table(table: dict):
    """Print cross-model comparison table."""
    rows = [
        ("AR greedy", "ar_greedy"),
        ("AR calibrated beam", "ar_calibrated"),
        ("AR + KenLM (rescore)", "ar_rescore_kenlm"),
        ("AR + KenLM (shallow)", "ar_shallow_kenlm"),
        ("AR + Neural LM (rescore)", "ar_rescore_neural"),
        ("CTC greedy", "ctc_greedy"),
        ("CTC beam", "ctc_beam"),
        ("CTC + KenLM", "ctc_kenlm"),
    ]

    print("\n" + "=" * 130)
    print("CROSS-MODEL DECODING STUDY — MAIN ABLATION TABLE")
    print("=" * 130)
    header = f"{'Decoding Method':<30s}"
    for model in ["hybrid", "ar", "rewi_ctc"]:
        header += f" | {'CER':>8s} {'WER':>8s}"
    print(header)
    print("-" * 130)

    for label, key in rows:
        line = f"{label:<30s}"
        for model in ["hybrid", "ar", "rewi_ctc"]:
            s = table.get(model, {}).get(key)
            if s:
                line += f" | {s['cer']:>8.4f} {s['wer']:>8.4f}"
            else:
                line += f" | {'---':>8s} {'---':>8s}"
        print(line)
        if key == "ar_shallow_kenlm" or key == "ar_rescore_neural":
            print("-" * 130)

    print("=" * 130)


def generate_main_latex(table: dict):
    """Generate LaTeX for the main cross-model ablation table."""
    rows = [
        ("AR greedy", "ar_greedy", "A0"),
        ("AR calibrated beam", "ar_calibrated", "A2/A3"),
        ("AR + KenLM rescore", "ar_rescore_kenlm", "C1"),
        ("AR + KenLM shallow", "ar_shallow_kenlm", "C2"),
        ("AR + Neural LM rescore", "ar_rescore_neural", "D1"),
        ("CTC greedy", "ctc_greedy", "B1"),
        ("CTC beam (no LM)", "ctc_beam", "B2"),
        ("CTC + KenLM", "ctc_kenlm", "B3"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Decoding study: cross-model CER/WER comparison on OnHW500 WI (word-level, 5-fold CV).}")
    lines.append(r"  \label{tab:decode_study_cross_model}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l c @{\hskip 4pt} c | c @{\hskip 4pt} c | c @{\hskip 4pt} c}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{2}{c|}{Hybrid (CTC+AR)} & \multicolumn{2}{c|}{AR-only} & \multicolumn{2}{c}{REWI (CTC)} \\")
    lines.append(r"    Method & CER & WER & CER & WER & CER & WER \\")
    lines.append(r"    \midrule")

    for label, key, stage in rows:
        parts = [f"    {label}"]
        for model in ["hybrid", "ar", "rewi_ctc"]:
            s = table.get(model, {}).get(key)
            if s and s["n"] >= 5:
                parts.append(f"${s['cer']:.3f}" + r" \pm " + f"{s['cer_std']:.3f}$")
                parts.append(f"${s['wer']:.3f}" + r" \pm " + f"{s['wer_std']:.3f}$")
            elif s:
                parts.append(f"${s['cer']:.3f}$")
                parts.append(f"${s['wer']:.3f}$")
            else:
                parts.append("---")
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")
        if key == "ar_shallow_kenlm":
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(OUTDIR / "table_cross_model.tex", "w") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX saved to {OUTDIR / 'table_cross_model.tex'}")
    return latex


# ═══════════════════════════════════════════════════════════════════════
# 2. Per-Stage Sweep Tables
# ═══════════════════════════════════════════════════════════════════════

def generate_sweep_latex(model_name: str, groups: dict[str, list[dict]]):
    """Generate per-stage sweep LaTeX tables for a given model."""

    # Stage A — AR Calibration
    stage_a_data = []
    for key in sorted(groups.keys()):
        if not key.startswith("stageA"):
            continue
        if "minlen" in key:
            continue  # catastrophic, skip
        g = groups[key]
        if len(g) < 5:
            continue
        s = cv_stats(g)
        # Parse parameters
        short = key.replace("stageA0_", "").replace("stageA1_", "").replace("stageA2_", "").replace("stageA3_", "")
        stage_a_data.append({"key": key, "short": short, **s})

    if stage_a_data:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(f"  \\caption{{AR calibration sweep ({model_name} model).}}")
        lines.append(f"  \\label{{tab:decode_ar_cal_{model_name}}}")
        lines.append(r"  \small")
        lines.append(r"  \begin{tabular}{l c c c c}")
        lines.append(r"    \toprule")
        lines.append(r"    Configuration & CER$\downarrow$ & WER$\downarrow$ & NED P95 & ms/sample \\")
        lines.append(r"    \midrule")
        for d in stage_a_data:
            cer_s = f"${d['cer']:.4f}" + r" \pm " + f"{d['cer_std']:.4f}$"
            lines.append(f"    {d['short']} & {cer_s} & ${d['wer']:.4f}$ & ${d['p95']:.3f}$ & ${d['rt_ms']:.0f}$ \\\\")
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        with open(OUTDIR / f"table_ar_calibration_{model_name}.tex", "w") as f:
            f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# 3. CER vs LM Weight Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_cer_vs_lm_weight():
    """Plot CER vs LM weight λ for CTC+KenLM and AR+KenLM across models."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CTC + KenLM (hybrid and rewi_ctc)
    ax = axes[0]
    ax.set_title("CTC + KenLM: CER vs LM Weight λ", fontsize=13)
    for model_name, color, marker in [("hybrid", "#2196F3", "o"), ("rewi_ctc", "#4CAF50", "s")]:
        groups = load_all(RESULT_DIRS[model_name])
        points = []
        for key, grp in groups.items():
            if not key.startswith("stageB3_") or len(grp) < 5:
                continue
            lw_m = re.search(r"lw([\d.]+)", key)
            ib_m = re.search(r"ib([\d.-]+)", key)
            if lw_m and ib_m:
                lw = float(lw_m.group(1))
                ib = float(ib_m.group(1))
                s = cv_stats(grp)
                points.append((lw, ib, s["cer"], s["cer_std"]))

        # Group by insertion_bonus, plot best IB per lw
        by_lw = defaultdict(list)
        for lw, ib, cer, std in points:
            by_lw[lw].append((ib, cer, std))

        xs, ys, errs = [], [], []
        for lw in sorted(by_lw.keys()):
            best = min(by_lw[lw], key=lambda x: x[1])
            xs.append(lw)
            ys.append(best[1])
            errs.append(best[2])

        # Add greedy baseline
        grdy = groups.get("stageB1_ctc_greedy")
        if grdy and len(grdy) >= 5:
            gs = cv_stats(grdy)
            ax.axhline(gs["cer"], color=color, linestyle="--", alpha=0.5, label=f"{model_name} greedy")

        ax.errorbar(xs, ys, yerr=errs, marker=marker, color=color,
                    label=f"{model_name} best", capsize=3, linewidth=1.5)

    ax.set_xlabel("LM Weight λ")
    ax.set_ylabel("CER (↓)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # AR + KenLM rescoring (hybrid and ar)
    ax = axes[1]
    ax.set_title("AR + KenLM Rescore: CER vs LM Weight λ", fontsize=13)
    for model_name, color, marker in [("hybrid", "#2196F3", "o"), ("ar", "#FF5722", "^")]:
        groups = load_all(RESULT_DIRS[model_name])
        points = []
        for key, grp in groups.items():
            if not key.startswith("stageC1_") or len(grp) < 5:
                continue
            lw_m = re.search(r"lw([\d.]+)", key)
            lb_m = re.search(r"lb([\d.-]+)", key)
            if lw_m and lb_m:
                lw = float(lw_m.group(1))
                lb = float(lb_m.group(1))
                s = cv_stats(grp)
                points.append((lw, lb, s["cer"], s["cer_std"]))

        # Best lb per lw (exclude positive lb which is catastrophic)
        by_lw = defaultdict(list)
        for lw, lb, cer, std in points:
            if lb <= 0:  # positive lb causes degenerate results
                by_lw[lw].append((lb, cer, std))

        xs, ys, errs = [], [], []
        for lw in sorted(by_lw.keys()):
            if by_lw[lw]:
                best = min(by_lw[lw], key=lambda x: x[1])
                xs.append(lw)
                ys.append(best[1])
                errs.append(best[2])

        grdy = groups.get("stageA0_ar_greedy")
        if grdy and len(grdy) >= 5:
            gs = cv_stats(grdy)
            ax.axhline(gs["cer"], color=color, linestyle="--", alpha=0.5, label=f"{model_name} greedy")

        ax.errorbar(xs, ys, yerr=errs, marker=marker, color=color,
                    label=f"{model_name} best", capsize=3, linewidth=1.5)

    ax.set_xlabel("LM Weight λ")
    ax.set_ylabel("CER (↓)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cer_vs_lm_weight.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "cer_vs_lm_weight.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'cer_vs_lm_weight.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# 4. AR Beam Size & Length Norm Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_ar_calibration():
    """Plot AR calibration sweeps: beam size and length normalisation."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Beam size sweep
    ax = axes[0]
    ax.set_title("AR Beam Size Sweep", fontsize=13)
    for model_name, color, marker in [("hybrid", "#2196F3", "o"), ("ar", "#FF5722", "^")]:
        groups = load_all(RESULT_DIRS[model_name])
        beams, cers, rts = [], [], []
        for B in [1, 2, 4, 8, 16]:
            key = f"stageA1_ar_beam_B{B}"
            if key in groups and len(groups[key]) >= 5:
                s = cv_stats(groups[key])
                beams.append(B)
                cers.append(s["cer"])
                rts.append(s["rt_ms"])
        ax.plot(beams, cers, marker=marker, color=color, label=model_name, linewidth=1.5)
    ax.set_xlabel("Beam Size B")
    ax.set_ylabel("CER (↓)")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.legend()
    ax.grid(alpha=0.3)

    # Length normalisation sweep
    ax = axes[1]
    ax.set_title("AR Length Normalisation α Sweep", fontsize=13)
    for model_name, color, marker in [("hybrid", "#2196F3", "o"), ("ar", "#FF5722", "^")]:
        groups = load_all(RESULT_DIRS[model_name])
        alphas, cers = [], []
        for a in [0.0, 0.2, 0.4, 0.6, 0.8]:
            key = f"stageA2_ar_lenorm_a{a}"
            if key in groups and len(groups[key]) >= 5:
                s = cv_stats(groups[key])
                alphas.append(a)
                cers.append(s["cer"])
        ax.plot(alphas, cers, marker=marker, color=color, label=model_name, linewidth=1.5)
    ax.set_xlabel("Length Normalisation α")
    ax.set_ylabel("CER (↓)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "ar_calibration.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "ar_calibration.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'ar_calibration.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# 5. Runtime vs CER Pareto Plot
# ═══════════════════════════════════════════════════════════════════════

def plot_runtime_vs_cer():
    """Plot runtime vs CER Pareto frontier across methods."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Runtime vs CER Trade-off", fontsize=14)

    colors = {"hybrid": "#2196F3", "ar": "#FF5722", "rewi_ctc": "#4CAF50"}
    markers = {"hybrid": "o", "ar": "^", "rewi_ctc": "s"}

    for model_name in ["hybrid", "ar", "rewi_ctc"]:
        groups = load_all(RESULT_DIRS[model_name])
        points = []
        for key, grp in groups.items():
            if len(grp) < 5:
                continue
            if "minlen" in key:
                continue  # skip catastrophic configs
            s = cv_stats(grp)
            if s["cer"] > 0.15:
                continue  # skip outliers
            # Determine label
            if "greedy" in key:
                lbl = "greedy"
            elif "beam_lm" in key or "shallow" in key:
                lbl = "+LM"
            elif "rescore" in key:
                lbl = "rescore"
            elif "beam" in key:
                lbl = "beam"
            else:
                lbl = "other"
            points.append((s["rt_ms"], s["cer"], key, lbl))

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, c=colors[model_name], marker=markers[model_name],
                       alpha=0.5, s=20, label=model_name)

            # Annotate Pareto front (greedy and best overall)
            greedy_pts = [p for p in points if p[3] == "greedy"]
            if greedy_pts:
                gp = greedy_pts[0]
                ax.annotate(f"{model_name}\ngreedy", (gp[0], gp[1]),
                            fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Runtime (ms/sample)")
    ax.set_ylabel("CER (↓)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "runtime_vs_cer.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "runtime_vs_cer.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'runtime_vs_cer.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# 6. CTC Beam + LM Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_ctc_lm_heatmap():
    """Heatmap of CER for CTC beam + LM sweep (lw × ib) for hybrid model."""
    if not HAS_MPL:
        return

    groups = load_all(RESULT_DIRS["hybrid"])
    lws = [0.1, 0.2, 0.4, 0.8, 1.2]
    ibs = [-0.5, 0.0, 0.5]
    matrix = np.full((len(lws), len(ibs)), np.nan)

    for i, lw in enumerate(lws):
        for j, ib in enumerate(ibs):
            key = f"stageB3_ctc_beam_lm_lw{lw}_ib{ib}"
            if key in groups and len(groups[key]) >= 5:
                matrix[i, j] = cv_stats(groups[key])["cer"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(ibs)))
    ax.set_xticklabels([f"{ib}" for ib in ibs])
    ax.set_yticks(range(len(lws)))
    ax.set_yticklabels([f"{lw}" for lw in lws])
    ax.set_xlabel("Insertion Bonus β")
    ax.set_ylabel("LM Weight λ")
    ax.set_title("CTC + KenLM: CER Heatmap (Hybrid Model)")

    # Annotate cells
    for i in range(len(lws)):
        for j in range(len(ibs)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        fontsize=9, color="white" if matrix[i, j] > 0.095 else "black")

    plt.colorbar(im, ax=ax, label="CER")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ctc_lm_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "ctc_lm_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'ctc_lm_heatmap.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# 7. AR Rescoring Heatmap (KenLM, lw × lb for N=8)
# ═══════════════════════════════════════════════════════════════════════

def plot_ar_rescore_heatmap():
    """Heatmap of AR rescoring CER for lw × lb with N=8."""
    if not HAS_MPL:
        return

    groups = load_all(RESULT_DIRS["hybrid"])
    lws = [0.1, 0.2, 0.4, 0.8]
    lbs = [-0.2, 0.0, 0.2]
    matrix = np.full((len(lws), len(lbs)), np.nan)

    for i, lw in enumerate(lws):
        for j, lb in enumerate(lbs):
            key = f"stageC1_ar_rescore_N8_lw{lw}_lb{lb}"
            if key in groups and len(groups[key]) >= 5:
                matrix[i, j] = cv_stats(groups[key])["cer"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(lbs)))
    ax.set_xticklabels([f"{lb}" for lb in lbs])
    ax.set_yticks(range(len(lws)))
    ax.set_yticklabels([f"{lw}" for lw in lws])
    ax.set_xlabel("Length Bonus β")
    ax.set_ylabel("LM Weight λ")
    ax.set_title("AR + KenLM Rescore (N=8): CER Heatmap (Hybrid)")

    for i in range(len(lws)):
        for j in range(len(lbs)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        fontsize=9, color="white" if matrix[i, j] > 0.08 else "black")

    plt.colorbar(im, ax=ax, label="CER")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ar_rescore_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "ar_rescore_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {FIG_DIR / 'ar_rescore_heatmap.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# 8. Summary JSON
# ═══════════════════════════════════════════════════════════════════════

def save_summary_json(table: dict):
    """Save comprehensive summary as JSON."""
    # Convert numpy types
    def to_json(obj):
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    json_table = {}
    for model, entries in table.items():
        json_table[model] = {}
        for key, val in entries.items():
            if val is not None:
                json_table[model][key] = {k: to_json(v) for k, v in val.items()}
            else:
                json_table[model][key] = None

    with open(OUTDIR / "cross_model_summary.json", "w") as f:
        json.dump(json_table, f, indent=2)
    print(f"Summary JSON saved: {OUTDIR / 'cross_model_summary.json'}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Decoding Study — Thesis Analysis")
    print("=" * 60)

    # 1. Main table
    table = build_main_table()
    print_main_table(table)
    latex = generate_main_latex(table)
    save_summary_json(table)

    # 2. Per-model sweep tables
    for model_name in ["hybrid", "ar"]:
        groups = load_all(RESULT_DIRS[model_name])
        generate_sweep_latex(model_name, groups)

    # 3. Figures
    plot_cer_vs_lm_weight()
    plot_ar_calibration()
    plot_runtime_vs_cer()
    plot_ctc_lm_heatmap()
    plot_ar_rescore_heatmap()

    print("\n" + "=" * 60)
    print("Analysis complete. Outputs in:", OUTDIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
