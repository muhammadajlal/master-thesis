#!/usr/bin/env python3
"""
Decoding Study — Thesis Figures (HWRFormer / ar_transformer_xs run)
===================================================================
Regenerates the four decoding-study thesis figures from the CURRENT
xs decode runs (July recompute), replacing the stale HWRFormer-L
(March) versions:

  1. cer_vs_lm_weight.pdf   — CER vs LM weight (CTC+KenLM, AR+KenLM rescore)
  2. runtime_vs_cer.pdf     — runtime/CER trade-off scatter
  3. ctc_lm_heatmap.pdf     — REWI CTC+KenLM sweep heatmap + hybrid CTC head bars
  4. ar_rescore_heatmap.pdf — AR decoding methods (Stage C) bar comparison

Data sources (all on disk, no training):
  - results/hwr2/decode_study_xs_full_hybrid   (5 folds, confirmation configs)
  - results/hwr2/decode_study_xs_full_ar       (5 folds, confirmation configs)
  - results/hwr2/decode_study_rewi_ctc         (REWI baseline; model-independent
                                                of the L->xs change, full B2/B3 sweep)

NOTE: the xs runs evaluate a single configuration per stage (the operating
points selected on the earlier sweep), so no lambda x beta grid exists for
the hybrid xs model. The former hybrid-L heatmaps are therefore replaced by
(a) the REWI sweep heatmap and (b) bar comparisons of the xs operating points.

Usage:
    python scripts/decode_study_thesis_analysis_xs.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work")
RESULT_DIRS = {
    "hybrid": BASE / "results/hwr2/decode_study_xs_full_hybrid",
    "ar": BASE / "results/hwr2/decode_study_xs_full_ar",
    "rewi_ctc": BASE / "results/hwr2/decode_study_rewi_ctc",
}

# Display names (no internal codenames in figure text)
DISPLAY = {"hybrid": "Hybrid", "ar": "AR-only", "rewi_ctc": "REWI"}
COLORS = {"hybrid": "#2196F3", "ar": "#FF5722", "rewi_ctc": "#4CAF50"}
MARKERS = {"hybrid": "o", "ar": "^", "rewi_ctc": "s"}


def load_all(results_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    pattern = str(results_dir / "**" / "metrics.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            m = json.load(f)
        tag = os.path.basename(os.path.dirname(path))
        key = re.sub(r"__fold\d+$", "", tag)
        groups[key].append(m)
    return dict(groups)


def cv_stats(group: list[dict]) -> dict:
    n = len(group)
    cers = [r["cer"] for r in group]
    wers = [r["wer"] for r in group]
    rt = [r.get("runtime_per_sample_ms", 0) for r in group]
    return {
        "n": n,
        "cer": float(np.mean(cers)),
        "cer_std": float(np.std(cers)) if n > 1 else 0.0,
        "wer": float(np.mean(wers)),
        "wer_std": float(np.std(wers)) if n > 1 else 0.0,
        "rt_ms": float(np.mean(rt)),
    }


def stats_of(groups, key):
    if key in groups and len(groups[key]) >= 5:
        return cv_stats(groups[key])
    return None


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: CER vs LM weight
# ═══════════════════════════════════════════════════════════════════════

def plot_cer_vs_lm_weight(g_hyb, g_ar, g_rewi, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: CTC + KenLM ──
    ax = axes[0]
    ax.set_title("CTC + KenLM: CER vs LM Weight λ", fontsize=13)

    # REWI: full lw x ib sweep — best insertion bonus per lw
    points = []
    for key, grp in g_rewi.items():
        if not key.startswith("stageB3_") or len(grp) < 5:
            continue
        lw_m = re.search(r"lw([\d.]+)", key)
        ib_m = re.search(r"ib([\d.-]+)", key)
        if lw_m and ib_m:
            s = cv_stats(grp)
            points.append((float(lw_m.group(1)), float(ib_m.group(1)),
                           s["cer"], s["cer_std"]))
    by_lw = defaultdict(list)
    for lw, ib, cer, std in points:
        by_lw[lw].append((ib, cer, std))
    xs_, ys_, errs = [], [], []
    for lw in sorted(by_lw):
        best = min(by_lw[lw], key=lambda x: x[1])
        xs_.append(lw)
        ys_.append(best[1])
        errs.append(best[2])

    rewi_grdy = stats_of(g_rewi, "stageB1_ctc_greedy")
    if rewi_grdy:
        ax.axhline(rewi_grdy["cer"], color=COLORS["rewi_ctc"], linestyle="--",
                   alpha=0.6, label="REWI greedy")
    ax.errorbar(xs_, ys_, yerr=errs, marker="s", color=COLORS["rewi_ctc"],
                label="REWI + KenLM (best β per λ)", capsize=3, linewidth=1.5)

    # Hybrid CTC head: greedy baseline + single evaluated KenLM operating point
    hyb_ctc_grdy = stats_of(g_hyb, "stageB1_ctc_greedy")
    if hyb_ctc_grdy:
        ax.axhline(hyb_ctc_grdy["cer"], color=COLORS["hybrid"], linestyle="--",
                   alpha=0.6, label="Hybrid CTC-head greedy")
    hyb_lm = stats_of(g_hyb, "stageB3_ctc_kenlm_lw0p4")
    if hyb_lm:
        lw_used = g_hyb["stageB3_ctc_kenlm_lw0p4"][0]["config"]["lm_weight"]
        ax.errorbar([lw_used], [hyb_lm["cer"]], yerr=[hyb_lm["cer_std"]],
                    marker="o", color=COLORS["hybrid"], linestyle="none",
                    capsize=3, markersize=8,
                    label=f"Hybrid + KenLM (λ={lw_used})")

    ax.set_xlabel("LM Weight λ")
    ax.set_ylabel("CER (↓)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: AR + KenLM rescoring ──
    ax = axes[1]
    ax.set_title("AR + KenLM Rescore: CER vs LM Weight λ", fontsize=13)
    for model, g in [("hybrid", g_hyb), ("ar", g_ar)]:
        grdy = stats_of(g, "stageA0_ar_greedy")
        if grdy:
            ax.axhline(grdy["cer"], color=COLORS[model], linestyle="--",
                       alpha=0.6, label=f"{DISPLAY[model]} greedy")
        key = "stageC1_ar_kenlm_rescore_N8_lw0p1"
        res = stats_of(g, key)
        if res:
            lw_used = g[key][0]["config"]["lm_weight"]
            ax.errorbar([lw_used], [res["cer"]], yerr=[res["cer_std"]],
                        marker=MARKERS[model], color=COLORS[model],
                        linestyle="none", capsize=3, markersize=8,
                        label=f"{DISPLAY[model]} + KenLM rescore (λ={lw_used})")
    ax.set_xlim(0.0, 0.2)
    ax.set_xlabel("LM Weight λ")
    ax.set_ylabel("CER (↓)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "cer_vs_lm_weight.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(fig_dir / "cer_vs_lm_weight.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'cer_vs_lm_weight.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Runtime vs CER
# ═══════════════════════════════════════════════════════════════════════

def plot_runtime_vs_cer(g_all, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Runtime vs CER Trade-off", fontsize=14)

    annots = []  # (x, y, text, dx, dy, ha, va)
    for model in ["hybrid", "ar", "rewi_ctc"]:
        groups = g_all[model]
        points = []
        for key, grp in groups.items():
            if len(grp) < 5 or "minlen" in key:
                continue
            s = cv_stats(grp)
            if s["cer"] > 0.15:
                continue
            points.append((s["rt_ms"], s["cer"], key))
        if not points:
            continue
        xs_ = [p[0] for p in points]
        ys_ = [p[1] for p in points]
        ax.scatter(xs_, ys_, c=COLORS[model], marker=MARKERS[model],
                   alpha=0.5, s=22, label=DISPLAY[model])

        # collect greedy points for annotation
        for rt, cer, key in points:
            if key == "stageA0_ar_greedy" and model == "hybrid":
                annots.append((rt, cer, "Hybrid greedy", -8, 10, "right", "bottom"))
            elif key == "stageA0_ar_greedy" and model == "ar":
                annots.append((rt, cer, "AR-only greedy", 8, -12, "left", "top"))
            elif key == "stageB1_ctc_greedy" and model == "hybrid":
                annots.append((rt, cer, "Hybrid CTC-head\ngreedy", 8, 0, "left", "center"))
            elif key == "stageB1_ctc_greedy" and model == "rewi_ctc":
                annots.append((rt, cer, "REWI greedy", 8, 0, "left", "center"))

    for x, y, txt, dx, dy, ha, va in annots:
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=ha, va=va)

    ax.set_xlabel("Runtime (ms/sample)")
    ax.set_ylabel("CER (↓)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "runtime_vs_cer.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(fig_dir / "runtime_vs_cer.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'runtime_vs_cer.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: CTC + KenLM — REWI sweep heatmap + hybrid CTC head bars
# ═══════════════════════════════════════════════════════════════════════

def plot_ctc_lm_heatmap(g_hyb, g_rewi, fig_dir: Path):
    lws = [0.1, 0.2, 0.4, 0.8, 1.2]
    ibs = [-0.5, 0.0, 0.5]
    matrix = np.full((len(lws), len(ibs)), np.nan)
    for i, lw in enumerate(lws):
        for j, ib in enumerate(ibs):
            key = f"stageB3_ctc_beam_lm_lw{lw}_ib{ib}"
            s = stats_of(g_rewi, key)
            if s:
                matrix[i, j] = s["cer"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(ibs)))
    ax.set_xticklabels([f"{ib}" for ib in ibs])
    ax.set_yticks(range(len(lws)))
    ax.set_yticklabels([f"{lw}" for lw in lws])
    ax.set_xlabel("Insertion Bonus β")
    ax.set_ylabel("LM Weight λ")
    ax.set_title("REWI: CTC + KenLM CER (λ × β sweep)", fontsize=12)
    thresh = np.nanmin(matrix) + 0.6 * (np.nanmax(matrix) - np.nanmin(matrix))
    for i in range(len(lws)):
        for j in range(len(ibs)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if matrix[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax, label="CER")

    # Right: hybrid CTC-head operating points (single-config xs run)
    labels, vals, stds = [], [], []
    for lbl, key in [("Greedy", "stageB1_ctc_greedy"),
                     ("Beam\n(B=50, no LM)", "stageB2_ctc_beam_B50"),
                     ("+ KenLM\n(λ=0.4, β=0)", "stageB3_ctc_kenlm_lw0p4")]:
        s = stats_of(g_hyb, key)
        if s:
            labels.append(lbl)
            vals.append(s["cer"])
            stds.append(s["cer_std"])
    bars = ax2.bar(labels, vals, yerr=stds, capsize=4,
                   color=["#90CAF9", "#64B5F6", "#1976D2"], width=0.55)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("CER (↓)")
    ax2.set_title("Hybrid CTC head (HWRFormer)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(fig_dir / "ctc_lm_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(fig_dir / "ctc_lm_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'ctc_lm_heatmap.pdf'}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: AR decoding with KenLM (Stage C operating points)
# ═══════════════════════════════════════════════════════════════════════

def plot_ar_rescore(g_hyb, g_ar, fig_dir: Path):
    methods = [
        ("Greedy", "stageA0_ar_greedy"),
        ("Calibrated beam\n(B=4, α=0)", "stageA2_ar_lenorm_a0.0"),
        ("+ KenLM rescore\n(N=8, λ=0.1)", "stageC1_ar_kenlm_rescore_N8_lw0p1"),
        ("+ KenLM shallow\n(B=4, λ=0.2)", "stageC2_ar_kenlm_shallow_B4_lw0p2"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.36
    x = np.arange(len(methods))
    for off, model, g in [(-width / 2, "hybrid", g_hyb), (width / 2, "ar", g_ar)]:
        vals, stds = [], []
        for _, key in methods:
            s = stats_of(g, key)
            vals.append(s["cer"] if s else np.nan)
            stds.append(s["cer_std"] if s else 0.0)
        bars = ax.bar(x + off, vals, width, yerr=stds, capsize=3,
                      color=COLORS[model], alpha=0.85, label=DISPLAY[model],
                      error_kw={"alpha": 0.45, "lw": 1.0})
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                # offset left of the bar centre so the error bar line
                # does not strike through the value label
                ax.text(b.get_x() + b.get_width() / 2 - 0.035, v + 0.0025,
                        f"{v:.4f}", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax.set_ylabel("CER (↓)")
    ax.set_title("AR Decoding with KenLM (Stage C): CER by Method", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(fig_dir / "ar_rescore_heatmap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(fig_dir / "ar_rescore_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'ar_rescore_heatmap.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(
        BASE / "results/hwr2/decode_study_xs_thesis_analysis/figures"))
    args = ap.parse_args()
    fig_dir = Path(args.outdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    g_all = {m: load_all(d) for m, d in RESULT_DIRS.items()}
    g_hyb, g_ar, g_rewi = g_all["hybrid"], g_all["ar"], g_all["rewi_ctc"]

    # sanity printout of headline numbers
    for lbl, g, key in [
        ("hybrid CTC-head greedy", g_hyb, "stageB1_ctc_greedy"),
        ("hybrid CTC+KenLM", g_hyb, "stageB3_ctc_kenlm_lw0p4"),
        ("hybrid AR greedy", g_hyb, "stageA0_ar_greedy"),
        ("AR-only greedy", g_ar, "stageA0_ar_greedy"),
        ("hybrid AR+KenLM rescore", g_hyb, "stageC1_ar_kenlm_rescore_N8_lw0p1"),
        ("REWI greedy", g_rewi, "stageB1_ctc_greedy"),
    ]:
        s = stats_of(g, key)
        print(f"{lbl:30s}: CER={s['cer']:.4f} WER={s['wer']:.4f}" if s else f"{lbl}: MISSING")

    plot_cer_vs_lm_weight(g_hyb, g_ar, g_rewi, fig_dir)
    plot_runtime_vs_cer(g_all, fig_dir)
    plot_ctc_lm_heatmap(g_hyb, g_rewi, fig_dir)
    plot_ar_rescore(g_hyb, g_ar, fig_dir)


if __name__ == "__main__":
    main()
