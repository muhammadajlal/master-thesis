#!/usr/bin/env python3
"""Aggregate Levenshtein histograms across folds.

This script is meant for metrics that *should* be aggregated across folds
(e.g., Levenshtein distance histograms). It reads the per-fold exported JSONs
created by `export_val_full: true` and concatenates all samples.

It plots:
  - Raw Levenshtein distance histogram (y = #samples)
  - Normalized Levenshtein distance histogram where d_norm = d / |ref|

Example (AR-only vs Hybrid AR-decode):

  python analysis/scripts/aggregate_levenshtein_folds.py \
    --folds 0,1,2,3,4 \
    --ar_export_pattern  "results/hwr2/element_word_onhw500_new/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0.json" \
    --hyb_export_pattern "results/hwr2/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0_ar.json" \
    --outdir figures/ar_vs_hybrid/all_folds

Notes:
- Patterns may be exact paths or glob patterns.
- Relative paths are resolved against the repo root (imu-hwr/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Ensure REWI_work is on the path
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

REPO_ROOT = os.path.abspath(os.path.join(PROJ, "..", ".."))

from rewi.analysis.metrics import lev_dist


def _resolve_maybe_relative(path_or_pattern: str) -> str:
    """Resolve a path or glob-pattern relative to repo root if needed."""
    p = str(path_or_pattern)
    # absolute patterns: keep
    if os.path.isabs(p):
        return p
    # if exists relative to CWD: keep
    if glob.glob(p):
        return p
    # try repo root
    p2 = os.path.join(REPO_ROOT, p)
    return p2


def _find_one(pattern: str) -> str:
    """Find exactly one file for a pattern; if multiple, pick newest."""
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    if len(matches) == 1:
        return matches[0]

    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    logger.warning("Multiple matches for pattern; using newest: {}", matches[0])
    return matches[0]


def _parse_folds(s: str) -> list[int]:
    s = str(s).strip()
    if s in {"-1", "all"}:
        return [0, 1, 2, 3, 4]
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


@dataclass
class ExportData:
    preds: list[str]
    labels: list[str]


def load_export_json(path: str) -> ExportData:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    preds = d.get("predictions", [])
    labels = d.get("labels", [])
    if not isinstance(preds, list) or not isinstance(labels, list):
        raise ValueError(f"Bad export format in {path}: expected lists")
    if len(preds) != len(labels):
        raise ValueError(f"Length mismatch in {path}: {len(preds)} preds vs {len(labels)} labels")
    return ExportData(preds=preds, labels=labels)


def compute_distances(preds: list[str], labels: list[str]) -> tuple[list[int], list[float]]:
    raw: list[int] = []
    norm_ref: list[float] = []
    for p, y in zip(preds, labels):
        d = int(lev_dist(str(p), str(y)))
        raw.append(d)
        norm_ref.append(d / max(1, len(str(y))))
    return raw, norm_ref


def plot_raw_hist(raw_a: list[int], raw_b: list[int], *, label_a: str, label_b: str, outpath: str) -> None:
    max_d = int(max(max(raw_a, default=0), max(raw_b, default=0)))
    bins = np.arange(0, max_d + 2) - 0.5

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.hist(raw_a, bins=bins, alpha=0.55, label=f"{label_a} (n={len(raw_a)})", color="C0", edgecolor="none")
    ax.hist(raw_b, bins=bins, alpha=0.55, label=f"{label_b} (n={len(raw_b)})", color="C1", edgecolor="none")

    ax.set_xlabel("Levenshtein distance (raw edit count)")
    ax.set_ylabel("# validation samples")
    ax.set_title("Levenshtein Distance Histogram (all folds)")
    ax.set_xticks(range(0, max_d + 1))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved raw LD histogram: {}", outpath)


def plot_norm_hist(norm_a: list[float], norm_b: list[float], *, label_a: str, label_b: str, outpath: str) -> None:
    # Most values should lie in [0, 1], but allow a small overshoot if weird strings exist.
    hi = float(max(max(norm_a, default=0.0), max(norm_b, default=0.0), 1e-6))
    hi = min(max(hi, 0.5), 2.0)
    bins = np.linspace(0.0, hi, 40)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.hist(norm_a, bins=bins, alpha=0.55, label=f"{label_a} (n={len(norm_a)})", color="C0", edgecolor="none")
    ax.hist(norm_b, bins=bins, alpha=0.55, label=f"{label_b} (n={len(norm_b)})", color="C1", edgecolor="none")

    ax.set_xlabel("Normalized Levenshtein distance (d / |ref|)")
    ax.set_ylabel("# validation samples")
    ax.set_title("Normalized Levenshtein Distance Histogram (all folds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved normalized LD histogram: {}", outpath)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds or -1/all")
    ap.add_argument("--ar_export_pattern", type=str, required=True)
    ap.add_argument("--hyb_export_pattern", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="figures/levenshtein_all_folds")
    ap.add_argument("--label_ar", type=str, default="AR-only")
    ap.add_argument("--label_hyb", type=str, default="Hybrid (AR decode)")
    args = ap.parse_args()

    folds = _parse_folds(args.folds)
    os.makedirs(args.outdir, exist_ok=True)

    preds_ar_all: list[str] = []
    labels_ar_all: list[str] = []
    preds_hyb_all: list[str] = []
    labels_hyb_all: list[str] = []

    for k in folds:
        pat_ar = args.ar_export_pattern
        pat_h = args.hyb_export_pattern
        if "{fold}" in pat_ar:
            pat_ar = pat_ar.format(fold=k)
        if "{fold}" in pat_h:
            pat_h = pat_h.format(fold=k)

        pat_ar = _resolve_maybe_relative(pat_ar)
        pat_h = _resolve_maybe_relative(pat_h)

        fp_ar = _find_one(pat_ar)
        fp_h = _find_one(pat_h)

        d_ar = load_export_json(fp_ar)
        d_h = load_export_json(fp_h)

        preds_ar_all.extend(d_ar.preds)
        labels_ar_all.extend(d_ar.labels)
        preds_hyb_all.extend(d_h.preds)
        labels_hyb_all.extend(d_h.labels)

        logger.info(
            "Loaded fold {}: AR n={} | Hybrid n={}",
            k,
            len(d_ar.preds),
            len(d_h.preds),
        )

    raw_ar, norm_ar = compute_distances(preds_ar_all, labels_ar_all)
    raw_h, norm_h = compute_distances(preds_hyb_all, labels_hyb_all)

    plot_raw_hist(raw_ar, raw_h, label_a=args.label_ar, label_b=args.label_hyb,
                  outpath=os.path.join(args.outdir, "levenshtein_hist_counts.pdf"))
    plot_norm_hist(norm_ar, norm_h, label_a=args.label_ar, label_b=args.label_hyb,
                   outpath=os.path.join(args.outdir, "levenshtein_hist_counts_norm.pdf"))

    # Save quick summary
    summary = {
        "folds": folds,
        "ar": {
            "n": len(raw_ar),
            "mean_raw": float(np.mean(raw_ar)) if raw_ar else 0.0,
            "mean_norm_ref": float(np.mean(norm_ar)) if norm_ar else 0.0,
            "p90_norm_ref": float(np.quantile(norm_ar, 0.90)) if norm_ar else 0.0,
        },
        "hybrid_ar": {
            "n": len(raw_h),
            "mean_raw": float(np.mean(raw_h)) if raw_h else 0.0,
            "mean_norm_ref": float(np.mean(norm_h)) if norm_h else 0.0,
            "p90_norm_ref": float(np.quantile(norm_h, 0.90)) if norm_h else 0.0,
        },
    }
    out_json = os.path.join(args.outdir, "levenshtein_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: {}", out_json)


if __name__ == "__main__":
    main()
