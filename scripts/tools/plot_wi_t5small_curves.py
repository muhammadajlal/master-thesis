#!/usr/bin/env python3
"""Plot CER/WER curves for WI experiments (T5-small multimodal).

Reads RunManager JSON outputs (train_*.json) under each experiment folder and:
- plots mean±std across folds for CER and WER vs epoch
- writes per-epoch aggregated CSV
- writes a summary table (Markdown + CSV)

Usage (from repo root or work/REWI_work):
  python3 work/REWI_work/tools/plot_wi_t5small_curves.py

You can also override output dir:
  OUT_DIR=results/hwr2/plots_wi_t5small python3 ...
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Experiment:
    name: str
    path: str  # user-provided base path


EXPERIMENTS: List[Experiment] = [
    Experiment("Unfreeze@0 (0-600)", "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/multimodal_t5_small-0-600_word"),
    Experiment("Unfreeze@60 (60-300)", "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/multimodal_t5_small-60-300"),
    Experiment("Unfreeze@80 (80-600)", "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/multimodal_t5_small-80-600"),
    Experiment("Unfreeze@200 (200-600)", "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/multimodal_t5_small-200-600"),
    Experiment("FreezeLM", "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/multimodal_t5_small_FreezeLM"),
]

FOLDS = list(range(5))
DATASET_SUBDIR = "t5-small__onhw_wi_word_rh"


def _resolve_base_dir(user_path: str) -> str:
    """User passes either <exp_root> or <exp_root>/<t5-small__...>.

    Normalize to the directory that contains fold_0..fold_4.
    """
    p = os.path.abspath(user_path)
    if os.path.isdir(os.path.join(p, "fold_0")):
        return p
    sub = os.path.join(p, DATASET_SUBDIR)
    if os.path.isdir(os.path.join(sub, "fold_0")):
        return sub
    raise FileNotFoundError(f"Could not find fold_0 under: {p} (or {sub})")


def _latest_train_json(base_dir: str, fold: int) -> str:
    pat = os.path.join(base_dir, f"fold_{fold}", str(fold), "train_*.json")
    cands = sorted(glob.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No train_*.json for fold {fold}: {pat}")
    return cands[-1]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_epoch_metrics(obj: dict) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return (cer_by_epoch, wer_by_epoch)."""
    cer: Dict[int, float] = {}
    wer: Dict[int, float] = {}

    for k, v in obj.items():
        if not str(k).isdigit():
            continue
        if not isinstance(v, dict):
            continue
        ev = v.get("evaluation")
        if not isinstance(ev, dict):
            continue
        e = int(k)
        if "character_error_rate" in ev:
            cer[e] = float(ev["character_error_rate"])
        if "word_error_rate" in ev:
            wer[e] = float(ev["word_error_rate"])

    return cer, wer


def _extract_fold_best(obj: dict) -> Tuple[float | None, int | None, float | None, int | None]:
    """Try to read results['best'] written by RunManager.

    Returns: (best_cer, best_cer_epoch, best_wer, best_wer_epoch)
    """
    best = obj.get("best")
    if not isinstance(best, dict):
        return None, None, None, None

    def _get(metric: str):
        v = best.get(metric)
        if isinstance(v, list) and len(v) >= 2:
            try:
                return float(v[1]), int(v[0])
            except Exception:
                return None, None
        return None, None

    bcer, ecer = _get("character_error_rate")
    bwer, ewer = _get("word_error_rate")
    return bcer, ecer, bwer, ewer


def _aggregate_curves(fold_curves: List[Dict[int, float]]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Aggregate list of {epoch->value} dicts into mean/std dicts."""
    epochs = sorted(set().union(*[set(c.keys()) for c in fold_curves if c]))
    mean: Dict[int, float] = {}
    std: Dict[int, float] = {}
    for e in epochs:
        vals = [c[e] for c in fold_curves if e in c]
        if not vals:
            continue
        mean[e] = float(np.mean(vals))
        std[e] = float(np.std(vals))
    return mean, std


def _best_from_mean_curve(mean: Dict[int, float]) -> Tuple[float | None, int | None]:
    if not mean:
        return None, None
    e_best = min(mean.keys(), key=lambda e: mean[e])
    return float(mean[e_best]), int(e_best)


def _plot(ax, series: Dict[str, Tuple[Dict[int, float], Dict[int, float]]], title: str) -> None:
    for name, (mean, std) in series.items():
        xs = np.array(sorted(mean.keys()), dtype=np.int64)
        ys = np.array([mean[e] for e in xs], dtype=np.float64)
        ax.plot(xs, ys, label=name, linewidth=2)
        ss = np.array([std.get(int(e), 0.0) for e in xs], dtype=np.float64)
        ax.fill_between(xs, ys - ss, ys + ss, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def main() -> int:
    out_dir = os.environ.get(
        "OUT_DIR",
        "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/plots_wi_t5small",
    )
    os.makedirs(out_dir, exist_ok=True)

    per_epoch_rows: List[dict] = []
    summary_rows: List[dict] = []

    cer_series: Dict[str, Tuple[Dict[int, float], Dict[int, float]]] = {}
    wer_series: Dict[str, Tuple[Dict[int, float], Dict[int, float]]] = {}

    for exp in EXPERIMENTS:
        base = _resolve_base_dir(exp.path)

        fold_cer: List[Dict[int, float]] = []
        fold_wer: List[Dict[int, float]] = []
        fold_best_cer: List[float] = []
        fold_best_wer: List[float] = []

        for f in FOLDS:
            p = _latest_train_json(base, f)
            obj = _load_json(p)
            cer, wer = _extract_epoch_metrics(obj)
            fold_cer.append(cer)
            fold_wer.append(wer)

            bcer, _ecer, bwer, _ewer = _extract_fold_best(obj)
            if bcer is not None:
                fold_best_cer.append(float(bcer))
            if bwer is not None:
                fold_best_wer.append(float(bwer))

        cer_mean, cer_std = _aggregate_curves(fold_cer)
        wer_mean, wer_std = _aggregate_curves(fold_wer)
        cer_series[exp.name] = (cer_mean, cer_std)
        wer_series[exp.name] = (wer_mean, wer_std)

        # per-epoch CSV rows
        epochs = sorted(set(cer_mean.keys()) | set(wer_mean.keys()))
        for e in epochs:
            per_epoch_rows.append(
                {
                    "experiment": exp.name,
                    "epoch": e,
                    "cer_mean": cer_mean.get(e, ""),
                    "cer_std": cer_std.get(e, ""),
                    "wer_mean": wer_mean.get(e, ""),
                    "wer_std": wer_std.get(e, ""),
                }
            )

        # summary stats
        best_cer_mean, best_cer_epoch = _best_from_mean_curve(cer_mean)
        best_wer_mean, best_wer_epoch = _best_from_mean_curve(wer_mean)

        row = {
            "experiment": exp.name,
            "base_dir": base,
            "best_cer_mean_curve": best_cer_mean,
            "best_cer_epoch": best_cer_epoch,
            "best_wer_mean_curve": best_wer_mean,
            "best_wer_epoch": best_wer_epoch,
        }

        if fold_best_cer:
            row["avg_fold_best_cer"] = float(np.mean(fold_best_cer))
            row["std_fold_best_cer"] = float(np.std(fold_best_cer))
        else:
            row["avg_fold_best_cer"] = None
            row["std_fold_best_cer"] = None

        if fold_best_wer:
            row["avg_fold_best_wer"] = float(np.mean(fold_best_wer))
            row["std_fold_best_wer"] = float(np.std(fold_best_wer))
        else:
            row["avg_fold_best_wer"] = None
            row["std_fold_best_wer"] = None

        summary_rows.append(row)

    # --- Plots ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    _plot(axs[0], cer_series, "WI / T5-small: CER vs epoch (mean±std over folds)")
    _plot(axs[1], wer_series, "WI / T5-small: WER vs epoch (mean±std over folds)")
    fig.suptitle("Multimodal T5-small freeze/unfreeze ablations (WI)")
    plt.tight_layout()

    out_png = os.path.join(out_dir, "wi_t5small_cer_wer_curves.png")
    fig.savefig(out_png, dpi=200)

    # Also save separate plots (handy for slides)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    _plot(ax2, cer_series, "WI / T5-small: CER vs epoch")
    out_cer = os.path.join(out_dir, "wi_t5small_cer_curves.png")
    fig2.savefig(out_cer, dpi=200, bbox_inches="tight")

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    _plot(ax3, wer_series, "WI / T5-small: WER vs epoch")
    out_wer = os.path.join(out_dir, "wi_t5small_wer_curves.png")
    fig3.savefig(out_wer, dpi=200, bbox_inches="tight")

    # --- CSV: per-epoch ---
    out_csv = os.path.join(out_dir, "wi_t5small_curves.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["experiment", "epoch", "cer_mean", "cer_std", "wer_mean", "wer_std"],
        )
        w.writeheader()
        w.writerows(per_epoch_rows)

    # --- CSV + MD: summary ---
    out_sum_csv = os.path.join(out_dir, "wi_t5small_summary.csv")
    sum_fields = [
        "experiment",
        "best_cer_mean_curve",
        "best_cer_epoch",
        "best_wer_mean_curve",
        "best_wer_epoch",
        "avg_fold_best_cer",
        "std_fold_best_cer",
        "avg_fold_best_wer",
        "std_fold_best_wer",
        "base_dir",
    ]

    with open(out_sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in sum_fields})

    out_md = os.path.join(out_dir, "wi_t5small_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# WI / T5-small ablation summary\n\n")
        f.write("Curves are aggregated as mean±std across folds (0..4).\n\n")
        f.write("| Experiment | Best CER (mean curve) | Epoch | Best WER (mean curve) | Epoch | Avg fold-best CER ± std | Avg fold-best WER ± std |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            def fmt(x):
                return "" if x is None else f"{x:.6f}"

            cer_best = fmt(r.get("best_cer_mean_curve"))
            wer_best = fmt(r.get("best_wer_mean_curve"))

            cer_e = r.get("best_cer_epoch")
            wer_e = r.get("best_wer_epoch")

            cer_fold = r.get("avg_fold_best_cer")
            cer_fold_s = r.get("std_fold_best_cer")
            wer_fold = r.get("avg_fold_best_wer")
            wer_fold_s = r.get("std_fold_best_wer")

            cer_fold_txt = "" if cer_fold is None else f"{cer_fold:.6f} ± {cer_fold_s:.6f}"
            wer_fold_txt = "" if wer_fold is None else f"{wer_fold:.6f} ± {wer_fold_s:.6f}"

            f.write(
                f"| {r['experiment']} | {cer_best} | {cer_e if cer_e is not None else ''} | {wer_best} | {wer_e if wer_e is not None else ''} | {cer_fold_txt} | {wer_fold_txt} |\n"
            )

    print("Wrote plots:")
    print(" ", out_png)
    print(" ", out_cer)
    print(" ", out_wer)
    print("Wrote tables:")
    print(" ", out_csv)
    print(" ", out_sum_csv)
    print(" ", out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
