#!/usr/bin/env python3
"""Plot T5 unfreezing ablation results.

This script is designed for directories like:
  results/hwr2/multimodal_t5_small-60-600/
  results/hwr2/multimodal_t5_small-80-600/
  results/hwr2/multimodal_t5_small-200-600/

Each directory is expected to contain:
  - results.json (from evaluate.py)
  - per-fold train_*.json files somewhere below (optional, for learning curves)

Example:
  cd work/REWI_work
  python3 scripts/tools/plot_t5_unfreeze_ablation.py \
    --runs ../../results/hwr2/multimodal_t5_small-60-600 \
           ../../results/hwr2/multimodal_t5_small-80-600 \
           ../../results/hwr2/multimodal_t5_small-200-600 \
    --out figures/t5_unfreeze_ablation_summary.png \
    --curve-out figures/t5_unfreeze_ablation_curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    label: str
    unfreeze_epoch: Optional[int]
    cer_mean: float
    cer_std: float
    wer_mean: float
    wer_std: float


def _infer_unfreeze_epoch(path: str) -> Optional[int]:
    # matches both: multimodal_t5_small-200-600 and t5_small-200-600
    m = re.search(r"t5_small-(\d+)-(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def _common_out_dir(run_dirs: List[str]) -> str:
    """Choose a stable output directory for figures.

    For typical usage where run dirs live under `results/hwr2/...`, this returns:
      results/hwr2/figures
    """
    common = os.path.commonpath([os.path.abspath(r) for r in run_dirs])
    # If the common path is exactly a run directory, fall back to its parent.
    if os.path.isfile(os.path.join(common, "results.json")):
        common = os.path.dirname(common)
    return os.path.join(common, "figures")


def _format_label(run_dir: str) -> str:
    unfreeze = _infer_unfreeze_epoch(run_dir)
    # Standardized naming across experiments
    base = "CNN-t5Small"
    if unfreeze is None:
        return base
    return f"{base} (unfreeze@{unfreeze})"


def load_results_json(run_dir: str) -> RunSummary:
    path = os.path.join(run_dir, "results.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cer = data["cer"]
    wer = data["wer"]

    return RunSummary(
        run_dir=run_dir,
        label=_format_label(run_dir),
        unfreeze_epoch=_infer_unfreeze_epoch(run_dir),
        cer_mean=float(cer["mean"]),
        cer_std=float(cer["std"]),
        wer_mean=float(wer["mean"]),
        wer_std=float(wer["std"]),
    )


def iter_train_json_paths(run_dir: str, dataset_substr: Optional[str]) -> Iterable[str]:
    for root, _dirs, files in os.walk(run_dir):
        if dataset_substr and dataset_substr not in root:
            continue
        for fn in files:
            if fn.startswith("train_") and fn.endswith(".json"):
                yield os.path.join(root, fn)


def load_cv_curve(
    run_dir: str,
    metric: str,
    dataset_substr: Optional[str],
) -> Dict[int, Tuple[float, float]]:
    """Returns epoch -> (mean, std) for a metric across folds."""

    metric_key = {
        "cer": "character_error_rate",
        "wer": "word_error_rate",
    }[metric]

    vals: Dict[int, List[float]] = defaultdict(list)

    any_file = False
    for path in iter_train_json_paths(run_dir, dataset_substr):
        any_file = True
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        for k, v in d.items():
            if not isinstance(k, str) or not k.isdigit():
                continue
            if not isinstance(v, dict) or "evaluation" not in v:
                continue
            ev = v.get("evaluation")
            if not isinstance(ev, dict) or metric_key not in ev:
                continue
            epoch = int(k)
            vals[epoch].append(float(ev[metric_key]))

    if not any_file:
        return {}

    curve: Dict[int, Tuple[float, float]] = {}
    for epoch in sorted(vals.keys()):
        xs = vals[epoch]
        if not xs:
            continue
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        curve[epoch] = (mean, var**0.5)

    return curve


def plot_summary(runs: List[RunSummary], out_path: str) -> None:
    # Prefer ordering by unfreeze epoch if available
    runs_sorted = sorted(runs, key=lambda r: (r.unfreeze_epoch is None, r.unfreeze_epoch or 0))

    x = [r.unfreeze_epoch if r.unfreeze_epoch is not None else i for i, r in enumerate(runs_sorted)]
    labels = [r.label for r in runs_sorted]

    cer = [100.0 * r.cer_mean for r in runs_sorted]
    cer_err = [100.0 * r.cer_std for r in runs_sorted]

    wer = [100.0 * r.wer_mean for r in runs_sorted]
    wer_err = [100.0 * r.wer_std for r in runs_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=150)
    ax.errorbar(x, cer, yerr=cer_err, fmt="o-", capsize=3, label="CER")
    ax.errorbar(x, wer, yerr=wer_err, fmt="s-", capsize=3, label="WER")

    ax.set_xlabel("Unfreeze epoch (decoder-side LM)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("T5 unfreezing ablation (mean ± std over folds)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add labels near points if x is not categorical
    # Try to avoid overlapping text: stagger annotation offsets.
    for j, (xi, yi, name) in enumerate(zip(x, cer, labels)):
        dy = 4 + (j % 2) * 10
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, dy), fontsize=8, alpha=0.85)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_curves(
    run_dirs: List[str],
    out_path: str,
    dataset_substr: Optional[str],
) -> None:
    # Plot mean CER curves; WER is often noisier for word datasets, but include both.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharex=True)

    for run_dir in run_dirs:
        label = _format_label(run_dir)
        unfreeze = _infer_unfreeze_epoch(run_dir)

        for ax, metric in zip(axes, ("cer", "wer")):
            curve = load_cv_curve(run_dir, metric=metric, dataset_substr=dataset_substr)
            if not curve:
                continue

            epochs = sorted(curve.keys())
            mean = [100.0 * curve[e][0] for e in epochs]
            ax.plot(epochs, mean, label=label)

            if unfreeze is not None:
                ax.axvline(unfreeze, color="k", lw=1, alpha=0.15)

    axes[0].set_title("Validation CER (mean over folds)")
    axes[1].set_title("Validation WER (mean over folds)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error rate (%)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories containing results.json (and optionally train_*.json).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="If set, write plots into this directory (filenames are auto-chosen).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path for summary plot.",
    )
    p.add_argument(
        "--curve-out",
        default=None,
        help="If set, also write learning-curve plot to this path.",
    )
    p.add_argument(
        "--dataset-substr",
        default=None,
        help="Optional substring filter for train_*.json discovery (e.g., onhw_wi_word_rh).",
    )

    args = p.parse_args()

    runs = [os.path.abspath(r) for r in args.runs]
    for r in runs:
        if not os.path.isfile(os.path.join(r, "results.json")):
            raise SystemExit(f"Missing results.json in: {r}")

    summaries = [load_results_json(r) for r in runs]

    # Default output location: <common_runs_parent>/figures
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _common_out_dir(runs)
    os.makedirs(out_dir, exist_ok=True)

    out_summary = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(out_dir, "t5_unfreeze_ablation_summary.png")
    )
    out_curves = (
        os.path.abspath(args.curve_out)
        if args.curve_out
        else os.path.join(out_dir, "t5_unfreeze_ablation_curves.png")
    )

    plot_summary(summaries, out_summary)

    # Always write curves unless explicitly disabled by passing --curve-out '' (not supported)
    plot_curves(runs, out_curves, dataset_substr=args.dataset_substr)

    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_curves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
