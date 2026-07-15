#!/usr/bin/env python3
"""Reproduce the hybrid HWRFormer noise-injection matrix in the thesis appendix."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


MODES = (
    "uniform",
    "bigram_right",
    "bigram_left",
    "self_confusion",
    "adjacent_swap",
)
DATASETS = (
    "onhw_wi_word_rh",
    "onhw_wd_word_rh",
    "wi_word_hw6_meta",
    "wi_sent_hw6_meta",
)
MAX_EPOCH = 299


def load_fold(directory: Path, fold: int) -> dict[str, object]:
    pattern = directory / f"fold_{fold}" / str(fold) / "train_*.json"
    candidates: list[tuple[float, int, str, float]] = []
    for filename in sorted(glob.glob(str(pattern))):
        data = json.loads(Path(filename).read_text())
        for key, values in data.items():
            if not key.isdigit() or int(key) > MAX_EPOCH or "evaluation" not in values:
                continue
            evaluation = values["evaluation"]
            candidates.append(
                (
                    float(evaluation["character_error_rate"]) * 100.0,
                    int(key),
                    filename,
                    float(evaluation["word_error_rate"]) * 100.0,
                )
            )
    if not candidates:
        raise FileNotFoundError(f"No epoch 0--{MAX_EPOCH} metrics under {pattern}")
    cer, epoch, filename, wer = min(candidates)
    return {
        "cer_percent": cer,
        "wer_percent_at_selected_cer_epoch": wer,
        "selected_epoch": epoch,
        "source_json": str(Path(filename).relative_to(directory.parents[1])),
    }


def analyze(results_root: Path) -> dict[str, object]:
    matrix: dict[str, object] = {}
    for mode in MODES:
        matrix[mode] = {}
        family = results_root / f"HybridInputCorruption-XS-L01_{mode}"
        for dataset in DATASETS:
            directory = family / f"ar_transformer_xs__{dataset}"
            folds = [load_fold(directory, fold) for fold in range(5)]
            cer = [float(item["cer_percent"]) for item in folds]
            wer = [float(item["wer_percent_at_selected_cer_epoch"]) for item in folds]
            matrix[mode][dataset] = {
                "folds": folds,
                "cer_percent": {
                    "mean": statistics.mean(cer),
                    "sample_std": statistics.stdev(cer),
                },
                "wer_percent_at_selected_cer_epoch": {
                    "mean": statistics.mean(wer),
                    "sample_std": statistics.stdev(wer),
                },
            }
    return {
        "selection_rule": (
            "For each fold, select the minimum validation CER among retained epochs "
            "0--299 across all train JSON segments; take WER from the same epoch."
        ),
        "lambda_ctc": 0.1,
        "p_replace": 0.15,
        "matrix": matrix,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_results = Path(__file__).resolve().parents[4] / "results" / "hwr2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_results)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "thesis_hybrid_noise_l01.json",
    )
    args = parser.parse_args()

    report = analyze(args.results_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
