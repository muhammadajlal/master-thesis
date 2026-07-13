#!/usr/bin/env python3
"""Reproduce the fold-level statistics reported in thesis Chapter 6.

The script reads only consolidated ``results.json`` files. Paired effects are
reported as condition B minus condition A in percentage points. The confidence
interval is a two-sided Student-t interval over the five paired fold effects.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


T_CRIT_DF4_975 = 2.7764451051977987
DATASETS = ("onhw", "private")


@dataclass(frozen=True)
class Condition:
    label: str
    onhw: str
    private: str


CONDITIONS = {
    "mlp_pretrained": Condition(
        "MLP + pretrained GPT-2",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/"
        "vlm__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "mlp_random": Condition(
        "MLP + random GPT-2",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/"
        "vlm__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "pool_noctc": Condition(
        "Pool-MLP without CTC",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/"
        "vlm__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "qformer_noctc": Condition(
        "Lightweight Q-Former without CTC",
        "Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/"
        "vlm__onhw_wi_word_rh_noctc/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/"
        "vlm__wi_word_hw6_meta_noctc/results.json",
    ),
    "gated_noctc": Condition(
        "Gated Multi-View without CTC",
        "Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/"
        "vlm__onhw_wi_word_rh_noctc/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/"
        "vlm__wi_word_hw6_meta_noctc/results.json",
    ),
    "mlp_ctc": Condition(
        "MLP + CTC",
        "H1_LambdaSweep/H1_hybrid_mlp_lam02__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "pool_ctc": Condition(
        "Pool-MLP + CTC",
        "Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/"
        "vlm__onhw_wi_word_rh_lam02/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "qformer_ctc": Condition(
        "Lightweight Q-Former + CTC",
        "Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/"
        "vlm__onhw_wi_word_rh_lam02/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "gated_ctc": Condition(
        "Gated Multi-View + CTC",
        "Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/"
        "vlm__onhw_wi_word_rh_lam02/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "byt5": Condition(
        "ByT5 hybrid MLP",
        "Ablations-MMLM/byt5-small/Hybrid/M1_byt5_hybrid_mlp/"
        "vlm__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/byt5-small/Hybrid/M1_byt5_hybrid_mlp/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
    "conformer": Condition(
        "Shallow Conformer hybrid MLP",
        "Ablations-MMLM/GPT-2/Hybrid/N1_conformer_hybrid_mlp/"
        "vlm__onhw_wi_word_rh/results.json",
        "Ablations-MMLM/GPT-2/Hybrid/N1_conformer_hybrid_mlp/"
        "vlm__wi_word_hw6_meta/results.json",
    ),
}


COMPARISONS = {
    "pretrained_minus_random": ("mlp_random", "mlp_pretrained"),
    "mlp_ctc_effect": ("mlp_pretrained", "mlp_ctc"),
    "pool_ctc_effect": ("pool_noctc", "pool_ctc"),
    "qformer_ctc_effect": ("qformer_noctc", "qformer_ctc"),
    "gated_ctc_effect": ("gated_noctc", "gated_ctc"),
    "qformer_minus_mlp": ("mlp_ctc", "qformer_ctc"),
    "qformer_minus_pool": ("pool_ctc", "qformer_ctc"),
    "qformer_minus_gated": ("gated_ctc", "qformer_ctc"),
}


def load_result(path: Path) -> dict[str, dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    result: dict[str, dict[int, float]] = {}
    for metric in ("cer", "wer"):
        raw = data.get(metric, {}).get("raw", {})
        folds = {int(key): float(value) * 100.0 for key, value in raw.items()}
        if sorted(folds) != list(range(5)):
            raise ValueError(f"{path}: {metric} must contain folds 0--4")
        stored_mean = float(data[metric]["mean"]) * 100.0
        calculated_mean = statistics.mean(folds.values())
        if not math.isclose(stored_mean, calculated_mean, abs_tol=1e-10):
            raise ValueError(f"{path}: inconsistent {metric} mean")
        result[metric] = folds
    return result


def paired_summary(a: dict[int, float], b: dict[int, float]) -> dict[str, object]:
    differences = [b[fold] - a[fold] for fold in range(5)]
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    half_width = T_CRIT_DF4_975 * sem
    signs = "".join("+" if value > 0 else "-" if value < 0 else "0" for value in differences)
    return {
        "mean_pp": mean,
        "ci95_pp": [mean - half_width, mean + half_width],
        "fold_signs": signs,
        "fold_differences_pp": differences,
    }


def analyze(results_root: Path) -> dict[str, object]:
    loaded: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    means: dict[str, object] = {}
    for key, condition in CONDITIONS.items():
        loaded[key] = {}
        means[key] = {"label": condition.label}
        for dataset in DATASETS:
            result = load_result(results_root / getattr(condition, dataset))
            loaded[key][dataset] = result
            means[key][dataset] = {
                metric: statistics.mean(result[metric].values())
                for metric in ("cer", "wer")
            }

    comparisons: dict[str, object] = {}
    for name, (a_key, b_key) in COMPARISONS.items():
        comparisons[name] = {
            "definition": f"{CONDITIONS[b_key].label} minus {CONDITIONS[a_key].label}",
        }
        for dataset in DATASETS:
            comparisons[name][dataset] = {
                metric: paired_summary(
                    loaded[a_key][dataset][metric], loaded[b_key][dataset][metric]
                )
                for metric in ("cer", "wer")
            }
    return {"means_percent": means, "paired_comparisons": comparisons}


def print_text(report: dict[str, object]) -> None:
    print("CONDITION MEANS (%)")
    for key, values in report["means_percent"].items():
        print(f"{key}: {values['label']}")
        for dataset in DATASETS:
            metrics = values[dataset]
            print(f"  {dataset:7s} CER={metrics['cer']:.2f} WER={metrics['wer']:.2f}")

    print("\nPAIRED EFFECTS (B - A, percentage points)")
    for name, values in report["paired_comparisons"].items():
        print(f"{name}: {values['definition']}")
        for dataset in DATASETS:
            fields = []
            for metric in ("cer", "wer"):
                item = values[dataset][metric]
                lower, upper = item["ci95_pp"]
                fields.append(
                    f"{metric.upper()}={item['mean_pp']:+.2f} "
                    f"[{lower:+.2f},{upper:+.2f}] signs={item['fold_signs']}"
                )
            print(f"  {dataset:7s} " + "; ".join(fields))


def main() -> None:
    default_root = Path(__file__).resolve().parents[3] / "results" / "hwr2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_root)
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON")
    args = parser.parse_args()

    report = analyze(args.results_root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
