#!/usr/bin/env python3
"""Collect experiment results from all training runs.

Scans results/hwr2/ for completed experiments, extracts best CER/WER
per fold from train_*.json files, and produces a consolidated table.

Usage:
    python3 scripts/collect_results.py                    # print table
    python3 scripts/collect_results.py --csv results.csv  # save CSV
    python3 scripts/collect_results.py --json results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "results" / "hwr2"
GPT2_AR_ROOT = "Ablations-MMLM/GPT-2/AR-only"
GPT2_HYBRID_ROOT = "Ablations-MMLM/GPT-2/Hybrid"

# (result_dir_name, dataset_suffix, description, trainable_params)
EXPERIMENTS = {
    # Baselines
    "Baseline-CTC":        ("Baseline-REWI",                     "bilstm_wide__onhw_wi_word_rh",        "CTC BiLSTM",                          "3.15M"),
    "Baseline-CTC-hw6":    ("Baseline-REWI",                     "bilstm_wide__wi_word_hw6_meta",       "CTC BiLSTM",                          "3.15M"),
    "Baseline-AR":         ("Baseline-AR",                        "ar_transformer_s__onhw_wi_word_rh",   "AR Transformer",                      "4.82M"),
    "Baseline-AR-hw6":     ("Baseline-AR",                        "ar_transformer_s__wi_word_hw6_meta",  "AR Transformer",                      "4.82M"),
    # GPT-2 VLM ablations
    "A1-MLP":              (f"{GPT2_AR_ROOT}/vlm_ablation_A1_mlp_pretrained",     "vlm__onhw_wi_word_rh",                "MLP + pretrained GPT-2",              "5.09M"),
    "A1-MLP-hw6":          (f"{GPT2_AR_ROOT}/vlm_ablation_A1_mlp_pretrained",     "vlm__wi_word_hw6_meta",               "MLP + pretrained GPT-2",              "5.09M"),
    "A2-MLP-rand":         (f"{GPT2_AR_ROOT}/vlm_ablation_A2_mlp_random",         "vlm__onhw_wi_word_rh",                "MLP + random GPT-2",                  "5.09M"),
    "A2-MLP-rand-hw6":     (f"{GPT2_AR_ROOT}/vlm_ablation_A2_mlp_random",         "vlm__wi_word_hw6_meta",               "MLP + random GPT-2",                  "5.09M"),
    "A3-Linear":           (f"{GPT2_AR_ROOT}/vlm_ablation_A3_linear_pretrained",   "vlm__onhw_wi_word_rh",               "Linear + pretrained GPT-2",           "4.50M"),
    "A3-Linear-hw6":       (f"{GPT2_AR_ROOT}/vlm_ablation_A3_linear_pretrained",   "vlm__wi_word_hw6_meta",              "Linear + pretrained GPT-2",           "4.50M"),
    "B1-Pool":             (f"{GPT2_AR_ROOT}/vlm_ablation_B1_pooling_pretrained",  "vlm__onhw_wi_word_rh",               "PoolingMLP + pretrained GPT-2",       "5.09M"),
    "B1-Pool-hw6":         (f"{GPT2_AR_ROOT}/vlm_ablation_B1_pooling_pretrained",  "vlm__wi_word_hw6_meta",              "PoolingMLP + pretrained GPT-2",       "5.09M"),
    # QFormer
    "QF-v2":               (f"{GPT2_AR_ROOT}/vlm_Qformer_gpt2_word_v2",           "vlm__onhw_wi_word_rh",               "QFormer + GPT-2",                     "23.4M"),
    "QF-v2-hw6":           (f"{GPT2_AR_ROOT}/vlm_Qformer_gpt2_word_v2",           "vlm__wi_word_hw6_meta",              "QFormer + GPT-2",                     "23.4M"),
    "QF-v3":               (f"{GPT2_AR_ROOT}/vlm_gpt2_word_v3",                    "vlm__onhw_wi_word_rh",               "QFormer v3",                          "23.4M"),
    "QF-v3-s10":           (f"{GPT2_AR_ROOT}/vlm_gpt2_word_v3_soft10",            "vlm__onhw_wi_word_rh",               "QFormer v3 (10 soft)",                "23.4M"),
    # Follow-ups
    "B2-Pool32-hw6":       (f"{GPT2_AR_ROOT}/vlm_followup_B2_pooling_lora32",     "vlm__wi_word_hw6_meta",              "PoolingMLP + LoRA r=32",              "8.18M"),
    "C1-MiniQF-hw6":       (f"{GPT2_AR_ROOT}/vlm_followup_C1_mini_qformer",       "vlm__wi_word_hw6_meta",              "Mini QFormer (2L)",                   "13.96M"),
    "Q2-QF32-hw6":         (f"{GPT2_AR_ROOT}/vlm_followup_Q2_qformer_lora32",     "vlm__wi_word_hw6_meta",              "QFormer + LoRA r=32",                 "23.4M"),
    # New experiments
    "F1-Frozen":           (f"{GPT2_AR_ROOT}/F1_frozen_enc_mlp",                   "vlm__onhw_wi_word_rh",               "Frozen enc + MLP + LoRA GPT-2",       "~2.6M"),
    "F1-Frozen-hw6":       (f"{GPT2_AR_ROOT}/F1_frozen_enc_mlp",                   "vlm__wi_word_hw6_meta",              "Frozen enc + MLP + LoRA GPT-2",       "~2.6M"),
    "F2-EncAR":            (f"{GPT2_AR_ROOT}/F2_vlm_enc_ar",                       "ar_transformer_s__onhw_wi_word_rh",  "VLM enc -> AR decoder (frozen)",      "~2.4M"),
    "F2-EncAR-hw6":        (f"{GPT2_AR_ROOT}/F2_vlm_enc_ar",                       "ar_transformer_s__wi_word_hw6_meta", "VLM enc -> AR decoder (frozen)",      "~2.4M"),
    "G1a-Lin":             (f"{GPT2_AR_ROOT}/G1a_linear",                          "vlm__onhw_wi_word_rh",               "Linear connector (minimal)",          "4.50M"),
    "G1a-Lin-hw6":         (f"{GPT2_AR_ROOT}/G1a_linear",                          "vlm__wi_word_hw6_meta",              "Linear connector (minimal)",          "4.50M"),
    "G1b-Conv":            (f"{GPT2_AR_ROOT}/G1b_conv_pool",                       "vlm__onhw_wi_word_rh",               "ConvPool connector (minimal)",        "4.50M"),
    "G1b-Conv-hw6":        (f"{GPT2_AR_ROOT}/G1b_conv_pool",                       "vlm__wi_word_hw6_meta",              "ConvPool connector (minimal)",        "4.50M"),
    "H1-Hybrid":           (f"{GPT2_HYBRID_ROOT}/H1_hybrid_mlp",                       "vlm__onhw_wi_word_rh",               "Hybrid CTC+LM (MLP)",                "~5.1M"),
    "H1-Hybrid-hw6":       (f"{GPT2_HYBRID_ROOT}/H1_hybrid_mlp",                       "vlm__wi_word_hw6_meta",              "Hybrid CTC+LM (MLP)",                "~5.1M"),
}


def _best_from_json(json_path: Path) -> Optional[Tuple[float, float]]:
    """Extract best CER/WER from a train_*.json file.

    Format: dict keyed by epoch strings, each with an optional
    'evaluation' dict containing 'character_error_rate' and 'word_error_rate'.
    """
    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    best_cer, best_wer = None, None
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        ev = val.get("evaluation") or val.get("evaluation_lm")
        if not isinstance(ev, dict):
            continue
        cer = ev.get("character_error_rate")
        if cer is None:
            cer = ev.get("CER")
        if cer is None:
            continue
        cer = float(cer)
        wer = float(ev.get("word_error_rate", ev.get("WER", 0)))
        if best_cer is None or cer < best_cer:
            best_cer = cer
            best_wer = wer

    if best_cer is not None:
        return best_cer, best_wer or 0.0
    return None


def _find_fold_results(base_dir: Path) -> Dict[int, Tuple[float, float]]:
    """Find per-fold best CER/WER from an experiment directory.

    Directory structure: base_dir/fold_{i}/{subdir}/train_*.json
    where subdir is typically '0' or '{fold_idx}'.
    """
    results = {}
    for fold_idx in range(5):
        fold_dir = base_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            continue

        # Search subdirs for train_*.json files
        best = None
        for sub in sorted(fold_dir.iterdir()):
            if not sub.is_dir() or not sub.name.isdigit():
                continue
            for jf in sorted(sub.glob("train_*.json")):
                m = _best_from_json(jf)
                if m and (best is None or m[0] < best[0]):
                    best = m

        # Also check fold_dir directly (some experiments don't have numeric subdirs)
        for jf in sorted(fold_dir.glob("train_*.json")):
            m = _best_from_json(jf)
            if m and (best is None or m[0] < best[0]):
                best = m

        if best is not None:
            results[fold_idx] = best
    return results


def collect_all(root: Path = RESULTS_ROOT) -> List[dict]:
    """Collect results from all registered experiments."""
    rows = []
    for exp_id, (dir_name, ds_suffix, desc, params) in sorted(EXPERIMENTS.items()):
        base_dir = root / dir_name / ds_suffix
        fold_results = _find_fold_results(base_dir)

        dataset = "onhw" if "onhw" in ds_suffix else "hw6"
        num_folds = len(fold_results)

        if num_folds > 0:
            mean_cer = sum(v[0] for v in fold_results.values()) / num_folds
            mean_wer = sum(v[1] for v in fold_results.values()) / num_folds
        else:
            mean_cer = mean_wer = None

        rows.append({
            "exp_id": exp_id,
            "description": desc,
            "params": params,
            "dataset": dataset,
            "dir_name": dir_name,
            "folds_complete": num_folds,
            "mean_cer": mean_cer,
            "mean_wer": mean_wer,
            "per_fold": {k: {"cer": v[0], "wer": v[1]} for k, v in fold_results.items()},
            "status": "done" if num_folds == 5 else ("partial" if num_folds > 0 else "pending"),
        })
    return rows


def print_table(rows: List[dict]) -> None:
    """Print a formatted results table grouped by dataset."""
    for dataset in ["onhw", "hw6"]:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        ds_label = "OnHW (public, word-level)" if dataset == "onhw" else "HW6 (private, word-level)"
        print(f"\n{'='*90}")
        print(f"  {ds_label}")
        print(f"{'='*90}")
        print(f"{'Experiment':<20} {'Params':<8} {'St':<8} {'F':<4} {'CER%':<8} {'WER%':<8} {'Description'}")
        print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*30}")

        ds_rows.sort(key=lambda r: (r["mean_cer"] is None, r["mean_cer"] or 999))

        for r in ds_rows:
            c = f"{r['mean_cer']*100:.2f}" if r["mean_cer"] is not None else "---"
            w = f"{r['mean_wer']*100:.2f}" if r["mean_wer"] is not None else "---"
            print(f"{r['exp_id']:<20} {r['params']:<8} {r['status']:<8} {r['folds_complete']}/5  {c:<8} {w:<8} {r['description']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--csv", help="Save results to CSV file")
    parser.add_argument("--json", help="Save results to JSON file")
    parser.add_argument("--root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    rows = collect_all(Path(args.root))
    print_table(rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "exp_id", "dataset", "params", "status", "folds_complete",
                "mean_cer", "mean_wer", "description", "dir_name",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: v for k, v in r.items() if k != "per_fold"})
        print(f"Saved CSV to {args.csv}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"Saved JSON to {args.json}")


if __name__ == "__main__":
    main()
