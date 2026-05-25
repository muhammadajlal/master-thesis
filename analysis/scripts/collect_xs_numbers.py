#!/usr/bin/env python3
"""Collect every XS result needed by paper 2 into one staging JSON.

Reads `best.character_error_rate` / `best.word_error_rate` from
train_*.json across 5 folds per experiment, computes mean and std,
and writes the result to xs_numbers.json. Treat the output as the
single source of truth for all paper-2 table cells and inline numbers
when refilling under the XS swap.

Re-runnable: call this script periodically as SLURM jobs land and the
staging file will update. Cells with incomplete folds are reported as
PENDING so the migration plan can wait for them before editing the
corresponding .tex.

Run from anywhere:
    python collect_xs_numbers.py [--out path]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Optional

RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
DEFAULT_OUT = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/xs_numbers.json"
)

DATASET_LABELS = {
    "onhw_wi_word_rh":           "OnHW-WI",
    "onhw_wd_word_rh":           "OnHW-WD",
    "onhw_equations_wi_word_rh": "Eq-WI",
    "onhw_equations_wd_word_rh": "Eq-WD",
    "wi_word_hw6_meta":          "Priv-words",
    "wi_sent_hw6_meta":          "Priv-sent",
}


def read_fold(model_dir: Path) -> tuple[Optional[float], Optional[float], int]:
    """Read 5-fold CER/WER from train_*.json. Returns (cer_list, wer_list, n_folds)."""
    cers, wers = [], []
    n = 0
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"fold_{k}/{k}/train_*.json")))
        if not files:
            continue
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
            wers.append(float(d["best"]["word_error_rate"][1]) * 100.0)
            n += 1
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return cers, wers, n


def cell(model_dir: Path) -> dict:
    cers, wers, n = read_fold(model_dir)
    if n == 0:
        return {"status": "missing", "n_folds": 0, "path": str(model_dir)}
    if n < 5:
        return {
            "status": "pending",
            "n_folds": n,
            "path": str(model_dir),
            "cer_mean": round(statistics.mean(cers), 2),
            "wer_mean": round(statistics.mean(wers), 2),
        }
    return {
        "status": "ready",
        "n_folds": 5,
        "path": str(model_dir),
        "cer_mean": round(statistics.mean(cers), 2),
        "cer_std":  round(statistics.stdev(cers), 2),
        "wer_mean": round(statistics.mean(wers), 2),
        "wer_std":  round(statistics.stdev(wers), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out: dict = {
        "_meta": {
            "results_root": str(RESULTS),
            "schema": "result cells keyed by [section][row], status=ready|pending|missing",
        }
    }

    # ----- Table 1: OnHW-WI migration -----
    out["table1_onhw_migration"] = {
        "REWI_BiLSTM_CTC": {
            "note": "Reference from REWI paper (Li 2025); not recomputed",
            "cer_mean": 7.30, "wer_mean": 15.16, "params_M": 4.64, "macs_M": 695,
        },
        "HWRFormer_CTC": {
            **cell(RESULTS / "Baseline-Transformer-XS-CTC/transformer_xs__onhw_wi_word_rh"),
        },
        "HWRFormer_AR_no_gating": {
            **cell(RESULTS / "Baseline-AR-XS-Ungated/ar_transformer_xs__onhw_wi_word_rh"),
        },
        "HWRFormer_AR_elementwise": {
            **cell(RESULTS / "Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh"),
        },
        "HWRFormer_AR_headwise": {
            **cell(RESULTS / "Baseline-AR-XS-HeadwiseGating/ar_transformer_xs__onhw_wi_word_rh"),
        },
    }

    # ----- Table 2: Cross-dataset (private words + sentences) -----
    out["table2_private"] = {}
    for variant_key, root in [
        ("HWRFormer_AR_elementwise", "Baseline-AR-XS-blconv_b"),
        ("HWRFormer_AR_headwise",    "Baseline-AR-XS-HeadwiseGating"),
    ]:
        out["table2_private"][variant_key] = {
            "private_words": cell(RESULTS / root / "ar_transformer_xs__wi_word_hw6_meta"),
            "private_sent":  cell(RESULTS / root / "ar_transformer_xs__wi_sent_hw6_meta"),
        }
    out["table2_private"]["REWI_BiLSTM_CTC"] = {
        "note": "Reference from REWI paper",
        "private_words": {"cer_mean": 9.39, "cer_std": 0.53, "wer_mean": 31.82, "wer_std": 0.62},
        "private_sent":  {"cer_mean": 6.55, "cer_std": 0.63, "wer_mean": 23.52, "wer_std": 2.03},
    }

    # ----- Table 3: Hybrid ablation (OnHW-WI), λ_ctc = 0.1 (XS optimum from λ sweep) -----
    out["table3_hybrid"] = {
        "AR_only":         cell(RESULTS / "Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh"),
        "hybrid_independent": cell(RESULTS / "train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh"),
        "hybrid_weight_tied": cell(RESULTS / "train_element_word_hybrid_01_xs_onhw_wi_ctc_to_ar_outproj/ar_transformer_xs__onhw_wi_word_rh"),
    }

    # ----- Table 4: Corruption modes × 6 datasets -----
    out["table4_corruption_modes"] = {}
    MODE_DIR = {
        "uniform":      "Baseline-AR-XS-InputCorruption-uniform",
        "bigramright":  "Baseline-AR-XS-InputCorruption-bigramright",
        "bigramleft":   "Baseline-AR-XS-InputCorruption-bigramleft",
        "selfconf":     "Baseline-AR-XS-InputCorruption-selfconf",
        "adjacentswap": "Baseline-AR-XS-InputCorruption-adjacentswap",
    }
    BASELINE_DIR = {
        "onhw_wi_word_rh":           "Baseline-AR-XS-blconv_b",
        "onhw_wd_word_rh":           "Baseline-AR-XS-blconv_b",       # may differ; check after run
        "onhw_equations_wi_word_rh": "Baseline-AR-XS-blconv_b",
        "onhw_equations_wd_word_rh": "Baseline-AR-XS-blconv_b",
        "wi_word_hw6_meta":          "Baseline-AR-XS-blconv_b",
        "wi_sent_hw6_meta":          "Baseline-AR-XS-blconv_b",
    }
    for dataset, _label in DATASET_LABELS.items():
        out["table4_corruption_modes"][dataset] = {}
        # No-corruption baseline
        out["table4_corruption_modes"][dataset]["__baseline__"] = cell(
            RESULTS / BASELINE_DIR[dataset] / f"ar_transformer_xs__{dataset}"
        )
        for mode, mode_root in MODE_DIR.items():
            out["table4_corruption_modes"][dataset][mode] = cell(
                RESULTS / mode_root / f"ar_transformer_xs__{dataset}"
            )

    # ----- Figure: lambda sweep on OnHW-WI -----
    out["figure_lambda_sweep"] = {}
    for k in range(1, 11):
        lam = k / 10.0
        d = RESULTS / f"train_element_word_hybrid_{k:02d}_xs_onhw_wi" / "ar_transformer_xs__onhw_wi_word_rh"
        out["figure_lambda_sweep"][f"lambda_{lam:.1f}"] = cell(d)

    # ----- Figure: p_ic sweep on OnHW-WI words, OnHW-WD words, Priv-sent -----
    out["figure_pic_sweep"] = {}
    for dataset_key in ["onhw_wi_word_rh", "onhw_wd_word_rh", "wi_sent_hw6_meta"]:
        out["figure_pic_sweep"][dataset_key] = {}
        # p=0.15 is the main XS uniform corruption run
        out["figure_pic_sweep"][dataset_key]["p_0.15"] = cell(
            RESULTS / "Baseline-AR-XS-InputCorruption-uniform" / f"ar_transformer_xs__{dataset_key}"
        )
        for p in (0.05, 0.10, 0.20, 0.30):
            pstr = f"p0p{int(round(p*100)):02d}"
            d = RESULTS / "Baseline-AR-XS-InputCorruption-Sweep-blconv_b" / f"ar_transformer_xs__{dataset_key}__{pstr}"
            out["figure_pic_sweep"][dataset_key][f"p_{p:.2f}"] = cell(d)

    # ----- Summary -----
    statuses: dict[str, int] = {"ready": 0, "pending": 0, "missing": 0}

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("status") in statuses:
                statuses[obj["status"]] += 1
                return
            for v in obj.values():
                walk(v)

    walk(out)
    out["_meta"]["summary_cell_counts"] = statuses

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"saved: {args.out}")
    print(f"summary: {statuses}")


if __name__ == "__main__":
    main()
