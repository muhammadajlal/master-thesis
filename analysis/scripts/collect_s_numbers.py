#!/usr/bin/env python3
"""Collect HWRFormer (scaled) results needed by the thesis Ch.5 noise-injection
figures into ``s_numbers.json``.

Mirrors the per-fold aggregation convention of ``collect_xs_numbers.py``:
each cell pulls ``best.character_error_rate[1]`` / ``best.word_error_rate[1]``
from the last ``train_*.json`` under each ``fold_k/k/`` directory, then
reports ``cer_mean`` / ``cer_std`` / ``wer_mean`` / ``wer_std`` (and ``n_folds``)
in percent.

Output JSON keys:
    table4_corruption_modes_s   mode -> dataset -> stats
    figure_pic_sweep_s          dataset -> p_value -> stats
    figure_lambda_sweep_s       single sparse entry (onhw_wi_word_rh -> 0.6)
    baseline_no_noise_s         dataset -> stats (no-noise scaled anchor;
                                 also reused as p=0.00 of figure_pic_sweep_s)

Run:
    /home/woody/iwso/iwso214h/imu-hwr/envs/rewi26/bin/python \
        /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/scripts/collect_s_numbers.py
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
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/s_numbers.json"
)

DATASETS = [
    "onhw_wi_word_rh",
    "onhw_wd_word_rh",
    "wi_word_hw6_meta",
    "wi_sent_hw6_meta",
]

MODES = ["uniform", "bigramright", "bigramleft", "selfconf", "adjacentswap"]

# Matrix A scaled mode directories.
# WI/private datasets live under a single per-mode dir; WD lives in its
# own per-mode dir (dataset-suffixed).
MODE_DIRS_WI_PRIV: dict[str, str] = {
    "uniform":      "Baseline-AR-InputCorruption-blconv_b",
    "bigramright":  "Baseline-AR-InputCorruption-Bigram-blconv_b",
    "bigramleft":   "Baseline-AR-InputCorruption-BigramLeft-blconv_b",
    "selfconf":     "Baseline-AR-InputCorruption-SelfConf-blconv_b",
    "adjacentswap": "Baseline-AR-InputCorruption-AdjacentSwap-blconv_b",
}
MODE_DIRS_WD: dict[str, str] = {
    "uniform":      "Baseline-AR-InputCorruption-WD-uniform-blconv_b",
    "bigramright":  "Baseline-AR-InputCorruption-WD-bigramright-blconv_b",
    "bigramleft":   "Baseline-AR-InputCorruption-WD-bigramleft-blconv_b",
    "selfconf":     "Baseline-AR-InputCorruption-WD-selfconf-blconv_b",
    "adjacentswap": "Baseline-AR-InputCorruption-WD-adjacentswap-blconv_b",
}

# Canonical no-noise scaled group per dataset.
NO_NOISE_DIR: dict[str, str] = {
    "onhw_wi_word_rh":  "Baseline-AR-ElementwiseGating",
    "onhw_wd_word_rh":  "Baseline-AR-ElementwiseGating-WD",
    "wi_word_hw6_meta": "Baseline-AR-ElementwiseGating",
    "wi_sent_hw6_meta": "Baseline-AR-ElementwiseGating",
}

# Matrix B: p_ic sweep groups (default p=0.15 plus the Sweep group for other p).
P_DEFAULT_DIR: dict[str, str] = {
    "onhw_wi_word_rh":  "Baseline-AR-InputCorruption-blconv_b",
    "onhw_wd_word_rh":  "Baseline-AR-InputCorruption-WD-uniform-blconv_b",
    "wi_word_hw6_meta": "Baseline-AR-InputCorruption-blconv_b",
    "wi_sent_hw6_meta": "Baseline-AR-InputCorruption-blconv_b",
}
P_SWEEP_DIR = "Baseline-AR-InputCorruption-Sweep-blconv_b"

# Matrix C: lambda=0.6 scaled, OnHW-WI words only.
LAMBDA_06_ONHW_WI_DIR = (
    "Baseline-Hybrid/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh"
)


def read_fold(model_dir: Path) -> tuple[list[float], list[float], int]:
    """Read 5-fold CER/WER (percent) from ``train_*.json``.

    Returns ``(cers, wers, n_folds)``. Missing folds are silently skipped.
    """
    cers: list[float] = []
    wers: list[float] = []
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
    """Aggregate a single experiment-group cell.

    Always returns the 5 fields expected by the thesis figure scripts.
    Missing/partial folds return null means/stds (n_folds < 5 will be
    plot-faded by the downstream figure scripts).
    """
    cers, wers, n = read_fold(model_dir)
    if n == 0:
        return {
            "n_folds": 0,
            "cer_mean": None,
            "cer_std":  None,
            "wer_mean": None,
            "wer_std":  None,
            "path":     str(model_dir),
        }
    cer_mean = round(statistics.mean(cers), 2)
    wer_mean = round(statistics.mean(wers), 2)
    if n >= 2:
        cer_std = round(statistics.stdev(cers), 2)
        wer_std = round(statistics.stdev(wers), 2)
    else:
        cer_std = None
        wer_std = None
    return {
        "n_folds": n,
        "cer_mean": cer_mean,
        "cer_std":  cer_std,
        "wer_mean": wer_mean,
        "wer_std":  wer_std,
        "path":     str(model_dir),
    }


def describe(name: str, c: dict) -> str:
    cer = c.get("cer_mean")
    cer_s = f"{cer:.2f}" if isinstance(cer, (int, float)) else "  -- "
    wer = c.get("wer_mean")
    wer_s = f"{wer:.2f}" if isinstance(wer, (int, float)) else "  -- "
    return f"  {name:<48s}  n={c['n_folds']}  CER={cer_s}  WER={wer_s}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out: dict = {
        "_meta": {
            "results_root": str(RESULTS),
            "schema": "scaled (ar_transformer_s) cells; 5-fold mean+std in percent",
            "notes": (
                "no-noise baseline group = Baseline-AR-ElementwiseGating "
                "(and -WD for onhw_wd_word_rh); same numbers reused as "
                "p=0.00 anchor of figure_pic_sweep_s."
            ),
        }
    }

    # -----------------------------------------------------------------
    # Baseline no-noise (also figure_pic_sweep_s p=0.00 anchor).
    # -----------------------------------------------------------------
    print("[no-noise baseline (scaled)]")
    baseline_no_noise: dict = {}
    for ds in DATASETS:
        d = RESULTS / NO_NOISE_DIR[ds] / f"ar_transformer_s__{ds}"
        c = cell(d)
        baseline_no_noise[ds] = c
        print(describe(ds, c))
    out["baseline_no_noise_s"] = baseline_no_noise

    # -----------------------------------------------------------------
    # Table 4: corruption modes (mode -> dataset).
    # -----------------------------------------------------------------
    print("\n[table4_corruption_modes_s]")
    out["table4_corruption_modes_s"] = {}
    for mode in MODES:
        out["table4_corruption_modes_s"][mode] = {}
        print(f"  mode={mode}")
        for ds in DATASETS:
            if ds == "onhw_wd_word_rh":
                root = MODE_DIRS_WD[mode]
            else:
                root = MODE_DIRS_WI_PRIV[mode]
            d = RESULTS / root / f"ar_transformer_s__{ds}"
            c = cell(d)
            out["table4_corruption_modes_s"][mode][ds] = c
            print(describe(f"{mode}/{ds}", c))

    # -----------------------------------------------------------------
    # Figure: p_ic sweep (dataset -> p_value).
    # -----------------------------------------------------------------
    print("\n[figure_pic_sweep_s]")
    out["figure_pic_sweep_s"] = {}
    P_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    for ds in DATASETS:
        out["figure_pic_sweep_s"][ds] = {}
        print(f"  dataset={ds}")
        for p in P_VALUES:
            pkey = f"{p:.2f}"
            if p == 0.00:
                # Reuse the no-noise baseline cell exactly.
                c = dict(baseline_no_noise[ds])
            elif p == 0.15:
                d = RESULTS / P_DEFAULT_DIR[ds] / f"ar_transformer_s__{ds}"
                c = cell(d)
            else:
                pstr = f"p0p{int(round(p * 100)):02d}"
                d = RESULTS / P_SWEEP_DIR / f"ar_transformer_s__{ds}__{pstr}"
                c = cell(d)
            out["figure_pic_sweep_s"][ds][pkey] = c
            print(describe(f"{ds}/p={pkey}", c))

    # -----------------------------------------------------------------
    # Figure: lambda sweep (sparse, single entry).
    # -----------------------------------------------------------------
    print("\n[figure_lambda_sweep_s]")
    d = RESULTS / LAMBDA_06_ONHW_WI_DIR
    c = cell(d)
    out["figure_lambda_sweep_s"] = {"onhw_wi_word_rh": {"0.6": c}}
    print(describe("onhw_wi_word_rh/lambda=0.6", c))

    # -----------------------------------------------------------------
    # Summary of per-cell coverage.
    # -----------------------------------------------------------------
    full = 0
    partial: list[str] = []
    missing: list[str] = []

    def walk(prefix: str, obj) -> None:
        nonlocal full
        if isinstance(obj, dict):
            if "n_folds" in obj and "cer_mean" in obj:
                n = obj["n_folds"]
                if n == 5:
                    full += 1
                elif n == 0:
                    missing.append(prefix)
                else:
                    partial.append(f"{prefix} (n={n})")
                return
            for k, v in obj.items():
                walk(f"{prefix}/{k}" if prefix else k, v)

    for top in ("baseline_no_noise_s", "table4_corruption_modes_s",
                "figure_pic_sweep_s", "figure_lambda_sweep_s"):
        walk(top, out[top])

    out["_meta"]["coverage"] = {
        "cells_full":    full,
        "cells_partial": partial,
        "cells_missing": missing,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Coverage summary ===")
    print(f"  full (n=5):   {full}")
    print(f"  partial:      {len(partial)}")
    for p in partial:
        print(f"    {p}")
    print(f"  missing:      {len(missing)}")
    for m in missing:
        print(f"    {m}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
