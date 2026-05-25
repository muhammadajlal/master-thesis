#!/usr/bin/env python3
"""Re-measure MACs at a configurable generate len_max for paper 2 Table 1.

Usage:
    python measure_macs_at_lenmax.py --len_max 6  # mean OnHW-Words500
"""
from __future__ import annotations

import argparse
import json
import sys
import yaml
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from evaluate import get_macs_params  # noqa: E402

CONFIGS = [
    ("REWI BiLSTM-CTC",         "configs/decode_study/base_rewi_ctc.yaml"),
    ("HWRFormer CTC",           "configs/AR-Baseline/train-transformer-xs-ctc-onhw-word.yaml"),
    ("HWRFormer AR no-gate",    "configs/AR-Baseline/train-ar-xs-ungated-onhw-word.yaml"),
    ("HWRFormer AR elementwise","configs/AR-Baseline/train-ar-baseline-xs-onhw-word.yaml"),
    ("HWRFormer AR headwise",   "configs/AR-Baseline/train-ar-xs-headwise-onhw-word.yaml"),
    ("Hybrid indep lam=0.1",    "configs/hybrid-xs/train_element_word_01_xs_onhw_wi.yaml"),
    ("Hybrid tied  lam=0.1",    "configs/hybrid-xs/train_element_word_01_xs_onhw_wi_ctc_to_ar_outproj.yaml"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len_max", type=int, default=6,
                    help="Generation budget for AR/hybrid models (default 6 = OnHW-Words500 mean ceil)")
    args = ap.parse_args()

    print(f"{'Model':30s}  {'Params (M)':>10s}  {'MACs (G)':>10s}  len_max={args.len_max}")
    print("-" * 65)

    for label, cfg_path in CONFIGS:
        full = PROJ / cfg_path
        if not full.exists():
            print(f"{label:30s}  NOT FOUND: {full}")
            continue
        with open(full) as f:
            cfg = yaml.safe_load(f)
        cfg["generate_len_max"] = int(args.len_max)
        out = get_macs_params(cfg)
        params_m = out["params"] / 1e6
        macs_g = out["macs"] / 1e9
        print(f"{label:30s}  {params_m:10.3f}  {macs_g:10.3f}")


if __name__ == "__main__":
    main()
