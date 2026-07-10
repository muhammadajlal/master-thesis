#!/usr/bin/env python3
"""Batch-recompute MACs and params for every architectural combo in the thesis.

Per evaluate.py:get_macs_params(), the encoder signal length is T=1024 for
'word' datasets and T=4096 for sentences, and the AR decode budget defaults to
the rounded-up mean output length: 6 characters at the OnHW-words500 word
reference (used for every word task) and 19 for private sentences — the thesis
reporting convention that matches the paper's Table 1. So MACs/params depend
only on (arch_en, arch_de, gating, hybrid, task_type). We compute each unique
combo once.

Output: results/thesis_macs.csv

Usage:
    envs/rewi26/bin/python work/REWI_work/analysis/scripts/batch_macs_for_thesis.py
"""
from __future__ import annotations

import csv
import sys
import time
import yaml
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from evaluate import get_macs_params  # noqa: E402

# Each entry: (thesis_label, config_path). Path RELATIVE to PROJ.
# We pick ONE config per (arch, task) and rely on T=1024/4096 hardcoding in
# get_macs_params to make MACs comparable across same-task datasets.
CONFIGS: list[tuple[str, str]] = [
    # --- REWI / BiLSTM-CTC (uses configs/train.yaml as canonical) ----------
    ("REWI BiLSTM-CTC | word",          "configs/train.yaml"),

    # --- Transformer-CTC (s) ----------------------------------------------
    ("Transformer-CTC s | word",        "configs/AR-Baseline/train-transformer-ctc-onhw-word.yaml"),
    ("Transformer-CTC s | priv sent",   "configs/CTC_Primary/train_element_word_02.yaml"),

    # --- Transformer-CTC (xs) ---------------------------------------------
    ("Transformer-CTC xs | word",       "configs/AR-Baseline/train-transformer-xs-ctc-onhw-word.yaml"),
    ("Transformer-CTC xs | priv sent",  "configs/AR-Baseline/train-transformer-xs-ctc-stabilo-sent.yaml"),

    # --- AR Transformer (s) ----------------------------------------------
    ("AR s ungated  | OnHW word",       "configs/AR-Baseline/train-ar-ungated-onhw-word.yaml"),
    ("AR s elem     | OnHW word",       "configs/train_element_word.yaml"),
    ("AR s elem     | priv sent",       "configs/train_sent.yaml"),
    ("AR s headwise | priv word",       "configs/train_head_word.yaml"),
    ("AR s headwise | priv sent",       "configs/train_head_sent.yaml"),

    # --- AR Transformer (xs) ---------------------------------------------
    ("AR xs elem     | OnHW word",      "configs/AR-Baseline/train-ar-baseline-xs-onhw-word.yaml"),
    ("AR xs elem     | priv sent",      "configs/AR-Baseline/train-ar-baseline-xs-stabilo-sent.yaml"),
    ("AR xs ungated  | OnHW word",      "configs/AR-Baseline/train-ar-xs-ungated-onhw-word.yaml"),
    ("AR xs ungated  | priv sent",      "configs/AR-Baseline/train-ar-xs-ungated-stabilo-sent.yaml"),
    ("AR xs headwise | OnHW word",      "configs/AR-Baseline/train-ar-xs-headwise-onhw-word.yaml"),
    ("AR xs headwise | priv sent",      "configs/AR-Baseline/train-ar-xs-headwise-stabilo-sent.yaml"),

    # --- Hybrid CTC-AR (s, lambda=0.6) -----------------------------------
    ("Hybrid s lam=0.6 | OnHW word",    "configs/hybrid/train_element_word_06.yaml"),
    ("Hybrid s lam=0.6 | priv sent",    "configs/hybrid/train_element_word_06_stabilo_sent.yaml"),
    ("Hybrid s lam=0.6 | tied",         "configs/hybrid/train_element_word_06_fulltying.yaml"),

    # --- Hybrid CTC-AR (xs) ----------------------------------------------
    ("Hybrid xs lam=0.6 | OnHW word",       "configs/hybrid-xs/train_element_word_06_xs_onhw_wi.yaml"),
    ("Hybrid xs lam=0.6 | priv sent",       "configs/hybrid-xs/train_element_word_06_xs_stabilo_sent.yaml"),
    ("Hybrid xs lam=0.1 | OnHW word",       "configs/hybrid-xs/train_element_word_01_xs_onhw_wi.yaml"),
    ("Hybrid xs lam=0.1 | priv sent",       "configs/hybrid-xs/train_element_word_01_xs_stabilo_sent.yaml"),
    ("Hybrid xs lam=0.1 tied | OnHW word",  "configs/hybrid-xs/train_element_word_01_xs_onhw_wi_ctc_to_ar_outproj.yaml"),
]


def main() -> None:
    out_path = PROJ.parent.parent / "results" / "thesis_macs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"Running {len(CONFIGS)} unique architectural combos...")
    print(f"{'Label':40s}  {'Params (M)':>11s}  {'MACs (G)':>10s}  {'time':>6s}")
    print("-" * 80)

    for label, cfg_path in CONFIGS:
        full = PROJ / cfg_path
        if not full.exists():
            print(f"{label:40s}  [NOT FOUND] {cfg_path}")
            continue
        with open(full) as f:
            cfg = yaml.safe_load(f)
        t0 = time.time()
        try:
            out = get_macs_params(cfg, {})
        except Exception as exc:
            print(f"{label:40s}  [ERROR] {exc.__class__.__name__}: {exc}")
            continue
        dt = time.time() - t0
        params_m = out["params"] / 1e6
        macs_g = out["macs"] / 1e9
        print(f"{label:40s}  {params_m:11.3f}  {macs_g:10.3f}  {dt:5.1f}s")
        rows.append({
            "label": label,
            "config": cfg_path,
            "arch_en": cfg.get("arch_en", ""),
            "arch_de": cfg.get("arch_de", ""),
            "dir_dataset": cfg.get("dir_dataset", "") or "",
            "use_gated_attention": cfg.get("use_gated_attention", ""),
            "gating_type": cfg.get("gating_type", ""),
            "dual_head_enabled": (cfg.get("dual_head") or {}).get("enabled", ""),
            "lambda_ctc": (cfg.get("dual_head") or {}).get("lambda_ctc", ""),
            "use_bpe": cfg.get("use_bpe", ""),
            "params": out["params"],
            "macs": out["macs"],
            "flops": out["flops"],
            "generate_len_max": out["generate_len_max"],
        })

    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
