#!/usr/bin/env python3
"""Combine two per-model CSVs into one multi-task CSV.

Goal: reuse existing quantitative analysis scripts by treating *model type*
(e.g., AR-only vs Hybrid) as the `task` column.

Inputs are expected to be semicolon-separated CSVs with at least:
  task, fold, json_path, sample_index, prediction, label, levenshtein_distance

The output keeps the same canonical columns, and overwrites `task` with the
provided method labels. The original `task` is preserved as `dataset_task`.

Example (run from work/REWI_work):

python3 analysis/scripts/combine_task_csvs.py \
  --ar_csv analysis/ar_only_all_folds.csv \
  --hyb_csv analysis/hybrid_ar_all_folds.csv \
  --out_csv analysis/quant_all_val_predictions_ar_vs_hybrid.csv \
  --ar_name "AR-only" \
  --hyb_name "Hybrid (AR head)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


_REQUIRED = [
    "task",
    "fold",
    "json_path",
    "sample_index",
    "prediction",
    "label",
    "levenshtein_distance",
]


def _read_csv(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Found: {list(df.columns)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_csv", required=True)
    ap.add_argument("--hyb_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ar_name", default="AR-only")
    ap.add_argument("--hyb_name", default="Hybrid (AR head)")
    ap.add_argument("--sep", default=";")

    args = ap.parse_args()

    ar = _read_csv(args.ar_csv, args.sep).copy()
    hyb = _read_csv(args.hyb_csv, args.sep).copy()

    # Preserve original dataset marker (often "word" / "sent").
    ar["dataset_task"] = ar["task"].astype(str)
    hyb["dataset_task"] = hyb["task"].astype(str)

    ar["task"] = args.ar_name
    hyb["task"] = args.hyb_name

    out = pd.concat([ar, hyb], ignore_index=True)

    # Keep deterministic ordering for sanity-checking across tools.
    out = out.sort_values(["task", "fold", "json_path", "sample_index"], kind="mergesort")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep=args.sep, index=False)

    print("Wrote:", str(out_path))
    print("Rows:", len(out))
    print("Tasks:", sorted(out["task"].unique().tolist()))


if __name__ == "__main__":
    main()
