#!/usr/bin/env python3
"""Aggregate XS val_full export JSONs into a single CSV.

Mirrors the schema of `analysis/quant_all_val_predictions_ar_vs_hybrid.csv`
but for the XS swap. Source JSONs live under
  `results/hwr2/Baseline-AR-XS-blconv_b/<arch>/fold_<k>/exports/val_full_fold<k>_epoch0.json`
  `results/hwr2/train_element_word_hybrid_06_xs_onhw_wi/<arch>/fold_<k>/exports/val_full_fold<k>_epoch0_ar.json`
(plus the matching onhw_wd directories).

Output: `analysis/quant_all_val_predictions_ar_vs_hybrid_xs.csv` (";"-separated).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_hybrid_xs.csv"

# (task_label, model_root, arch, dataset_task) tuples.
SOURCES: list[tuple[str, str, str, str]] = [
    ("AR-only", "Baseline-AR-XS-blconv_b",
     "ar_transformer_xs__onhw_wi_word_rh", "word"),
    ("AR-only", "Baseline-AR-XS-blconv_b",
     "ar_transformer_xs__onhw_wd_word_rh", "word"),
    ("Hybrid (AR Decoding)", "train_element_word_hybrid_06_xs_onhw_wi",
     "ar_transformer_xs__onhw_wi_word_rh", "word"),
    ("Hybrid (AR Decoding)", "train_element_word_hybrid_06_xs_onhw_wd",
     "ar_transformer_xs__onhw_wd_word_rh", "word"),
]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def find_export(model_root: str, arch: str, fold: int, is_hybrid: bool) -> Path | None:
    exports = RESULTS / model_root / arch / f"fold_{fold}" / "exports"
    suffix = "_ar" if is_hybrid else ""
    candidate = exports / f"val_full_fold{fold}_epoch0{suffix}.json"
    if candidate.exists():
        return candidate
    # Fall back to the smallest-epoch export available.
    pattern = f"val_full_fold{fold}_epoch*{suffix}.json"
    options = sorted(exports.glob(pattern))
    return options[0] if options else None


def main() -> None:
    rows = []
    missing: list[str] = []
    total = 0
    for task, model_root, arch, dataset_task in SOURCES:
        is_hybrid = task.startswith("Hybrid")
        for fold in range(5):
            export = find_export(model_root, arch, fold, is_hybrid)
            if export is None:
                missing.append(f"{task} {model_root} fold {fold}")
                continue
            with open(export) as fp:
                d = json.load(fp)
            preds = d.get("predictions", [])
            labels = d.get("labels", [])
            n = min(len(preds), len(labels))
            for idx in range(n):
                p = "" if preds[idx] is None else str(preds[idx])
                l = "" if labels[idx] is None else str(labels[idx])
                rows.append({
                    "task": task,
                    "fold": fold,
                    "json_path": str(export),
                    "sample_index": idx,
                    "prediction": p,
                    "label": l,
                    "levenshtein_distance": levenshtein(p, l),
                    "dataset_task": dataset_task,
                })
            total += n
            print(f"  + {task:<22s} fold {fold} {arch:<48s} n={n}")

    if missing:
        print("Missing exports (skipped):")
        for m in missing:
            print(f"  - {m}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "fold", "json_path", "sample_index",
        "prediction", "label", "levenshtein_distance", "dataset_task",
    ]
    with open(OUT_CSV, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows ({total} predictions) to {OUT_CSV}")


if __name__ == "__main__":
    main()
