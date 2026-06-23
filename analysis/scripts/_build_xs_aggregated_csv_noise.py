#!/usr/bin/env python3
"""Aggregate XS val_full export JSONs for the AR-only versus AR + noise
injection paired comparison.

Mirrors _build_xs_aggregated_csv.py but with sources restricted to the
AR-only blconv_b baseline and the AR + noise injection (uniform,
p_replace=0.15) checkpoint on OnHW WI word. Both are AR-only architectures
so the export filename carries no _ar suffix.

Output: analysis/quant_all_val_predictions_ar_vs_noise_xs.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_noise_xs.csv"

DATASETS: list[tuple[str, str]] = [
    ("ar_transformer_xs__onhw_wi_word_rh", "onhw_wi_word"),
    ("ar_transformer_xs__onhw_wd_word_rh", "onhw_wd_word"),
    ("ar_transformer_xs__wi_word_hw6_meta", "priv_word"),
    ("ar_transformer_xs__wi_sent_hw6_meta", "priv_sent"),
]

SOURCES: list[tuple[str, str, str, str]] = []
for arch, dataset_task in DATASETS:
    SOURCES.append(("AR-only", "Baseline-AR-XS-blconv_b", arch, dataset_task))
    SOURCES.append(("Noise (uniform p=0.15)",
                    "Baseline-AR-XS-InputCorruption-uniform",
                    arch, dataset_task))


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


def find_export(model_root: str, arch: str, fold: int) -> Path | None:
    exports = RESULTS / model_root / arch / f"fold_{fold}" / "exports"
    candidate = exports / f"val_full_fold{fold}_epoch0.json"
    if candidate.exists():
        return candidate
    pattern = f"val_full_fold{fold}_epoch*.json"
    # Exclude the _ar suffix files so we do not accidentally pick a hybrid
    # export sharing the same directory.
    options = [p for p in sorted(exports.glob(pattern)) if not p.stem.endswith("_ar")]
    return options[0] if options else None


def main() -> None:
    rows = []
    missing: list[str] = []
    total = 0
    for task, model_root, arch, dataset_task in SOURCES:
        for fold in range(5):
            export = find_export(model_root, arch, fold)
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
            print(f"  + {task:<28s} fold {fold} {arch:<48s} n={n}")

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
