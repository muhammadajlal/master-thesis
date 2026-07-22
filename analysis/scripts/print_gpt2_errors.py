#!/usr/bin/env python3
"""Aggregate the regenerated HWR-GPT (Q-Former + auxiliary CTC) validation
predictions into per-dataset error lists.

Reads the val_full exports written by export_gpt2_errors.sbatch:
    <root>/<tag>/fold_<k>/exports/val_full_fold<k>_*_lm.json   {"predictions", "labels"}

For each dataset (onhw_wi, priv_word) it emits, over all available folds:
  - a readable .txt in the exact `reference -> prediction (d=...)` format;
  - a .csv with columns fold, sample_id, reference, prediction, levenshtein.
Only incorrect predictions (Levenshtein distance > 0) are listed. Each
validation sample appears once (folds are writer-disjoint). No output is
labelled English-like; these are the raw errors.

Run from work/REWI_work:
    python analysis/scripts/print_gpt2_errors.py --root /home/woody/iwso/iwso214h/gpt2_error_export
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import Levenshtein

TAGS = [("onhw_wi", "OnHW WI (Q-Former + aux CTC, lambda_ctc=0.2)"),
        ("priv_word", "Private word (Q-Former + aux CTC, lambda_ctc=0.6)")]


def load_fold(root: Path, tag: str, fold: int):
    pat = str(root / tag / f"fold_{fold}" / "exports" / f"val_full_fold{fold}_*_lm.json")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    d = json.load(open(files[-1], encoding="utf-8"))
    return d["predictions"], d["labels"]


def build(root: Path, tag: str):
    rows = []
    folds_found = []
    for k in range(5):
        got = load_fold(root, tag, k)
        if got is None:
            continue
        folds_found.append(k)
        preds, labels = got
        for idx, (p, y) in enumerate(zip(preds, labels)):
            d = Levenshtein.distance(p, y)
            if d > 0:
                rows.append({"fold": k, "sample_id": idx,
                             "reference": y, "prediction": p, "levenshtein": d})
    return rows, folds_found


def write_outputs(root: Path, tag: str, desc: str, rows, folds):
    out_txt = root / f"gpt2_errors_{tag}.txt"
    out_csv = root / f"gpt2_errors_{tag}.csv"
    n_total_est = "?"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"# HWR-GPT errors — {desc}\n")
        f.write(f"# folds included: {folds}\n")
        f.write(f"# incorrect samples (Levenshtein d>0): {len(rows)}\n")
        f.write("# format: reference -> prediction (d=<levenshtein>)  [fold k, id N]\n\n")
        for r in rows:
            f.write(f"{r['reference']} -> {r['prediction']} "
                    f"(d={r['levenshtein']})  [fold {r['fold']}, id {r['sample_id']}]\n")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "sample_id", "reference",
                                          "prediction", "levenshtein"],
                           delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{tag}] folds {folds} | {len(rows)} errors -> {out_txt.name}, {out_csv.name}")
    return out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/home/woody/iwso/iwso214h/gpt2_error_export"))
    ap.add_argument("--preview", type=int, default=0,
                    help="Print the first N lines of each .txt to stdout.")
    args = ap.parse_args()

    for tag, desc in TAGS:
        rows, folds = build(args.root, tag)
        if not folds:
            print(f"[{tag}] no exports found yet under {args.root}/{tag}")
            continue
        txt = write_outputs(args.root, tag, desc, rows, folds)
        if args.preview:
            print(f"\n----- {desc} (first {args.preview}) -----")
            with open(txt, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= args.preview + 5:
                        break
                    print(line.rstrip())
            print("-----\n")


if __name__ == "__main__":
    main()
