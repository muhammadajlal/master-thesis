#!/usr/bin/env python3
"""Export summary tables from prediction export JSONs.

This script reads one or more export JSON files (or fold templates) of the form:

  {"predictions": [...], "labels": [...], ...}

and computes character-level edit distance summaries, including:
- Exact match rate
- Micro normalized edit distance (equivalent to micro CER)
- Macro normalized edit distance
- Quantiles of per-sample normalized edit distance
- Tail probabilities P(e > t)

It can optionally aggregate across cross-validation folds when the export path
contains a "{fold}" placeholder.

Example (run from work/REWI_work):

python3 analysis/scripts/export_error_tables.py \
  --folds 0,1,2,3,4 \
  --row "AR-only=../../results/hwr2/element_word_onhw500_new/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0.json" \
  --row "Hybrid (AR)=../../results/hwr2/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0_ar.json" \
  --row "Hybrid (CTC)=../../results/hwr2/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_{fold}/exports/val_full_fold{fold}_epoch0_ctc.json" \
  --outdir analysis/tables/ar_vs_hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep DP width to the shorter string.
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    cur[j - 1] + 1,  # insertion
                    prev[j] + 1,  # deletion
                    prev[j - 1] + (0 if ca == cb else 1),  # substitute
                )
            )
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Summary:
    N: int
    exact_pct: float
    mean_ref_len: float
    micro: float
    macro: float
    quantiles: Dict[str, float]
    tails_pct: Dict[str, float]


def _summarize(
    preds: Sequence[str],
    refs: Sequence[str],
    quantiles: Sequence[float],
    tail_thresholds: Sequence[float],
) -> Summary:
    preds = ["" if p is None else str(p) for p in preds]
    refs = ["" if r is None else str(r) for r in refs]

    # Character-level edit distance, normalized by |ref|.
    edits = np.fromiter(
        (_levenshtein_distance(p, r) for p, r in zip(preds, refs)),
        dtype=np.float64,
        count=len(refs),
    )
    ref_lens = np.fromiter((max(1, len(r)) for r in refs), dtype=np.float64, count=len(refs))
    norm = edits / ref_lens

    quantile_map = {
        f"p{int(q * 100):02d}": float(np.quantile(norm, q)) for q in quantiles
    }
    tail_map = {
        f"P(e>{t})": float((norm > t).mean() * 100.0) for t in tail_thresholds
    }

    return Summary(
        N=int(len(refs)),
        exact_pct=float((norm == 0).mean() * 100.0),
        mean_ref_len=float(np.mean([len(r) for r in refs])),
        micro=float(edits.sum() / ref_lens.sum()),
        macro=float(norm.mean()),
        quantiles=quantile_map,
        tails_pct=tail_map,
    )


def _parse_folds(arg: str) -> List[int]:
    if not arg.strip():
        return []
    return [int(x) for x in arg.split(",")]


def _parse_floats_csv(arg: str) -> List[float]:
    if not arg.strip():
        return []
    return [float(x) for x in arg.split(",")]


def _parse_rows(rows: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for r in rows:
        if "=" not in r:
            raise ValueError(f"Invalid --row {r!r}. Expected NAME=PATH.")
        name, path = r.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --row {r!r}. Expected NAME=PATH.")
        parsed.append((name, path))
    return parsed


def _read_export(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "predictions" not in data or "labels" not in data:
        raise KeyError(
            f"Export JSON missing keys 'predictions'/'labels': {path}"
        )

    preds = data["predictions"]
    refs = data["labels"]
    if len(preds) != len(refs):
        raise ValueError(
            f"predictions/labels length mismatch in {path}: {len(preds)} vs {len(refs)}"
        )
    return list(preds), list(refs)


def _fmt(x: float, ndigits: int = 4) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    return f"{x:.{ndigits}f}"


def _render_overall_table(methods: List[str], summaries: Mapping[str, Summary]) -> str:
    cols = [
        "Method",
        "Exact (\\%)",
        "Micro",
        "Macro",
        "p50",
        "p90",
        "p95",
        "p99",
    ]

    lines: List[str] = []
    lines.append("\\begin{tabular}{lrrrrrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + r" \\")
    lines.append("\\hline")

    for name in methods:
        s = summaries[name]
        q = s.quantiles
        row = [
            name,
            _fmt(s.exact_pct, 2),
            _fmt(s.micro, 4),
            _fmt(s.macro, 4),
            _fmt(q.get("p50", float("nan"))),
            _fmt(q.get("p90", float("nan"))),
            _fmt(q.get("p95", float("nan"))),
            _fmt(q.get("p99", float("nan"))),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _render_per_fold_table(
    folds: Sequence[int],
    methods: List[str],
    per_fold: Mapping[int, Mapping[str, Summary]],
) -> str:
    cols = ["Fold", "Method", "Exact (\\%)", "Micro", "Macro", "p90", "p99"]

    lines: List[str] = []
    lines.append("\\begin{tabular}{r l r r r r r}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + r" \\")
    lines.append("\\hline")

    for fold in folds:
        for name in methods:
            s = per_fold[fold][name]
            q = s.quantiles
            row = [
                str(fold),
                name,
                _fmt(s.exact_pct, 2),
                _fmt(s.micro, 4),
                _fmt(s.macro, 4),
                _fmt(q.get("p90", float("nan"))),
                _fmt(q.get("p99", float("nan"))),
            ]
            lines.append(" & ".join(row) + r" \\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--row",
        action="append",
        default=[],
        help="Method row as NAME=PATH. PATH may include {fold} placeholder.",
    )
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--quantiles", default="0.50,0.90,0.95,0.99")
    ap.add_argument("--tail_thresholds", default="0.5,1.0")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--no_per_fold",
        action="store_true",
        help="Skip per-fold table output.",
    )

    args = ap.parse_args()
    rows = _parse_rows(args.row)
    if not rows:
        raise SystemExit("At least one --row NAME=PATH is required.")

    folds = _parse_folds(args.folds)
    qs = _parse_floats_csv(args.quantiles)
    tails = _parse_floats_csv(args.tail_thresholds)

    os.makedirs(args.outdir, exist_ok=True)

    methods = [name for name, _ in rows]
    overall_preds: Dict[str, List[str]] = {name: [] for name in methods}
    overall_refs: Dict[str, List[str]] = {name: [] for name in methods}
    per_fold: Dict[int, Dict[str, Summary]] = {}

    for fold in folds:
        per_fold[fold] = {}
        for name, path_tmpl in rows:
            path = path_tmpl.format(fold=fold)
            preds, refs = _read_export(path)
            per_fold[fold][name] = _summarize(preds, refs, qs, tails)
            overall_preds[name].extend(preds)
            overall_refs[name].extend(refs)

    overall_summary = {
        name: _summarize(overall_preds[name], overall_refs[name], qs, tails)
        for name in methods
    }

    # Write machine-readable JSON
    json_out: Dict[str, Any] = {
        "folds": folds,
        "quantiles": qs,
        "tail_thresholds": tails,
        "overall": {
            name: {
                **{
                    "N": s.N,
                    "exact_pct": s.exact_pct,
                    "mean_ref_len": s.mean_ref_len,
                    "micro": s.micro,
                    "macro": s.macro,
                },
                "quantiles": s.quantiles,
                "tails_pct": s.tails_pct,
            }
            for name, s in overall_summary.items()
        },
        "per_fold": {
            str(fold): {
                name: {
                    **{
                        "N": s.N,
                        "exact_pct": s.exact_pct,
                        "mean_ref_len": s.mean_ref_len,
                        "micro": s.micro,
                        "macro": s.macro,
                    },
                    "quantiles": s.quantiles,
                    "tails_pct": s.tails_pct,
                }
                for name, s in per_fold[fold].items()
            }
            for fold in folds
        },
    }
    with open(os.path.join(args.outdir, "error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    # Write LaTeX
    with open(os.path.join(args.outdir, "table_overall.tex"), "w", encoding="utf-8") as f:
        f.write(_render_overall_table(methods, overall_summary))

    if not args.no_per_fold:
        with open(os.path.join(args.outdir, "table_per_fold.tex"), "w", encoding="utf-8") as f:
            f.write(_render_per_fold_table(folds, methods, per_fold))


if __name__ == "__main__":
    main()
