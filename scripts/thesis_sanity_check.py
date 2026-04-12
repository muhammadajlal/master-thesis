#!/usr/bin/env python3
"""Thesis sanity-check: recompute CV results from raw train_*.json logs.

Fixes two bugs in top-level evaluate.py:
  1. Multiple train_*.json per fold (resumes/resubmits) are treated as
     separate folds, inflating CV count.
  2. `best` field in each train_*.json is local to that run and does not
     respect an epoch cap; no way to restrict to first N epochs.

This script globs each experiment directory, groups resume files per fold,
merges epoch-level `evaluation` blocks across resumes, caps the search at
MAX_EPOCH (default 299 = first 300 epochs), and recomputes CER/WER mean/std.
It writes a `results.recomputed.json` alongside the existing `results.json`
and prints a diff so tables can be updated.
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from statistics import mean, pstdev

import yaml

MAX_EPOCH_DEFAULT = 299  # inclusive cap -> "first 300 epochs" (0..299)


def _infer_primary_head(cfg: dict) -> str:
    dual = cfg.get("dual_head", {}) or {}
    if not bool(dual.get("enabled", False)):
        return "ar"
    primary_raw = dual.get("primary", dual.get("primary_head", "auto"))
    primary = str(primary_raw).lower() if primary_raw is not None else "auto"
    if primary in {"ar", "ctc"}:
        return primary
    try:
        la = float(dual.get("lambda_ar", 1.0))
    except Exception:
        la = 1.0
    try:
        lc = float(dual.get("lambda_ctc", 0.0))
    except Exception:
        lc = 0.0
    sched = dual.get("lambda_ctc_schedule", {}) or {}
    if bool(sched.get("enabled", False)) and "max" in sched:
        try:
            lc = float(sched.get("max", lc))
        except Exception:
            pass
    return "ctc" if lc >= la else "ar"


def _eval_keys_for(primary: str):
    if primary == "ctc":
        return ["evaluation_ctc", "evaluation"]
    return ["evaluation", "evaluation_ctc"]


def _load_yaml_if_any(fold_dir: str):
    yamls = sorted(glob(os.path.join(fold_dir, "train_*.yaml")))
    if not yamls:
        return {}
    # pick the newest; all resumes should share the same head config
    with open(yamls[-1]) as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}


def _merge_epoch_evals(fold_dir: str, eval_keys, max_epoch: int):
    """Return dict: epoch_int -> (cer, wer) from all train_*.json in fold_dir,
    scanning only epochs <= max_epoch and using the first available key.
    If the same epoch appears in multiple files, the newest file wins
    (by mtime)."""
    files = sorted(
        glob(os.path.join(fold_dir, "train_*.json")),
        key=lambda p: os.path.getmtime(p),
    )
    merged = {}  # epoch -> (cer, wer, src_file)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [WARN] failed to parse {f}: {e}", file=sys.stderr)
            continue
        for k, v in d.items():
            if not (isinstance(k, str) and k.isdigit()):
                continue
            ep = int(k)
            if ep > max_epoch:
                continue
            if not isinstance(v, dict):
                continue
            ev = None
            for ek in eval_keys:
                if ek in v and isinstance(v[ek], dict):
                    ev = v[ek]
                    break
            if ev is None:
                continue
            cer = ev.get("character_error_rate")
            wer = ev.get("word_error_rate")
            if cer is None or wer is None:
                continue
            merged[ep] = (float(cer), float(wer), os.path.basename(f))
    return merged, [os.path.basename(x) for x in files]


def recompute_experiment(exp_dir: str, max_epoch: int = MAX_EPOCH_DEFAULT):
    fold_dirs = sorted(glob(os.path.join(exp_dir, "fold_*", "*/")))
    fold_dirs = [d.rstrip("/") for d in fold_dirs if os.path.isdir(d)]
    if not fold_dirs:
        return None

    cer_list, wer_list = [], []
    details = []
    for fd in fold_dirs:
        cfg = _load_yaml_if_any(fd)
        primary = _infer_primary_head(cfg) if cfg else "ar"
        ekeys = _eval_keys_for(primary)
        merged, files = _merge_epoch_evals(fd, ekeys, max_epoch)
        if not merged:
            details.append(
                {
                    "fold_dir": fd,
                    "files": files,
                    "primary": primary,
                    "note": "no evaluation entries <= max_epoch",
                    "best_cer": None,
                }
            )
            continue
        best_ep = min(merged.keys(), key=lambda e: merged[e][0])
        best_cer, best_wer, src = merged[best_ep]
        cer_list.append(best_cer)
        wer_list.append(best_wer)
        details.append(
            {
                "fold_dir": fd,
                "files": files,
                "primary": primary,
                "best_epoch": best_ep,
                "best_cer": best_cer,
                "best_wer": best_wer,
                "best_src": src,
                "n_eval_epochs": len(merged),
                "max_epoch_seen": max(merged.keys()),
            }
        )

    if not cer_list:
        return {"details": details, "n_folds": 0}

    result = {
        "max_epoch_cap": max_epoch,
        "n_folds": len(cer_list),
        "cer": {
            "raw": {str(i): v for i, v in enumerate(cer_list)},
            "mean": mean(cer_list),
            "std": pstdev(cer_list),
        },
        "wer": {
            "raw": {str(i): v for i, v in enumerate(wer_list)},
            "mean": mean(wer_list),
            "std": pstdev(wer_list),
        },
        "details": details,
    }
    return result


def _load_existing(exp_dir: str):
    p = os.path.join(exp_dir, "results.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _find_experiments(root: str):
    """Experiment = deepest dir that directly contains fold_*/ children."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        if any(re.fullmatch(r"fold_\d+", d) for d in dirnames):
            out.append(dirpath)
            # do not descend into fold_* subtrees
            dirnames[:] = [d for d in dirnames if not re.fullmatch(r"fold_\d+", d)]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/hwr2",
                    help="Search root (relative to cwd or absolute)")
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH_DEFAULT)
    ap.add_argument("--write", action="store_true",
                    help="Write results.recomputed.json next to each results.json")
    ap.add_argument("--only-diffs", action="store_true",
                    help="Print only experiments whose recomputed numbers differ")
    ap.add_argument("--threshold", type=float, default=1e-6,
                    help="Min abs CER mean diff to count as changed")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    exps = _find_experiments(root)
    print(f"Found {len(exps)} experiments under {root}", file=sys.stderr)

    changed = []
    for exp in exps:
        rec = recompute_experiment(exp, max_epoch=args.max_epoch)
        if rec is None or rec.get("n_folds", 0) == 0:
            continue
        existing = _load_existing(exp)
        ex_mean = None
        ex_n = None
        if existing and isinstance(existing.get("cer"), dict):
            ex_mean = existing["cer"].get("mean")
            raw = existing["cer"].get("raw") or {}
            ex_n = len(raw) if isinstance(raw, dict) else None

        new_mean = rec["cer"]["mean"]
        diff = None if ex_mean is None else (new_mean - ex_mean)
        is_diff = (
            ex_mean is None
            or (ex_n is not None and ex_n != rec["n_folds"])
            or (diff is not None and abs(diff) >= args.threshold)
        )

        if args.only_diffs and not is_diff:
            continue

        rel = os.path.relpath(exp, root)
        marker = "!! " if is_diff else "   "
        ex_str = "-" if ex_mean is None else f"{ex_mean:.4f}(n={ex_n})"
        print(
            f"{marker}{rel}\n"
            f"      old_cer={ex_str}  new_cer={new_mean:.4f}(n={rec['n_folds']})"
            + (f"  diff={diff:+.4f}" if diff is not None else "")
        )
        # Flag per-fold epoch cap issues
        capped = [
            d for d in rec["details"]
            if d.get("max_epoch_seen") is not None
            and d.get("max_epoch_seen") == args.max_epoch
        ]
        beyond = [
            d for d in rec["details"]
            if d.get("best_src") and len(d.get("files", [])) > 1
        ]
        if beyond:
            print(f"      resumes merged in {len(beyond)}/{rec['n_folds']} folds")
        if is_diff:
            changed.append((rel, ex_mean, new_mean, diff, ex_n, rec["n_folds"]))

        if args.write:
            outp = os.path.join(exp, "results.recomputed.json")
            with open(outp, "w") as f:
                json.dump(rec, f, indent=2)

    print(f"\n{len(changed)} experiments have changed numbers "
          f"(threshold={args.threshold}).", file=sys.stderr)


if __name__ == "__main__":
    main()
