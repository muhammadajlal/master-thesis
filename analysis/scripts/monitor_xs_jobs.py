#!/usr/bin/env python3
"""Monitor the XS gap-filling SLURM jobs and flag silent failures.

For each tracked array job:
  - confirm it's in squeue (or in sacct as completed)
  - scan its .out / .err logs for known failure signatures
  - check progress: latest epoch reached vs target (300)
  - for completed array tasks, verify train_*.json landed in the expected dir

Re-runnable. Run from anywhere:
    python monitor_xs_jobs.py [--verbose]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

LOG_DIR = Path("/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/logs/slurm")
RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")

# These job-name prefixes are the ones we submitted in Phase A (plus the original hybrid).
TRACKED_PREFIXES = (
    "hyb-xs-bl-",   # original hybrid base λ=0.6 OnHW-WI (1612631)
    "hyb-xs-l",     # lambda sweep (l01..l10) + λ=0.1 cross-dataset
    "hyb-xs-tied",  # weight-tied variant
    "xs-hw-",       # headwise gating
    "xs-sw-",       # p_ic sweep
    "hxc01-",       # hybrid+corruption combo at λ=0.1 (5 modes × 6 datasets)
)

# Failure signatures we scan log files for.
FAIL_PATTERNS = [
    (re.compile(r"\bTraceback \(most recent call last\)"), "python traceback"),
    (re.compile(r"\bCUDA out of memory\b", re.I),           "CUDA OOM"),
    (re.compile(r"\bRuntimeError\b"),                       "RuntimeError"),
    (re.compile(r"\b(srun|slurmstepd): error\b"),           "slurm error"),
    (re.compile(r"\bsbatch: error\b"),                       "sbatch error"),
    (re.compile(r"\bAssertionError\b"),                      "AssertionError"),
    (re.compile(r"\bNaN\b"),                                 "NaN loss"),
    (re.compile(r"\bDataLoader worker .* killed\b"),         "worker killed"),
    (re.compile(r"\bsegmentation fault\b", re.I),            "segfault"),
    (re.compile(r"\bKilled\b\s*$", re.M),                    "process killed"),
    (re.compile(r"\bcore dumped\b", re.I),                   "core dumped"),
]

# Patterns to *ignore* (false positives in normal training output).
BENIGN_PATTERNS = [
    re.compile(r"will be killed by OOM-killer"),  # warning in env, not actual OOM
]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout


def squeue_state() -> dict[str, dict]:
    """Return jobname -> {state, jobid, runtime, nodelist} from squeue."""
    out = run(["squeue", "-u", "iwso214h", "-h", "-o", "%i|%j|%T|%M|%R"])
    out_jobs: dict[str, dict] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            jobid, name, state, runtime, reason = line.split("|", 4)
        except ValueError:
            continue
        # Some entries are array-task ids like "1612638_0", others "1612638_[0-4]".
        # We index by base array id for grouping.
        base = jobid.split("_")[0]
        if not any(name.startswith(p) for p in TRACKED_PREFIXES):
            continue
        out_jobs.setdefault(base, {"name": name, "tasks": []})["tasks"].append({
            "jobid": jobid, "state": state, "runtime": runtime, "reason": reason,
        })
    return out_jobs


def sacct_finished(since: str = "2026-05-17") -> list[dict]:
    """Return recent completed/failed array tasks for tracked job names."""
    out = run([
        "sacct", "-u", "iwso214h",
        "-S", since,
        "--noheader",
        "--format=JobID,JobName,State,ExitCode,Elapsed",
        "-P",
    ])
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jobid, name, state, exitcode, elapsed = parts[:5]
        # Filter to tracked array tasks (skip .batch / .extern internal steps)
        if "." in jobid:
            continue
        if not any(name.startswith(p) for p in TRACKED_PREFIXES):
            continue
        rows.append({"jobid": jobid, "name": name, "state": state, "exitcode": exitcode, "elapsed": elapsed})
    return rows


def scan_log(path: Path) -> tuple[list[str], int | None]:
    """Return (list of failure tags, latest epoch number reached if any)."""
    failures: list[str] = []
    latest_epoch: int | None = None
    if not path.exists():
        return failures, latest_epoch
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return failures, latest_epoch
    for pat, tag in FAIL_PATTERNS:
        for m in pat.finditer(text):
            line = text[max(0, m.start()-50):m.end()+200]
            if any(b.search(line) for b in BENIGN_PATTERNS):
                continue
            failures.append(tag)
            break
    # Track training progress by the most recent "Epoch <N>" line.
    epochs = re.findall(r"Epoch[\s:]+(\d+)", text)
    if epochs:
        latest_epoch = max(int(e) for e in epochs)
    return failures, latest_epoch


def expected_result_dir_for(jobname: str) -> Path | None:
    """Heuristically map a tracked job name back to its result dir."""
    if jobname.startswith("hyb-xs-bl-onhw-wi-word"):
        return RESULTS / "train_element_word_hybrid_06_xs_onhw_wi" / "ar_transformer_xs__onhw_wi_word_rh"
    if jobname == "hyb-xs-tied":
        return RESULTS / "train_element_word_hybrid_06_xs_onhw_wi_ctc_to_ar_outproj" / "ar_transformer_xs__onhw_wi_word_rh"
    if jobname == "hyb-xs-l01-tied":
        return RESULTS / "train_element_word_hybrid_01_xs_onhw_wi_ctc_to_ar_outproj" / "ar_transformer_xs__onhw_wi_word_rh"
    # λ=0.1 hybrid on non-OnHW-WI datasets (cross-dataset hybrid evidence)
    m_cross = re.match(r"hyb-xs-l(\d{2})-(.+)$", jobname)
    if m_cross:
        k = int(m_cross.group(1))
        suffix = m_cross.group(2)
        ds_short_to_long = {
            "onhw-wd":    ("onhw_wd", "onhw_wd_word_rh"),
            "eq-wi":      ("equations_wi", "onhw_equations_wi_word_rh"),
            "eq-wd":      ("equations_wd", "onhw_equations_wd_word_rh"),
            "stabilo-w":  ("stabilo", "wi_word_hw6_meta"),
            "stabilo-s":  ("stabilo_sent", "wi_sent_hw6_meta"),
        }
        if suffix in ds_short_to_long:
            dir_suffix, ds = ds_short_to_long[suffix]
            return RESULTS / f"train_element_word_hybrid_{k:02d}_xs_{dir_suffix}" / f"ar_transformer_xs__{ds}"
    # λ sweep on OnHW-WI (existing 01..10)
    m = re.match(r"hyb-xs-l(\d{2})$", jobname)
    if m:
        k = int(m.group(1))
        return RESULTS / f"train_element_word_hybrid_{k:02d}_xs_onhw_wi" / "ar_transformer_xs__onhw_wi_word_rh"
    # Hybrid+corruption combo at λ=0.1: hxc01-{mode_tag}-{ds_tag}
    m_hxc = re.match(r"hxc01-(unif|bigr|bigl|self|adj)-(onhw-wi|onhw-wd|eq-wi|eq-wd|priv-w|priv-s)$", jobname)
    if m_hxc:
        mode_tag, ds_tag = m_hxc.group(1), m_hxc.group(2)
        mode_dir = {"unif": "uniform", "bigr": "bigram_right", "bigl": "bigram_left",
                    "self": "self_confusion", "adj": "adjacent_swap"}[mode_tag]
        ds = {"onhw-wi": "onhw_wi_word_rh", "onhw-wd": "onhw_wd_word_rh",
              "eq-wi": "onhw_equations_wi_word_rh", "eq-wd": "onhw_equations_wd_word_rh",
              "priv-w": "wi_word_hw6_meta", "priv-s": "wi_sent_hw6_meta"}[ds_tag]
        return RESULTS / f"HybridInputCorruption-XS-L01_{mode_dir}" / f"ar_transformer_xs__{ds}"
    if jobname.startswith("xs-hw-"):
        short = jobname[len("xs-hw-"):]
        ds = {
            "onhw-wd":        "onhw_wd_word_rh",
            "equations-wi":   "onhw_equations_wi_word_rh",
            "equations-wd":   "onhw_equations_wd_word_rh",
            "stabilo":        "wi_word_hw6_meta",
            "stabilo-sent":   "wi_sent_hw6_meta",
        }.get(short)
        if ds:
            return RESULTS / "Baseline-AR-XS-HeadwiseGating" / f"ar_transformer_xs__{ds}"
    if jobname.startswith("xs-sw-"):
        m = re.match(r"xs-sw-(.+?)-p0p(\d+)", jobname)
        if m:
            short, p = m.group(1), m.group(2)
            ds = {"onhw-word": "onhw_wi_word_rh", "stabilo-sent": "wi_sent_hw6_meta"}.get(short)
            if ds:
                return RESULTS / "Baseline-AR-XS-InputCorruption-Sweep-blconv_b" / f"ar_transformer_xs__{ds}__p0p{p}"
    return None


def fold_progress(result_root: Path) -> tuple[int, list[int], list[str]]:
    """Return (folds with train_*.json, per-fold max-epoch list, silent-fail tags)."""
    completed = 0
    max_epochs: list[int] = []
    silent_fails: list[str] = []
    for k in range(5):
        fold_dir = result_root / f"fold_{k}/{k}"
        train_files = sorted(glob.glob(str(fold_dir / "train_*.json")))
        test_only_files = sorted(glob.glob(str(fold_dir / "test_*.json")))
        if not train_files:
            max_epochs.append(0)
            # Silent-fail signature: fold dir exists with test_*.json but no train_*.json
            # (means test=true config skipped training entirely).
            if test_only_files:
                silent_fails.append(f"fold{k}_test_only")
            continue
        try:
            with open(train_files[-1]) as f:
                d = json.load(f)
            keys = [int(k2) for k2 in d.keys() if k2.isdigit()]
            max_epochs.append(max(keys) if keys else 0)
            if d.get("best", {}).get("character_error_rate"):
                completed += 1
        except Exception:
            max_epochs.append(0)
    return completed, max_epochs, silent_fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    queue = squeue_state()
    finished_rows = sacct_finished()
    # Map jobid -> rows for fast lookup
    finished_by_base: dict[str, list[dict]] = {}
    for r in finished_rows:
        base = r["jobid"].split("_")[0]
        finished_by_base.setdefault(base, []).append(r)

    print(f"{'jobid':>10}  {'name':<24}  {'queue':>14}  {'completed':>10}  {'epochs':>20}  flags")
    print("-" * 110)

    all_bases = sorted(set(queue.keys()) | set(finished_by_base.keys()))
    problems: list[str] = []
    total_running = 0
    total_pending = 0
    total_completed = 0

    for base in all_bases:
        q = queue.get(base)
        finished = finished_by_base.get(base, [])
        if q:
            name = q["name"]
            states = [t["state"] for t in q["tasks"]]
            running = sum(1 for s in states if s == "RUNNING")
            pending = sum(1 for s in states if s == "PENDING")
            queue_str = f"R{running}/P{pending}"
            total_running += running
            total_pending += pending
        else:
            name = finished[0]["name"] if finished else "?"
            queue_str = "-"
        # sacct counts
        finished_states = {r["state"]: 0 for r in finished}
        for r in finished:
            finished_states[r["state"]] = finished_states.get(r["state"], 0) + 1
        completed_str = ""
        for s, c in finished_states.items():
            completed_str += f"{s[:3]}{c} "
        completed_str = completed_str.strip() or "-"

        # Result-dir progress
        result_root = expected_result_dir_for(name)
        epoch_summary = ""
        completed_folds = 0
        silent_fail_tags: list[str] = []
        if result_root:
            completed_folds, max_epochs, silent_fail_tags = fold_progress(result_root)
            epoch_summary = "/".join(str(e) for e in max_epochs)
            total_completed += completed_folds

        # Log scan for failures (most-recent log per array task)
        flag_tags: set[str] = set(silent_fail_tags)
        log_pattern = LOG_DIR / f"{name}_{base}_*.out"
        for log in sorted(glob.glob(str(log_pattern))):
            fails, _ = scan_log(Path(log))
            flag_tags.update(fails)
        # Also scan .err files
        err_pattern = LOG_DIR / f"{name}_{base}_*.err"
        for log in sorted(glob.glob(str(err_pattern))):
            fails, _ = scan_log(Path(log))
            flag_tags.update(fails)

        flags = ",".join(sorted(flag_tags)) if flag_tags else ""
        if flags:
            problems.append(f"{base} ({name}): {flags}")

        # Only print job lines that match tracked prefixes (sanity)
        print(f"{base:>10}  {name[:24]:<24}  {queue_str:>14}  {completed_str:<10}  {epoch_summary:>20}  {flags}")

    print("-" * 110)
    print(f"TOTAL tasks: running={total_running}  pending={total_pending}  completed_folds={total_completed} / {len(all_bases)*5}")
    if problems:
        print("\n*** PROBLEMS DETECTED ***")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    else:
        print("\nNo failure signatures detected.")

if __name__ == "__main__":
    main()
