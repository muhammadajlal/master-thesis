import argparse
import json
import os
import re
import glob
import csv

# Try fast Levenshtein; fall back to pure python
try:
    import Levenshtein  # pip install python-Levenshtein
    def lev(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except Exception:
    def lev(a: str, b: str) -> int:
        # Pure-Python DP
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

FOLD_RE_1 = re.compile(r"(?:^|/)(?:fold_)(\d+)(?:/|$)")
FOLD_RE_2 = re.compile(r"(?:^|/)(?:fold)(\d+)(?:/|$)")
FOLD_RE_3 = re.compile(r"val_full_fold(\d+)_", re.IGNORECASE)

def infer_fold(path: str):
    p = path.replace("\\", "/")
    for rx in (FOLD_RE_1, FOLD_RE_2, FOLD_RE_3):
        m = rx.search(p)
        if m:
            return int(m.group(1))
    return None

def infer_task(path: str):
    p = path.lower().replace("\\", "/")
    # Adjust these heuristics to match your folder names
    if "inference_results_word" in p or "/word" in p or "wi_word" in p:
        return "word"
    if "inference_results_sent" in p or "/sent" in p or "wi_sent" in p:
        return "sent"
    return "unknown"

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more directories to search (recursively) for val_full_*.json files.",
    )
    ap.add_argument(
        "--pattern",
        default="**/exports/val_full_*.json",
        help="Glob pattern under each root (default: **/exports/val_full_*.json).",
    )
    ap.add_argument(
        "--out",
        default="quant_all_val_predictions.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--task",
        default="auto",
        choices=["auto", "word", "sent"],
        help="Force a task label or infer from path.",
    )
    ap.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default ';' to match your earlier scripts).",
    )
    args = ap.parse_args()

    # Collect json files
    files = []
    for r in args.roots:
        r_abs = os.path.abspath(r)
        files.extend(glob.glob(os.path.join(r_abs, args.pattern), recursive=True))

    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No JSONs found under roots={args.roots} with pattern={args.pattern}")

    rows = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv, delimiter=args.delimiter)
        w.writerow([
            "task", "fold", "json_path",
            "sample_index", "prediction", "label",
            "levenshtein_distance"
        ])

        for jp in files:
            fold = infer_fold(jp)
            task = args.task if args.task != "auto" else infer_task(jp)

            d = load_json(jp)
            preds = d.get("predictions", [])
            labs  = d.get("labels", [])

            n = min(len(preds), len(labs))
            if len(preds) != len(labs):
                print(f"[WARN] length mismatch in {jp}: preds={len(preds)} labels={len(labs)} (using n={n})")

            for i in range(n):
                p = preds[i]
                y = labs[i]
                w.writerow([task, fold, jp, i, p, y, lev(str(p), str(y))])
                rows += 1

    print(f"Wrote {rows} rows to {args.out}")
    print(f"Read {len(files)} JSON files.")

if __name__ == "__main__":
    main()
