import os
import re
import csv
import glob
import ast
import argparse
import Levenshtein  # make sure python-Levenshtein is installed

def levenshtein_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance between two strings.

    Uses the same implementation as in get_levenshtein_distance
    (Levenshtein.distance from python-Levenshtein).
    """
    return Levenshtein.distance(a, b)


BEST_RE = re.compile(r"best:\s*(\{.*\})")
METRICS_RE = re.compile(
    r"levenshtein_distance:\s*([0-9.eE+-]+)\s*,\s*"
    r"average_sentence_length:\s*([0-9.eE+-]+)\s*,\s*"
    r"character_error_rate:\s*([0-9.eE+-]+)\s*,\s*"
    r"word_error_rate:\s*([0-9.eE+-]+)"
)

PRED_RE = re.compile(r"predictions:\s*(\[.*\])\s*$")
LABELS_RE = re.compile(r"labels:\s*(\[.*\])\s*$")


def find_best_dict(lines):
    """Parse the last 'best: {...}' line in the log and return it as a Python dict."""
    for line in reversed(lines):
        m = BEST_RE.search(line)
        if m:
            try:
                best_dict = ast.literal_eval(m.group(1))
                return best_dict
            except Exception as e:
                print("Failed to parse best dict:", e)
                return None
    return None


def find_block_for_best(lines, best_dict, metric_key="character_error_rate", tol=1e-6):
    """
    Find the metrics + predictions + labels block corresponding to the best metric.
    Assumes:
      - metrics line (METRICS_RE)
      - then 'predictions:' line
      - then 'labels:' line
    """
    if metric_key not in best_dict:
        raise ValueError(f"Metric '{metric_key}' not found in best dict: {best_dict.keys()}")

    best_epoch, best_value = best_dict[metric_key]
    best_value = float(best_value)

    for i, line in enumerate(lines):
        m = METRICS_RE.search(line)
        if not m:
            continue

        lev = float(m.group(1))
        avg_len = float(m.group(2))
        cer = float(m.group(3))
        wer = float(m.group(4))

        metric_val = {
            "levenshtein_distance": lev,
            "average_sentence_length": avg_len,
            "character_error_rate": cer,
            "word_error_rate": wer,
        }[metric_key]

        if abs(metric_val - best_value) < tol:
            preds_line = lines[i + 1] if i + 1 < len(lines) else ""
            labels_line = lines[i + 2] if i + 2 < len(lines) else ""

            mp = PRED_RE.search(preds_line)
            ml = LABELS_RE.search(labels_line)

            if not (mp and ml):
                continue

            try:
                preds = ast.literal_eval(mp.group(1))
                labels = ast.literal_eval(ml.group(1))
            except Exception as e:
                print("Failed to parse predictions/labels:", e)
                continue

            return {
                "metrics": {
                    "levenshtein_distance": lev,
                    "average_sentence_length": avg_len,
                    "character_error_rate": cer,
                    "word_error_rate": wer,
                    "epoch_from_best": best_epoch,
                },
                "predictions": preds,
                "labels": labels,
                "metrics_line_index": i,
            }

    return None


def extract_from_log(log_path, metric_key="character_error_rate"):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    best_dict = find_best_dict(lines)
    if best_dict is None:
        print(f"[WARN] No 'best:' line found in {log_path}")
        return None

    block = find_block_for_best(lines, best_dict, metric_key=metric_key)
    if block is None:
        print(f"[WARN] No matching metrics/preds/labels block found for best {metric_key} in {log_path}")
        return None

    return block


def main():
    # Script in: .../imu-hwr/work/REWI_work
    # Results in: .../imu-hwr/results/hwr2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, "..", "..", "results", "hwr2"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[default_root],
        help="Root directories to search for train.log files (recursive).",
    )
    parser.add_argument(
        "--metric",
        default="character_error_rate",
        choices=[
            "levenshtein_distance",
            "average_sentence_length",
            "character_error_rate",
            "word_error_rate",
        ],
        help="Metric to match between best line and metrics line.",
    )
    parser.add_argument(
        "--output",
        default="best_predictions_all_folds_word_only.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    # Find all train.log files under the given roots
    log_files = []
    for root in args.roots:
        root_abs = os.path.abspath(root)
        print(f"Searching in root: {root_abs}")
        if not os.path.isdir(root_abs):
            print(f"[WARN] Root does not exist or is not a directory: {root_abs}")
            continue

        pattern = os.path.join(root_abs, "**", "train*.log")
        print(f"  Using pattern: {pattern}")
        matches = glob.glob(pattern, recursive=True)
        print(f"  Found {len(matches)} train.log files in this root.")
        log_files.extend(matches)

    if not log_files:
        print("No train.log files found. Check your --roots paths.")
        return

    log_files = sorted(set(log_files))
    print(f"Total unique train.log files found: {len(log_files)}")

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        # Use ';' as delimiter as requested
        writer = csv.writer(csvfile, delimiter=';')

        # Columns:
        #  - dataset (only wi_word_hw6_meta variants)
        #  - fold_index
        #  - epoch_from_best
        #  - average_sentence_length
        #  - character_error_rate
        #  - word_error_rate
        #  - sample_index
        #  - prediction
        #  - label
        #  - eval_levenshtein_distance   (from metrics in log)
        #  - ind_levenshtein_distance    (computed per sample)
        writer.writerow([
            "dataset",
            "fold_index",
            "epoch_from_best",
            "average_sentence_length",
            "character_error_rate",
            "word_error_rate",
            "sample_index",
            "prediction",
            "label",
            #"eval_levenshtein_distance",
            "ind_levenshtein_distance",
        ])

        for log_path in log_files:
            parts = log_path.split(os.sep)
            # Expect: .../results/hwr2/<model_name>/<dataset_name>/fold_0/0/train.log
            if len(parts) < 6:
                print(f"[WARN] Path too short to parse model/dataset: {log_path}")
                continue

            model_name = parts[-5]
            dataset_name = parts[-4]

            # Only models ending in 'no_tokenizer'
            if not model_name.endswith("no_tokenizer"):
                continue

            # Only the WORD datasets: e.g. ar_transformer_s__wi_word_hw6_meta
            if "wi_word_hw6_meta" not in dataset_name:
                continue

            # fold_* directory
            fold_match = re.search(r"fold_(\d+)", log_path)
            fold_index = int(fold_match.group(1)) if fold_match else None

            print(f"Processing {log_path} (dataset={dataset_name}, fold={fold_index})")
            block = extract_from_log(log_path, metric_key=args.metric)
            if block is None:
                continue

            metrics = block["metrics"]
            preds = block["predictions"]
            labels = block["labels"]

            if len(preds) != len(labels):
                print(f"[WARN] predictions and labels length mismatch in {log_path}: {len(preds)} vs {len(labels)}")
                length = min(len(preds), len(labels))
            else:
                length = len(preds)

            for idx in range(length):
                pred = preds[idx]
                label = labels[idx]
                ind_lev = levenshtein_distance(pred, label)

                writer.writerow([
                    dataset_name,
                    fold_index,
                    metrics["epoch_from_best"],
                    metrics["average_sentence_length"],
                    metrics["character_error_rate"],
                    metrics["word_error_rate"],
                    idx,
                    pred,
                    label,
                    #metrics["levenshtein_distance"],  # eval_levenshtein_distance
                    ind_lev,                           # ind_levenshtein_distance
                ])

    print(f"Done. CSV written to {args.output}")


if __name__ == "__main__":
    main()
