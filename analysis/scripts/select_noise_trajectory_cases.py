#!/usr/bin/env python3
"""Select three deterministic noise-trajectory cases and write the LaTeX table.

Three pre-specified cases for the qualitative §5.6 addition:

    C1 = argmax(edit_dist_AR - edit_dist_noise)   over all four datasets pooled
                                                  (largest improvement under noise)
    C2 = argmin(edit_dist_AR - edit_dist_noise)   over all four datasets pooled
                                                  (largest degradation under noise)
    C3 = sample with median(edit_dist_AR - edit_dist_noise)
         restricted to dataset_task == 'priv_sent' AND edit_dist_AR > edit_dist_noise
         (representative private-sentence improvement)

Ties broken by smaller len_ref, then lex order on (dataset_task, fold, sample_index).
Lower index on even-count median.

Outputs:
    thesis/tables/noise_trajectory_examples.tex   (\\input'd from chapter)
    analysis/noise_trajectory_cases.csv           (audit trail)

Run from work/REWI_work:
    python analysis/scripts/select_noise_trajectory_cases.py
"""
from __future__ import annotations

from pathlib import Path

import Levenshtein
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent.parent.parent
THESIS_DIR = WORK_DIR.parent.parent / "thesis"
IN_PAIRED_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_noise_xs.csv"
IN_PER_SAMPLE = WORK_DIR / "analysis" / "aligned_edit_per_sample.csv"
OUT_TABLE = THESIS_DIR / "tables" / "noise_trajectory_examples.tex"
OUT_AUDIT = WORK_DIR / "analysis" / "noise_trajectory_cases.csv"

TASK_AR = "AR-only"
TASK_NOISE = "Noise (uniform p=0.15)"

DATASET_LABELS = {
    "onhw_wi_word": "OnHW \\gls{wi} word",
    "onhw_wd_word": "OnHW \\gls{wd} word",
    "priv_word": "Priv.\\ word",
    "priv_sent": "Priv.\\ sent.",
}

# No hard truncation. The p{}-column wraps long cells naturally; truncating
# the AR-only or ground-truth string would hide the first-error position when
# it falls near the end (a common case on long-form private sentences).


def first_err_ref_pos(label: str, pred: str) -> int:
    """Index in label of the first replace or delete op; -1 if none."""
    ops = Levenshtein.editops(label, pred)
    for op, i, _ in ops:
        if op in ("replace", "delete"):
            return i
    return -1


def latex_escape(s: str) -> str:
    """Escape LaTeX special characters inside \\texttt cells."""
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def render_text_cell(text: str, underline_pos: int) -> str:
    """Wrap text in \\texttt; underline the character at underline_pos if any.

    No truncation: the p{}-column wraps long cells naturally.
    """
    if 0 <= underline_pos < len(text):
        prefix = text[:underline_pos]
        marked = text[underline_pos]
        suffix = text[underline_pos + 1:]
        return (
            "\\texttt{"
            + latex_escape(prefix)
            + "\\underline{" + latex_escape(marked) + "}"
            + latex_escape(suffix)
            + "}"
        )
    return "\\texttt{" + latex_escape(text) + "}"


def load_paired() -> pd.DataFrame:
    """Return the inner AR-vs-noise join with prediction/label strings.

    Columns: dataset_task, fold, sample_index,
             label, prediction_ar, prediction_noise,
             edit_dist_ar, edit_dist_noise, len_ref.
    """
    df = pd.read_csv(IN_PAIRED_CSV, sep=";")
    df["fold"] = df["fold"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)
    ar = df[df["task"] == TASK_AR][
        ["dataset_task", "fold", "sample_index", "prediction", "label",
         "levenshtein_distance"]
    ].rename(columns={"prediction": "prediction_ar",
                      "levenshtein_distance": "edit_dist_ar"})
    no = df[df["task"] == TASK_NOISE][
        ["dataset_task", "fold", "sample_index", "prediction",
         "levenshtein_distance"]
    ].rename(columns={"prediction": "prediction_noise",
                      "levenshtein_distance": "edit_dist_noise"})
    j = ar.merge(no, on=["dataset_task", "fold", "sample_index"], how="inner")
    j["label"] = j["label"].fillna("").astype(str)
    j["prediction_ar"] = j["prediction_ar"].fillna("").astype(str)
    j["prediction_noise"] = j["prediction_noise"].fillna("").astype(str)
    j["len_ref"] = j["label"].str.len()
    j["delta_edit"] = j["edit_dist_ar"] - j["edit_dist_noise"]
    return j


def pick_case(df: pd.DataFrame, ascending: bool) -> pd.Series:
    """Pick global argmax (ascending=False) or argmin (ascending=True) of
    delta_edit. Ties broken by smaller len_ref, then by lex order on
    (dataset_task, fold, sample_index). Always return the top row after the
    requested sort, so iloc[0] is the desired extremum."""
    df_sorted = df.sort_values(
        by=["delta_edit", "len_ref", "dataset_task", "fold", "sample_index"],
        ascending=[ascending, True, True, True, True],
    )
    return df_sorted.iloc[0]


def pick_median(df: pd.DataFrame) -> pd.Series:
    """Pick the median delta_edit row from the positive-improvement subset.
    Lower index on even count."""
    sub = df[df["delta_edit"] > 0].sort_values(
        by=["delta_edit", "len_ref", "dataset_task", "fold", "sample_index"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    n = len(sub)
    if n == 0:
        raise RuntimeError("no positive-improvement private-sentence samples")
    idx = (n - 1) // 2  # lower index on even count
    return sub.iloc[idx]


def main() -> None:
    df = load_paired()
    print(f"loaded {len(df)} paired rows across {sorted(df['dataset_task'].unique())}")

    # C1 argmax delta_edit globally
    case1 = pick_case(df, ascending=False)
    # C2 argmin delta_edit globally
    case2 = pick_case(df, ascending=True)
    # C3 median positive on priv_sent
    case3 = pick_median(df[df["dataset_task"] == "priv_sent"])

    cases = [("C1", case1, "largest improvement"),
             ("C2", case2, "largest degradation"),
             ("C3", case3, "median private-sentence improvement")]

    # Audit CSV
    audit_rows = []
    for tag, row, desc in cases:
        audit_rows.append({
            "case": tag,
            "description": desc,
            "dataset_task": row["dataset_task"],
            "fold": int(row["fold"]),
            "sample_index": int(row["sample_index"]),
            "len_ref": int(row["len_ref"]),
            "edit_dist_ar": int(row["edit_dist_ar"]),
            "edit_dist_noise": int(row["edit_dist_noise"]),
            "delta_edit": int(row["delta_edit"]),
            "label": row["label"],
            "prediction_ar": row["prediction_ar"],
            "prediction_noise": row["prediction_noise"],
        })
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)
    print(f"wrote audit: {OUT_AUDIT}")

    # LaTeX table
    table_rows = []
    for tag, row, _ in cases:
        ds_label = DATASET_LABELS.get(row["dataset_task"], row["dataset_task"])
        label = row["label"]
        pred_ar = row["prediction_ar"]
        pred_no = row["prediction_noise"]
        first_err_ar = first_err_ref_pos(label, pred_ar)
        first_err_no = first_err_ref_pos(label, pred_no)
        # Ground-truth underline marks the earliest divergent reference
        # position across either prediction.
        candidates = [p for p in (first_err_ar, first_err_no) if p >= 0]
        gt_under = min(candidates) if candidates else -1
        # Per-sample noise decomposition
        ops_no = Levenshtein.editops(label, pred_no)
        sub_n = sum(1 for op, _, _ in ops_no if op == "replace")
        ins_n = sum(1 for op, _, _ in ops_no if op == "insert")
        del_n = sum(1 for op, _, _ in ops_no if op == "delete")
        table_rows.append({
            "tag": tag,
            "ds_label": ds_label,
            "gt_cell": render_text_cell(label, gt_under),
            "ar_cell": render_text_cell(pred_ar, first_err_ar),
            "noise_cell": (
                render_text_cell(pred_no, first_err_no)
                + "\\newline\\itshape\\small (sub=" + str(sub_n)
                + ", ins=" + str(ins_n) + ", del=" + str(del_n)
                + ", first-err=" + (str(first_err_no) if first_err_no >= 0 else "--")
                + ")\\upshape\\normalsize"
            ),
        })

    lines = []
    lines.append("% Auto-generated by analysis/scripts/select_noise_trajectory_cases.py")
    lines.append("% Three deterministic noise-trajectory examples for sec:noise-trajectory-examples.")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Three noise-injection trajectory examples selected "
        r"deterministically from the paired prediction CSV of "
        r"\cref{sec:cascade-noise}. C1 is the sample with the largest "
        r"reduction in Levenshtein edit distance under noise injection. "
        r"C2 is the sample with the largest increase. C3 is the median "
        r"of the positive-improvement subset restricted to the private "
        r"sentence dataset. The underlined character marks the first "
        r"reference position assigned a replace or delete by the "
        r"Levenshtein alignment.}"
    )
    lines.append(r"\label{tab:noise-trajectory-examples}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}llp{3.2cm}p{3.2cm}p{3.4cm}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Case & Dataset & Ground truth & AR-only output & "
                 r"Noise + edit decomposition \\")
    lines.append(r"\midrule")
    for r in table_rows:
        lines.append(
            f"{r['tag']} & {r['ds_label']} & {r['gt_cell']} & {r['ar_cell']} "
            f"& {r['noise_cell']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote table: {OUT_TABLE}")
    for tag, row, desc in cases:
        print(
            f"  {tag} ({desc}): {row['dataset_task']}/fold{int(row['fold'])}"
            f"/sample{int(row['sample_index'])} "
            f"  delta_edit={int(row['delta_edit'])}"
            f"  AR_dist={int(row['edit_dist_ar'])}"
            f"  Noise_dist={int(row['edit_dist_noise'])}"
            f"  len_ref={int(row['len_ref'])}"
        )


if __name__ == "__main__":
    main()
