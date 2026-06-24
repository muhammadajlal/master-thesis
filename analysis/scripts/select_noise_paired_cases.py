#!/usr/bin/env python3
"""Select three deterministic paired output examples for the qualitative addition.

All three cases are restricted to the private-sentence dataset and selected on
the per-sample paired change in length-normalized edit distance
    delta_tilde_e = e_noise - e_AR,   with e = edit_dist / max(1, len_ref).

    C1 = argmin(delta_tilde_e) over priv_sent   (largest improvement under noise)
    C2 = argmax(delta_tilde_e) over priv_sent   (largest degradation under noise)
    C3 = median(delta_tilde_e) over the priv_sent subset with delta_tilde_e < 0
                                                (representative improvement)

Ties broken by smaller len_ref, then lex order on (fold, sample_index).
Lower index on even-count median.

Restricting the selection to priv_sent removes a length confound that would
otherwise pool extrema from sentences (mean length 19) against words (mean
length 5). The supervisor's "trajectory" wording is reserved for the
single-corruption recovery profile; here we report paired free-running
greedy outputs.

Underline rendering uses the aligned editop indices, not the reference
index applied to the prediction. The reference cell underlines label[i]
at the earliest replace/delete position over either model. The prediction
cell underlines pred[j] for a substitution (op=replace) and inserts a
'[del]' marker at pred[j] for a deletion (op=delete) so the reader can
see that label[i] was missing in the prediction.

Outputs:
    thesis/tables/noise_paired_examples.tex   (\\input'd from chapter)
    analysis/noise_paired_cases.csv           (audit trail)

Run from work/REWI_work:
    python analysis/scripts/select_noise_paired_cases.py
"""
from __future__ import annotations

from pathlib import Path

import Levenshtein
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent.parent.parent
THESIS_DIR = WORK_DIR.parent.parent / "thesis"
IN_PAIRED_CSV = WORK_DIR / "analysis" / "quant_all_val_predictions_ar_vs_noise_xs.csv"
OUT_TABLE = THESIS_DIR / "tables" / "noise_paired_examples.tex"
OUT_AUDIT = WORK_DIR / "analysis" / "noise_paired_cases.csv"

TASK_AR = "AR-only"
TASK_NOISE = "Noise (uniform p=0.15)"

DATASET_LABELS = {
    "onhw_wi_word": "OnHW \\gls{wi} word",
    "onhw_wd_word": "OnHW \\gls{wd} word",
    "priv_word": "Priv.\\ word",
    "priv_sent": "Priv.\\ sent.",
}


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
    return "".join(repl.get(ch, ch) for ch in s)


def first_label_error_op(label: str, pred: str):
    """Return (op, i, j) for the first replace or delete op against label.

    i is the reference (label) index, j is the prediction index. Returns
    None if the prediction matches the label exactly or only inserts.
    """
    for op, i, j in Levenshtein.editops(label, pred):
        if op in ("replace", "delete"):
            return (op, int(i), int(j))
    return None


def render_label_cell(label: str, ar_first, no_first) -> str:
    """Underline label[i_min] where i_min is the earliest reference index of
    a first replace/delete error over either model. The annotation marks
    where the earliest model failure consumes the reference, not where each
    model's prediction diverges."""
    candidates = [op[1] for op in (ar_first, no_first) if op is not None]
    if not candidates:
        return "\\texttt{" + latex_escape(label) + "}"
    i = min(candidates)
    if not (0 <= i < len(label)):
        return "\\texttt{" + latex_escape(label) + "}"
    return (
        "\\texttt{"
        + latex_escape(label[:i])
        + "\\underline{" + latex_escape(label[i]) + "}"
        + latex_escape(label[i + 1:])
        + "}"
    )


def render_pred_cell(pred: str, first_op) -> str:
    """Render the prediction cell. If the first label-side error is a
    substitution at (i, j), underline pred[j]. If it is a deletion at
    (i, j), insert a '[del]' marker at prediction position j to mark that
    label[i] was missing; no character in pred is underlined for a
    deletion. If first_op is None, render the prediction verbatim."""
    if first_op is None:
        return "\\texttt{" + latex_escape(pred) + "}"
    op, _i, j = first_op
    if op == "replace":
        if not (0 <= j < len(pred)):
            return "\\texttt{" + latex_escape(pred) + "}"
        return (
            "\\texttt{"
            + latex_escape(pred[:j])
            + "\\underline{" + latex_escape(pred[j]) + "}"
            + latex_escape(pred[j + 1:])
            + "}"
        )
    # delete: label[i] missing in pred; mark the gap at pred position j
    j = max(0, min(j, len(pred)))
    return (
        "\\texttt{"
        + latex_escape(pred[:j])
        + "\\textbf{[del]}"
        + latex_escape(pred[j:])
        + "}"
    )


def load_paired() -> pd.DataFrame:
    """Inner AR-vs-noise join with prediction/label strings."""
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
    # Normalized paired edit-distance change: matches the headline metric
    # of tab:cascade-noise-mcnemar (mean negative = noise wins).
    j["e_ar_norm"] = j["edit_dist_ar"] / j["len_ref"].clip(lower=1)
    j["e_noise_norm"] = j["edit_dist_noise"] / j["len_ref"].clip(lower=1)
    j["delta_e_norm"] = j["e_noise_norm"] - j["e_ar_norm"]
    return j


def pick_extremum(df: pd.DataFrame, smallest: bool) -> pd.Series:
    """Pick min (smallest=True, largest improvement under noise) or max
    (smallest=False, largest degradation) of delta_e_norm. Ties broken by
    smaller len_ref, then lex order on (fold, sample_index)."""
    df_sorted = df.sort_values(
        by=["delta_e_norm", "len_ref", "fold", "sample_index"],
        ascending=[smallest, True, True, True],
    )
    return df_sorted.iloc[0]


def pick_median_improvement(df: pd.DataFrame) -> pd.Series:
    """Median delta_e_norm over the improvement subset (delta_e_norm < 0).
    Lower index on even count."""
    sub = df[df["delta_e_norm"] < 0].sort_values(
        by=["delta_e_norm", "len_ref", "fold", "sample_index"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    if sub.empty:
        raise RuntimeError("no priv_sent samples with delta_e_norm < 0")
    return sub.iloc[(len(sub) - 1) // 2]


def main() -> None:
    df = load_paired()
    print(f"loaded {len(df)} paired rows across {sorted(df['dataset_task'].unique())}")
    sent = df[df["dataset_task"] == "priv_sent"].copy()
    print(f"  priv_sent rows: {len(sent)}")

    case1 = pick_extremum(sent, smallest=True)   # largest improvement
    case2 = pick_extremum(sent, smallest=False)  # largest degradation
    case3 = pick_median_improvement(sent)        # median improvement

    cases = [("C1", case1, "largest improvement (priv. sent.)"),
             ("C2", case2, "largest degradation (priv. sent.)"),
             ("C3", case3, "median improvement (priv. sent.)")]

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
            "delta_edit_raw": int(row["edit_dist_ar"] - row["edit_dist_noise"]),
            "delta_e_norm": float(row["delta_e_norm"]),
            "label": row["label"],
            "prediction_ar": row["prediction_ar"],
            "prediction_noise": row["prediction_noise"],
        })
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)
    print(f"wrote audit: {OUT_AUDIT}")

    # LaTeX table
    table_rows = []
    for tag, row, _desc in cases:
        ds_label = DATASET_LABELS.get(row["dataset_task"], row["dataset_task"])
        label = row["label"]
        pred_ar = row["prediction_ar"]
        pred_no = row["prediction_noise"]
        ar_first = first_label_error_op(label, pred_ar)
        no_first = first_label_error_op(label, pred_no)

        ops_no = Levenshtein.editops(label, pred_no)
        sub_n = sum(1 for op, _, _ in ops_no if op == "replace")
        ins_n = sum(1 for op, _, _ in ops_no if op == "insert")
        del_n = sum(1 for op, _, _ in ops_no if op == "delete")

        if no_first is not None:
            first_err_tag = f"first-err ref={no_first[1]}"
        else:
            first_err_tag = "first-err --"
        decomp = (
            "\\newline\\itshape\\small ("
            f"sub={sub_n}, ins={ins_n}, del={del_n}, {first_err_tag})"
            "\\upshape\\normalsize"
        )

        table_rows.append({
            "tag": tag,
            "ds_label": ds_label,
            "gt_cell": render_label_cell(label, ar_first, no_first),
            "ar_cell": render_pred_cell(pred_ar, ar_first),
            "noise_cell": render_pred_cell(pred_no, no_first) + decomp,
        })

    lines = []
    lines.append("% Auto-generated by analysis/scripts/select_noise_paired_cases.py")
    lines.append("% Three deterministic paired examples for sec:noise-paired-examples.")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Three deterministic paired output examples on the "
        r"private-sentence validation set, selected on the per-sample paired "
        r"change in length-normalized edit distance "
        r"$\Delta\tilde{e} = \tilde{e}_{\mathrm{noise}} - \tilde{e}_{\mathrm{AR}}$. "
        r"C1 is the sample with the most negative $\Delta\tilde{e}$ "
        r"(largest improvement under noise). C2 is the most positive "
        r"$\Delta\tilde{e}$ (largest degradation). C3 is the median of the "
        r"$\Delta\tilde{e} < 0$ subset. The reference cell underlines the "
        r"earliest reference position assigned a replace or delete by either "
        r"model. Prediction cells use the aligned prediction index of the "
        r"first replace (underlined) or mark a deletion with \textbf{[del]} "
        r"to indicate the missing reference character; no underline is "
        r"placed on an unrelated prediction character.}"
    )
    lines.append(r"\label{tab:noise-paired-examples}")
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
            f"/sample{int(row['sample_index'])}"
            f"  delta_e_norm={float(row['delta_e_norm']):+.4f}"
            f"  delta_edit_raw={int(row['edit_dist_ar'] - row['edit_dist_noise']):+d}"
            f"  AR_dist={int(row['edit_dist_ar'])}"
            f"  Noise_dist={int(row['edit_dist_noise'])}"
            f"  len_ref={int(row['len_ref'])}"
        )


if __name__ == "__main__":
    main()
