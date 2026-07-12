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
    thesis/tables/noise_trajectory_examples.tex   (\\input'd from chapter)
    analysis/noise_trajectory_cases.csv            (audit trail; gitignored *.csv,
                                                    regenerated deterministically here)

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


def latex_escape(s: str) -> str:
    """Escape LaTeX special characters inside \\texttt cells, and inject
    invisible mid-token break opportunities so long monospace tokens do not
    overflow narrow p{} columns. \\allowbreak emits no glyph but lets LaTeX
    wrap between characters of an otherwise unbreakable run."""
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
    break_after = 8
    out = []
    run = 0
    for ch in s:
        out.append(repl.get(ch, ch))
        if ch in " \t-/()[].,;:!?":
            run = 0
        else:
            run += 1
            if run >= break_after:
                out.append(r"\allowbreak{}")
                run = 0
    return "".join(out)


def first_edit_op(label: str, pred: str):
    """Return (op, i, j) for the FIRST edit operation (replace, delete, or
    insert) between label and pred. i is the source-side (label) index, j
    is the destination-side (prediction) index. Convention from
    Levenshtein.editops(label, pred):
        replace (i, j): label[i] is replaced by pred[j]
        delete  (i, j): label[i] is removed from pred; j is the insertion
                        point in pred (no character of pred is consumed)
        insert  (i, j): pred[j] is inserted before label[i] (i may equal
                        len(label) for a trailing insert)
    Returns None if label == pred exactly (no editops).
    """
    ops = Levenshtein.editops(label, pred)
    if not ops:
        return None
    op, i, j = ops[0]
    return (op, int(i), int(j))


def render_label_cell(label: str, ar_first, no_first) -> str:
    """Underline label[i_min] where i_min is the earliest reference index of
    the first edit operation over either model. For replace/delete, i is
    the consumed reference position. For insert, i is the reference index
    BEFORE which the insertion occurred (so label[i] is the character that
    appears immediately after the insert in the alignment). The underline
    annotates the earliest reference position whose alignment to a
    prediction is non-trivial. If i == len(label) for both models (only
    trailing inserts), no character is underlined."""
    candidates = [op[1] for op in (ar_first, no_first)
                  if op is not None]
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
    """Render the prediction cell with a plain underline on the first edit
    position. Substitution and insertion underline pred[j]. Deletion has no
    in-cell marker (the reference cell already shows the deletion via its
    own underline). Verbatim rendering when there is no first edit op."""
    if first_op is None:
        return "\\texttt{" + latex_escape(pred) + "}"
    op, _i, j = first_op
    if op in ("replace", "insert"):
        if not (0 <= j < len(pred)):
            return "\\texttt{" + latex_escape(pred) + "}"
        return (
            "\\texttt{"
            + latex_escape(pred[:j])
            + "\\underline{" + latex_escape(pred[j]) + "}"
            + latex_escape(pred[j + 1:])
            + "}"
        )
    # delete: no in-cell marker; the reference cell shows the missing position.
    return "\\texttt{" + latex_escape(pred) + "}"


def format_op_tag(first_op) -> str:
    """Compact textual representation of the first edit operation, used in
    audit CSV and in the per-row metadata of the LaTeX table."""
    if first_op is None:
        return "none"
    op, i, j = first_op
    if op == "replace":
        return f"sub@ref{i}->pred{j}"
    if op == "delete":
        return f"del@ref{i}"
    return f"ins@pred{j}/before ref{i}"


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

    # Reader-facing labels (col 1 + col 2 in the rendered table). The col-6
    # outcome describes the paired FINAL greedy outputs only -- these are
    # free-running results, not decoding trajectories, so no cascade/rescue
    # wording is used (the "trajectory" framing is reserved for the
    # single-corruption influence profile).
    cases = [
        ("Case 1", case1, "Largest improvement", "Largest paired edit-distance reduction under noise."),
        ("Case 2", case2, "Largest degradation", "Largest paired edit-distance increase under noise."),
        ("Case 3", case3, "Typical improvement", "Typical paired improvement; both outputs remain imperfect."),
    ]

    # Audit CSV. Includes the first edit operation for both models with
    # both reference and prediction indices so reviewers can reproduce the
    # rendered underline placement.
    audit_rows = []
    for tag, row, role, _outcome in cases:
        desc = f"{role.lower()} (priv. sent.)"
        label = row["label"]
        ar_first = first_edit_op(label, row["prediction_ar"])
        no_first = first_edit_op(label, row["prediction_noise"])
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
            "ar_first_op": format_op_tag(ar_first),
            "ar_first_ref_idx": ar_first[1] if ar_first else -1,
            "ar_first_pred_idx": ar_first[2] if ar_first else -1,
            "noise_first_op": format_op_tag(no_first),
            "noise_first_ref_idx": no_first[1] if no_first else -1,
            "noise_first_pred_idx": no_first[2] if no_first else -1,
            "label": label,
            "prediction_ar": row["prediction_ar"],
            "prediction_noise": row["prediction_noise"],
        })
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT, index=False)
    print(f"wrote audit: {OUT_AUDIT}")

    # LaTeX table. Six reader-facing columns: Case identifier, descriptive
    # selection role with Delta_tilde_e value, ground truth, both predictions
    # with one short edit-count subline each, and a one-clause Outcome.
    # First-edit underlines stay; the (sub, ins, del) decomposition is shown
    # without the first-op coordinate. The audit CSV preserves the full
    # decomposition with reference and prediction indices.
    table_rows = []
    for tag, row, role, outcome in cases:
        label = row["label"]
        pred_ar = row["prediction_ar"]
        pred_no = row["prediction_noise"]
        ar_first = first_edit_op(label, pred_ar)
        no_first = first_edit_op(label, pred_no)

        ops_ar = Levenshtein.editops(label, pred_ar)
        ops_no = Levenshtein.editops(label, pred_no)
        sub_a = sum(1 for op, _, _ in ops_ar if op == "replace")
        ins_a = sum(1 for op, _, _ in ops_ar if op == "insert")
        del_a = sum(1 for op, _, _ in ops_ar if op == "delete")
        sub_n = sum(1 for op, _, _ in ops_no if op == "replace")
        ins_n = sum(1 for op, _, _ in ops_no if op == "insert")
        del_n = sum(1 for op, _, _ in ops_no if op == "delete")

        decomp_ar = (
            f"\\newline\\itshape\\footnotesize sub={sub_a}, ins={ins_a}, "
            f"del={del_a}\\upshape\\normalsize"
        )
        decomp_no = (
            f"\\newline\\itshape\\footnotesize sub={sub_n}, ins={ins_n}, "
            f"del={del_n}\\upshape\\normalsize"
        )

        # Per-sample Delta_e_norm shown as a unitless ratio (not pp), since
        # the per-sample value can exceed 1 when len_ref is small relative to
        # the prediction edits. The pp scaling in tab:cascade-noise-mcnemar
        # applies only to the per-fold MEAN.
        delta = float(row["delta_e_norm"])
        role_cell = f"{role} ($\\Delta e={delta:+.2f}$)"

        table_rows.append({
            "tag": tag,
            "role_cell": role_cell,
            "gt_cell": render_label_cell(label, ar_first, no_first),
            "ar_cell": render_pred_cell(pred_ar, ar_first) + decomp_ar,
            "noise_cell": render_pred_cell(pred_no, no_first) + decomp_no,
            "outcome_cell": outcome,
        })

    lines = []
    lines.append("% Auto-generated by analysis/scripts/select_noise_paired_cases.py")
    lines.append("% Three deterministic paired output examples for sec:noise-trajectory-examples.")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Paired private-sentence examples comparing HWRFormer and "
        r"HWRFormer + noise injection. Cases are selected by the paired "
        r"change in length-normalized edit distance, "
        r"$\Delta e = e_{\mathrm{noise}} - "
        r"e_{\mathrm{HWRFormer}}$, where negative values favor the "
        r"noise-injection model. The three rows show the largest "
        r"improvement, largest degradation, and a typical improvement case. "
        r"Underlining marks the first aligned edit in each output. Edit "
        r"counts below each prediction decompose the Levenshtein distance "
        r"into substitutions, insertions, and deletions.}"
    )
    lines.append(r"\label{tab:noise-trajectory-examples}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}p{3.0cm}p{2.2cm}p{2.8cm}p{2.8cm}p{2.6cm}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Selection role & Ground truth & "
                 r"HWRFormer output & HWRFormer + noise output & Outcome \\")
    lines.append(r"\midrule")
    for r in table_rows:
        lines.append(
            f"{r['role_cell']} & {r['gt_cell']} & "
            f"{r['ar_cell']} & {r['noise_cell']} & {r['outcome_cell']} \\\\[4pt]"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote table: {OUT_TABLE}")
    for tag, row, role, _outcome in cases:
        print(
            f"  {tag} ({role}): {row['dataset_task']}/fold{int(row['fold'])}"
            f"/sample{int(row['sample_index'])}"
            f"  delta_e_norm={float(row['delta_e_norm']):+.4f}"
            f"  delta_edit_raw={int(row['edit_dist_ar'] - row['edit_dist_noise']):+d}"
            f"  AR_dist={int(row['edit_dist_ar'])}"
            f"  Noise_dist={int(row['edit_dist_noise'])}"
            f"  len_ref={int(row['len_ref'])}"
        )


if __name__ == "__main__":
    main()
