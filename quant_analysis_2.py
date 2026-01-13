import os
import numpy as np
import pandas as pd
import unicodedata

# ============================================================
# Quantile-example extraction (word + sentence) with:
# - NFC normalization for sanity checks (does NOT change d from CSV)
# - report-safe truncated strings + full strings
# - debug repr columns to reveal hidden chars
# - optional recompute check of Levenshtein ONLY on suspicious rows
# - CSV outputs only (no LaTeX/text files)
# ============================================================

# -----------------------
# Config
# -----------------------
CSV_PATH = "./quant_all_val_predictions_new.csv"   # adjust if needed
CSV_SEP  = ";"

OUT_DIR  = "./quant_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# How many examples near each quantile (per task)
K_PER_QUANTILE = 5
QUANTILES = [0.50, 0.90, 0.99]

# Optional extra tail examples (largest normalized errors per task)
TOPK_TAIL = 8

# For report display (truncated columns). Full strings are also kept.
MAX_SHOW = 35

# If True: recompute Levenshtein on *suspicious* rows only.
# Suspicious := pred_nfc == label_nfc but d>0 OR pred_nfc != label_nfc but d==0
VERIFY_D_WITH_RECOMPUTE = True

# -----------------------
# Helpers
# -----------------------
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", "" if pd.isna(s) else str(s))

def show_repr(s: str) -> str:
    # shows hidden characters like \n \t \xa0 etc.
    return repr("" if pd.isna(s) else str(s))

def trunc(s: str, n: int = MAX_SHOW) -> str:
    s = "" if pd.isna(s) else str(s)
    return s if len(s) <= n else (s[:n] + "...")

def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein (edit distance). Use only for small/suspicious subsets."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Ensure b is the shorter for less memory
    if lb > la:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur
    return prev[lb]

def pick_k_closest(nonzero_df: pd.DataFrame, target: float, k: int, used_uids: set) -> pd.DataFrame:
    """Pick k samples with lev_norm closest to target, excluding those already used."""
    gg = nonzero_df[~nonzero_df["uid"].isin(used_uids)].copy()
    if gg.empty:
        return gg
    gg["abs_diff"] = (gg["lev_norm"] - target).abs()
    gg = gg.sort_values(["abs_diff", "lev_norm"], ascending=[True, True]).head(k).copy()
    return gg

def quantile_examples_for_task(task_df: pd.DataFrame, k_per_q: int, quantiles: list[float]) -> tuple[pd.DataFrame, dict]:
    """
    Quantiles computed on lev_norm > 0 (nonzero errors) ONLY.
    Examples chosen as closest-to-quantile targets.
    """
    nz = task_df[task_df["lev_norm"] > 0].copy()
    if len(nz) == 0:
        return pd.DataFrame(), {}

    vals = nz["lev_norm"].to_numpy()
    qvals = {q: float(np.quantile(vals, q)) for q in quantiles}

    used = set()
    picked_chunks = []
    for q in quantiles:
        target = qvals[q]
        picked = pick_k_closest(nz, target, k_per_q, used)
        if not picked.empty:
            picked["target_quantile"] = q
            picked["target_value"] = target
            picked_chunks.append(picked)
            used.update(picked["uid"].tolist())

    out = pd.concat(picked_chunks, ignore_index=True) if picked_chunks else pd.DataFrame()
    return out, qvals

def tail_topk_for_task(task_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    nz = task_df[task_df["lev_norm"] > 0].copy()
    if nz.empty:
        return pd.DataFrame()
    return nz.sort_values("lev_norm", ascending=False).head(topk).copy()

# -----------------------
# Load + normalize columns
# -----------------------
df = pd.read_csv(CSV_PATH, sep=CSV_SEP)

# Your exact headers (capitalized)
rename_map = {
    "Task": "task",
    "Fold": "fold",
    "Json_path": "json_path",
    "Sample_index": "sample_index",
    "Prediction": "prediction",
    "Label": "label",
    "Levenshtein_distance": "levenshtein_distance",
}
df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

required = ["task", "fold", "json_path", "sample_index", "prediction", "label", "levenshtein_distance"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

df["task"] = df["task"].astype(str)
df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")

df = df.dropna(subset=["fold", "sample_index", "levenshtein_distance"]).copy()
df["fold"] = df["fold"].astype(int)
df["sample_index"] = df["sample_index"].astype(int)
df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

# Keep originals for reporting, but also create NFC versions for checks
df["prediction"] = df["prediction"].astype(str)
df["label"] = df["label"].astype(str)

df["pred_nfc"] = df["prediction"].map(nfc)
df["label_nfc"] = df["label"].map(nfc)

# Length on NFC labels (more stable if Unicode composed/decomposed)
df["label_chars"] = df["label_nfc"].str.len().clip(lower=1)

# Normalized error from CSV distance (this is your metric)
df["lev_norm"] = df["levenshtein_distance"] / df["label_chars"]

# Exact match from CSV distance
df["exact"] = (df["levenshtein_distance"] == 0)

# -----------------------
# Correlation: length vs error (CSV-only)
# -----------------------
def corr_len_block(g: pd.DataFrame) -> dict:
    # All rows
    rho_d_all = g["levenshtein_distance"].corr(g["label_chars"], method="pearson")
    rho_e_all = g["lev_norm"].corr(g["label_chars"], method="pearson")

    # Incorrect only (d > 0)
    g_bad = g[g["levenshtein_distance"] > 0]
    if len(g_bad) >= 2:
        rho_d_dpos = g_bad["levenshtein_distance"].corr(g_bad["label_chars"], method="pearson")
        rho_e_dpos = g_bad["lev_norm"].corr(g_bad["label_chars"], method="pearson")
    else:
        rho_d_dpos = np.nan
        rho_e_dpos = np.nan

    return {
        "rho_d_all": float(rho_d_all) if pd.notna(rho_d_all) else np.nan,
        "rho_dtilde_all": float(rho_e_all) if pd.notna(rho_e_all) else np.nan,
        "rho_d_dpos": float(rho_d_dpos) if pd.notna(rho_d_dpos) else np.nan,
        "rho_dtilde_dpos": float(rho_e_dpos) if pd.notna(rho_e_dpos) else np.nan,
        "n_all": int(len(g)),
        "n_dpos": int((g["levenshtein_distance"] > 0).sum()),
    }

rows = []
for task, g in df.groupby("task", sort=True):
    r = corr_len_block(g)
    r["task"] = task
    rows.append(r)

corr_df = pd.DataFrame(rows)[
    ["task", "rho_d_all", "rho_dtilde_all", "rho_d_dpos", "rho_dtilde_dpos", "n_all", "n_dpos"]
].sort_values("task")


# Unique id per sample (task/fold/index)
df["uid"] = df["task"] + "|" + df["fold"].astype(str) + "|" + df["sample_index"].astype(str)

# Debug visibility
df["pred_repr"] = df["pred_nfc"].map(show_repr)
df["label_repr"] = df["label_nfc"].map(show_repr)
df["pred_len"] = df["pred_nfc"].str.len()
df["label_len"] = df["label_nfc"].str.len()

# Flag suspicious consistency cases
df["looks_equal_but_dpos"] = (df["pred_nfc"] == df["label_nfc"]) & (df["levenshtein_distance"] > 0)
df["looks_unequal_but_dzero"] = (df["pred_nfc"] != df["label_nfc"]) & (df["levenshtein_distance"] == 0)
df["is_suspicious"] = df["looks_equal_but_dpos"] | df["looks_unequal_but_dzero"]

# Optional recompute check only for suspicious rows
df["lev_recomputed_nfc"] = np.nan
df["d_mismatch_flag"] = False
if VERIFY_D_WITH_RECOMPUTE:
    sus_idx = df.index[df["is_suspicious"]].tolist()
    if len(sus_idx) > 0:
        # Recompute on NFC strings
        recomputed = []
        for i in sus_idx:
            a = df.at[i, "pred_nfc"]
            b = df.at[i, "label_nfc"]
            recomputed.append(levenshtein(a, b))
        df.loc[sus_idx, "lev_recomputed_nfc"] = recomputed
        df.loc[sus_idx, "d_mismatch_flag"] = (df.loc[sus_idx, "lev_recomputed_nfc"].astype(int) != df.loc[sus_idx, "levenshtein_distance"].astype(int))

# Report-safe truncations (keep both full + short)
df["prediction_full"] = df["prediction"]
df["label_full"] = df["label"]
df["prediction_short"] = df["prediction"].map(lambda s: trunc(s, MAX_SHOW))
df["label_short"] = df["label"].map(lambda s: trunc(s, MAX_SHOW))

# -----------------------
# Build ONE combined table (word + sentence)
# -----------------------
all_examples = []
all_qvals_rows = []
all_tail = []

for task, g in df.groupby("task", sort=True):
    ex_df, qvals = quantile_examples_for_task(g, K_PER_QUANTILE, QUANTILES)
    tail_df = tail_topk_for_task(g, TOPK_TAIL)

    # Save quantile targets (computed on NONZERO lev_norm only)
    n_nonzero = int((g["lev_norm"] > 0).sum())
    for q, v in qvals.items():
        all_qvals_rows.append({
            "task": task,
            "quantile": q,
            "quantile_value": v,
            "n_nonzero": n_nonzero,
            "n_total": int(len(g)),
            "exact_match_%": float(100.0 * (g["levenshtein_distance"] == 0).mean()),
        })

    if not ex_df.empty:
        all_examples.append(ex_df)

    if not tail_df.empty:
        all_tail.append(tail_df)

examples_all = pd.concat(all_examples, ignore_index=True) if all_examples else pd.DataFrame()
qvals_all = pd.DataFrame(all_qvals_rows)
tail_all = pd.concat(all_tail, ignore_index=True) if all_tail else pd.DataFrame()

# -----------------------
# Keep report-friendly columns in the combined examples CSV
# -----------------------
if not examples_all.empty:
    examples_all = examples_all[[
        "task",
        "target_quantile", "target_value",
        "fold", "sample_index",
        "levenshtein_distance", "label_chars", "lev_norm",
        "prediction_short", "label_short",
        "prediction_full", "label_full",
        "pred_len", "label_len",
        "pred_repr", "label_repr",
        "looks_equal_but_dpos", "looks_unequal_but_dzero", "is_suspicious",
        "lev_recomputed_nfc", "d_mismatch_flag",
        "json_path"
    ]].sort_values(["task", "target_quantile", "lev_norm"], ascending=[True, True, True])

if not tail_all.empty:
    tail_all = tail_all[[
        "task", "fold", "sample_index",
        "levenshtein_distance", "label_chars", "lev_norm",
        "prediction_short", "label_short",
        "prediction_full", "label_full",
        "pred_len", "label_len",
        "pred_repr", "label_repr",
        "looks_equal_but_dpos", "looks_unequal_but_dzero", "is_suspicious",
        "lev_recomputed_nfc", "d_mismatch_flag",
        "json_path"
    ]].sort_values(["task", "lev_norm"], ascending=[True, False])

# -----------------------
# Save ONLY CSV files
# -----------------------
out_examples = os.path.join(OUT_DIR, "examples_by_quantile_all_tasks.csv")
out_qvals = os.path.join(OUT_DIR, "quantile_targets_nonzero_by_task.csv")
out_tail = os.path.join(OUT_DIR, "examples_tail_topk_by_task.csv")
out_corr = os.path.join(OUT_DIR, "corr_len_vs_error_by_task.csv")

examples_all.to_csv(out_examples, index=False)
qvals_all.to_csv(out_qvals, index=False)
tail_all.to_csv(out_tail, index=False)
corr_df.to_csv(out_corr, index=False)
# Optional: a compact suspicious report as CSV (helps debug “Landen” cases)
out_susp = os.path.join(OUT_DIR, "suspicious_rows_debug.csv")
sus = df[df["is_suspicious"]].copy()
sus = sus[[
    "task","fold","sample_index",
    "levenshtein_distance","lev_norm",
    "prediction_short","label_short",
    "pred_repr","label_repr",
    "looks_equal_but_dpos","looks_unequal_but_dzero",
    "lev_recomputed_nfc","d_mismatch_flag",
    "json_path"
]].sort_values(["d_mismatch_flag","task","fold","sample_index"], ascending=[False, True, True, True])
sus.to_csv(out_susp, index=False)

print("Done.")
print("Wrote:", out_examples)
print("Wrote:", out_qvals)
print("Wrote:", out_tail)
print("Wrote:", out_corr)
print("Wrote:", out_susp)
print(f"Suspicious rows: {len(sus)} (see suspicious_rows_debug.csv)")
