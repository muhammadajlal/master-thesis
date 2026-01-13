import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "./quant_all_val_predictions_new.csv"
CSV_SEP = ";"
OUT_DIR = "./quant_analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Plot controls
X_CLIP = 1.5          # clipped histogram/CDF window
HIST_BINS = 80

# Regime definition: choose ONE
REGIME_MODE = "quantiles"   # "quantiles" OR "thresholds"
THR = [0.10, 0.30, 0.60]    # only used if REGIME_MODE == "thresholds"

# ============================================================
# HELPERS
# ============================================================
def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - capitalized: "Task","Fold","Json_path","Sample_index","Prediction","Label","Levenshtein_distance"
      - lowercase:   "task","fold","json_path","sample_index","prediction","label","levenshtein_distance"
    and normalizes to lowercase canonical names.
    """
    col_map = {c: re.sub(r"\s+", "", c).lower() for c in df.columns}

    # canonical targets
    targets = {
        "task": ["task"],
        "fold": ["fold"],
        "json_path": ["json_path", "jsonpath"],
        "sample_index": ["sample_index", "sampleindex"],
        "prediction": ["prediction", "pred"],
        "label": ["label", "gt", "groundtruth"],
        "levenshtein_distance": ["levenshtein_distance", "levenshteindistance", "lev", "distance"],
    }

    # build rename dict
    rename = {}
    for original, norm in col_map.items():
        for canon, aliases in targets.items():
            if norm in aliases:
                rename[original] = canon

    df = df.rename(columns=rename)

    required = ["task", "fold", "json_path", "sample_index", "prediction", "label", "levenshtein_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df

def summarize_group(g: pd.DataFrame) -> pd.Series:
    n = len(g)

    lev_sum = g["levenshtein_distance"].sum()
    char_sum = g["label_chars"].sum()
    lev_norm_micro = lev_sum / max(1, char_sum)
    lev_norm_macro = g["lev_norm"].mean()

    vals = g["lev_norm"].to_numpy()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        qs = {q: np.nan for q in [0.5, 0.9, 0.95, 0.99]}
    else:
        qs = {
            0.5: np.quantile(vals, 0.5),
            0.9: np.quantile(vals, 0.9),
            0.95: np.quantile(vals, 0.95),
            0.99: np.quantile(vals, 0.99),
        }

    return pd.Series({
        "N": n,
        "Exact_match_%": 100.0 * g["exact"].mean(),
        "Mean_label_chars": g["label_chars"].mean(),
        "Mean_label_words": g["label_words"].mean(),
        "LevNorm_micro": lev_norm_micro,
        "LevNorm_macro": lev_norm_macro,
        "LevNorm_p50": qs[0.5],
        "LevNorm_p90": qs[0.9],
        "LevNorm_p95": qs[0.95],
        "LevNorm_p99": qs[0.99],
        "Tail_P(LevNorm>0.5)%": 100.0 * (g["lev_norm"] > 0.5).mean(),
        "Tail_P(LevNorm>1.0)%": 100.0 * (g["lev_norm"] > 1.0).mean(),
    })

def regimes_quantiles(g: pd.DataFrame) -> pd.Series:
    vals = g.loc[g["lev_norm"] > 0, "lev_norm"].to_numpy()
    total = len(g)
    exact = (g["lev_norm"] == 0).sum()

    if len(vals) == 0:
        return pd.Series({
            "Exact": 100.0, "Near-miss": 0.0, "Moderate": 0.0, "Moderate-high": 0.0, "Catastrophic": 0.0,
            "q50_nonzero": np.nan, "q90_nonzero": np.nan, "q99_nonzero": np.nan
        })

    q50 = np.quantile(vals, 0.50)
    q90 = np.quantile(vals, 0.90)
    q99 = np.quantile(vals, 0.99)

    near = ((g["lev_norm"] > 0) & (g["lev_norm"] <= q50)).sum()
    mod = ((g["lev_norm"] > q50) & (g["lev_norm"] <= q90)).sum()
    mod_hi = ((g["lev_norm"] > q90) & (g["lev_norm"] <= q99)).sum()
    cat = (g["lev_norm"] > q99).sum()

    return pd.Series({
        "Exact": 100.0 * exact / total,
        "Near-miss": 100.0 * near / total,
        "Moderate": 100.0 * mod / total,
        "Moderate-high": 100.0 * mod_hi / total,
        "Catastrophic": 100.0 * cat / total,
        "q50_nonzero": q50,
        "q90_nonzero": q90,
        "q99_nonzero": q99,
    })

def regimes_thresholds(g: pd.DataFrame, thr=(0.10, 0.30, 0.60)) -> pd.Series:
    t1, t2, t3 = thr
    total = len(g)

    exact = (g["lev_norm"] == 0).sum()
    near = ((g["lev_norm"] > 0) & (g["lev_norm"] <= t1)).sum()
    mod = ((g["lev_norm"] > t1) & (g["lev_norm"] <= t2)).sum()
    mod_hi = ((g["lev_norm"] > t2) & (g["lev_norm"] <= t3)).sum()
    cat = (g["lev_norm"] > t3).sum()

    return pd.Series({
        "Exact": 100.0 * exact / total,
        "Near-miss": 100.0 * near / total,
        "Moderate": 100.0 * mod / total,
        "Moderate-high": 100.0 * mod_hi / total,
        "Catastrophic": 100.0 * cat / total,
        "thr1": t1, "thr2": t2, "thr3": t3,
    })

# ---------------- PLOTTING (single pipeline) ----------------
def plot_exact_rate_by_fold(task: str, g: pd.DataFrame, out_dir: str):
    per_fold = g.groupby("fold")["exact"].mean().sort_index() * 100.0
    plt.figure()
    plt.bar(per_fold.index.astype(int).astype(str), per_fold.values)
    plt.xlabel("Fold")
    plt.ylabel("Exact match (%)")
    plt.title(f"{task}: Exact match rate by fold")
    out = os.path.join(out_dir, f"{_slug(task)}_exact_rate_by_fold.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

def plot_histograms_and_cdfs(task: str, g: pd.DataFrame, out_dir: str, x_clip: float = 1.5):
    vals = g["lev_norm"].to_numpy()
    vals = vals[np.isfinite(vals)]
    task_id = _slug(task)

    # --- Unclipped histogram (raw range) ---
    plt.figure()
    plt.hist(vals, bins=HIST_BINS)
    plt.xlabel("Normalized Levenshtein (unclipped)")
    plt.ylabel("Count")
    plt.title(f"{task}: histogram of normalized Levenshtein (unclipped)")
    out = os.path.join(out_dir, f"{task_id}_levnorm_hist_unclipped.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Clipped histogram (force range to [0, x_clip]) ---
    plt.figure()
    plt.hist(np.clip(vals, 0, x_clip), bins=HIST_BINS, range=(0, x_clip))
    plt.xlabel(f"Normalized Levenshtein (clipped at {x_clip})")
    plt.ylabel("Count")
    plt.title(f"{task}: histogram of normalized Levenshtein (clipped)")
    out = os.path.join(out_dir, f"{task_id}_levnorm_hist_clip{x_clip}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Unclipped CDF (full x-range) ---
    s = np.sort(vals)
    y = np.arange(1, len(s) + 1) / len(s)
    plt.figure()
    plt.plot(s, y)
    plt.xlabel("Normalized Levenshtein (unclipped)")
    plt.ylabel("CDF")
    plt.title(f"{task}: CDF of normalized Levenshtein (unclipped)")
    out = os.path.join(out_dir, f"{task_id}_levnorm_cdf_unclipped.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

    # --- Clipped-window CDF (show only x <= x_clip) ---
    mask = s <= x_clip
    plt.figure()
    plt.plot(s[mask], y[mask])
    plt.xlabel(f"Normalized Levenshtein (x ≤ {x_clip})")
    plt.ylabel("CDF")
    plt.title(f"{task}: CDF of normalized Levenshtein (x ≤ {x_clip})")
    out = os.path.join(out_dir, f"{task_id}_levnorm_cdf_xmax{x_clip}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# LOAD + FEATURES
# ============================================================
df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
df = standardize_columns(df)

df["task"] = df["task"].astype(str)
df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")

df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")
df = df.dropna(subset=["levenshtein_distance", "fold"]).copy()
df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

df["prediction"] = df["prediction"].astype(str)
df["label"] = df["label"].astype(str)

# label lengths
df["label_chars"] = df["label"].str.len().clip(lower=1)
df["label_words"] = df["label"].str.split().apply(len).clip(lower=1)

# normalized per-sample distance
df["lev_norm"] = df["levenshtein_distance"] / df["label_chars"]

# exact match
df["exact"] = (df["levenshtein_distance"] == 0)

# ============================================================
# TABLES (OVERALL + FOLDWISE)
# ============================================================
overall = df.groupby("task", as_index=True).apply(summarize_group).sort_index()
overall.to_csv(os.path.join(OUT_DIR, "overall_summary_norm.csv"), index=True)

foldwise = df.groupby(["task", "fold"], as_index=True).apply(summarize_group).sort_index()
foldwise.to_csv(os.path.join(OUT_DIR, "foldwise_summary_norm.csv"), index=True)

# ============================================================
# REGIMES (OVERALL + FOLDWISE)
# ============================================================
if REGIME_MODE == "quantiles":
    regimes_overall = df.groupby("task", as_index=True).apply(regimes_quantiles).sort_index()
    regimes_fold = df.groupby(["task", "fold"], as_index=True).apply(regimes_quantiles).sort_index()
else:
    regimes_overall = df.groupby("task", as_index=True).apply(lambda g: regimes_thresholds(g, THR)).sort_index()
    regimes_fold = df.groupby(["task", "fold"], as_index=True).apply(lambda g: regimes_thresholds(g, THR)).sort_index()

regimes_overall.to_csv(os.path.join(OUT_DIR, f"regimes_overall_{REGIME_MODE}.csv"), index=True)
regimes_fold.to_csv(os.path.join(OUT_DIR, f"regimes_fold_{REGIME_MODE}.csv"), index=True)

# ============================================================
# CORRELATION (RAW vs NORM) WITH LENGTH
# ============================================================
corr_rows = []
for task, g in df.groupby("task"):
    # handle edge cases where variance is zero
    try:
        r_raw = np.corrcoef(g["levenshtein_distance"], g["label_chars"])[0, 1]
    except Exception:
        r_raw = np.nan
    try:
        r_norm = np.corrcoef(g["lev_norm"], g["label_chars"])[0, 1]
    except Exception:
        r_norm = np.nan
    corr_rows.append({"task": task, "pearson_rawLev_vs_len": r_raw, "pearson_normLev_vs_len": r_norm})

pd.DataFrame(corr_rows).to_csv(os.path.join(OUT_DIR, "length_correlation_raw_vs_norm.csv"), index=False)

# ============================================================
# PLOTS (SINGLE PIPELINE) -> produces BOTH clipped and unclipped
# ============================================================
for task, g in df.groupby("task"):
    plot_exact_rate_by_fold(task, g, OUT_DIR)
    plot_histograms_and_cdfs(task, g, OUT_DIR, x_clip=X_CLIP)

# ============================================================
# DONE
# ============================================================
print("Done. Outputs saved to:", OUT_DIR)
print("Wrote:")
print(" - overall_summary_norm.csv")
print(" - foldwise_summary_norm.csv")
print(f" - regimes_overall_{REGIME_MODE}.csv")
print(f" - regimes_fold_{REGIME_MODE}.csv")
print(" - length_correlation_raw_vs_norm.csv")
print("Plots per task:")
print(" - *_exact_rate_by_fold.png")
print(" - *_levnorm_hist_unclipped.png")
print(f" - *_levnorm_hist_clip{X_CLIP}.png")
print(" - *_levnorm_cdf_unclipped.png")
print(f" - *_levnorm_cdf_xmax{X_CLIP}.png")
