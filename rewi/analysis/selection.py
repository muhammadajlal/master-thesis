"""
Sample selection utilities for qualitative analysis.

Provides functions to select representative samples for visualization based on
error quantiles (correct, near-miss, catastrophic failures).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_partB_selection_unified_quantile(
    unified_csv_path: str,
    fold: int,
    task_name: str = "word",
    n_correct: int = 2,
    n_nearmiss: int = 2,
    n_catastrophic: int = 2,
    seed: int = 42,
    quantiles: Tuple[float, float] = (0.50, 0.99),
) -> Dict[int, Dict[str, Any]]:
    """
    Select qualitative examples from unified CSV based on error quantiles.

    Regimes (fold-local, task-local):
      - correct: levenshtein_distance == 0 (random sample)
      - near_miss: d>0, d_norm closest to nonzero p50
      - catastrophic: d>0, d_norm closest to nonzero p99

    Args:
        unified_csv_path: Path to CSV with predictions and errors.
        fold: Fold index to filter by.
        task_name: Task name ("word" or "sent").
        n_correct: Number of correct samples to select.
        n_nearmiss: Number of near-miss samples to select.
        n_catastrophic: Number of catastrophic samples to select.
        seed: Random seed for reproducibility.
        quantiles: Tuple of (near_miss_quantile, catastrophic_quantile).

    Returns:
        Dictionary mapping sample_index to metadata dict containing:
        - regime: "correct", "near_miss", or "catastrophic"
        - lev: Levenshtein distance
        - d_norm: Normalized distance (d / label_length)
        - csv_pred: Prediction from CSV
        - csv_label: Ground truth label
        - target_quantile: The quantile used for selection (if applicable)
        - target_value: The quantile threshold value (if applicable)
    """
    # Load and standardize column names
    df = pd.read_csv(unified_csv_path, sep=";")

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

    required = ["task", "fold", "sample_index", "prediction", "label", "levenshtein_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in unified CSV: {missing}\nFound: {list(df.columns)}")

    # Filter by fold and task
    df["task"] = df["task"].astype(str)
    df = df[df["task"] == str(task_name)].copy()

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")

    df = df.dropna(subset=["fold", "sample_index", "levenshtein_distance"]).copy()
    df["fold"] = df["fold"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

    df = df[df["fold"] == int(fold)].copy()
    if df.empty:
        return {}

    # Compute normalized error: d_norm = d / |y|
    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    # Create pools
    correct_pool = df[df["levenshtein_distance"] == 0].copy()
    nz = df[df["levenshtein_distance"] > 0].copy()
    
    if nz.empty:
        # No errors => only correct examples possible
        correct_sel = (
            correct_pool.sample(n=min(n_correct, len(correct_pool)), random_state=seed)
            if len(correct_pool) else correct_pool
        )
        return {
            int(r["sample_index"]): {
                "regime": "correct",
                "lev": float(r["levenshtein_distance"]),
                "d_norm": float(r["d_norm"]),
                "csv_pred": str(r["prediction"]),
                "csv_label": str(r["label"]),
                "target_quantile": None,
                "target_value": None,
            }
            for _, r in correct_sel.iterrows()
        }

    q_near, q_cat = quantiles
    q50 = float(nz["d_norm"].quantile(q_near))
    q99 = float(nz["d_norm"].quantile(q_cat))

    # Near-miss: closest-to-q50 among errors
    near_pool = nz.assign(abs_diff=(nz["d_norm"] - q50).abs()).sort_values(["abs_diff", "d_norm"])
    # Catastrophic: closest-to-q99 among errors (prefer higher d_norm if ties)
    cata_pool = nz.assign(abs_diff=(nz["d_norm"] - q99).abs()).sort_values(
        ["abs_diff", "d_norm"], ascending=[True, False]
    )

    # Select without overlap
    used: set = set()

    def pick_random(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        if len(pool) <= n:
            return pool.copy()
        return pool.sample(n=n, random_state=seed).copy()

    def pick_closest(pool: pd.DataFrame, n: int, used_set: set) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        pool2 = pool[~pool["sample_index"].isin(used_set)].copy()
        return pool2.head(min(n, len(pool2))).copy()

    correct_sel = pick_random(correct_pool, n_correct).assign(
        regime="correct", target_quantile=np.nan, target_value=np.nan
    )
    used |= set(correct_sel["sample_index"].tolist())

    near_sel = pick_closest(near_pool, n_nearmiss, used).assign(
        regime="near_miss", target_quantile=q_near, target_value=q50
    )
    used |= set(near_sel["sample_index"].tolist())

    cata_sel = pick_closest(cata_pool, n_catastrophic, used).assign(
        regime="catastrophic", target_quantile=q_cat, target_value=q99
    )

    sel = pd.concat([correct_sel, near_sel, cata_sel], ignore_index=True)
    if sel.empty:
        return {}

    # Build result dictionary
    sel_map: Dict[int, Dict[str, Any]] = {}
    for _, r in sel.iterrows():
        si = int(r["sample_index"])
        sel_map[si] = {
            "regime": str(r["regime"]),
            "lev": float(r["levenshtein_distance"]),
            "d_norm": float(r["d_norm"]),
            "csv_pred": str(r.get("prediction", "")),
            "csv_label": str(r.get("label", "")),
            "target_quantile": (
                None if pd.isna(r.get("target_quantile", np.nan)) else float(r["target_quantile"])
            ),
            "target_value": (
                None if pd.isna(r.get("target_value", np.nan)) else float(r["target_value"])
            ),
        }

    return sel_map


def load_partB_selection_by_indices(
    unified_csv_path: str,
    fold: int,
    task_name: str,
    indices: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Load sample metadata for specific indices from unified CSV.
    
    Use this when you have a pre-defined list of sample indices to analyze.
    
    Args:
        unified_csv_path: Path to CSV with predictions and errors.
        fold: Fold index to filter by.
        task_name: Task name ("word" or "sent").
        indices: List of sample indices to load.
    
    Returns:
        Dictionary mapping sample_index to metadata dict.
    """
    df = pd.read_csv(unified_csv_path, sep=";")

    rename_map = {
        "Task": "task",
        "Fold": "fold",
        "Sample_index": "sample_index",
        "Prediction": "prediction",
        "Label": "label",
        "Levenshtein_distance": "levenshtein_distance",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    df = df[df["task"].astype(str) == str(task_name)].copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")
    df = df.dropna(subset=["fold", "sample_index", "levenshtein_distance"]).copy()

    df["fold"] = df["fold"].astype(int)
    df["sample_index"] = df["sample_index"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)

    df = df[
        (df["fold"] == int(fold)) & (df["sample_index"].isin([int(i) for i in indices]))
    ].copy()
    
    if df.empty:
        return {}

    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    sel_map = {}
    for _, r in df.iterrows():
        si = int(r["sample_index"])
        sel_map[si] = {
            "regime": "table_example",
            "lev": float(r["levenshtein_distance"]),
            "d_norm": float(r["d_norm"]),
            "csv_pred": str(r.get("prediction", "")),
            "csv_label": str(r.get("label", "")),
        }
    return sel_map


def compute_fold_thresholds(
    unified_csv_path: str,
    fold: int,
    task_name: str,
    q_near: float = 0.5,
    q_cat: float = 0.99,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute error quantile thresholds for a specific fold.
    
    Args:
        unified_csv_path: Path to CSV with predictions and errors.
        fold: Fold index to compute thresholds for.
        task_name: Task name ("word" or "sent").
        q_near: Quantile for near-miss threshold (default: 0.5).
        q_cat: Quantile for catastrophic threshold (default: 0.99).
    
    Returns:
        Tuple of (q_near_value, q_cat_value) or (None, None) if no errors.
    """
    df = pd.read_csv(unified_csv_path, sep=";")
    df = df.rename(columns={
        "Task": "task",
        "Fold": "fold",
        "Sample_index": "sample_index",
        "Prediction": "prediction",
        "Label": "label",
        "Levenshtein_distance": "levenshtein_distance",
    })
    
    df = df[df["task"].astype(str) == str(task_name)].copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["levenshtein_distance"] = pd.to_numeric(df["levenshtein_distance"], errors="coerce")
    df = df.dropna(subset=["fold", "levenshtein_distance"]).copy()
    df["fold"] = df["fold"].astype(int)
    df["levenshtein_distance"] = df["levenshtein_distance"].astype(int)
    df = df[df["fold"] == int(fold)].copy()
    
    if df.empty:
        return None, None

    df["label"] = df["label"].astype(str)
    df["y_len"] = df["label"].str.len().clip(lower=1)
    df["d_norm"] = df["levenshtein_distance"] / df["y_len"]

    nz = df[df["levenshtein_distance"] > 0]
    if nz.empty:
        return 0.0, 0.0  # No errors in fold

    q50 = float(nz["d_norm"].quantile(q_near))
    q99 = float(nz["d_norm"].quantile(q_cat))
    return q50, q99
