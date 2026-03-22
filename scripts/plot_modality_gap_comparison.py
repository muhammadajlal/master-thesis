#!/usr/bin/env python3
"""
Modality Gap Comparison — Supervisor-style Fig. 3
====================================================
Produces a 2x2 grid:
  Top row:    Baseline (MLP, no CTC) — OnHW | Stabilo
  Bottom row: Hybrid (MLP + CTC)     — OnHW | Stabilo

Each panel shows:
  - Sensor embeddings (small colored dots)
  - Ground-truth text anchors (large transparent circles)

UMAP is fit jointly so all four panels share the same embedding space.

Usage:
    python scripts/plot_modality_gap_comparison.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA

# ── Config ──
BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz")
OUT_DIR = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_EXP = "vlm_ablation_A1_mlp_pretrained"   # A1: MLP, no CTC
HYBRID_EXP = "H1_hybrid_mlp"                       # H1: MLP + CTC

# Also do Pooling variants
BASELINE_POOL_EXP = "vlm_ablation_B1_pooling_pretrained"  # B1: Pooling-MLP, no CTC
HYBRID_POOL_EXP = "H1_hybrid_pooling"                      # H1p: Pooling-MLP + CTC


def load_npz(exp_name: str) -> dict:
    path = BASE / exp_name / "embeddings_fold0.npz"
    return dict(np.load(str(path), allow_pickle=True))


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> float:
    """Mean pairwise cosine similarity between matched pairs."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(a_norm * b_norm, axis=1)
    return float(np.mean(sims))


def make_comparison_figure(
    baseline_data: dict,
    hybrid_data: dict,
    method: str = "umap",
    baseline_label: str = "Baseline (MLP)",
    hybrid_label: str = "Hybrid CTC+MLP",
    out_name: str = "modality_gap_comparison",
):
    """Create 2x2 comparison figure."""
    # Collect all embeddings for joint fitting
    all_embs = []
    slices = {}
    idx = 0
    for row_label, data in [("baseline", baseline_data), ("hybrid", hybrid_data)]:
        for ds in ["onhw", "hw6"]:
            conn = data[f"connector_{ds}"]
            text = data[f"text_anchor_{ds}"]
            n = len(conn)
            slices[(row_label, ds, "imu")] = (idx, idx + n)
            idx += n
            all_embs.append(conn)
            slices[(row_label, ds, "text")] = (idx, idx + n)
            idx += n
            all_embs.append(text)

    combined = np.concatenate(all_embs, axis=0)

    # Fit reducer on all data jointly
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(combined)
        method_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(combined)
        var = reducer.explained_variance_ratio_
        method_label = f"PCA (PC1: {var[0]:.1%}, PC2: {var[1]:.1%})"

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ds_labels = {"onhw": "OnHW (public)", "hw6": "Stabilo (private)"}
    row_configs = [
        ("baseline", baseline_label),
        ("hybrid", hybrid_label),
    ]

    for row_idx, (row_key, row_title) in enumerate(row_configs):
        for col_idx, ds in enumerate(["onhw", "hw6"]):
            ax = axes[row_idx, col_idx]

            # Get coordinates
            s_imu = slices[(row_key, ds, "imu")]
            s_text = slices[(row_key, ds, "text")]
            c_imu = coords[s_imu[0]:s_imu[1]]
            c_text = coords[s_text[0]:s_text[1]]

            # Compute cosine similarity
            conn = combined[s_imu[0]:s_imu[1]]
            text = combined[s_text[0]:s_text[1]]
            cos_sim = cosine_sim_matrix(conn, text)

            # Plot text anchors (background, large transparent)
            ax.scatter(
                c_text[:, 0], c_text[:, 1],
                c="tab:green", alpha=0.20, s=100, marker="o",
                label="Text anchors (GT)",
                edgecolors="tab:green", linewidths=0.5, zorder=1,
            )
            # Plot IMU tokens (foreground, small solid)
            ax.scatter(
                c_imu[:, 0], c_imu[:, 1],
                c="tab:red", alpha=0.55, s=12,
                label="IMU embeddings (sensor)", zorder=2,
            )

            ax.set_title(
                f"{row_title} — {ds_labels[ds]}\n"
                f"cos sim = {cos_sim:.3f}",
                fontsize=12,
            )
            if row_idx == 1:
                ax.set_xlabel(f"{method_label} dim 1", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{method_label} dim 2", fontsize=10)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=8)

    fig.suptitle(
        "Modality Gap: Sensor Embeddings vs Text Anchors\n"
        f"Top: {baseline_label} (no CTC)  |  Bottom: {hybrid_label} (+ auxiliary CTC)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["pdf", "png"]:
        path = OUT_DIR / f"{out_name}_{method}.{ext}"
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("Loading embeddings...")
    baseline = load_npz(BASELINE_EXP)
    hybrid = load_npz(HYBRID_EXP)

    for method in ["umap", "pca"]:
        print(f"\n=== MLP: {method.upper()} ===")
        make_comparison_figure(
            baseline, hybrid,
            method=method,
            baseline_label="MLP Connector (A1)",
            hybrid_label="MLP + CTC Auxiliary (H1)",
            out_name="modality_gap_mlp",
        )

    # Also do Pooling variants if available
    try:
        baseline_pool = load_npz(BASELINE_POOL_EXP)
        hybrid_pool = load_npz(HYBRID_POOL_EXP)
        for method in ["umap", "pca"]:
            print(f"\n=== Pooling-MLP: {method.upper()} ===")
            make_comparison_figure(
                baseline_pool, hybrid_pool,
                method=method,
                baseline_label="Pooling-MLP Connector (B1)",
                hybrid_label="Pooling-MLP + CTC Auxiliary (H1p)",
                out_name="modality_gap_pooling",
            )
    except FileNotFoundError:
        print("Pooling variants not available, skipping.")

    # Print cosine similarity summary
    print("\n=== Cosine Similarity Summary ===")
    for name, exp in [(BASELINE_EXP, "A1"), (HYBRID_EXP, "H1"),
                       (BASELINE_POOL_EXP, "B1"), (HYBRID_POOL_EXP, "H1p")]:
        try:
            d = load_npz(name)
        except FileNotFoundError:
            continue
        for ds in ["onhw", "hw6"]:
            sim = cosine_sim_matrix(d[f"connector_{ds}"], d[f"text_anchor_{ds}"])
            print(f"  {exp} {ds}: cos_sim = {sim:.4f}")


if __name__ == "__main__":
    main()
