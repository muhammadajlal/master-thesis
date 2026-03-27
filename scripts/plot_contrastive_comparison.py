#!/usr/bin/env python3
"""
Contrastive Loss Comparison — H1 (CTC only) vs J2 (CTC + Contrastive)
=====================================================================
Produces a 2x2 grid:
  Left column:  H1 baseline (CTC only)
  Right column: J2 contrastive (CTC + InfoNCE)
  Top row:      MLP connector
  Bottom row:   Pooling-MLP connector

Each panel shows IMU tokens (small) vs text anchors (large) in UMAP/PCA space.
Joint fitting ensures comparable embedding spaces across panels.
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

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz")
OUT_DIR = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # (row_label, left_exp, right_exp, left_title, right_title)
    ("MLP", "H1_hybrid_mlp", "J2_contrastive_mlp",
     "H1: MLP + CTC (baseline)", "J2: MLP + CTC + Contrastive"),
    ("Pooling-MLP", "H1_hybrid_pooling", "J2_contrastive_pooling",
     "H1p: Pool-MLP + CTC (baseline)", "J2p: Pool-MLP + CTC + Contrastive"),
]


def load_npz(exp_name: str) -> dict:
    path = BASE / exp_name / "embeddings_fold0.npz"
    return dict(np.load(str(path), allow_pickle=True))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_n * b_n, axis=1)))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def make_figure(method: str = "umap", dataset: str = "both"):
    """Create 2x2 comparison figure for a given dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect all embeddings for joint fitting
    all_embs = []
    panel_info = []  # (row, col, imu_slice, text_slice, label, exp_data)

    idx = 0
    for row_idx, (row_label, left_exp, right_exp, left_title, right_title) in enumerate(CONFIGS):
        for col_idx, (exp_name, title) in enumerate([(left_exp, left_title), (right_exp, right_title)]):
            data = load_npz(exp_name)
            if dataset == "both":
                # Combine both datasets
                conn_list, text_list = [], []
                for ds in ["onhw", "hw6"]:
                    if f"connector_{ds}" in data:
                        conn_list.append(data[f"connector_{ds}"])
                        text_list.append(data[f"text_anchor_{ds}"])
                conn = np.concatenate(conn_list, axis=0)
                text = np.concatenate(text_list, axis=0)
            else:
                conn = data[f"connector_{dataset}"]
                text = data[f"text_anchor_{dataset}"]

            n = len(conn)
            imu_slice = (idx, idx + n)
            idx += n
            all_embs.append(conn)
            text_slice = (idx, idx + n)
            idx += n
            all_embs.append(text)

            sim = cos_sim(conn, text)
            dist = l2_dist(conn, text)
            panel_info.append((row_idx, col_idx, imu_slice, text_slice, title, sim, dist))

    combined = np.concatenate(all_embs, axis=0)

    # Fit reducer jointly
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(combined)
        method_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(combined)
        var = reducer.explained_variance_ratio_
        method_label = f"PCA (PC1: {var[0]:.1%}, PC2: {var[1]:.1%})"

    # Plot panels
    for row_idx, col_idx, imu_slice, text_slice, title, sim, dist in panel_info:
        ax = axes[row_idx, col_idx]
        c_imu = coords[imu_slice[0]:imu_slice[1]]
        c_text = coords[text_slice[0]:text_slice[1]]

        ax.scatter(c_text[:, 0], c_text[:, 1],
                   c="tab:green", alpha=0.20, s=100, marker="o",
                   label="Text anchors (GT)", edgecolors="tab:green",
                   linewidths=0.5, zorder=1)
        ax.scatter(c_imu[:, 0], c_imu[:, 1],
                   c="tab:red", alpha=0.55, s=12,
                   label="IMU embeddings (sensor)", zorder=2)

        ax.set_title(f"{title}\ncos_sim={sim:.3f}  L2={dist:.1f}",
                     fontsize=11, fontweight="bold")
        if row_idx == 1:
            ax.set_xlabel(f"{method_label} dim 1", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f"{method_label} dim 2", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=8)

    ds_name = {"both": "OnHW + Stabilo", "onhw": "OnHW", "hw6": "Stabilo"}[dataset]
    fig.suptitle(
        f"Effect of Contrastive Alignment Loss on Modality Gap ({ds_name})\n"
        f"Left: CTC only (baseline)  |  Right: CTC + In-Batch Contrastive (InfoNCE)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ["pdf", "png"]:
        path = OUT_DIR / f"contrastive_comparison_{dataset}_{method}.{ext}"
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    for dataset in ["both", "onhw", "hw6"]:
        for method in ["pca", "umap"]:
            print(f"\n=== {dataset} / {method.upper()} ===")
            try:
                make_figure(method=method, dataset=dataset)
            except Exception as e:
                print(f"  FAILED: {e}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Experiment':42s} {'Dataset':8s} {'Cos Sim':>8s} {'L2 Dist':>8s}")
    print("=" * 75)
    for row_label, left_exp, right_exp, left_title, right_title in CONFIGS:
        for exp_name, title in [(left_exp, left_title), (right_exp, right_title)]:
            data = load_npz(exp_name)
            for ds in ["onhw", "hw6"]:
                conn = data[f"connector_{ds}"]
                text = data[f"text_anchor_{ds}"]
                sim = cos_sim(conn, text)
                dist = l2_dist(conn, text)
                ds_label = "OnHW" if ds == "onhw" else "Stabilo"
                print(f"  {title:40s} {ds_label:8s} {sim:8.4f} {dist:8.2f}")
        print("-" * 75)


if __name__ == "__main__":
    main()
