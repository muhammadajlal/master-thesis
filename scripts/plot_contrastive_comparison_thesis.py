#!/usr/bin/env python3
"""Thesis-clean variant of the contrastive UMAP comparison figure (fig 6.3).

Differences from plot_contrastive_comparison.py:
  - no suptitle and no vendor/dataset codenames in the figure itself
  - panel titles use reader-facing condition names,
    no internal experiment codes and no embedded cos/L2 numbers
    (per-dataset metrics live in tab:contrastive-alignment)
  - legend wording matches the thesis caption
  - writes directly to thesis/figures/contrastive_comparison_umap.pdf

Inputs: saved fold-0 embeddings under analysis/embedding_viz/<exp>/embeddings_fold0.npz
(combined OnHW + private word validation samples, keys connector_{onhw,hw6} /
text_anchor_{onhw,hw6}).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz")
OUT_PDF = Path("/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/contrastive_comparison_umap.pdf")

CONFIGS = [
    ("H1_hybrid_mlp", "HWR-GPT (MLP + CTC)"),
    ("J2_contrastive_mlp", "HWR-GPT (MLP + CTC + seq. alignment)"),
    ("H1_hybrid_pooling", "HWR-GPT (Pool-MLP + CTC)"),
    ("J2_contrastive_pooling", "HWR-GPT (Pool-MLP + CTC + seq. alignment)"),
]


def load_combined(exp_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(str(BASE / exp_name / "embeddings_fold0.npz"), allow_pickle=True))
    conn = np.concatenate([data[f"connector_{ds}"] for ds in ("onhw", "hw6")], axis=0)
    text = np.concatenate([data[f"text_anchor_{ds}"] for ds in ("onhw", "hw6")], axis=0)
    return conn, text


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    all_embs, panel_info, idx = [], [], 0
    for exp_name, title in CONFIGS:
        conn, text = load_combined(exp_name)
        n = len(conn)
        imu_slice = (idx, idx + n); idx += n
        all_embs.append(conn)
        text_slice = (idx, idx + n); idx += n
        all_embs.append(text)
        panel_info.append((imu_slice, text_slice, title))

    combined = np.concatenate(all_embs, axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(combined)

    for i, (imu_slice, text_slice, title) in enumerate(panel_info):
        ax = axes[i // 2, i % 2]
        c_imu = coords[imu_slice[0]:imu_slice[1]]
        c_text = coords[text_slice[0]:text_slice[1]]
        ax.scatter(c_text[:, 0], c_text[:, 1], c="tab:green", alpha=0.20, s=100,
                   marker="o", label="Ground-truth text embeddings",
                   edgecolors="tab:green", linewidths=0.5, zorder=1)
        ax.scatter(c_imu[:, 0], c_imu[:, 1], c="tab:red", alpha=0.55, s=12,
                   label="Projected IMU embeddings", zorder=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if i // 2 == 1:
            ax.set_xlabel("UMAP dim 1", fontsize=10)
        if i % 2 == 0:
            ax.set_ylabel("UMAP dim 2", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PDF), dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
