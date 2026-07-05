#!/usr/bin/env python3
"""Dataset-aware thesis V3c figure for sequence-level contrastive alignment.

The figure compares MLP and Pool-MLP connector outputs before and after
sequence-level alignment. Color encodes dataset and marker encodes modality.
It writes directly to thesis/figures/v3c_contrastive_comparison_umap.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

BASE = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz")
OUT_PDF = Path("/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/v3c_contrastive_comparison_umap.pdf")

PANELS = [
    ("H1_hybrid_mlp", "HWR-GPT (MLP + CTC)"),
    ("J2_contrastive_mlp", "HWR-GPT (MLP + CTC + seq. alignment)"),
    ("H1_hybrid_pooling", "HWR-GPT (Pool-MLP + CTC)"),
    ("J2_contrastive_pooling", "HWR-GPT (Pool-MLP + CTC + seq. alignment)"),
]
DATASETS = [
    ("onhw", "OnHW", "tab:blue"),
    ("hw6", "Private", "tab:orange"),
]


def load_panel(exp_name: str) -> dict[str, np.ndarray]:
    return dict(np.load(str(BASE / exp_name / "embeddings_fold0.npz"), allow_pickle=True))


def main() -> None:
    all_embs: list[np.ndarray] = []
    slices: list[tuple[int, str, str, str, tuple[int, int], tuple[int, int]]] = []
    idx = 0

    for panel_idx, (exp_name, title) in enumerate(PANELS):
        data = load_panel(exp_name)
        for ds_key, ds_label, _ in DATASETS:
            conn = data[f"connector_{ds_key}"]
            text = data[f"text_anchor_{ds_key}"]
            conn_slice = (idx, idx + len(conn))
            idx += len(conn)
            all_embs.append(conn)
            text_slice = (idx, idx + len(text))
            idx += len(text)
            all_embs.append(text)
            slices.append((panel_idx, title, ds_key, ds_label, conn_slice, text_slice))

    coords = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(
        np.concatenate(all_embs, axis=0)
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=False, sharey=False)
    handles_seen: set[str] = set()

    for panel_idx, title in enumerate([title for _, title in PANELS]):
        ax = axes[panel_idx // 2, panel_idx % 2]
        for _, _, ds_key, ds_label, conn_slice, text_slice in [s for s in slices if s[0] == panel_idx]:
            color = next(c for key, _, c in DATASETS if key == ds_key)
            conn_xy = coords[conn_slice[0]:conn_slice[1]]
            text_xy = coords[text_slice[0]:text_slice[1]]
            text_label = f"{ds_label} text"
            imu_label = f"{ds_label} IMU"
            ax.scatter(
                text_xy[:, 0], text_xy[:, 1],
                marker="o", s=70, facecolors="none", edgecolors=color,
                linewidths=0.8, alpha=0.35,
                label=text_label if text_label not in handles_seen else None,
                zorder=1,
            )
            handles_seen.add(text_label)
            ax.scatter(
                conn_xy[:, 0], conn_xy[:, 1],
                marker="^", s=12, c=color, alpha=0.55,
                label=imu_label if imu_label not in handles_seen else None,
                zorder=2,
            )
            handles_seen.add(imu_label)

        ax.set_title(title, fontsize=12, fontweight="bold")
        if panel_idx // 2 == 1:
            ax.set_xlabel("UMAP dim 1", fontsize=10)
        if panel_idx % 2 == 0:
            ax.set_ylabel("UMAP dim 2", fontsize=10)
        ax.tick_params(labelsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(
        "Dataset and Modality Structure Before and After Sequence-Level Alignment (Fold 0)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PDF), dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
