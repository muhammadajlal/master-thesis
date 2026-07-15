
#!/usr/bin/env python3
"""
Modality Gap Analysis — Experiments Retained in the Thesis Evidence Chain
============================================================
Produces:
  1. Summary table: cosine similarity + L2 distance for all experiments x datasets
  2. Bar chart comparing cosine sim / L2 dist across experiments
  3. Grid scatter plot: IMU tokens vs text anchors for each experiment (joint UMAP/PCA)

Usage (after running viz_embed_all_gap.sbatch):
    python scripts/plot_modality_gap_all.py
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
OUT_DIR = Path("/home/woody/iwso/iwso214h/imu-hwr/analysis/embedding_viz/comparison_all")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All experiments to compare — (dir_name, display_label, category)
EXPERIMENTS = [
    ("H1_hybrid_mlp",           "H1 (baseline)",        "baseline"),
    ("J1_contrastive_mlp",      "J1-MLP (global)",      "global_contrastive"),
    ("J1_contrastive_pooling",  "J1-Pool (global)",     "global_contrastive"),
    ("J2_contrastive_mlp",      "J2-MLP (global)",      "global_contrastive"),
    ("J2_contrastive_pooling",  "J2-Pool (global)",     "global_contrastive"),
    ("K1_ctc_mse",              "K1 (MSE)",             "local_alignment"),
    ("K2_ctc_posterior",        "K2 (CTC posterior)",   "architectural"),
    ("K3_ec_loss",              "K3 (EC loss)",         "regularization"),
    ("K5_kv_multiview",         "K5 (KV inject)",       "architectural"),
    ("L1_mini_qformer",         "L1 (mini Q-Former)",   "architectural"),
    ("L2_kv_slim",              "L2 (KV slim)",         "architectural"),
    ("M1_byt5_hybrid_mlp",      "M1 (ByT5 dec)",        "architectural"),
    ("N1_conformer_hybrid_mlp", "N1 (Conformer enc)",   "architectural"),
]

CATEGORY_COLORS = {
    "baseline":            "tab:gray",
    "global_contrastive":  "tab:blue",
    "local_alignment":     "tab:orange",
    "architectural":       "tab:green",
    "regularization":      "tab:purple",
}


def load_npz(exp_name: str) -> dict | None:
    path = BASE / exp_name / "embeddings_fold0.npz"
    if not path.exists():
        return None
    return dict(np.load(str(path), allow_pickle=True))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_n * b_n, axis=1)))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


# ── 1. Summary table ──────────────────────────────────────

def print_summary_table():
    """Print a markdown-style summary table."""
    print("\n" + "=" * 90)
    print(f"{'Experiment':28s} | {'OnHW cos':>9s} {'OnHW L2':>8s} | {'HW6 cos':>9s} {'HW6 L2':>8s} | {'Category'}")
    print("=" * 90)
    for dir_name, label, category in EXPERIMENTS:
        data = load_npz(dir_name)
        if data is None:
            print(f"  {label:26s} |   --- MISSING ---")
            continue
        row = f"  {label:26s} |"
        for ds in ["onhw", "hw6"]:
            key_c = f"connector_{ds}"
            key_t = f"text_anchor_{ds}"
            if key_c in data and key_t in data:
                sim = cos_sim(data[key_c], data[key_t])
                dist = l2_dist(data[key_c], data[key_t])
                row += f" {sim:9.4f} {dist:8.2f} |"
            else:
                row += f"      ---      --- |"
        row += f" {category}"
        print(row)
    print("=" * 90)


# ── 2. Bar chart ──────────────────────────────────────────

def plot_bar_chart():
    """Bar chart comparing cosine sim and L2 distance across experiments."""
    for ds, ds_label in [("onhw", "OnHW"), ("hw6", "Stabilo")]:
        labels, sims, dists, colors = [], [], [], []
        for dir_name, label, category in EXPERIMENTS:
            data = load_npz(dir_name)
            if data is None:
                continue
            key_c = f"connector_{ds}"
            key_t = f"text_anchor_{ds}"
            if key_c not in data or key_t not in data:
                continue
            labels.append(label)
            sims.append(cos_sim(data[key_c], data[key_t]))
            dists.append(l2_dist(data[key_c], data[key_t]))
            colors.append(CATEGORY_COLORS.get(category, "tab:gray"))

        if not labels:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(labels))
        ax1.barh(x, sims, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_yticks(x)
        ax1.set_yticklabels(labels, fontsize=10)
        ax1.set_xlabel("Cosine Similarity (higher = closer)", fontsize=11)
        ax1.set_title(f"Cosine Similarity — {ds_label}", fontsize=12, fontweight="bold")
        ax1.invert_yaxis()
        # Add value labels
        for i, v in enumerate(sims):
            ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

        ax2.barh(x, dists, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_yticks(x)
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel("L2 Distance (lower = closer)", fontsize=11)
        ax2.set_title(f"L2 Distance — {ds_label}", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()
        for i, v in enumerate(dists):
            ax2.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=9)

        # Legend for categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, edgecolor="black", label=cat.replace("_", " ").title())
            for cat, c in CATEGORY_COLORS.items()
        ]
        ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            path = OUT_DIR / f"gap_barchart_{ds}.{ext}"
            fig.savefig(str(path), dpi=200, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


# ── 3. Grid scatter plot ──────────────────────────────────

def plot_scatter_grid(method: str = "pca", dataset: str = "onhw"):
    """Grid of scatter plots: IMU vs text anchors for each experiment."""
    ds_key = dataset
    ds_label = "OnHW" if ds_key == "onhw" else "Stabilo"

    # Load all available data
    loaded = []
    for dir_name, label, category in EXPERIMENTS:
        data = load_npz(dir_name)
        if data is None:
            continue
        key_c = f"connector_{ds_key}"
        key_t = f"text_anchor_{ds_key}"
        if key_c not in data or key_t not in data:
            continue
        loaded.append((label, category, data[key_c], data[key_t]))

    if not loaded:
        print(f"No data for {ds_label}")
        return

    n_exp = len(loaded)
    ncols = min(5, n_exp)
    nrows = (n_exp + ncols - 1) // ncols

    # Collect all embeddings for joint fitting
    all_embs = []
    slices = []
    idx = 0
    for label, category, conn, text in loaded:
        n = len(conn)
        imu_s = (idx, idx + n)
        idx += n
        all_embs.append(conn)
        text_s = (idx, idx + n)
        idx += n
        all_embs.append(text)
        slices.append((imu_s, text_s))

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

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, (label, category, conn, text) in enumerate(loaded):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        imu_s, text_s = slices[i]
        c_imu = coords[imu_s[0]:imu_s[1]]
        c_text = coords[text_s[0]:text_s[1]]

        sim = cos_sim(conn, text)
        dist = l2_dist(conn, text)

        ax.scatter(c_text[:, 0], c_text[:, 1],
                   c="tab:green", alpha=0.20, s=80, marker="o",
                   label="Text anchors", edgecolors="tab:green",
                   linewidths=0.3, zorder=1)
        ax.scatter(c_imu[:, 0], c_imu[:, 1],
                   c=CATEGORY_COLORS.get(category, "tab:red"),
                   alpha=0.55, s=10, zorder=2,
                   label="IMU embeddings")

        ax.set_title(f"{label}\ncos={sim:.3f}  L2={dist:.1f}",
                     fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for i in range(n_exp, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Modality Gap Across All Experiments — {ds_label} ({method_label})\n"
        f"Green circles = GPT-2 text embeddings | Colored dots = IMU connector output",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUT_DIR / f"gap_grid_{dataset}_{method}.{ext}"
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── 4. CER vs Cosine Similarity scatter ──────────────────

def plot_cer_vs_gap():
    """Scatter plot: CER vs cosine similarity — does closing the gap help?"""
    # CER results from evaluate.py (mean over 5 folds)
    CER_RESULTS = {
        "H1 (baseline)":      {"onhw": 7.40, "hw6": 16.87},
        "J1-MLP (global)":    {"onhw": 7.47, "hw6": 18.50},
        "J1-Pool (global)":   {"onhw": 7.49, "hw6": 18.11},
        "J2-MLP (global)":    {"onhw": 7.23, "hw6": 18.04},
        "J2-Pool (global)":   {"onhw": 7.47, "hw6": 17.52},
        "K1 (MSE)":           {"onhw": 7.52, "hw6": 18.30},
        "K2 (CTC posterior)": {"onhw": 11.59, "hw6": 38.09},
        "K3 (EC loss)":       {"onhw": 7.66, "hw6": 19.15},
        "K5 (KV inject)":     {"onhw": 7.00, "hw6": 16.89},
    }

    for ds, ds_label in [("onhw", "OnHW"), ("hw6", "Stabilo")]:
        fig, ax = plt.subplots(figsize=(10, 7))

        for dir_name, label, category in EXPERIMENTS:
            data = load_npz(dir_name)
            if data is None or label not in CER_RESULTS:
                continue
            key_c = f"connector_{ds}"
            key_t = f"text_anchor_{ds}"
            if key_c not in data or key_t not in data:
                continue

            sim = cos_sim(data[key_c], data[key_t])
            cer = CER_RESULTS[label].get(ds)
            if cer is None:
                continue

            color = CATEGORY_COLORS.get(category, "tab:gray")
            ax.scatter(sim, cer, c=color, s=120, edgecolors="black",
                       linewidths=0.8, zorder=3)
            ax.annotate(label, (sim, cer), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Cosine Similarity (connector output vs GPT-2 text embeddings)", fontsize=11)
        ax.set_ylabel("CER % (lower = better)", fontsize=11)
        ax.set_title(
            f"CER vs Modality Gap — {ds_label}\n"
            "Does closing the gap improve recognition?",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, edgecolor="black", label=cat.replace("_", " ").title())
            for cat, c in CATEGORY_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            path = OUT_DIR / f"cer_vs_gap_{ds}.{ext}"
            fig.savefig(str(path), dpi=200, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


def main():
    print_summary_table()
    plot_bar_chart()

    for ds in ["onhw", "hw6"]:
        for method in ["pca"]:
            print(f"\n=== Scatter grid: {ds} / {method.upper()} ===")
            try:
                plot_scatter_grid(method=method, dataset=ds)
            except Exception as e:
                print(f"  FAILED: {e}")

        if HAS_UMAP:
            try:
                plot_scatter_grid(method="umap", dataset=ds)
            except Exception as e:
                print(f"  UMAP FAILED: {e}")

    plot_cer_vs_gap()

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
