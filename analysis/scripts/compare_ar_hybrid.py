#!/usr/bin/env python3
"""
Comparative analysis of AR-only vs Hybrid CTC-AR encoder representations.

Generates:
  1. t-SNE / PCA scatter plots of encoder frame embeddings (colored by character)
  2. Levenshtein distance distribution comparison
  3. Cosine similarity matrices (intra-sample frame similarity)
  4. Per-character error rate comparison

Usage:
    python analysis/scripts/compare_ar_hybrid.py \
        --ar_ckpt   <path-to-ar-only-best_cer.pth> \
        --hyb_ckpt  <path-to-hybrid-best_cer.pth> \
        --ar_export <path-to-ar-export-json> \
        --hyb_export <path-to-hybrid-export-json> \
        --dataset   /path/to/onhw_wi_word_rh \
        --fold 0 \
        --outdir    figures/ar_vs_hybrid \
        [--max_samples 1000] [--device cuda]

All parts are optional — if you only provide exports, only distribution plots
are generated; if you only provide checkpoints, only embedding plots are
generated.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from loguru import logger

# Ensure REWI_work is on the path
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# Repo root (imu-hwr) for resolving paths like results/... and data/...
REPO_ROOT = os.path.abspath(os.path.join(PROJ, "..", ".."))


def _resolve_path(p: str | None) -> str | None:
    """Resolve a possibly-relative path.

    Many users run this script from work/REWI_work, but pass paths relative to
    the repository root (e.g. results/... or data/...).
    """
    if p is None:
        return None
    p = str(p)
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p

    candidates = [
        os.path.join(REPO_ROOT, p),
        os.path.join(PROJ, p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return p


def _require_exists(p: str | None, *, name: str) -> None:
    if p is None:
        return
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing {name}: {p}\n"
            f"Tip: if you run from {PROJ}, paths like 'results/...' should exist under {REPO_ROOT}/results."  # noqa: E501
        )


# ─────────────────────────── CLI ─────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(description="AR-only vs Hybrid encoder analysis")
    p.add_argument("--ar_ckpt", type=str, default=None, help="AR-only best checkpoint path")
    p.add_argument("--hyb_ckpt", type=str, default=None, help="Hybrid best checkpoint path")
    p.add_argument("--ar_export", type=str, default=None, help="AR-only export JSON (predictions+labels)")
    p.add_argument("--hyb_export", type=str, default=None, help="Hybrid export JSON")
    p.add_argument("--dataset", type=str, default=None, help="Dataset root (e.g., onhw_wi_word_rh)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--outdir", type=str, default="figures/ar_vs_hybrid")
    p.add_argument("--max_samples", type=int, default=1000, help="Max samples for feature extraction")
    p.add_argument("--n_qual_samples", type=int, default=8, help="N samples for qualitative visuals (cosine sim)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42)

    # Difficulty-based qualitative selection (cosine-sim)
    p.add_argument(
        "--difficulty_mode",
        type=str,
        default="auto",
        choices=["auto", "difficulty", "random"],
        help="auto: use difficulty if an export is provided, else random.",
    )
    p.add_argument(
        "--difficulty_from",
        type=str,
        default="auto",
        choices=["auto", "hyb", "ar"],
        help="Which export to use for difficulty (auto: prefer --hyb_export if set).",
    )
    p.add_argument(
        "--difficulty_quantiles",
        type=str,
        default="0.50,0.90,0.99",
        help="Comma-separated quantiles for difficulty picks (default: '0.50,0.90,0.99').",
    )
    p.add_argument(
        "--difficulty_n",
        type=int,
        default=4,
        help="Number of qualitative examples to pick in difficulty mode (default: 4).",
    )
    p.add_argument(
        "--difficulty_no_easy",
        action="store_true",
        help="Disable explicit easy example (normalized LD == 0) when selecting difficulty samples.",
    )
    p.add_argument(
        "--skip_checkpoints",
        action="store_true",
        help="Skip checkpoint-based analyses (t-SNE/PCA/cosine sim) and only run export-based metrics.",
    )
    # Model config (must match training)
    p.add_argument("--arch_en", type=str, default="blconv_b")
    p.add_argument("--arch_de", type=str, default="ar_transformer_s")
    p.add_argument("--num_channel", type=int, default=13)
    p.add_argument("--use_gated_attention", action="store_true", default=True)
    p.add_argument("--gating_type", type=str, default="elementwise")
    return p.parse_args()


# ─────────────────────── Categories ──────────────────────── #

CATEGORIES_WORD = [
    "", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "Ä", "Ö", "Ü",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "ä", "ö", "ü", "ß",
]


# ──────────────── 1. t-SNE / PCA of Encoder Frames ──────────────── #

def plot_tsne(
    features: np.ndarray,
    char_labels: list[str],
    title: str,
    save_path: str,
    perplexity: int = 30,
    seed: int = 42,
    max_points: int = 50000,
):
    """Plot t-SNE of encoder frame features, colored by character class."""
    from sklearn.manifold import TSNE
    import inspect

    N = features.shape[0]
    if N > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, max_points, replace=False)
        features = features[idx]
        char_labels = [char_labels[i] for i in idx]

    logger.info("Running t-SNE on {} points...", features.shape[0])
    # scikit-learn renamed/standardized `n_iter` -> `max_iter` (sklearn>=1.8).
    sig = inspect.signature(TSNE)
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = 1000
    elif "n_iter" in sig.parameters:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    emb = tsne.fit_transform(features)

    # Assign colors by unique character
    unique_chars = sorted(set(char_labels))
    cmap = cm.get_cmap("tab20", len(unique_chars))
    char_to_color = {c: cmap(i) for i, c in enumerate(unique_chars)}
    colors = [char_to_color[c] for c in char_labels]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=1, alpha=0.4, rasterized=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # Legend (show subset if too many)
    from matplotlib.lines import Line2D
    max_legend = 40
    legend_chars = unique_chars[:max_legend]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=char_to_color[c],
                       markersize=6, label=c if c != '' else '<blank>') for c in legend_chars]
    ax.legend(handles=handles, fontsize=6, ncol=4, loc='upper right', framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved t-SNE plot: {}", save_path)


def plot_pca(
    features: np.ndarray,
    char_labels: list[str],
    title: str,
    save_path: str,
    max_points: int = 50000,
    seed: int = 42,
):
    """Plot PCA of encoder frame features, colored by character class."""
    from sklearn.decomposition import PCA

    N = features.shape[0]
    if N > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(N, max_points, replace=False)
        features = features[idx]
        char_labels = [char_labels[i] for i in idx]

    logger.info("Running PCA on {} points...", features.shape[0])
    pca = PCA(n_components=2, random_state=seed)
    emb = pca.fit_transform(features)

    unique_chars = sorted(set(char_labels))
    cmap = cm.get_cmap("tab20", len(unique_chars))
    char_to_color = {c: cmap(i) for i, c in enumerate(unique_chars)}
    colors = [char_to_color[c] for c in char_labels]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=1, alpha=0.4, rasterized=True)
    ax.set_title(f"{title}\n(var explained: {pca.explained_variance_ratio_[:2].sum():.2%})", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    from matplotlib.lines import Line2D
    max_legend = 40
    legend_chars = unique_chars[:max_legend]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=char_to_color[c],
                       markersize=6, label=c if c != '' else '<blank>') for c in legend_chars]
    ax.legend(handles=handles, fontsize=6, ncol=4, loc='upper right', framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved PCA plot: {}", save_path)


def plot_tsne_side_by_side(
    feats_ar: dict, feats_hyb: dict,
    save_path: str, perplexity: int = 30, seed: int = 42,
    max_points: int = 30000,
):
    """Side-by-side t-SNE comparison of AR-only vs Hybrid encoder features."""
    from sklearn.manifold import TSNE
    import inspect

    # Subsample both to equal size
    N_ar = feats_ar['features'].shape[0]
    N_hyb = feats_hyb['features'].shape[0]
    N = min(N_ar, N_hyb, max_points)

    rng = np.random.RandomState(seed)
    idx_ar = rng.choice(N_ar, N, replace=False) if N_ar > N else np.arange(N_ar)
    idx_hyb = rng.choice(N_hyb, N, replace=False) if N_hyb > N else np.arange(N_hyb)

    f_ar = feats_ar['features'][idx_ar]
    c_ar = [feats_ar['char_labels'][i] for i in idx_ar]
    f_hyb = feats_hyb['features'][idx_hyb]
    c_hyb = [feats_hyb['char_labels'][i] for i in idx_hyb]

    # Fit t-SNE jointly for comparable embeddings
    combined = np.concatenate([f_ar, f_hyb], axis=0)
    logger.info("Running joint t-SNE on {} points...", combined.shape[0])
    sig = inspect.signature(TSNE)
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = 1000
    elif "n_iter" in sig.parameters:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    emb = tsne.fit_transform(combined)
    emb_ar = emb[:N]
    emb_hyb = emb[N:]

    all_chars = sorted(set(c_ar + c_hyb))
    cmap = cm.get_cmap("tab20", len(all_chars))
    char_to_color = {c: cmap(i) for i, c in enumerate(all_chars)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=150)

    for ax, emb_part, chars, title in [
        (ax1, emb_ar, c_ar, "AR-only Encoder"),
        (ax2, emb_hyb, c_hyb, "Hybrid CTC-AR Encoder"),
    ]:
        colors = [char_to_color[c] for c in chars]
        ax.scatter(emb_part[:, 0], emb_part[:, 1], c=colors, s=1, alpha=0.4, rasterized=True)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    from matplotlib.lines import Line2D
    legend_chars = all_chars[:40]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=char_to_color[c],
                       markersize=6, label=c if c != '' else '<blank>') for c in legend_chars]
    fig.legend(handles=handles, fontsize=6, ncol=8, loc='lower center', framealpha=0.7)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved side-by-side t-SNE: {}", save_path)


# ──────────────── 2. Levenshtein Distribution ──────────────── #

def plot_levenshtein_distribution(
    preds_ar: list[str], labels_ar: list[str],
    preds_hyb: list[str], labels_hyb: list[str],
    save_path: str,
    title: str = "Levenshtein Distance Distribution",
    *,
    normalized: bool = False,
):
    r"""Side-by-side histogram + CDF of per-sample Levenshtein distances.

    If normalized=True, plots distance divided by reference length, i.e.
    $d_\sim = d / |y|$, which is comparable across varying word lengths.
    """
    try:
        from Levenshtein import distance as lev_dist
    except ImportError:
        from rewi.analysis.metrics import levenshtein_distance as lev_dist

    dists_ar_raw = [lev_dist(p, l) for p, l in zip(preds_ar, labels_ar)]
    dists_hyb_raw = [lev_dist(p, l) for p, l in zip(preds_hyb, labels_hyb)]

    if normalized:
        dists_ar = [d / max(1, len(l)) for d, l in zip(dists_ar_raw, labels_ar)]
        dists_hyb = [d / max(1, len(l)) for d, l in zip(dists_hyb_raw, labels_hyb)]
        x_label = "Normalized Levenshtein Distance (d / |ref|)"
    else:
        dists_ar = dists_ar_raw
        dists_hyb = dists_hyb_raw
        x_label = "Levenshtein Distance"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Histogram
    if normalized:
        bins = np.linspace(0.0, max(max(dists_ar), max(dists_hyb), 1e-6), 40)
    else:
        bins = np.arange(0, max(max(dists_ar), max(dists_hyb)) + 2) - 0.5

    ax1.hist(dists_ar, bins=bins, alpha=0.6, label=f"AR-only (mean={np.mean(dists_ar):.3f})", density=True, color="C0")
    ax1.hist(dists_hyb, bins=bins, alpha=0.6, label=f"Hybrid (mean={np.mean(dists_hyb):.3f})", density=True, color="C1")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Density")
    ax1.set_title(f"{title}{' (normalized)' if normalized else ''} — Histogram")
    ax1.legend()

    # CDF
    for dists, label, color in [(dists_ar, "AR-only", "C0"), (dists_hyb, "Hybrid", "C1")]:
        sorted_d = np.sort(dists)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, label=label, color=color)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Cumulative Proportion")
    ax2.set_title(f"{title}{' (normalized)' if normalized else ''} — CDF")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Levenshtein distribution: {}", save_path)


# ──────────────── 3. Cosine Similarity Matrix ──────────────── #

def _segments_from_char_labels(chars: list[str]):
    """Return (boundaries, seg_chars) from per-frame char labels."""
    if len(chars) == 0:
        return [0], []

    boundaries = [0]
    seg_chars = [chars[0]]
    prev = chars[0]
    for t in range(1, len(chars)):
        if chars[t] != prev:
            boundaries.append(t)
            prev = chars[t]
            seg_chars.append(prev)
    boundaries.append(len(chars))
    return boundaries, seg_chars


def _levenshtein_distance(a: str, b: str) -> int:
    """Classic DP Levenshtein distance (no external deps)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is shorter for lower memory.
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _normalized_levenshtein(pred: str, ref: str) -> float:
    return float(_levenshtein_distance(pred, ref)) / float(max(1, len(ref)))


def _parse_quantiles(s: str, *, default: tuple[float, ...] = (0.50, 0.90, 0.99)) -> list[float]:
    s = str(s).strip()
    if not s:
        return list(default)
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out if out else list(default)


def _choose_difficulty_indices(
    preds: list[str],
    labels: list[str],
    *,
    n: int = 4,
    quantiles: list[float] | None = None,
    include_easy: bool = True,
    seed: int = 42,
) -> tuple[list[int], dict[str, object]]:
    """Pick indices by difficulty using normalized Levenshtein distance.

    Default picks: easy0 + {p50, p90, p99} (nearest in value, unique indices).
    """
    q = quantiles if quantiles is not None else [0.50, 0.90, 0.99]
    if len(preds) != len(labels):
        raise ValueError("preds/labels length mismatch")
    if len(preds) == 0:
        raise ValueError("empty preds/labels")

    d_norm = np.array([
        _normalized_levenshtein(str(p), str(r)) for p, r in zip(preds, labels)
    ], dtype=np.float32)

    chosen: list[int] = []
    tags: dict[int, str] = {}

    if include_easy:
        easy = np.where(d_norm == 0.0)[0]
        if easy.size > 0:
            idx_easy = int(easy[0])
            chosen.append(idx_easy)
            tags[idx_easy] = "easy0"

    for qq in q:
        qq = float(qq)
        target = float(np.quantile(d_norm, qq))
        order = np.argsort(np.abs(d_norm - target)).tolist()
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
                tags[int(idx)] = f"p{int(round(qq * 100)):02d}"
                break

    # Fill remaining with a stable, seeded choice among remaining.
    if len(chosen) < n:
        rng = np.random.RandomState(seed)
        remaining = [i for i in range(len(preds)) if i not in chosen]
        if remaining:
            k = min(n - len(chosen), len(remaining))
            extra = rng.choice(remaining, size=k, replace=False).tolist()
            chosen.extend([int(i) for i in extra])

    chosen = chosen[:n]
    meta: dict[str, object] = {
        "quantiles": q,
        "include_easy": include_easy,
        "chosen": [
            {
                "idx": int(i),
                "tag": tags.get(int(i), ""),
                "norm_ld": float(d_norm[int(i)]),
                "pred": str(preds[int(i)]),
                "label": str(labels[int(i)]),
            }
            for i in chosen
        ],
    }
    return chosen, meta


def plot_cosine_similarity(
    features: np.ndarray,
    char_labels: list[str],
    title: str,
    save_path: str,
):
    """Plot frame-to-frame cosine similarity matrix for a single sample."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(features)  # (T, T)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150, constrained_layout=True)
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Boundaries + segment labels
    boundaries, seg_chars = _segments_from_char_labels(char_labels)

    # Mark character boundaries (green grid)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)
        ax.axvline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)

    # Put segment labels JUST above the axes (x in data coords, y in axes coords)
    for i, ch in enumerate(seg_chars):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        seg_len = boundaries[i + 1] - boundaries[i]
        if seg_len > 1:
            ax.text(
                mid,
                1.01,
                ch,
                transform=ax.get_xaxis_transform(),  # x=data, y=axes fraction
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )

    ax.set_title(title, fontsize=12, pad=18)  # extra pad so title won't collide
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    ax.tick_params(axis="both", labelsize=8)

    fig.colorbar(im, ax=ax, shrink=0.85, label="Cosine similarity", pad=0.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_similarity_comparison(
    feats_ar: dict,
    feats_hyb: dict,
    save_dir: str,
    n_samples: int = 4,
    seed: int = 42,
    chosen_sample_ids: list[int] | None = None,
    difficulty_meta: dict[str, object] | None = None,
):
    """Side-by-side cosine similarity matrices for selected samples."""
    os.makedirs(save_dir, exist_ok=True)

    unique_ar = np.unique(feats_ar["sample_idx"])
    unique_hyb = np.unique(feats_hyb["sample_idx"])
    common = np.intersect1d(unique_ar, unique_hyb)

    if chosen_sample_ids is not None:
        common_set = set(common.tolist())
        chosen = [int(i) for i in chosen_sample_ids if int(i) in common_set]
        if len(chosen) == 0:
            raise ValueError("chosen_sample_ids provided but none are common between AR and Hybrid")
    else:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(common, min(n_samples, len(common)), replace=False)

    from sklearn.metrics.pairwise import cosine_similarity

    # Build sample_id -> difficulty info mapping
    diff_map = {}
    if difficulty_meta is not None and "chosen" in difficulty_meta:
        for item in difficulty_meta["chosen"]:
            idx = item.get("idx")
            if idx is not None:
                diff_map[int(idx)] = {
                    "tag": item.get("tag", ""),
                    "norm_ld": item.get("norm_ld", 0.0),
                }

    for sample_id in chosen:
        # AR
        mask_ar = feats_ar["sample_idx"] == sample_id
        f_ar = feats_ar["features"][mask_ar]
        c_ar = [feats_ar["char_labels"][i] for i, m in enumerate(mask_ar) if m]
        word_ar = feats_ar["word_labels"][np.where(mask_ar)[0][0]]

        # Hybrid
        mask_hyb = feats_hyb["sample_idx"] == sample_id
        f_hyb = feats_hyb["features"][mask_hyb]
        c_hyb = [feats_hyb["char_labels"][i] for i, m in enumerate(mask_hyb) if m]

        # Get difficulty info if available
        diff_info = diff_map.get(sample_id, {})
        diff_tag = diff_info.get("tag", "")
        norm_ld = diff_info.get("norm_ld", None)
        title_suffix = ""
        if diff_tag:
            title_suffix = f" [{diff_tag}"
            if norm_ld is not None:
                title_suffix += f", norm_LD={norm_ld:.3f}"
            title_suffix += "]"

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(16, 7),
            dpi=150,
            constrained_layout=True,
        )

        im = None
        panels = [
            (ax1, f_ar, c_ar, "AR-only"),
            (ax2, f_hyb, c_hyb, "Hybrid"),
        ]

        for ax, f, chars, label in panels:
            if f.shape[0] == 0 or len(chars) == 0:
                ax.set_axis_off()
                continue

            sim = cosine_similarity(f)
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

            ax.set_title(f'{label} — "{word_ar}"', fontsize=11, pad=18)
            ax.set_xlabel("Frame index", fontsize=10)
            ax.set_ylabel("Frame index", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)

            boundaries, seg_chars = _segments_from_char_labels(chars)

            # Green boundary grid
            for b in boundaries[1:-1]:
                ax.axhline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)
                ax.axvline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)

            # Segment labels above each subplot (no overlap)
            for i, ch in enumerate(seg_chars):
                mid = (boundaries[i] + boundaries[i + 1]) / 2
                seg_len = boundaries[i + 1] - boundaries[i]
                if seg_len > 1:
                    ax.text(
                        mid,
                        1.01,
                        ch,
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        clip_on=False,
                    )

        if im is not None:
            fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, label="Cosine similarity", pad=0.02)

        # Move suptitle higher so it can't collide with per-axis titles/labels
        fig.suptitle(
            f'Cosine Similarity — "{word_ar}" (sample {sample_id}){title_suffix}',
            fontsize=13,
            y=1.12,
        )

        path = os.path.join(save_dir, f"cosine_sim_sample{sample_id:04d}_{word_ar[:10]}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cosine similarity comparison: {}", path)


# ──────────────── 4. Per-Character Error Rate ──────────────── #

def compute_per_char_error(
    preds: list[str], labels: list[str], categories: list[str],
) -> dict[str, dict]:
    """Compute per-character-class error rates (substitution, deletion, insertion).

    Returns dict: char -> {total, correct, substituted, deleted, sub_rate, del_rate}
    """
    import jiwer

    stats = {c: {'total': 0, 'correct': 0, 'substituted': 0, 'deleted': 0}
             for c in categories if c != ''}

    out = jiwer.process_characters(labels, preds)
    for alignment, ref_str, hyp_str in zip(out.alignments, out.references, out.hypotheses):
        for event in alignment:
            if event.type == 'equal':
                for i in range(event.ref_start_idx, event.ref_end_idx):
                    c = ref_str[i]
                    if c in stats:
                        stats[c]['total'] += 1
                        stats[c]['correct'] += 1
            elif event.type == 'substitute':
                for i in range(event.ref_start_idx, event.ref_end_idx):
                    c = ref_str[i]
                    if c in stats:
                        stats[c]['total'] += 1
                        stats[c]['substituted'] += 1
            elif event.type == 'delete':
                for i in range(event.ref_start_idx, event.ref_end_idx):
                    c = ref_str[i]
                    if c in stats:
                        stats[c]['total'] += 1
                        stats[c]['deleted'] += 1

    # Compute rates
    for c, s in stats.items():
        tot = max(s['total'], 1)
        s['sub_rate'] = s['substituted'] / tot
        s['del_rate'] = s['deleted'] / tot
        s['error_rate'] = (s['substituted'] + s['deleted']) / tot

    return stats


def plot_per_char_error_comparison(
    stats_ar: dict, stats_hyb: dict,
    save_path: str,
    title: str = "Per-Character Error Rate: AR-only vs Hybrid",
):
    """Bar chart comparing per-character error rates."""
    chars = sorted(set(stats_ar.keys()) | set(stats_hyb.keys()))
    chars = [c for c in chars if c != '']

    er_ar = [stats_ar.get(c, {}).get('error_rate', 0) for c in chars]
    er_hyb = [stats_hyb.get(c, {}).get('error_rate', 0) for c in chars]

    x = np.arange(len(chars))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(chars) * 0.4), 6), dpi=150)
    ax.bar(x - w / 2, er_ar, w, label='AR-only', alpha=0.8, color='C0')
    ax.bar(x + w / 2, er_hyb, w, label='Hybrid', alpha=0.8, color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels(chars, fontsize=7)
    ax.set_ylabel("Error Rate (sub + del)")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved per-character error comparison: {}", save_path)


# ──────────────── Main ──────────────── #

def main():
    args = parse_args()

    # Graceful CPU fallback when CUDA isn't available.
    try:
        import torch
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to --device cpu")
            args.device = "cpu"
    except Exception:
        pass

    # Resolve user-provided paths.
    args.ar_ckpt = _resolve_path(args.ar_ckpt)
    args.hyb_ckpt = _resolve_path(args.hyb_ckpt)
    args.ar_export = _resolve_path(args.ar_export)
    args.hyb_export = _resolve_path(args.hyb_export)
    args.dataset = _resolve_path(args.dataset)

    _require_exists(args.ar_ckpt, name="--ar_ckpt")
    _require_exists(args.hyb_ckpt, name="--hyb_ckpt")
    _require_exists(args.ar_export, name="--ar_export")
    _require_exists(args.hyb_export, name="--hyb_export")
    _require_exists(args.dataset, name="--dataset")

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    categories = CATEGORIES_WORD

    # Decide qualitative sample indices for visuals (cosine similarity).
    qual_indices: list[int] | None = None
    qual_meta: dict[str, object] | None = None
    mode = str(args.difficulty_mode)
    export_source = None
    if str(args.difficulty_from) == "hyb":
        export_source = args.hyb_export
    elif str(args.difficulty_from) == "ar":
        export_source = args.ar_export
    else:
        export_source = args.hyb_export or args.ar_export

    if mode == "difficulty" or (mode == "auto" and export_source is not None):
        if export_source is None:
            raise ValueError("difficulty mode requires an export JSON (use --hyb_export or --ar_export)")
        with open(export_source) as f:
            exp = json.load(f)
        preds_exp = exp.get("predictions")
        labels_exp = exp.get("labels")
        if not isinstance(preds_exp, list) or not isinstance(labels_exp, list):
            raise ValueError("Export JSON must contain 'predictions' and 'labels' lists")
        quantiles = _parse_quantiles(args.difficulty_quantiles)
        qual_indices, qual_meta = _choose_difficulty_indices(
            [str(x) for x in preds_exp],
            [str(x) for x in labels_exp],
            n=int(args.difficulty_n),
            quantiles=quantiles,
            include_easy=(not bool(args.difficulty_no_easy)),
            seed=int(args.seed),
        )
        logger.info("Qualitative indices (difficulty): {}", qual_indices)

    # ── Part A: Export-based analyses (Levenshtein, per-char errors) ── #
    if args.ar_export and args.hyb_export:
        logger.info("=== Export-based analyses ===")
        with open(args.ar_export) as f:
            ar_data = json.load(f)
        with open(args.hyb_export) as f:
            hyb_data = json.load(f)

        preds_ar, labels_ar = ar_data['predictions'], ar_data['labels']
        preds_hyb, labels_hyb = hyb_data['predictions'], hyb_data['labels']

        # Levenshtein distribution
        plot_levenshtein_distribution(
            preds_ar, labels_ar, preds_hyb, labels_hyb,
            save_path=os.path.join(args.outdir, "levenshtein_distribution.pdf"),
        )

        # Normalized Levenshtein distribution (distance / |ref|)
        plot_levenshtein_distribution(
            preds_ar, labels_ar, preds_hyb, labels_hyb,
            save_path=os.path.join(args.outdir, "levenshtein_distribution_norm.pdf"),
            normalized=True,
        )

        # Per-character error rates
        stats_ar = compute_per_char_error(preds_ar, labels_ar, categories)
        stats_hyb = compute_per_char_error(preds_hyb, labels_hyb, categories)
        plot_per_char_error_comparison(
            stats_ar, stats_hyb,
            save_path=os.path.join(args.outdir, "per_char_error_rate.pdf"),
        )

    # ── Part B: Checkpoint-based analyses (t-SNE, cosine sim) ── #
    if not getattr(args, "skip_checkpoints", False) and (args.ar_ckpt or args.hyb_ckpt) and args.dataset:
        import torch
        from torch.utils.data import DataLoader, Subset
        from rewi.dataset import HRDataset
        from rewi.dataset.utils import fn_collate
        from rewi.model import BaseModel, DualHeadModel
        from rewi.analysis.encoder_features import (
            extract_encoder_features,
            load_model_from_checkpoint,
        )

        logger.info("=== Checkpoint-based analyses ===")

        # Build val dataloader
        dataset = HRDataset(
            os.path.join(args.dataset, 'val.json'),
            categories,
            8,  # ratio_ds for blconv_b
            args.fold,
            0,  # len_seq
            cache=True,
        )
        loader = DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=fn_collate)

        # Build a dedicated qualitative subset loader so ALL qualitative visuals use the same samples.
        if qual_indices is None:
            rng = np.random.RandomState(args.seed)
            n_pick = min(int(args.n_qual_samples), len(dataset))
            qual_indices = sorted(rng.choice(len(dataset), n_pick, replace=False).tolist())
            logger.info("Qualitative indices (random): {}", qual_indices)
        else:
            # Clamp to dataset range
            qual_indices = [int(i) for i in qual_indices if 0 <= int(i) < len(dataset)]
            if len(qual_indices) == 0:
                raise ValueError("Difficulty-picked qualitative indices are out of range for the dataset")

            # Optional sanity-check: export ordering should match dataset ordering.
            if qual_meta is not None and export_source is not None:
                try:
                    with open(export_source) as f:
                        exp = json.load(f)
                    exp_labels = exp.get("labels")
                    if isinstance(exp_labels, list) and len(exp_labels) == len(dataset):
                        mism = 0
                        for i in qual_indices:
                            if str(dataset.annos[int(i)]["label"]) != str(exp_labels[int(i)]):
                                mism += 1
                        if mism > 0:
                            logger.warning(
                                "{} / {} qualitative indices have label mismatch between export and dataset. "
                                "This suggests export ordering may not match dataset ordering.",
                                mism,
                                len(qual_indices),
                            )
                except Exception:
                    pass

        subset = Subset(dataset, qual_indices)
        loader_qual = DataLoader(subset, batch_size=1, num_workers=2, collate_fn=fn_collate, shuffle=False)

        feats_ar = None
        feats_hyb = None

        # AR-only model
        if args.ar_ckpt:
            logger.info("Loading AR-only model...")
            vocab_ar = len(categories) + 3  # +PAD, BOS, EOS
            model_ar = BaseModel(
                args.arch_en, args.arch_de, args.num_channel, vocab_ar, 0,
                use_gated_attention=args.use_gated_attention,
                gating_type=args.gating_type,
            )
            model_ar = load_model_from_checkpoint(args.ar_ckpt, model_ar, args.device)
            feats_ar = extract_encoder_features(model_ar, loader, args.device, categories, args.max_samples)

            plot_tsne(feats_ar['features'], feats_ar['char_labels'],
                      "AR-only Encoder Features (t-SNE)",
                      os.path.join(args.outdir, "tsne_ar_only.pdf"),
                      perplexity=args.perplexity, seed=args.seed)
            plot_pca(feats_ar['features'], feats_ar['char_labels'],
                     "AR-only Encoder Features (PCA)",
                     os.path.join(args.outdir, "pca_ar_only.pdf"),
                     seed=args.seed)
            del model_ar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Hybrid model
        if args.hyb_ckpt:
            logger.info("Loading Hybrid model...")
            vocab_ar = len(categories) + 3
            vocab_ctc = len(categories)
            model_hyb = DualHeadModel(
                args.arch_en, args.arch_de, 'linear', args.num_channel,
                vocab_ar, vocab_ctc, 0,
                use_gated_attention=args.use_gated_attention,
                gating_type=args.gating_type,
            )
            model_hyb = load_model_from_checkpoint(args.hyb_ckpt, model_hyb, args.device)
            feats_hyb = extract_encoder_features(model_hyb, loader, args.device, categories, args.max_samples)

            plot_tsne(feats_hyb['features'], feats_hyb['char_labels'],
                      "Hybrid CTC-AR Encoder Features (t-SNE)",
                      os.path.join(args.outdir, "tsne_hybrid.pdf"),
                      perplexity=args.perplexity, seed=args.seed)
            plot_pca(feats_hyb['features'], feats_hyb['char_labels'],
                     "Hybrid CTC-AR Encoder Features (PCA)",
                     os.path.join(args.outdir, "pca_hybrid.pdf"),
                     seed=args.seed)

            del model_hyb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Side-by-side t-SNE
        if feats_ar is not None and feats_hyb is not None:
            plot_tsne_side_by_side(
                feats_ar, feats_hyb,
                os.path.join(args.outdir, "tsne_comparison.pdf"),
                perplexity=args.perplexity, seed=args.seed,
            )

            # Cosine similarity comparison on the SAME qualitative subset.
            try:
                feats_ar_q = None
                feats_hyb_q = None

                if args.ar_ckpt:
                    vocab_ar = len(categories) + 3
                    model_ar_q = BaseModel(
                        args.arch_en, args.arch_de, args.num_channel, vocab_ar, 0,
                        use_gated_attention=args.use_gated_attention,
                        gating_type=args.gating_type,
                    )
                    model_ar_q = load_model_from_checkpoint(args.ar_ckpt, model_ar_q, args.device)
                    feats_ar_q = extract_encoder_features(model_ar_q, loader_qual, args.device, categories, max_samples=10**9)
                    del model_ar_q
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if args.hyb_ckpt:
                    vocab_ar = len(categories) + 3
                    vocab_ctc = len(categories)
                    model_hyb_q = DualHeadModel(
                        args.arch_en, args.arch_de, 'linear', args.num_channel,
                        vocab_ar, vocab_ctc, 0,
                        use_gated_attention=args.use_gated_attention,
                        gating_type=args.gating_type,
                    )
                    model_hyb_q = load_model_from_checkpoint(args.hyb_ckpt, model_hyb_q, args.device)
                    feats_hyb_q = extract_encoder_features(model_hyb_q, loader_qual, args.device, categories, max_samples=10**9)
                    del model_hyb_q
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if feats_ar_q is not None and feats_hyb_q is not None:
                    # In subset extraction, sample_idx is 0..(n-1) for both models.
                    chosen_ids = list(range(min(len(qual_indices), 4)))
                    plot_cosine_similarity_comparison(
                        feats_ar_q,
                        feats_hyb_q,
                        os.path.join(args.outdir, "cosine_similarity"),
                        n_samples=len(chosen_ids),
                        seed=args.seed,
                        chosen_sample_ids=chosen_ids,
                        difficulty_meta=qual_meta,
                    )
            except Exception as e:
                logger.warning("Cosine similarity qualitative plot skipped: {}", e)

    logger.info("=== Analysis complete. Outputs in: {} ===", args.outdir)


if __name__ == "__main__":
    main()
