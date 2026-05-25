#!/usr/bin/env python3
"""
Comparative analysis of AR-only vs Hybrid CTC-AR encoder representations.

Generates:
    1. t-SNE / PCA scatter plots of encoder frame embeddings (colored by character)
    2. Cosine similarity matrices (intra-sample frame similarity)
    3. Cross-attention maps (teacher-forced)

Usage:
    python analysis/scripts/compare_ar_hybrid.py \
        --ar_ckpt   <path-to-ar-only-best_cer.pth> \
        --hyb_ckpt  <path-to-hybrid-best_cer.pth> \
        --dataset   /path/to/onhw_wi_word_rh \
        --fold 0 \
        --outdir    figures/ar_vs_hybrid \
        [--max_samples 1000] [--device cuda]

If you want difficulty-based qualitative selection, provide a unified CSV
using --difficulty_csv along with --difficulty_task.
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
    p.add_argument("--dataset", type=str, default=None, help="Dataset root (e.g., onhw_wi_word_rh)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--outdir", type=str, default="figures/ar_vs_hybrid")
    p.add_argument("--max_samples", type=int, default=1000, help="Max samples for feature extraction")
    p.add_argument("--n_qual_samples", type=int, default=6, help="N samples for qualitative visuals (cosine sim, attention)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42)

    # Difficulty-based qualitative selection (cosine-sim)
    p.add_argument(
        "--difficulty_mode",
        type=str,
        default="auto",
        choices=["auto", "difficulty", "random"],
        help="auto: use difficulty if a CSV is provided, else random.",
    )
    p.add_argument(
        "--difficulty_csv",
        type=str,
        default=None,
        help="Unified CSV used to select difficulty samples.",
    )
    p.add_argument(
        "--difficulty_task",
        type=str,
        default=None,
        help="Task name in the unified CSV used for difficulty selection.",
    )
    p.add_argument(
        "--difficulty_sep",
        type=str,
        default=";",
        help="CSV separator for --difficulty_csv (default: ';').",
    )
    p.add_argument(
        "--difficulty_quantiles",
        type=str,
        default="0.50,0.50,0.90,0.90,0.99,0.99",
        help="Comma-separated quantiles for difficulty picks (default: '0.50,0.50,0.90,0.90,0.99,0.99').",
    )
    p.add_argument(
        "--difficulty_n",
        type=int,
        default=6,
        help="Number of qualitative examples to pick in difficulty mode (default: 6).",
    )
    p.add_argument(
        "--difficulty_no_easy",
        action="store_true",
        help="Disable explicit easy example (normalized LD == 0) when selecting difficulty samples.",
    )
    p.add_argument(
        "--skip_checkpoints",
        action="store_true",
        help="Skip checkpoint-based analyses (t-SNE/PCA/cosine sim).",
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
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Plot PCA of encoder frame features, colored by character class.

    If ``xlim`` / ``ylim`` are passed, the panel uses those axis limits
    instead of matplotlib's data-driven auto-scaling. This is how the
    AR-only and Hybrid PCA panels get rendered with identical canvases
    so they look symmetric on the page.
    """
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

    # Landscape aspect (wider than tall) so the two side-by-side panels
    # fill the row when embedded in the paper. Short title (just the
    # variance percentage) avoids truncation under narrow minipage widths;
    # the latex subcaption identifies which panel is AR vs hybrid.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=2, alpha=0.45, rasterized=True)
    ax.set_title(f"PC1 + PC2 = {pca.explained_variance_ratio_[:2].sum():.1%}", fontsize=18, pad=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=16)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    from matplotlib.lines import Line2D
    max_legend = 40
    legend_chars = unique_chars[:max_legend]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=char_to_color[c],
                       markersize=7, label=c if c != '' else '<blank>') for c in legend_chars]
    ax.legend(handles=handles, fontsize=8, ncol=4, loc='upper right', framealpha=0.75)

    # Fixed 6x6 canvas (no bbox_inches='tight') so both panels render at
    # identical physical size when embedded side-by-side. Return the PCA
    # embedding so the caller can compute shared axis limits across panels.
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved PCA plot: {}", save_path)
    return emb, pca.explained_variance_ratio_


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


def _standardize_columns(df):
    col_map = {c: "".join(str(c).split()).lower() for c in df.columns}

    targets = {
        "task": ["task"],
        "fold": ["fold"],
        "json_path": ["json_path", "jsonpath"],
        "sample_index": ["sample_index", "sampleindex"],
        "prediction": ["prediction", "pred"],
        "label": ["label", "gt", "groundtruth"],
        "levenshtein_distance": ["levenshtein_distance", "levenshteindistance", "lev", "distance"],
    }

    rename = {}
    for original, norm in col_map.items():
        for canon, aliases in targets.items():
            if norm in aliases:
                rename[original] = canon
    df = df.rename(columns=rename)

    required = ["task", "fold", "sample_index", "prediction", "label", "levenshtein_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV after normalization: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def _load_difficulty_samples_from_csv(
    csv_path: str,
    *,
    sep: str,
    task: str | None,
    fold: int,
) -> tuple[list[str], list[str], list[int], dict[int, str], str]:
    import pandas as pd

    df = pd.read_csv(csv_path, sep=sep)
    df = _standardize_columns(df)

    df["task"] = df["task"].astype(str)
    if task is None:
        tasks = sorted(set(df["task"].tolist()))
        if len(tasks) != 1:
            raise ValueError(
                "--difficulty_task is required when CSV has multiple tasks. "
                f"Found tasks: {tasks}"
            )
        task = tasks[0]

    df = df[df["task"] == task].copy()
    if df.empty:
        raise ValueError(f"No rows found for task '{task}' in difficulty CSV")

    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    if fold >= 0:
        df = df[df["fold"] == fold].copy()
        if df.empty:
            raise ValueError(f"No rows found for task '{task}' and fold {fold} in difficulty CSV")

    df["prediction"] = df["prediction"].astype(str)
    df["label"] = df["label"].astype(str)
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce")
    df = df.dropna(subset=["sample_index"])

    preds = df["prediction"].tolist()
    labels = df["label"].tolist()
    sample_ids = df["sample_index"].astype(int).tolist()
    label_map = {int(s): str(l) for s, l in zip(sample_ids, labels)}
    return preds, labels, sample_ids, label_map, task


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

    # If the user asked for exactly as many quantile picks as n (e.g. 6 picks for
    # 0.50,0.50,0.90,0.90,0.99,0.99), do NOT prepend an "easy" example; it would
    # otherwise crowd out the requested quantile picks.
    want_easy = bool(include_easy) and (n > len(q))

    if want_easy:
        easy = np.where(d_norm == 0.0)[0]
        if easy.size > 0:
            idx_easy = int(easy[0])
            chosen.append(idx_easy)
            tags[idx_easy] = "easy0"

    # Track duplicate quantiles for unique tags (e.g. p50a, p50b)
    _q_total: dict[int, int] = {}
    for qq in q:
        _q_total[int(round(qq * 100))] = _q_total.get(int(round(qq * 100)), 0) + 1
    _q_seen: dict[int, int] = {}

    for qq in q:
        qq = float(qq)
        qq_key = int(round(qq * 100))
        _q_seen[qq_key] = _q_seen.get(qq_key, 0) + 1
        suffix = chr(ord('a') + _q_seen[qq_key] - 1) if _q_total[qq_key] > 1 else ""
        target = float(np.quantile(d_norm, qq))
        order = np.argsort(np.abs(d_norm - target)).tolist()
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
                tags[int(idx)] = f"p{qq_key:02d}{suffix}"
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


def _choose_difficulty_sample_ids(
    preds: list[str],
    labels: list[str],
    sample_ids: list[int],
    *,
    n: int,
    quantiles: list[float],
    include_easy: bool,
    seed: int,
) -> tuple[list[int], dict[str, object]]:
    if len(sample_ids) != len(preds):
        raise ValueError("sample_ids length mismatch with preds")

    idxs, meta = _choose_difficulty_indices(
        preds,
        labels,
        n=n,
        quantiles=quantiles,
        include_easy=include_easy,
        seed=seed,
    )
    chosen_sample_ids = [int(sample_ids[i]) for i in idxs]
    for item, sid in zip(meta.get("chosen", []), chosen_sample_ids):
        item["sample_index"] = int(sid)
    return chosen_sample_ids, meta


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

    # Build subset_pos/sample_id -> difficulty info mapping.
    # NOTE: In qualitative subset extraction, sample_idx is typically 0..(n-1).
    # For CSV difficulty selection we also store `subset_pos` in meta.
    diff_map: dict[int, dict[str, object]] = {}
    if difficulty_meta is not None and "chosen" in difficulty_meta:
        for item in difficulty_meta["chosen"]:
            key = item.get("subset_pos")
            if key is None:
                key = item.get("sample_index")
            if key is None:
                key = item.get("idx")
            if key is None:
                continue
            diff_map[int(key)] = {
                "tag": item.get("tag", ""),
                "norm_ld": item.get("norm_ld", None),
                "label": item.get("label", None),
                "sample_index": item.get("sample_index", None),
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
        diff_info = diff_map.get(int(sample_id), {})
        diff_tag = str(diff_info.get("tag", "") or "")
        norm_ld = diff_info.get("norm_ld", None)
        title_suffix = ""
        # Always show norm_LD if present, even if tag is empty.
        if diff_tag or (norm_ld is not None):
            title_suffix = " ["
            if diff_tag:
                title_suffix += f"{diff_tag}"
            if norm_ld is not None:
                try:
                    title_suffix += (", " if diff_tag else "") + f"norm_LD={float(norm_ld):.3f}"
                except Exception:
                    title_suffix += (", " if diff_tag else "") + f"norm_LD={norm_ld}"
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
            f'Cosine Similarity — GT: "{word_ar}" (subset_pos {sample_id}){title_suffix}',
            fontsize=13,
            y=1.12,
        )

        path = os.path.join(save_dir, f"cosine_sim_sample{sample_id:04d}_{word_ar[:10]}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cosine similarity comparison: {}", path)


# ──────── 3b. Diagonal Dominance Metrics (DD, Leak, M_off) ──────── #

def compute_dd_leak_moff(
    sim: np.ndarray, boundaries: list[int],
) -> dict[str, float]:
    """Compute DD(w), Leak(w), M_off(w) from a cosine-similarity matrix.

    Args:
        sim: (T, T) cosine similarity matrix.
        boundaries: segment boundaries [b0, b1, ..., bK] where segment i = [b_i, b_{i+1}).

    Returns:
        dict with keys DD, Leak, M_off.
    """
    K = len(boundaries) - 1  # number of segments
    if K <= 1:
        return {"DD": float("nan"), "Leak": float("nan"), "M_off": float("nan")}

    dd_per_seg: list[float] = []
    leak_per_seg: list[float] = []
    off_values: list[np.ndarray] = []

    for i in range(K):
        si = slice(boundaries[i], boundaries[i + 1])
        self_sim = float(sim[si, si].mean())

        cross_vals: list[np.ndarray] = []
        adj_vals: list[np.ndarray] = []
        for j in range(K):
            if j == i:
                continue
            sj = slice(boundaries[j], boundaries[j + 1])
            block = sim[si, sj]
            cross_vals.append(block.ravel())
            off_values.append(block.ravel())
            if abs(j - i) == 1:
                adj_vals.append(block.ravel())

        cross_sim = float(np.concatenate(cross_vals).mean()) if cross_vals else 0.0
        dd_per_seg.append(self_sim - cross_sim)
        if adj_vals:
            leak_per_seg.append(float(np.concatenate(adj_vals).mean()))

    return {
        "DD": float(np.mean(dd_per_seg)),
        "Leak": float(np.mean(leak_per_seg)) if leak_per_seg else float("nan"),
        "M_off": float(np.concatenate(off_values).mean()) if off_values else float("nan"),
    }


def compute_dd_leak_moff_over_samples(feats: dict) -> dict[str, float]:
    """Compute DD/Leak/M_off averaged over all samples in a feature dict.

    Args:
        feats: dict from extract_encoder_features with keys
               'features', 'char_labels', 'sample_idx'.

    Returns:
        dict with mean/std for DD, Leak, M_off and n_samples.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    unique_samples = np.unique(feats["sample_idx"])
    dd_list: list[float] = []
    leak_list: list[float] = []
    moff_list: list[float] = []

    for sid in unique_samples:
        mask = feats["sample_idx"] == sid
        f = feats["features"][mask]
        chars = [feats["char_labels"][i] for i, m in enumerate(mask) if m]
        if f.shape[0] < 2:
            continue

        sim = cosine_similarity(f)
        boundaries, _ = _segments_from_char_labels(chars)
        metrics = compute_dd_leak_moff(sim, boundaries)

        if not np.isnan(metrics["DD"]):
            dd_list.append(metrics["DD"])
        if not np.isnan(metrics["Leak"]):
            leak_list.append(metrics["Leak"])
        if not np.isnan(metrics["M_off"]):
            moff_list.append(metrics["M_off"])

    return {
        "DD_mean": float(np.mean(dd_list)) if dd_list else float("nan"),
        "DD_std": float(np.std(dd_list)) if dd_list else float("nan"),
        "Leak_mean": float(np.mean(leak_list)) if leak_list else float("nan"),
        "Leak_std": float(np.std(leak_list)) if leak_list else float("nan"),
        "M_off_mean": float(np.mean(moff_list)) if moff_list else float("nan"),
        "M_off_std": float(np.std(moff_list)) if moff_list else float("nan"),
        "n_samples": len(dd_list),
    }


# ──────────────── 5. Cross-Attention Maps ──────────────── #

class UniversalCrossAttnCatcher:
    """Captures cross-attention weights from both vanilla and gated decoders.

    For **gated** (GatedMultiheadAttention): reads ``_last_attn_weights``
    attribute that is stored right after softmax inside the module.

    For **vanilla** (nn.MultiheadAttention): patches forward to enable
    ``need_weights=True`` and captures the returned attention tensor.
    """

    def __init__(self):
        self.weights: list["torch.Tensor"] = []
        self._patched: list[tuple] = []
        self._handles: list = []

    def clear(self):
        self.weights.clear()

    def _vanilla_hook(self, module, inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            self.weights.append(out[1].detach().cpu())

    def _gated_hook(self, module, inp, out):
        if hasattr(module, "_last_attn_weights"):
            self.weights.append(module._last_attn_weights.cpu())

    def patch_decoder(self, decoder: "nn.Module"):
        import torch.nn as nn
        from rewi.model.ARDecoder import GatedMultiheadAttention

        for name, m in decoder.named_modules():
            if "self_attn" in name:
                continue
            if "multihead_attn" not in name:
                continue

            if isinstance(m, GatedMultiheadAttention):
                h = m.register_forward_hook(self._gated_hook)
                self._handles.append(h)
            elif isinstance(m, nn.MultiheadAttention):
                orig = m.forward

                def _make_wrapped(original):
                    def wrapped(*args, **kwargs):
                        kwargs["need_weights"] = True
                        kwargs["average_attn_weights"] = False
                        return original(*args, **kwargs)
                    return wrapped

                m.forward = _make_wrapped(orig)
                self._patched.append((m, orig))
                h = m.register_forward_hook(self._vanilla_hook)
                self._handles.append(h)

    def unpatch(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for m, orig in self._patched:
            m.forward = orig
        self._patched.clear()


def _attn_list_to_matrix(
    attn_list: list["torch.Tensor"],
    expected_tgt_len: int | None = None,
) -> np.ndarray | None:
    """Average a list of per-layer attention tensors into (tgt, src)."""
    import torch

    if not attn_list:
        return None
    mats = []
    for a in attn_list:
        t = a
        while t.dim() > 2:
            t = t.mean(dim=0)
        mats.append(t)
    M = torch.stack(mats, dim=0).mean(dim=0)
    M = M - M.min()
    if M.max() > 0:
        M = M / M.max()
    M = M.numpy()
    if expected_tgt_len is not None:
        if M.shape[0] != expected_tgt_len and M.shape[1] == expected_tgt_len:
            M = M.T
    return M


def extract_cross_attention_teacher_forced(
    model: "nn.Module",
    dataloader: "DataLoader",
    device: str,
    categories: list[str],
) -> list[dict | None]:
    """Extract per-sample cross-attention maps under teacher forcing.

    Each result dict contains:
        attn: np.ndarray (T_tgt, T_src) averaged over heads & layers.
        label: str  ground-truth word.
        tgt_tokens: list[str]  target token labels (BOS + chars).
    """
    import torch

    vocab_ctc = len(categories)
    BOS_ID = vocab_ctc + 1  # layout: PAD=Vctc, BOS=Vctc+1, EOS=Vctc+2

    catcher = UniversalCrossAttnCatcher()
    if hasattr(model, "decoder"):
        catcher.patch_decoder(model.decoder)
    elif hasattr(model, "dec"):
        catcher.patch_decoder(model.dec)

    model.eval()
    results: list[dict | None] = []

    with torch.no_grad():
        for x, y, len_x, len_y in dataloader:
            B = x.size(0)
            x = x.to(device)

            L = int(len_y[0])
            lab_ids = y[0, :L].tolist() if y.dim() == 2 else []
            y_inp = torch.tensor(
                [[BOS_ID] + lab_ids], dtype=torch.long, device=device,
            )

            catcher.clear()

            # Forward (works for both BaseModel and DualHeadModel)
            from rewi.model import DualHeadModel
            if isinstance(model, DualHeadModel):
                model(x, in_lengths=len_x.to(device), y_inp=y_inp)
            else:
                model(x, in_lengths=len_x.to(device), y_inp=y_inp)

            if catcher.weights:
                M = _attn_list_to_matrix(catcher.weights, expected_tgt_len=y_inp.size(1))
                label_str = "".join(
                    categories[i] for i in lab_ids
                    if 0 <= i < len(categories) and i != 0
                )
                tgt_tokens = ["<bos>"] + [
                    categories[i] if 0 <= i < len(categories) else "?"
                    for i in lab_ids
                ]
                results.append(
                    {"attn": M, "label": label_str, "tgt_tokens": tgt_tokens}
                )
            else:
                results.append(None)

    catcher.unpatch()
    return results


def plot_attention_comparison(
    attn_ar: list[dict | None],
    attn_hyb: list[dict | None],
    save_dir: str,
    difficulty_meta: dict[str, object] | None = None,
) -> None:
    """Side-by-side cross-attention heatmaps for AR-only vs Hybrid."""
    os.makedirs(save_dir, exist_ok=True)

    # Build subset_pos/sample_id -> difficulty info mapping.
    diff_map: dict[int, dict[str, object]] = {}
    if difficulty_meta is not None and "chosen" in difficulty_meta:
        for item in difficulty_meta["chosen"]:
            key = item.get("subset_pos")
            if key is None:
                key = item.get("sample_index")
            if key is None:
                key = item.get("idx")
            if key is None:
                continue
            diff_map[int(key)] = {
                "tag": item.get("tag", ""),
                "norm_ld": item.get("norm_ld", None),
                "label": item.get("label", None),
                "sample_index": item.get("sample_index", None),
            }

    n = min(len(attn_ar), len(attn_hyb))
    for i in range(n):
        a_ar = attn_ar[i]
        a_hyb = attn_hyb[i]
        if a_ar is None or a_hyb is None:
            continue

        label = a_ar["label"]
        tgt_ar = a_ar["tgt_tokens"]
        tgt_hyb = a_hyb["tgt_tokens"]

        diff_info = diff_map.get(int(i), {})
        diff_tag = str(diff_info.get("tag", "") or "")
        norm_ld = diff_info.get("norm_ld", None)
        title_suffix = ""
        if diff_tag or (norm_ld is not None):
            title_suffix = " ["
            if diff_tag:
                title_suffix += f"{diff_tag}"
            if norm_ld is not None:
                try:
                    title_suffix += (", " if diff_tag else "") + f"norm_LD={float(norm_ld):.3f}"
                except Exception:
                    title_suffix += (", " if diff_tag else "") + f"norm_LD={norm_ld}"
            title_suffix += "]"

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18, 6), dpi=150, constrained_layout=True,
        )

        for ax, M, tgt, name in [
            (ax1, a_ar["attn"], tgt_ar, "AR-only"),
            (ax2, a_hyb["attn"], tgt_hyb, "Hybrid"),
        ]:
            if M is None:
                ax.set_axis_off()
                continue
            im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xlabel("Encoder frame", fontsize=10)
            ax.set_ylabel("Decoder token", fontsize=10)
            ax.set_title(f'{name} — "{label}"', fontsize=11)
            if len(tgt) <= 30:
                ax.set_yticks(range(len(tgt)))
                ax.set_yticklabels(tgt, fontsize=7)
            ax.tick_params(axis="both", labelsize=8)

        fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, label="Attention weight", pad=0.02)
        fig.suptitle(
            f'Cross-Attention — GT: "{label}"{title_suffix}',
            fontsize=13, y=1.05,
        )

        safe_label = label[:10].replace("/", "_")
        path = os.path.join(save_dir, f"attn_sample{i:04d}_{safe_label}.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved attention comparison: {}", path)


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
    args.dataset = _resolve_path(args.dataset)
    args.difficulty_csv = _resolve_path(args.difficulty_csv)

    _require_exists(args.ar_ckpt, name="--ar_ckpt")
    _require_exists(args.hyb_ckpt, name="--hyb_ckpt")
    _require_exists(args.dataset, name="--dataset")
    _require_exists(args.difficulty_csv, name="--difficulty_csv")

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    categories = CATEGORIES_WORD

    # Decide qualitative sample indices for visuals (cosine similarity).
    qual_indices: list[int] | None = None
    qual_meta: dict[str, object] | None = None
    qual_label_map: dict[int, str] | None = None
    mode = str(args.difficulty_mode)

    if mode == "difficulty" or (mode == "auto" and args.difficulty_csv is not None):
        if args.difficulty_csv is None:
            raise ValueError("difficulty mode requires --difficulty_csv")
        preds_exp, labels_exp, sample_ids, label_map, task_name = _load_difficulty_samples_from_csv(
            args.difficulty_csv,
            sep=str(args.difficulty_sep),
            task=args.difficulty_task,
            fold=int(args.fold),
        )
        quantiles = _parse_quantiles(args.difficulty_quantiles)
        qual_indices, qual_meta = _choose_difficulty_sample_ids(
            [str(x) for x in preds_exp],
            [str(x) for x in labels_exp],
            sample_ids,
            n=int(args.difficulty_n),
            quantiles=quantiles,
            include_easy=(not bool(args.difficulty_no_easy)),
            seed=int(args.seed),
        )
        qual_meta["task"] = task_name
        qual_meta["fold"] = int(args.fold)
        qual_label_map = label_map
        logger.info("Qualitative indices (difficulty from CSV): {}", qual_indices)

        # Provide a stable mapping for visualization annotations.
        # In the qualitative subset loader we preserve order = qual_indices.
        if isinstance(qual_meta, dict) and "chosen" in qual_meta:
            pos_map = {int(ds_idx): int(pos) for pos, ds_idx in enumerate(qual_indices)}
            for item in qual_meta.get("chosen", []):
                try:
                    ds_idx = item.get("sample_index", None)
                    if ds_idx is not None:
                        item["subset_pos"] = int(pos_map[int(ds_idx)])
                except Exception:
                    pass

    # ── Checkpoint-based analyses (t-SNE, cosine sim) ── #
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

            # Optional sanity-check: CSV labels should match dataset labels.
            if qual_label_map is not None:
                mism = 0
                for i in qual_indices:
                    csv_label = qual_label_map.get(int(i))
                    if csv_label is None:
                        continue
                    if str(dataset.annos[int(i)]["label"]) != str(csv_label):
                        mism += 1
                if mism > 0:
                    logger.warning(
                        "{} / {} qualitative indices have label mismatch between CSV and dataset. "
                        "This suggests CSV ordering may not match dataset ordering.",
                        mism,
                        len(qual_indices),
                    )

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

            # Re-render the two PCA panels with shared axis limits so they
            # have identical canvases for the paper figure. The first
            # plot_pca call (per checkpoint) returns the projected emb,
            # but we re-run here to keep the helper stateless.
            from sklearn.decomposition import PCA as _PCA
            emb_ar_  = _PCA(n_components=2, random_state=args.seed).fit_transform(feats_ar['features'])
            emb_hyb_ = _PCA(n_components=2, random_state=args.seed).fit_transform(feats_hyb['features'])
            x_lo = min(emb_ar_[:, 0].min(), emb_hyb_[:, 0].min())
            x_hi = max(emb_ar_[:, 0].max(), emb_hyb_[:, 0].max())
            y_lo = min(emb_ar_[:, 1].min(), emb_hyb_[:, 1].min())
            y_hi = max(emb_ar_[:, 1].max(), emb_hyb_[:, 1].max())
            pad_x = 0.05 * (x_hi - x_lo)
            pad_y = 0.05 * (y_hi - y_lo)
            shared_xlim = (x_lo - pad_x, x_hi + pad_x)
            shared_ylim = (y_lo - pad_y, y_hi + pad_y)
            logger.info("Re-rendering PCAs with shared limits x=({:.1f},{:.1f}) y=({:.1f},{:.1f})",
                        *shared_xlim, *shared_ylim)
            plot_pca(feats_ar['features'], feats_ar['char_labels'],
                     "AR-only Encoder Features (PCA)",
                     os.path.join(args.outdir, "pca_ar_only.pdf"),
                     seed=args.seed, xlim=shared_xlim, ylim=shared_ylim)
            plot_pca(feats_hyb['features'], feats_hyb['char_labels'],
                     "Hybrid CTC-AR Encoder Features (PCA)",
                     os.path.join(args.outdir, "pca_hybrid.pdf"),
                     seed=args.seed, xlim=shared_xlim, ylim=shared_ylim)

        # ── DD / Leak / M_off (averaged over all extracted samples) ── #
        dd_ar = dd_hyb = None
        if feats_ar is not None:
            dd_ar = compute_dd_leak_moff_over_samples(feats_ar)
            logger.info(
                "AR-only  DD={DD_mean:.4f}±{DD_std:.4f}  Leak={Leak_mean:.4f}±{Leak_std:.4f}  "
                "M_off={M_off_mean:.4f}±{M_off_std:.4f}  (n={n_samples})",
                **dd_ar,
            )
        if feats_hyb is not None:
            dd_hyb = compute_dd_leak_moff_over_samples(feats_hyb)
            logger.info(
                "Hybrid   DD={DD_mean:.4f}±{DD_std:.4f}  Leak={Leak_mean:.4f}±{Leak_std:.4f}  "
                "M_off={M_off_mean:.4f}±{M_off_std:.4f}  (n={n_samples})",
                **dd_hyb,
            )
        if dd_ar is not None or dd_hyb is not None:
            dd_report = {}
            if dd_ar is not None:
                dd_report["ar"] = dd_ar
            if dd_hyb is not None:
                dd_report["hyb"] = dd_hyb
            with open(os.path.join(args.outdir, "dd_leak_moff.json"), "w") as f:
                json.dump(dd_report, f, indent=2)
            logger.info("Saved DD/Leak/M_off report: {}", os.path.join(args.outdir, "dd_leak_moff.json"))

        # ── Qualitative visual analyses on the SAME subset ── #
        if feats_ar is not None and feats_hyb is not None:
            try:
                feats_ar_q = None
                feats_hyb_q = None
                attn_ar_q = None
                attn_hyb_q = None

                if args.ar_ckpt:
                    vocab_ar = len(categories) + 3
                    model_ar_q = BaseModel(
                        args.arch_en, args.arch_de, args.num_channel, vocab_ar, 0,
                        use_gated_attention=args.use_gated_attention,
                        gating_type=args.gating_type,
                    )
                    model_ar_q = load_model_from_checkpoint(args.ar_ckpt, model_ar_q, args.device)
                    feats_ar_q = extract_encoder_features(model_ar_q, loader_qual, args.device, categories, max_samples=10**9)
                    attn_ar_q = extract_cross_attention_teacher_forced(
                        model_ar_q, loader_qual, args.device, categories,
                    )
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
                    attn_hyb_q = extract_cross_attention_teacher_forced(
                        model_hyb_q, loader_qual, args.device, categories,
                    )
                    del model_hyb_q
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if feats_ar_q is not None and feats_hyb_q is not None:
                    # In subset extraction, sample_idx is 0..(n-1) for both models.
                    chosen_ids = list(range(len(qual_indices)))
                    plot_cosine_similarity_comparison(
                        feats_ar_q,
                        feats_hyb_q,
                        os.path.join(args.outdir, "cosine_similarity"),
                        n_samples=len(chosen_ids),
                        seed=args.seed,
                        chosen_sample_ids=chosen_ids,
                        difficulty_meta=qual_meta,
                    )

                if attn_ar_q is not None and attn_hyb_q is not None:
                    plot_attention_comparison(
                        attn_ar_q,
                        attn_hyb_q,
                        os.path.join(args.outdir, "attention"),
                        difficulty_meta=qual_meta,
                    )
            except Exception as e:
                logger.warning("Qualitative visual analyses skipped: {}", e)

    logger.info("=== Analysis complete. Outputs in: {} ===", args.outdir)


if __name__ == "__main__":
    main()
