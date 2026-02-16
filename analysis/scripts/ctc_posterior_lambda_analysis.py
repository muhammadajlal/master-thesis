#!/usr/bin/env python3
"""CTC posterior diagnostics across hybrid training regimes (lambda_ctc sweep).

This script is intentionally *CTC-only* (internal diagnostic). It compares the
CTC posteriors between different hybrid checkpoints trained with different
lambda_ctc values.

Outputs:
  - Qualitative heatmaps:
      * per-sample grid (columns = lambdas)
      * compact grid (rows = samples, columns = lambdas)
  - Quantitative metrics vs lambda_ctc:
      * mean per-frame entropy (lower = sharper)
      * mean per-frame max probability (higher = peakier)
      * blank occupancy (argmax blank fraction)
      * optional entropy histogram overlay
      * optional peakiness-over-time curves

Key design: deterministic sample selection.
We select fixed dataset indices (seeded) and extract *exactly those samples*
for each checkpoint so comparisons are meaningful.

Example (fold 0, 3 lambdas):

  python analysis/scripts/ctc_posterior_lambda_analysis.py \
    --lambdas 0.1 0.6 1.0 \
    --ckpts \
      results/hwr2/train_element_word_hybrid_01/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth \
      results/hwr2/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth \
      results/hwr2/train_element_word_hybrid_10/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth \
    --dataset data/onhw_wi_word_rh --fold 0 \
    --outdir figures/ctc_posterior_lambda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Ensure REWI_work is on the path
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# Repo root (imu-hwr) for resolving paths like results/... and data/...
REPO_ROOT = os.path.abspath(os.path.join(PROJ, "..", ".."))


def _resolve_path(p: str | None) -> str | None:
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return p

    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)

    for base in (REPO_ROOT, PROJ):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand

    return p


def _require_exists(p: str, *, name: str) -> None:
    if not os.path.exists(p):
        hint = ""
        # Common failure mode: user uses ${BASE}/${AR} without exporting them, so the shell expands to empty.
        # That yields paths like: /train_element_word_hybrid_01//fold_0/...
        if p.startswith("/") and "train_element_word_hybrid_" in p and "/results/" not in p:
            hint = (
                "\nHint: this path looks like ${BASE}/${AR} expanded to empty. "
                "Either define them (e.g. BASE=results/hwr2 and AR=ar_transformer_s__onhw_wi_word_rh) "
                "or pass explicit paths starting with 'results/hwr2/...'."
            )
        raise FileNotFoundError(
            f"Missing {name}: {p}\n"
            f"Tip: if you run from {PROJ}, paths like 'results/...' should exist under {REPO_ROOT}/results."
            f"{hint}"
        )


CATEGORIES_WORD = [
    "",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "Ä",
    "Ö",
    "Ü",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "ä",
    "ö",
    "ü",
    "ß",
]


@dataclass
class PosteriorSample:
    posteriors: np.ndarray  # (T, V)
    label: str
    enc_length: int
    sample_id: int


@dataclass
class QuantMetrics:
    lambda_ctc: float

    # Over ALL frames (blank-dominated, keep for reference)
    mean_entropy: float
    std_entropy: float
    mean_max_prob: float
    std_max_prob: float

    # Over NON-BLANK frames only (more sensitive)
    mean_entropy_nb: float
    std_entropy_nb: float
    mean_max_prob_nb: float
    std_max_prob_nb: float
    nonblank_ratio: float  # (# nonblank frames) / (# total frames)

    # Blank occupancy (argmax blank fraction)
    blank_occupancy: float

    # Greedy CTC CER (string from argmax path)
    cer_greedy_mean: float     # mean over samples
    cer_greedy_std: float      # std over samples
    cer_greedy_micro: float    # total_edits / total_ref_len

    n_samples: int

def _levenshtein_distance(a: str, b: str) -> int:
    """Classic DP Levenshtein distance (no external deps)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure b is the shorter for lower memory.
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


def _ctc_greedy_decode_from_posteriors(P: np.ndarray, categories: list[str], blank_idx: int = 0) -> str:
    """Greedy CTC decode: argmax per frame -> collapse repeats -> remove blanks -> to string."""
    if P.size == 0:
        return ""
    path = np.argmax(P, axis=1).tolist()

    # Collapse repeats
    collapsed = []
    prev = None
    for idx in path:
        if prev is None or idx != prev:
            collapsed.append(idx)
        prev = idx

    # Remove blanks and map to chars
    out_chars = []
    for idx in collapsed:
        if idx == blank_idx:
            continue
        if 0 <= idx < len(categories):
            out_chars.append(categories[idx])
    return "".join(out_chars)



def _normalized_levenshtein(pred: str, ref: str) -> float:
    d = _levenshtein_distance(pred, ref)
    return float(d) / float(max(1, len(ref)))


def _parse_quantiles(s: str | None, *, default: tuple[float, ...]) -> list[float]:
    if s is None:
        return list(default)
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


def _choose_difficulty_sample_ids_from_export(
    export_path: str,
    *,
    n: int = 4,
    quantiles: list[float] | None = None,
    include_easy: bool = True,
) -> tuple[list[int], dict[str, object]]:
    """Pick dataset indices for qualitative examples by difficulty.

    Strategy (default): 4 examples
      - 1 easy: normalized LD == 0 (if exists)
      - then nearest to quantiles of normalized LD (defaults: 0.50, 0.90, 0.99)

    Returns:
      (indices, meta)
    """
    q = quantiles if quantiles is not None else [0.50, 0.90, 0.99]
    with open(export_path, "r") as f:
        data = json.load(f)

    preds = data.get("predictions")
    labels = data.get("labels")
    if not isinstance(preds, list) or not isinstance(labels, list):
        raise ValueError(f"Export must contain 'predictions' and 'labels' lists: {export_path}")
    if len(preds) != len(labels):
        raise ValueError(f"Export predictions/labels length mismatch: {len(preds)} vs {len(labels)}")
    if len(preds) == 0:
        raise ValueError(f"Export has 0 samples: {export_path}")

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

    # Quantile picks
    for qq in q:
        qq = float(qq)
        target = float(np.quantile(d_norm, qq))
        order = np.argsort(np.abs(d_norm - target))
        picked = None
        for idx in order.tolist():
            if idx not in chosen:
                picked = int(idx)
                break
        if picked is not None:
            chosen.append(picked)
            tags[picked] = f"p{int(round(qq * 100)):02d}"

    # Fill remaining if any duplicates / missing
    if len(chosen) < n:
        order = np.argsort(d_norm).tolist()
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= n:
                break

    chosen = chosen[:n]

    meta: dict[str, object] = {
        "export": export_path,
        "n_export": len(preds),
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


def _select_sample_ids(n_dataset: int, n_samples: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    if n_samples >= n_dataset:
        return list(range(n_dataset))
    return sorted(rng.choice(n_dataset, n_samples, replace=False).tolist())


def compute_quant_metrics(
    posteriors_list: list[PosteriorSample],
    *,
    lambda_ctc: float,
    categories: list[str],
    blank_idx: int = 0,
) -> QuantMetrics:
    ent_all = []
    mx_all = []

    ent_nb = []
    mx_nb = []

    blank_frames = 0
    nonblank_frames = 0
    total_frames = 0

    # Greedy CER aggregates
    cer_per_sample = []
    total_edits = 0
    total_ref_len = 0

    for s in posteriors_list:
        P = s.posteriors  # (T,V)
        T = int(P.shape[0])
        total_frames += T

        if T == 0:
            # Still count sample for CER (empty pred vs label)
            pred = ""
            ref = s.label
            d = _levenshtein_distance(pred, ref)
            total_edits += d
            total_ref_len += max(1, len(ref))
            cer_per_sample.append(d / max(1, len(ref)))
            continue

        P_clipped = np.clip(P, 1e-12, 1.0)

        # per-frame entropy + max prob
        H = -np.sum(P_clipped * np.log(P_clipped), axis=1)  # (T,)
        mx = np.max(P, axis=1)                              # (T,)

        ent_all.extend(H.tolist())
        mx_all.extend(mx.tolist())

        argmax = np.argmax(P, axis=1)
        blank_frames += int(np.sum(argmax == blank_idx))

        nb_mask = (argmax != blank_idx)
        nb_count = int(np.sum(nb_mask))
        nonblank_frames += nb_count
        if nb_count > 0:
            ent_nb.extend(H[nb_mask].tolist())
            mx_nb.extend(mx[nb_mask].tolist())

        # Greedy decode CER
        pred = _ctc_greedy_decode_from_posteriors(P, categories, blank_idx=blank_idx)
        ref = s.label
        d = _levenshtein_distance(pred, ref)
        total_edits += d
        total_ref_len += len(ref)
        cer_per_sample.append(d / max(1, len(ref)))

    ent_all = np.asarray(ent_all, dtype=np.float64)
    mx_all = np.asarray(mx_all, dtype=np.float64)
    ent_nb = np.asarray(ent_nb, dtype=np.float64)
    mx_nb = np.asarray(mx_nb, dtype=np.float64)
    cer_per_sample = np.asarray(cer_per_sample, dtype=np.float64)

    # Micro CER: total edits / total ref length (avoid div0)
    cer_micro = float(total_edits / max(1, total_ref_len))

    return QuantMetrics(
        lambda_ctc=float(lambda_ctc),

        mean_entropy=float(np.mean(ent_all)) if ent_all.size else float("nan"),
        std_entropy=float(np.std(ent_all)) if ent_all.size else float("nan"),
        mean_max_prob=float(np.mean(mx_all)) if mx_all.size else float("nan"),
        std_max_prob=float(np.std(mx_all)) if mx_all.size else float("nan"),

        mean_entropy_nb=float(np.mean(ent_nb)) if ent_nb.size else float("nan"),
        std_entropy_nb=float(np.std(ent_nb)) if ent_nb.size else float("nan"),
        mean_max_prob_nb=float(np.mean(mx_nb)) if mx_nb.size else float("nan"),
        std_max_prob_nb=float(np.std(mx_nb)) if mx_nb.size else float("nan"),
        nonblank_ratio=float(nonblank_frames / max(1, total_frames)),

        blank_occupancy=float(blank_frames / max(1, total_frames)),

        cer_greedy_mean=float(np.mean(cer_per_sample)) if cer_per_sample.size else float("nan"),
        cer_greedy_std=float(np.std(cer_per_sample)) if cer_per_sample.size else float("nan"),
        cer_greedy_micro=cer_micro,

        n_samples=int(len(posteriors_list)),
    )



def plot_metrics_vs_lambda(metrics: list[QuantMetrics], save_path: str) -> None:
    """Plot posterior diagnostics vs λ_ctc for a single fold."""
    metrics = sorted(metrics, key=lambda m: m.lambda_ctc)
    lambdas = [m.lambda_ctc for m in metrics]

    fig, axes = plt.subplots(
        2, 4, figsize=(22, 9), dpi=150, constrained_layout=True
    )
    axes = axes.ravel()

    # 1) Entropy (ALL frames)
    mu = [m.mean_entropy for m in metrics]
    sd = [m.std_entropy for m in metrics]
    axes[0].errorbar(lambdas, mu, yerr=sd, marker="o", capsize=4, linewidth=2)
    axes[0].set_xlabel("λ_ctc")
    axes[0].set_ylabel("Mean per-frame entropy (nats)")
    axes[0].set_title("Entropy (all frames) ↓")
    axes[0].axhline(uniform_entropy, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_entropy:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2) Entropy (NON-BLANK frames)
    mu = [m.mean_entropy_nb for m in metrics]
    sd = [m.std_entropy_nb for m in metrics]
    axes[1].errorbar(lambdas, mu, yerr=sd, marker="o", capsize=4, linewidth=2)
    axes[1].set_xlabel("λ_ctc")
    axes[1].set_ylabel("Mean per-frame entropy (nats)")
    axes[1].set_title("Entropy (non-blank frames) ↓")
    axes[1].axhline(uniform_entropy, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_entropy:.2f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3) Max prob (ALL frames)
    mu = [m.mean_max_prob for m in metrics]
    sd = [m.std_max_prob for m in metrics]
    axes[2].errorbar(lambdas, mu, yerr=sd, marker="s", capsize=4, linewidth=2)
    axes[2].set_xlabel("λ_ctc")
    axes[2].set_ylabel("Mean max P(c|t)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Peakiness (all frames) ↑")
    axes[2].axhline(uniform_max_prob, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_max_prob:.3f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # 4) Max prob (NON-BLANK frames)
    mu = [m.mean_max_prob_nb for m in metrics]
    sd = [m.std_max_prob_nb for m in metrics]
    axes[3].errorbar(lambdas, mu, yerr=sd, marker="s", capsize=4, linewidth=2)
    axes[3].set_xlabel("λ_ctc")
    axes[3].set_ylabel("Mean max P(c|t)")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("Peakiness (non-blank frames) ↑")
    axes[3].axhline(uniform_max_prob, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_max_prob:.3f}")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # 5) Blank occupancy (bar)
    blanks = [m.blank_occupancy for m in metrics]
    axes[4].bar(lambdas, blanks, width=0.06, alpha=0.85, edgecolor="black")
    axes[4].set_xlabel("λ_ctc")
    axes[4].set_ylabel("Argmax blank ratio")
    axes[4].set_ylim(0, 1.0)
    axes[4].set_title("Blank occupancy ↑/≈")
    axes[4].grid(True, alpha=0.3, axis="y")

    # 6) Non-blank ratio (line)
    nb_ratio = [m.nonblank_ratio for m in metrics]
    axes[5].plot(lambdas, nb_ratio, marker="^", linewidth=2)
    axes[5].set_xlabel("λ_ctc")
    axes[5].set_ylabel("Non-blank frame ratio")
    axes[5].set_ylim(0, 1.0)
    axes[5].set_title("Non-blank ratio")
    axes[5].grid(True, alpha=0.3)

    # 7) Greedy CER (micro) (line)
    cer_micro = [m.cer_greedy_micro for m in metrics]
    axes[6].plot(lambdas, cer_micro, marker="D", linewidth=2)
    axes[6].set_xlabel("λ_ctc")
    axes[6].set_ylabel("CER (micro)")
    axes[6].set_title("CTC greedy CER (micro) ↓")
    axes[6].grid(True, alpha=0.3)

    # 8) Greedy CER (mean ± std over samples) (errorbar)
    cer_mean = [m.cer_greedy_mean for m in metrics]
    cer_std = [m.cer_greedy_std for m in metrics]
    axes[7].errorbar(lambdas, cer_mean, yerr=cer_std, marker="D", capsize=4, linewidth=2)
    axes[7].set_xlabel("λ_ctc")
    axes[7].set_ylabel("CER (mean over samples)")
    axes[7].set_title("CTC greedy CER (mean±std) ↓")
    axes[7].grid(True, alpha=0.3)

    fig.suptitle("CTC posterior diagnostics vs λ_ctc", fontsize=14, y=1.02)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: {}", save_path)


def plot_metrics_vs_lambda_fold_averaged(
    per_fold_metrics: dict[int, list[QuantMetrics]],
    save_path: str,
    *,
    show_per_fold: bool = True,
) -> None:
    """Plot fold-averaged metrics vs λ_ctc (mean±std over folds)."""
    if not per_fold_metrics:
        raise ValueError("per_fold_metrics is empty")

    folds = sorted(per_fold_metrics.keys())
    lam_sorted = sorted({m.lambda_ctc for m in per_fold_metrics[folds[0]]})

    def _to_matrix(getter):
        rows = []
        for fold in folds:
            by_lam = {m.lambda_ctc: m for m in per_fold_metrics[fold]}
            rows.append([getter(by_lam[lam]) for lam in lam_sorted])
        return np.asarray(rows, dtype=np.float64)  # (F, L)

    # Build (F,L) matrices for each metric we want
    ent_all = _to_matrix(lambda m: m.mean_entropy)
    ent_nb = _to_matrix(lambda m: m.mean_entropy_nb)
    mx_all = _to_matrix(lambda m: m.mean_max_prob)
    mx_nb = _to_matrix(lambda m: m.mean_max_prob_nb)
    blk = _to_matrix(lambda m: m.blank_occupancy)
    nbr = _to_matrix(lambda m: m.nonblank_ratio)
    cer_micro = _to_matrix(lambda m: m.cer_greedy_micro)
    cer_mean = _to_matrix(lambda m: m.cer_greedy_mean)

    # Fold mean±std (across folds)
    def _mu_sd(M):
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0)

    ent_all_mu, ent_all_sd = _mu_sd(ent_all)
    ent_nb_mu, ent_nb_sd = _mu_sd(ent_nb)
    mx_all_mu, mx_all_sd = _mu_sd(mx_all)
    mx_nb_mu, mx_nb_sd = _mu_sd(mx_nb)
    blk_mu, blk_sd = _mu_sd(blk)
    nbr_mu, nbr_sd = _mu_sd(nbr)
    cer_micro_mu, cer_micro_sd = _mu_sd(cer_micro)
    cer_mean_mu, cer_mean_sd = _mu_sd(cer_mean)

    fig, axes = plt.subplots(
        2, 4, figsize=(22, 9), dpi=150, constrained_layout=True
    )
    axes = axes.ravel()

    # Uniform-distribution baselines
    V = len(CATEGORIES_WORD)
    uniform_entropy = float(np.log(V))
    uniform_max_prob = 1.0 / V

    # Helper: optionally draw faint per-fold curves
    def _maybe_draw_per_fold(ax, M):
        if not show_per_fold:
            return
        for i, fold in enumerate(folds):
            ax.plot(lam_sorted, M[i], alpha=0.25, linewidth=1)

    # 1) Entropy (ALL)
    _maybe_draw_per_fold(axes[0], ent_all)
    axes[0].errorbar(lam_sorted, ent_all_mu, yerr=ent_all_sd, marker="o", capsize=4, linewidth=2)
    axes[0].set_xlabel("λ_ctc")
    axes[0].set_ylabel("Mean per-frame entropy (nats)")
    axes[0].set_title("Entropy (all frames, fold-avg) ↓")
    axes[0].axhline(uniform_entropy, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_entropy:.2f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 2) Entropy (NON-BLANK)
    _maybe_draw_per_fold(axes[1], ent_nb)
    axes[1].errorbar(lam_sorted, ent_nb_mu, yerr=ent_nb_sd, marker="o", capsize=4, linewidth=2)
    axes[1].set_xlabel("λ_ctc")
    axes[1].set_ylabel("Mean per-frame entropy (nats)")
    axes[1].set_title("Entropy (non-blank, fold-avg) ↓")
    axes[1].axhline(uniform_entropy, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_entropy:.2f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3) Max prob (ALL)
    _maybe_draw_per_fold(axes[2], mx_all)
    axes[2].errorbar(lam_sorted, mx_all_mu, yerr=mx_all_sd, marker="s", capsize=4, linewidth=2)
    axes[2].set_xlabel("λ_ctc")
    axes[2].set_ylabel("Mean max P(c|t)")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Peakiness (all frames, fold-avg) ↑")
    axes[2].axhline(uniform_max_prob, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_max_prob:.3f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # 4) Max prob (NON-BLANK)
    _maybe_draw_per_fold(axes[3], mx_nb)
    axes[3].errorbar(lam_sorted, mx_nb_mu, yerr=mx_nb_sd, marker="s", capsize=4, linewidth=2)
    axes[3].set_xlabel("λ_ctc")
    axes[3].set_ylabel("Mean max P(c|t)")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("Peakiness (non-blank, fold-avg) ↑")
    axes[3].axhline(uniform_max_prob, ls="--", color="gray", alpha=0.6, label=f"uniform={uniform_max_prob:.3f}")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # 5) Blank occupancy
    _maybe_draw_per_fold(axes[4], blk)
    axes[4].errorbar(lam_sorted, blk_mu, yerr=blk_sd, marker="^", capsize=4, linewidth=2)
    axes[4].set_xlabel("λ_ctc")
    axes[4].set_ylabel("Argmax blank ratio")
    axes[4].set_ylim(0, 1.0)
    axes[4].set_title("Blank occupancy (fold-avg)")
    axes[4].grid(True, alpha=0.3, axis="y")

    # 6) Non-blank ratio
    _maybe_draw_per_fold(axes[5], nbr)
    axes[5].errorbar(lam_sorted, nbr_mu, yerr=nbr_sd, marker="^", capsize=4, linewidth=2)
    axes[5].set_xlabel("λ_ctc")
    axes[5].set_ylabel("Non-blank frame ratio")
    axes[5].set_ylim(0, 1.0)
    axes[5].set_title("Non-blank ratio (fold-avg)")
    axes[5].grid(True, alpha=0.3)

    # 7) CER micro
    _maybe_draw_per_fold(axes[6], cer_micro)
    axes[6].errorbar(lam_sorted, cer_micro_mu, yerr=cer_micro_sd, marker="D", capsize=4, linewidth=2)
    axes[6].set_xlabel("λ_ctc")
    axes[6].set_ylabel("CER (micro)")
    axes[6].set_title("CTC greedy CER (micro, fold-avg) ↓")
    axes[6].grid(True, alpha=0.3)

    # 8) CER mean (over samples)
    _maybe_draw_per_fold(axes[7], cer_mean)
    axes[7].errorbar(lam_sorted, cer_mean_mu, yerr=cer_mean_sd, marker="D", capsize=4, linewidth=2)
    axes[7].set_xlabel("λ_ctc")
    axes[7].set_ylabel("CER (mean over samples)")
    axes[7].set_title("CTC greedy CER (mean, fold-avg) ↓")
    axes[7].grid(True, alpha=0.3)

    fig.suptitle(f"CTC posterior diagnostics vs λ_ctc (avg over {len(folds)} folds)", fontsize=14, y=1.02)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: {}", save_path)


def plot_entropy_histogram_overlay(
    all_posteriors: dict[float, list[PosteriorSample]],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, constrained_layout=True)

    for lam in sorted(all_posteriors.keys()):
        entropies = []
        for s in all_posteriors[lam]:
            P = np.clip(s.posteriors, 1e-12, 1.0)
            H = -np.sum(P * np.log(P), axis=1)
            entropies.extend(H.tolist())
        entropies = np.asarray(entropies, dtype=np.float64)
        if entropies.size == 0:
            continue
        ax.hist(entropies, bins=80, alpha=0.35, density=True, label=f"λ={lam} (μ={entropies.mean():.2f})")

    ax.set_xlabel("Per-frame entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy distribution across λ_ctc")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: {}", save_path)


def plot_peakiness_over_time(
    all_posteriors: dict[float, list[PosteriorSample]],
    save_path: str,
    n_samples: int,
) -> None:
    lambdas = sorted(all_posteriors.keys())
    first = all_posteriors[lambdas[0]]
    n_show = min(n_samples, len(first))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), dpi=150, constrained_layout=True)
    if n_show == 1:
        axes = [axes]

    for row in range(n_show):
        ax = axes[row]
        label = first[row].label
        sample_id = first[row].sample_id

        for lam in lambdas:
            if row >= len(all_posteriors[lam]):
                continue
            P = all_posteriors[lam][row].posteriors
            max_p = np.max(P, axis=1)
            ax.plot(max_p, label=f"λ={lam}", alpha=0.85, linewidth=1.2)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("max P(c|t)")
        ax.set_title(f'"{label}" (sample {sample_id})', fontsize=10)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, ncol=max(1, len(lambdas)), loc="upper right")

    axes[-1].set_xlabel("Encoder timestep")

    fig.suptitle("Per-timestep peakiness across λ_ctc", fontsize=13, y=1.02)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: {}", save_path)


def plot_posterior_grid(
    all_posteriors: dict[float, list[PosteriorSample]],
    categories: list[str],
    save_dir: str,
    n_samples: int,
    difficulty_meta: dict[str, object] | None = None,
) -> None:
    """One figure per sample; columns = lambdas."""
    os.makedirs(save_dir, exist_ok=True)

    lambdas = sorted(all_posteriors.keys())
    n_lam = len(lambdas)
    first = all_posteriors[lambdas[0]]
    n_show = min(n_samples, len(first))

    cats_display = [c if c != "" else "ε" for c in categories]

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

    for s_idx in range(n_show):
        sample_id = first[s_idx].sample_id
        label = first[s_idx].label

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

        fig, axes = plt.subplots(
            1,
            n_lam,
            figsize=(max(4 * n_lam, 12), 6),
            dpi=150,
            constrained_layout=True,
            sharey=True,
        )
        if n_lam == 1:
            axes = [axes]

        im = None
        for col, lam in enumerate(lambdas):
            ax = axes[col]
            sample = all_posteriors[lam][s_idx]
            probs = sample.posteriors
            im = ax.imshow(probs.T, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"λ_ctc={lam}", fontsize=11, pad=6)
            ax.set_xlabel("Timestep", fontsize=9)

            if col == 0:
                ax.set_ylabel("Char class", fontsize=9)
                if len(categories) <= 70:
                    ax.set_yticks(range(len(categories)))
                    ax.set_yticklabels(cats_display, fontsize=4)
            else:
                ax.set_yticks([])

        fig.suptitle(f'CTC posteriors — GT: "{label}" (sample {sample_id}){title_suffix}', fontsize=13, y=1.02)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.75, label="P(char|t)", pad=0.01)

        out_path = os.path.join(save_dir, f"ctc_grid_sample{sample_id:04d}_{label[:10]}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info("Saved: {}", out_path)


def plot_posterior_grid_compact(
    all_posteriors: dict[float, list[PosteriorSample]],
    categories: list[str],
    save_path: str,
    n_samples: int,
    difficulty_meta: dict[str, object] | None = None,
) -> None:
    """Single compact figure: rows=samples, cols=lambdas."""
    lambdas = sorted(all_posteriors.keys())
    first = all_posteriors[lambdas[0]]
    n_show = min(n_samples, len(first))

    cats_display = [c if c != "" else "ε" for c in categories]

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

    fig, axes = plt.subplots(
        n_show,
        len(lambdas),
        figsize=(max(4 * len(lambdas), 12), max(3 * n_show, 6)),
        dpi=150,
        constrained_layout=True,
        sharey=True,
    )

    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(lambdas) == 1:
        axes = np.expand_dims(axes, axis=1)

    im = None
    for r in range(n_show):
        sample_id = first[r].sample_id
        label = first[r].label

        # Get difficulty info if available
        diff_info = diff_map.get(sample_id, {})
        diff_tag = diff_info.get("tag", "")
        norm_ld = diff_info.get("norm_ld", None)
        ylabel = f'"{label}"'
        if diff_tag:
            ylabel += f" [{diff_tag}"
            if norm_ld is not None:
                ylabel += f", {norm_ld:.3f}"
            ylabel += "]"

        for c, lam in enumerate(lambdas):
            ax = axes[r, c]
            sample = all_posteriors[lam][r]
            probs = sample.posteriors
            im = ax.imshow(probs.T, aspect="auto", origin="lower", cmap="hot", vmin=0, vmax=1)

            if r == 0:
                ax.set_title(f"λ_ctc={lam}", fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8, rotation=0, labelpad=60, ha="right", va="center")
                if len(categories) <= 70:
                    ax.set_yticks(range(len(categories)))
                    ax.set_yticklabels(cats_display, fontsize=3)
            else:
                ax.set_yticks([])

            if r == n_show - 1:
                ax.set_xlabel("Timestep", fontsize=8)

    fig.suptitle("CTC posterior heatmaps across λ_ctc", fontsize=14, y=1.02)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="P(char|t)", pad=0.01)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved: {}", save_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTC posterior analysis across λ_ctc sweep")
    p.add_argument("--lambdas", type=float, nargs="+", required=True)
    p.add_argument("--ckpts", type=str, nargs="+", required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--fold", type=int, default=0, help="Single fold index (ignored if --folds is set)")
    p.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated folds for fold-averaged metrics (e.g. '0,1,2,3,4'). If set, --ckpts should include '{fold}'.",
    )
    p.add_argument(
        "--qual_fold",
        type=int,
        default=None,
        help="Which fold to use for qualitative heatmaps when --folds is set (default: first fold).",
    )
    p.add_argument("--outdir", type=str, default="figures/ctc_posterior_lambda")
    p.add_argument("--n_qual_samples", type=int, default=6)
    p.add_argument("--n_quant_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Difficulty-based qualitative selection (thesis-quality figures)
    p.add_argument(
        "--difficulty_export",
        type=str,
        default=None,
        help=(
            "Optional export JSON (predictions+labels) used to pick qualitative samples by normalized Levenshtein. "
            "May include '{fold}' placeholder in fold-averaging mode. If set, overrides random qualitative indices."
        ),
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
        help="Number of qualitative examples to pick when --difficulty_export is set (default: 6).",
    )
    p.add_argument(
        "--difficulty_no_easy",
        action="store_true",
        help="Disable the explicit easy example (normalized LD == 0) when selecting difficulty samples.",
    )

    # Model config (must match training)
    p.add_argument("--arch_en", type=str, default="blconv_b")
    p.add_argument("--arch_de", type=str, default="ar_transformer_s")
    p.add_argument("--num_channel", type=int, default=13)
    p.add_argument("--use_gated_attention", action="store_true", default=True)
    p.add_argument("--gating_type", type=str, default="elementwise")

    p.add_argument("--skip_entropy_hist", action="store_true")
    p.add_argument("--skip_peakiness_time", action="store_true")
    p.add_argument(
        "--no_per_fold_curves",
        action="store_true",
        help="In fold-averaged plots, do not draw faint per-fold curves.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.lambdas) != len(args.ckpts):
        raise ValueError("--lambdas and --ckpts must have the same length")

    # Resolve paths
    args.dataset = _resolve_path(args.dataset)
    if args.dataset is None:
        raise ValueError("--dataset is required")
    _require_exists(args.dataset, name="--dataset")

    # NOTE: checkpoint existence validation happens later.
    # In fold-averaging mode we expand templates like .../fold_{fold}/{fold}/... per fold.
    args.ckpts = [_resolve_path(c) for c in args.ckpts]
    for c in args.ckpts:
        if c is None:
            raise ValueError("Invalid checkpoint path")

    os.makedirs(args.outdir, exist_ok=True)

    # CPU fallback
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        args.device = "cpu"

    categories = CATEGORIES_WORD

    # Parse folds for fold-averaging.
    if args.folds:
        folds = [int(x) for x in str(args.folds).split(",") if str(x).strip() != ""]
        folds = sorted(set(folds))
        if not folds:
            raise ValueError("--folds provided but empty")
    else:
        folds = [int(args.fold)]

    qual_fold = int(args.qual_fold) if args.qual_fold is not None else folds[0]

    from rewi.dataset import HRDataset

    def _build_dataset(fold: int):
        return HRDataset(
            os.path.join(args.dataset, "val.json"),
            categories,
            8,  # ratio_ds for blconv_b
            fold,
            0,  # len_seq
            cache=True,
        )

    def _expand_ckpt(ckpt: str, fold: int) -> str:
        # Allow templates like .../fold_{fold}/{fold}/...
        if "{fold}" in ckpt:
            return ckpt.format(fold=fold)
        return ckpt

    def _safe_ids_for_dataset(base_ids: list[int], dataset_len: int) -> list[int]:
        # Clamp to avoid out-of-range if some fold has fewer samples.
        return [i for i in base_ids if 0 <= i < dataset_len]

    # Use a dedicated fold for qualitative sample identity.
    dataset_qual = _build_dataset(qual_fold)
    logger.info("Val dataset size (qual_fold={}): {}", qual_fold, len(dataset_qual))

    # Choose qualitative indices.
    qual_meta = None
    if args.difficulty_export:
        exp_path = _resolve_path(_expand_ckpt(str(args.difficulty_export), qual_fold))
        if exp_path is None:
            raise ValueError("Invalid --difficulty_export path")
        _require_exists(exp_path, name="--difficulty_export")

        quantiles = _parse_quantiles(args.difficulty_quantiles, default=(0.50, 0.90, 0.99))
        qual_ids_base, qual_meta = _choose_difficulty_sample_ids_from_export(
            exp_path,
            n=int(args.difficulty_n),
            quantiles=quantiles,
            include_easy=(not bool(args.difficulty_no_easy)),
        )

        # Sanity-check mapping: export order should match HRDataset order.
        try:
            mism = []
            with open(exp_path, "r") as f:
                exp = json.load(f)
            exp_labels = exp.get("labels", [])
            if isinstance(exp_labels, list) and len(exp_labels) == len(dataset_qual):
                for i in qual_ids_base:
                    ds_lab = str(dataset_qual.annos[int(i)]["label"])  # type: ignore[attr-defined]
                    ex_lab = str(exp_labels[int(i)])
                    if ds_lab != ex_lab:
                        mism.append((int(i), ex_lab, ds_lab))
                if mism:
                    logger.warning(
                        "Difficulty export label mismatch for {} / {} picked samples. "
                        "This suggests export ordering may not match dataset ordering; qualitative sample mapping may be wrong.",
                        len(mism),
                        len(qual_ids_base),
                    )
        except Exception:
            pass

        args.n_qual_samples = len(qual_ids_base)
        logger.info("Using difficulty-picked qualitative indices: {}", qual_ids_base)
    else:
        qual_ids_base = _select_sample_ids(len(dataset_qual), args.n_qual_samples, seed=args.seed)

    # For fold-averaging, we only need determinism per fold (not same physical sample across folds).
    # But we do want IDs to always be valid; use dataset_qual length to pick IDs and clamp per fold if needed.
    quant_ids_base = _select_sample_ids(len(dataset_qual), args.n_quant_samples, seed=args.seed)
    quant_ids_base = sorted(set(quant_ids_base) | set(qual_ids_base))

    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from rewi.dataset.utils import fn_collate
    from rewi.model import DualHeadModel
    from rewi.analysis.encoder_features import load_model_from_checkpoint

    @torch.no_grad()
    def extract_ctc_posteriors_for_ids(
        model: nn.Module,
        dataset,
        sample_ids: list[int],
    ) -> list[PosteriorSample]:
        subset = Subset(dataset, sample_ids)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=fn_collate,
        )

        model.eval()
        model.to(args.device)

        out: list[PosteriorSample] = []
        cursor = 0

        for x, y, len_x, len_y in loader:
            x = x.to(args.device)
            len_x_dev = len_x.to(args.device)

            if not hasattr(model, "compute_ctc_logits"):
                raise ValueError("Model has no compute_ctc_logits (not a DualHeadModel?)")

            feats, enc_pad, enc_lengths = model._encode_with_mask(x, len_x_dev)
            logits = model.compute_ctc_logits(feats)
            probs = torch.softmax(logits, dim=-1)

            B = x.size(0)
            for b in range(B):
                T = int(enc_lengths[b].item())
                probs_b = probs[b, :T].detach().cpu().numpy()

                L = int(len_y[b])
                lab_ids = y[b, :L].tolist() if y.dim() == 2 else []
                label_str = "".join(
                    categories[i] for i in lab_ids if 0 <= i < len(categories) and i != 0
                )

                out.append(
                    PosteriorSample(
                        posteriors=probs_b,
                        label=label_str,
                        enc_length=T,
                        sample_id=sample_ids[cursor],
                    )
                )
                cursor += 1

        return out

    vocab_ar = len(categories) + 3
    vocab_ctc = len(categories)

    # Per-fold run
    per_fold_metrics: dict[int, list[QuantMetrics]] = {}
    qual_posteriors: dict[float, list[PosteriorSample]] | None = None

    for fold in folds:
        dataset_fold = _build_dataset(fold)
        logger.info("Fold {}: {} samples", fold, len(dataset_fold))

        qual_ids = _safe_ids_for_dataset(qual_ids_base, len(dataset_fold))
        quant_ids = _safe_ids_for_dataset(quant_ids_base, len(dataset_fold))
        if not qual_ids:
            raise ValueError(f"No valid qualitative sample ids for fold {fold}")
        if not quant_ids:
            raise ValueError(f"No valid quantitative sample ids for fold {fold}")

        metrics_fold: list[QuantMetrics] = []

        for lam, ckpt in zip(args.lambdas, args.ckpts):
            ckpt_fold = _expand_ckpt(ckpt, fold)
            ckpt_fold = _resolve_path(ckpt_fold)
            if ckpt_fold is None:
                raise ValueError("Invalid checkpoint path")
            _require_exists(ckpt_fold, name=f"checkpoint (fold {fold})")

            logger.info("=== fold={} | λ_ctc={} ===", fold, lam)

            model = DualHeadModel(
                args.arch_en,
                args.arch_de,
                "linear",
                args.num_channel,
                vocab_ar,
                vocab_ctc,
                0,
                use_gated_attention=args.use_gated_attention,
                gating_type=args.gating_type,
            )
            model = load_model_from_checkpoint(ckpt_fold, model, args.device)

            # Qualitative only for one fold
            if fold == qual_fold:
                qual_posts = extract_ctc_posteriors_for_ids(model, dataset_fold, qual_ids)
                if qual_posteriors is None:
                    qual_posteriors = {}
                qual_posteriors[float(lam)] = qual_posts

            quant_posts = extract_ctc_posteriors_for_ids(model, dataset_fold, quant_ids)
            qm = compute_quant_metrics(quant_posts,
                lambda_ctc=float(lam),
                categories=categories,
                blank_idx=0,
            )

            metrics_fold.append(qm)

            logger.info(
                "ALL: H={:.3f}±{:.3f}  maxP={:.3f}±{:.3f} | "
                "NB: H={:.3f}±{:.3f}  maxP={:.3f}±{:.3f} (nb_ratio={:.2f}) | "
                "blank={:.3f} | CER_greedy={:.3f}±{:.3f} (micro={:.3f})",
                qm.mean_entropy, qm.std_entropy, qm.mean_max_prob, qm.std_max_prob,
                qm.mean_entropy_nb, qm.std_entropy_nb, qm.mean_max_prob_nb, qm.std_max_prob_nb, qm.nonblank_ratio,
                qm.blank_occupancy,
                qm.cer_greedy_mean, qm.cer_greedy_std, qm.cer_greedy_micro,
            )


            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_fold_metrics[fold] = metrics_fold

    # Qualitative plots (single fold)
    if qual_posteriors is not None:
        plot_posterior_grid(
            qual_posteriors,
            categories,
            os.path.join(args.outdir, f"heatmap_grids_fold{qual_fold}"),
            args.n_qual_samples,
            difficulty_meta=qual_meta,
        )
        plot_posterior_grid_compact(
            qual_posteriors,
            categories,
            os.path.join(args.outdir, f"ctc_heatmap_grid_compact_fold{qual_fold}.pdf"),
            n_samples=min(args.n_qual_samples, 4),
            difficulty_meta=qual_meta,
        )
        if not args.skip_entropy_hist:
            plot_entropy_histogram_overlay(
                qual_posteriors,
                os.path.join(args.outdir, f"entropy_histogram_by_lambda_fold{qual_fold}.pdf"),
            )
        if not args.skip_peakiness_time:
            plot_peakiness_over_time(
                qual_posteriors,
                os.path.join(args.outdir, f"peakiness_over_time_fold{qual_fold}.pdf"),
                n_samples=min(args.n_qual_samples, 4),
            )

    # Quantitative plots + JSON
    if len(folds) == 1:
        plot_metrics_vs_lambda(per_fold_metrics[folds[0]], os.path.join(args.outdir, "metrics_vs_lambda.pdf"))
        out_json = {str(m.lambda_ctc): asdict(m) for m in sorted(per_fold_metrics[folds[0]], key=lambda m: m.lambda_ctc)}
        json_path = os.path.join(args.outdir, "ctc_posterior_metrics.json")
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        logger.info("Saved: {}", json_path)
    else:
        plot_metrics_vs_lambda_fold_averaged(
            per_fold_metrics,
            os.path.join(args.outdir, "metrics_vs_lambda_foldavg.pdf"),
            show_per_fold=not bool(args.no_per_fold_curves),
        )
        agg_json: dict[str, object] = {
            "folds": folds,
            "lambdas": sorted([float(x) for x in args.lambdas]),
            "per_fold": {
                str(fold): {str(m.lambda_ctc): asdict(m) for m in per_fold_metrics[fold]}
                for fold in folds
            },
        }
        json_path = os.path.join(args.outdir, "ctc_posterior_metrics_foldavg.json")
        with open(json_path, "w") as f:
            json.dump(agg_json, f, indent=2)
        logger.info("Saved: {}", json_path)

    logger.info("Done. Outputs in: {}", args.outdir)


if __name__ == "__main__":
    main()
