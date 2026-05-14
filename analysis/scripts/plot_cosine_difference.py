"""Cosine-similarity *difference* visualization for AR-only vs Hybrid encoders.

The standard side-by-side cosine plot (in compare_ar_hybrid.py::plot_cosine_similarity_comparison)
asks the reader to mentally subtract two heatmaps. This script does the
subtraction directly: it produces a single divergent-colormap heatmap of
S_hybrid - S_AR per sample, optionally with the absolute matrices shown
above for reference.

Positive cells (red) = hybrid is more similar than AR-only (typically the
diagonal band, indicating temporally smoother encoding).
Negative cells (blue) = AR-only is more similar (typically off-diagonal
blocks where AR-only has spurious frame coupling).

Usage (from within the analysis pipeline, after extract_encoder_features):

    from rewi.analysis.encoder_features import extract_encoder_features
    from analysis.scripts.plot_cosine_difference import plot_cosine_difference

    feats_ar  = extract_encoder_features(model_ar,  loader, device, categories, max_samples=10**9)
    feats_hyb = extract_encoder_features(model_hyb, loader, device, categories, max_samples=10**9)
    plot_cosine_difference(
        feats_ar=feats_ar,
        feats_hyb=feats_hyb,
        save_dir="figures/cosine_difference",
        sample_ids=[0, 1, 2, 3],
        difficulty_meta=...,  # optional, same dict as in compare_ar_hybrid.py
        also_show_absolutes=True,
    )

CLI use is also supported if features have been pickled to disk:

    python plot_cosine_difference.py \
        --feats-ar /path/feats_ar.npz --feats-hyb /path/feats_hyb.npz \
        --save-dir figures/cosine_difference \
        --sample-ids 0,1,2,3
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def _safe_label(word: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in word)[:24]


def plot_cosine_difference(
    feats_ar: dict,
    feats_hyb: dict,
    save_dir: str,
    sample_ids: Optional[list[int]] = None,
    difficulty_meta: Optional[dict] = None,
    also_show_absolutes: bool = False,
) -> list[str]:
    """Produce per-sample cosine-difference heatmaps (S_hybrid - S_AR).

    Each output PDF contains either one panel (the difference matrix) or
    three panels (AR cosine, hybrid cosine, difference) depending on
    `also_show_absolutes`.

    Returns the list of saved PDF paths.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    os.makedirs(save_dir, exist_ok=True)

    unique_ar = np.unique(feats_ar["sample_idx"])
    unique_hyb = np.unique(feats_hyb["sample_idx"])
    common = np.intersect1d(unique_ar, unique_hyb)

    if sample_ids is None:
        sample_ids = list(common[:4])
    else:
        common_set = set(common.tolist())
        sample_ids = [int(i) for i in sample_ids if int(i) in common_set]
        if not sample_ids:
            raise ValueError("None of the requested sample_ids are common to both feature dicts")

    diff_map: dict[int, dict] = {}
    if difficulty_meta is not None and "chosen" in difficulty_meta:
        for item in difficulty_meta["chosen"]:
            key = item.get("subset_pos") or item.get("sample_index") or item.get("idx")
            if key is None:
                continue
            diff_map[int(key)] = {
                "tag": item.get("tag", ""),
                "norm_ld": item.get("norm_ld", None),
                "label": item.get("label", None),
            }

    saved: list[str] = []

    for sid in sample_ids:
        mask_ar = feats_ar["sample_idx"] == sid
        f_ar = feats_ar["features"][mask_ar]
        chars_ar = [feats_ar["char_labels"][i] for i, m in enumerate(mask_ar) if m]
        word = feats_ar["word_labels"][np.where(mask_ar)[0][0]]

        mask_hyb = feats_hyb["sample_idx"] == sid
        f_hyb = feats_hyb["features"][mask_hyb]

        if f_ar.shape[0] == 0 or f_hyb.shape[0] == 0:
            continue
        if f_ar.shape[0] != f_hyb.shape[0]:
            T = min(f_ar.shape[0], f_hyb.shape[0])
            f_ar = f_ar[:T]
            f_hyb = f_hyb[:T]
            chars_ar = chars_ar[:T]

        sim_ar = cosine_similarity(f_ar)
        sim_hyb = cosine_similarity(f_hyb)
        diff = sim_hyb - sim_ar
        vmax = float(np.max(np.abs(diff)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        info = diff_map.get(int(sid), {})
        suffix_bits = []
        if info.get("tag"):
            suffix_bits.append(str(info["tag"]))
        if info.get("norm_ld") is not None:
            try:
                suffix_bits.append(f"norm_LD={float(info['norm_ld']):.3f}")
            except Exception:
                suffix_bits.append(f"norm_LD={info['norm_ld']}")
        suffix = f" [{', '.join(suffix_bits)}]" if suffix_bits else ""
        title = f'Cosine difference (Hybrid $-$ AR-only): "{word}"{suffix}'

        if also_show_absolutes:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150,
                                     constrained_layout=True)
            for ax, mat, lab, cmap, norm_arg in [
                (axes[0], sim_ar, f'AR-only "{word}"', 'RdBu_r', None),
                (axes[1], sim_hyb, f'Hybrid "{word}"', 'RdBu_r', None),
                (axes[2], diff, 'Hybrid $-$ AR (red: hybrid more similar)',
                 'RdBu_r', norm),
            ]:
                im = ax.imshow(mat, cmap=cmap, norm=norm_arg,
                               vmin=(-1 if norm_arg is None else None),
                               vmax=(1 if norm_arg is None else None),
                               aspect='auto')
                ax.set_title(lab, fontsize=11)
                ax.set_xlabel('Frame index')
                ax.set_ylabel('Frame index')
                fig.colorbar(im, ax=ax, fraction=0.04)
            fig.suptitle(title, fontsize=12)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6.2), dpi=150,
                                   constrained_layout=True)
            im = ax.imshow(diff, cmap='RdBu_r', norm=norm, aspect='auto')
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Frame index')
            ax.set_ylabel('Frame index')
            cbar = fig.colorbar(im, ax=ax, fraction=0.04)
            cbar.set_label(r'$\cos(h^{\mathrm{hyb}}_i, h^{\mathrm{hyb}}_j) - \cos(h^{\mathrm{AR}}_i, h^{\mathrm{AR}}_j)$',
                           fontsize=9)

        path = os.path.join(save_dir,
                            f'cosine_diff_sample{int(sid):04d}_{_safe_label(word)}.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)

    return saved


def _load_features_npz(path: str) -> dict:
    """Load features previously saved by np.savez_compressed.

    Expected keys: 'features', 'sample_idx', 'char_labels', 'word_labels'.
    """
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feats-ar", required=True, type=str,
                   help="Path to .npz with AR-only encoder features")
    p.add_argument("--feats-hyb", required=True, type=str,
                   help="Path to .npz with hybrid encoder features")
    p.add_argument("--save-dir", required=True, type=str)
    p.add_argument("--sample-ids", type=str, default=None,
                   help="Comma-separated sample ids; default: first 4 common")
    p.add_argument("--also-show-absolutes", action="store_true",
                   help="Render 3-panel layout (AR | Hybrid | Diff)")
    args = p.parse_args()

    feats_ar = _load_features_npz(args.feats_ar)
    feats_hyb = _load_features_npz(args.feats_hyb)
    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(s) for s in args.sample_ids.split(",")]
    paths = plot_cosine_difference(
        feats_ar=feats_ar,
        feats_hyb=feats_hyb,
        save_dir=args.save_dir,
        sample_ids=sample_ids,
        also_show_absolutes=args.also_show_absolutes,
    )
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
