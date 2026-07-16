#!/usr/bin/env python3
"""Re-render the thesis cosine-similarity grids offline from the cache written
by compare_ar_hybrid.py (no GPU / no checkpoints needed).

The cache (cosine_grid_cache.pkl) holds, per selected sample, the two cosine
matrices (HWRFormer, hybrid HWRFormer), their per-frame character labels, and
the reference word. This lets us iterate on example selection, colour scaling,
and panel size cheaply.

Scale modes:
  shared : one colour scale for the whole figure (values comparable across
           every panel; the scientifically safe default). vmax is pinned to
           the cosine self-similarity ceiling of 1; vmin is the global minimum
           so the range is not wasted on the empty [-1, vmin] band.
  panel  : each panel auto-scales to its own [min, 1] with its own colourbar.
           Vivid, but colours are NOT comparable across panels -- in a model
           comparison this can hide the very off-segment-leakage difference the
           figure is meant to show. Use only for purely illustrative panels.

Run from work/REWI_work, e.g.:
  python analysis/scripts/render_cosine_grid_offline.py \
      --words beim schon erhält --scale shared --panel-in 2.1 \
      --out ../../thesis/figures/baseline_xs/cosine_similarity/cosine_grid_main.pdf
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
DEFAULT_CACHE = (REPO / "thesis" / "figures" / "baseline_xs"
                 / "cosine_similarity" / "cosine_grid_cache.pkl")


def _segments(chars: list[str]) -> list[int]:
    """Boundary frame indices from per-frame character labels."""
    if not chars:
        return [0]
    b = [0]
    prev = chars[0]
    for t in range(1, len(chars)):
        if chars[t] != prev:
            b.append(t)
            prev = chars[t]
    b.append(len(chars))
    return b


def _dd_leak(sim: np.ndarray, chars: list[str]) -> tuple[float, float]:
    """Diagonal-block (within-character) mean and off-block mean similarity."""
    bnds = _segments(chars)
    within, off = [], []
    for i in range(len(bnds) - 1):
        a0, a1 = bnds[i], bnds[i + 1]
        within.append(sim[a0:a1, a0:a1])
    within_mean = float(np.mean([w.mean() for w in within])) if within else 0.0
    mask = np.ones_like(sim, dtype=bool)
    for i in range(len(bnds) - 1):
        a0, a1 = bnds[i], bnds[i + 1]
        mask[a0:a1, a0:a1] = False
    off_mean = float(sim[mask].mean()) if mask.any() else 0.0
    return within_mean, off_mean


def interestingness(item: dict) -> float:
    """Rank samples by how much the two models differ in block structure --
    the most illustrative examples for a HWRFormer-vs-hybrid comparison."""
    dd_a, off_a = _dd_leak(item["sim_ar"], item["chars_ar"])
    dd_h, off_h = _dd_leak(item["sim_hyb"], item["chars_hyb"])
    return abs(dd_a - dd_h) + abs(off_a - off_h)


def render(items: list[dict], out_path: str, scale: str = "shared",
           panel_in: float = 2.1,
           col_labels: tuple[str, str] = ("HWRFormer", "hybrid HWRFormer")) -> None:
    n = len(items)
    if n == 0:
        raise SystemExit("no items to render")
    # Width: 2 panels + colourbar gutter; height: n panels + header strip.
    fig_w = 2 * panel_in + (1.15 if scale != "panel" else 1.6)
    fig_h = n * panel_in + 0.55
    fig, axes = plt.subplots(n, 2, figsize=(fig_w, fig_h),
                             constrained_layout=True, squeeze=False)

    # Symmetric diverging bound centred at 0 so cosine 0 (orthogonal frames)
    # maps to white; blue and red then honestly encode sign. Bound = 95th
    # percentile of |off-diagonal| (the always-1 diagonal saturates red).
    absoff = []
    for it in items:
        for m in (it["sim_ar"], it["sim_hyb"]):
            off = m[~np.eye(m.shape[0], dtype=bool)]
            absoff.append(np.abs(off).ravel())
    vabs = float(np.percentile(np.concatenate(absoff), 95))
    vabs = float(min(1.0, max(0.4, np.ceil(vabs * 10) / 10.0)))

    shared_im = None
    for r, it in enumerate(items):
        row_min = min(float(np.min(it["sim_ar"])), float(np.min(it["sim_hyb"])))
        row_min = np.floor(row_min * 10) / 10.0
        for c, (sim, chars) in enumerate(
                [(it["sim_ar"], it["chars_ar"]), (it["sim_hyb"], it["chars_hyb"])]):
            ax = axes[r][c]
            if scale == "shared":
                vmin, vmax = -vabs, vabs
            elif scale == "row":
                vmin, vmax = row_min, 1.0
            else:  # panel
                vmin, vmax = np.floor(float(np.min(sim)) * 10) / 10.0, 1.0
            im = ax.imshow(sim, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                           aspect="equal")
            shared_im = im
            for b in _segments(chars)[1:-1]:
                ax.axhline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)
                ax.axvline(b - 0.5, color="lime", linewidth=0.5, alpha=0.7)
            ax.tick_params(axis="both", labelsize=8)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=12, pad=6)
            if r == n - 1:
                ax.set_xlabel("Frame index", fontsize=10)
            if c == 0:
                ax.set_ylabel(f'"{it["word"]}"\nFrame index', fontsize=10)
            if scale == "panel":
                cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
                cb.ax.tick_params(labelsize=7)
        if scale == "row":
            cb = fig.colorbar(im, ax=list(axes[r]), shrink=0.9, pad=0.02)
            cb.set_label("Cosine similarity", fontsize=9)
            cb.ax.tick_params(labelsize=8)

    if scale == "shared" and shared_im is not None:
        cb = fig.colorbar(shared_im, ax=axes, shrink=0.85,
                          label="Cosine similarity", pad=0.02)
        cb.set_label("Cosine similarity", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}  ({fig_w:.2f} x {fig_h:.2f} in, scale={scale})")


def load_cache(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def select(cache: list[dict], words: list[str] | None, top: int | None) -> list[dict]:
    if words:
        chosen = []
        for w in words:
            hit = next((it for it in cache if it["word"].lower() == w.lower()), None)
            if hit is None:
                raise SystemExit(f"word not in cache: {w} "
                                 f"(have: {[it['word'] for it in cache]})")
            chosen.append(hit)
        return chosen
    ranked = sorted(cache, key=interestingness, reverse=True)
    return ranked[: (top or 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--words", nargs="*", default=None,
                    help="Reference words to include as rows (in order).")
    ap.add_argument("--top", type=int, default=3,
                    help="If --words is omitted, take the N most interesting.")
    ap.add_argument("--scale", choices=["shared", "row", "panel"], default="shared")
    ap.add_argument("--panel-in", type=float, default=2.1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--rank", action="store_true",
                    help="Print the interestingness ranking and exit.")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    if args.rank:
        for it in sorted(cache, key=interestingness, reverse=True):
            dd_a, off_a = _dd_leak(it["sim_ar"], it["chars_ar"])
            dd_h, off_h = _dd_leak(it["sim_hyb"], it["chars_hyb"])
            print(f"{it['word']:>12}  score={interestingness(it):.3f}  "
                  f"DD {dd_a:.2f}->{dd_h:.2f}  off {off_a:.2f}->{off_h:.2f}")
        return

    items = select(cache, args.words, args.top)
    render(items, args.out, scale=args.scale, panel_in=args.panel_in)


if __name__ == "__main__":
    main()
