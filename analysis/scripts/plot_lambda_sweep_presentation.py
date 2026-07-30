#!/usr/bin/env python3
"""Presentation variant of the lambda_ctc sweep: HWRFormer only (no -L row).

Same canonical sources as plot_lambda_sweep_dual.py, restricted to the
baseline (xs = thesis "HWRFormer") sweep on OnHW-WI with its AR-only
reference and the selected lambda_ctc = 0.1.

Output: presentation/figures/ctc_lambda_sweep_hwrformer.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path("/home/woody/iwso/iwso214h/imu-hwr/results/hwr2")
OUT_PDF = Path("/home/woody/iwso/iwso214h/imu-hwr/presentation/figures/"
               "ctc_lambda_sweep_hwrformer.pdf")

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTED = 0.1
COLOR = "#1f77b4"
SEL_COLOR = "#9467bd"


def _read(base: Path):
    d = json.loads((base / "results.json").read_text())
    n = len(d["cer"]["raw"])
    sem = lambda s: s * 100 / math.sqrt(n) if n > 1 else 0.0
    return (d["cer"]["mean"] * 100, sem(d["cer"]["std"]),
            d["wer"]["mean"] * 100, sem(d["wer"]["std"]))


def main() -> None:
    pts = []
    for lam in LAMBDAS:
        idx = f"{int(round(lam*10)):02d}"
        base = (RESULTS / f"train_element_word_hybrid_{idx}_xs_onhw_wi"
                / "ar_transformer_xs__onhw_wi_word_rh")
        pts.append(_read(base))
    ar = _read(RESULTS / "Baseline-AR-XS-blconv_b"
               / "ar_transformer_xs__onhw_wi_word_rh")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)
    for ax, im, isem, ar_ref, ylab, title in [
        (axes[0], 0, 1, ar[0], "CER (%)", r"Downstream AR CER vs $\lambda_{\mathrm{ctc}}$"),
        (axes[1], 2, 3, ar[2], "WER (%)", r"Downstream AR WER vs $\lambda_{\mathrm{ctc}}$"),
    ]:
        ms = [p[im] for p in pts]
        ss = [p[isem] for p in pts]
        ax.axvspan(SELECTED - 0.02, SELECTED + 0.02, color=SEL_COLOR,
                   alpha=0.2, linewidth=0, zorder=1)
        ax.axvline(SELECTED, color=SEL_COLOR, linestyle="-", linewidth=1.8,
                   alpha=0.95, zorder=2)
        ax.plot(LAMBDAS, ms, color=COLOR, marker="o", lw=2, ms=6, zorder=5)
        ax.fill_between(LAMBDAS, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)],
                        color=COLOR, alpha=0.18, linewidth=0, zorder=3)
        ax.axhline(ar_ref, color="black", linestyle=":", lw=1.4, zorder=4)
        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAMBDAS)

    handles = [
        Line2D([0], [0], color=COLOR, marker="o", markersize=6, lw=2,
               label=r"Hybrid sweep, 5-fold mean $\pm$ SEM"),
        Line2D([0], [0], color="black", linestyle=":", lw=1.4,
               label="HWRFormer AR-only reference"),
        Line2D([0], [0], color=SEL_COLOR, lw=1.8,
               label=r"selected $\lambda_{\mathrm{ctc}}=0.1$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize=10, frameon=False, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
