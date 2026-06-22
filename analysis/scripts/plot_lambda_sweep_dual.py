#!/usr/bin/env python3
"""Generate thesis/figures/ctc_lambda_sweep_cer_wer.pdf.

Lambda_ctc sweep on OnHW WI word, dual-variant (HWRFormer baseline xs +
scaled s) full 10-point curves visualisation.

Data sources (canonical, read from disk via results.json — WER reported
at best-CER epoch per evaluate.py:188):
  - Baseline xs sweep:
    results/hwr2/train_element_word_hybrid_{01..10}_xs_onhw_wi/
    ar_transformer_xs__onhw_wi_word_rh/results.json
  - Scaled s sweep:
    results/hwr2/Baseline-Hybrid/train_element_word_hybrid_{01..10}/
    ar_transformer_s__onhw_wi_word_rh/results.json
  - AR-only reference:
    results/hwr2/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/results.json
    results/hwr2/Baseline-AR-ElementwiseGating/ar_transformer_s__onhw_wi_word_rh/results.json
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
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/ctc_lambda_sweep_cer_wer.pdf"
)

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTED_XS = 0.1
SELECTED_S = 0.6

XS_COLOR = "#1f77b4"
S_COLOR = "#ff7f0e"
AR_COLOR = "0.45"


def _sem(std_pct: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return std_pct / math.sqrt(n)


def _read_results(base: Path):
    """Read results.json and return (cer_mean_pct, cer_sem_pct, wer_mean_pct, wer_sem_pct, n)."""
    p = base / "results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    n = len(d["cer"]["raw"])
    cer_mean = d["cer"]["mean"] * 100
    cer_std = d["cer"]["std"] * 100
    wer_mean = d["wer"]["mean"] * 100
    wer_std = d["wer"]["std"] * 100
    return cer_mean, _sem(cer_std, n), wer_mean, _sem(wer_std, n), n


def _load_sweep(arch: str, group_pat: str):
    """Return four parallel lists (cer_mean, cer_sem, wer_mean, wer_sem) indexed by LAMBDAS."""
    cer_means, cer_sems, wer_means, wer_sems = [], [], [], []
    for lam in LAMBDAS:
        idx = f"{int(round(lam*10)):02d}"
        base = RESULTS / group_pat.format(idx=idx) / f"{arch}__onhw_wi_word_rh"
        r = _read_results(base)
        if r is None:
            cer_means.append(None); cer_sems.append(None)
            wer_means.append(None); wer_sems.append(None)
        else:
            cer_means.append(r[0]); cer_sems.append(r[1])
            wer_means.append(r[2]); wer_sems.append(r[3])
    return cer_means, cer_sems, wer_means, wer_sems


def main() -> None:
    xs_cer_m, xs_cer_s, xs_wer_m, xs_wer_s = _load_sweep(
        "ar_transformer_xs",
        "train_element_word_hybrid_{idx}_xs_onhw_wi",
    )
    s_cer_m, s_cer_s, s_wer_m, s_wer_s = _load_sweep(
        "ar_transformer_s",
        "Baseline-Hybrid/train_element_word_hybrid_{idx}",
    )

    ar_xs = _read_results(
        RESULTS / "Baseline-AR-XS-blconv_b" / "ar_transformer_xs__onhw_wi_word_rh"
    )
    ar_s = _read_results(
        RESULTS / "Baseline-AR-ElementwiseGating" / "ar_transformer_s__onhw_wi_word_rh"
    )
    ar_xs_cer = ar_xs[0] if ar_xs else 6.94
    ar_xs_wer = ar_xs[2] if ar_xs else 10.51
    ar_s_cer = ar_s[0] if ar_s else 6.53
    ar_s_wer = ar_s[2] if ar_s else 9.97

    fig, (ax_cer, ax_wer) = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)

    def _draw(ax, xs_means, xs_sems, s_means, s_sems,
              ar_xs_value, ar_s_value, y_label):
        # baseline curve
        xs_pts = [(lam, m, sm) for lam, m, sm in zip(LAMBDAS, xs_means, xs_sems) if m is not None]
        if xs_pts:
            lams = [p[0] for p in xs_pts]
            ms = [p[1] for p in xs_pts]
            ss = [p[2] or 0.0 for p in xs_pts]
            ax.plot(lams, ms, color=XS_COLOR, marker="o", linewidth=2.0,
                    markersize=6, linestyle="-", zorder=5)
            ax.fill_between(lams,
                            [m - sm for m, sm in zip(ms, ss)],
                            [m + sm for m, sm in zip(ms, ss)],
                            color=XS_COLOR, alpha=0.18, linewidth=0, zorder=3)
            if SELECTED_XS in lams:
                idx = lams.index(SELECTED_XS)
                ax.plot([SELECTED_XS], [ms[idx]], marker="o", markersize=11,
                        color=XS_COLOR, markerfacecolor=XS_COLOR,
                        markeredgecolor="black", markeredgewidth=1.2,
                        linestyle="None", zorder=8)

        # scaled curve
        s_pts = [(lam, m, sm) for lam, m, sm in zip(LAMBDAS, s_means, s_sems) if m is not None]
        if s_pts:
            lams = [p[0] for p in s_pts]
            ms = [p[1] for p in s_pts]
            ss = [p[2] or 0.0 for p in s_pts]
            ax.plot(lams, ms, color=S_COLOR, marker="s", linewidth=2.0,
                    markersize=6, linestyle="-", zorder=5)
            ax.fill_between(lams,
                            [m - sm for m, sm in zip(ms, ss)],
                            [m + sm for m, sm in zip(ms, ss)],
                            color=S_COLOR, alpha=0.18, linewidth=0, zorder=3)
            if SELECTED_S in lams:
                idx = lams.index(SELECTED_S)
                ax.plot([SELECTED_S], [ms[idx]], marker="*", markersize=18,
                        color=S_COLOR, markerfacecolor=S_COLOR,
                        markeredgecolor="black", markeredgewidth=1.2,
                        linestyle="None", zorder=8)

        # AR-only references (per configuration)
        ax.axhline(ar_xs_value, color=XS_COLOR, linestyle=":", linewidth=1.2,
                   alpha=0.65, zorder=2)
        ax.axhline(ar_s_value, color=S_COLOR, linestyle=":", linewidth=1.2,
                   alpha=0.65, zorder=2)

        # vertical selection markers
        ax.axvline(SELECTED_XS, color=XS_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.55, zorder=2)
        ax.axvline(SELECTED_S, color=S_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.55, zorder=2)

        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAMBDAS)

        all_y = []
        for pts, sems in [(xs_pts, xs_sems), (s_pts, s_sems)]:
            for _, m, sm in pts:
                sm = sm or 0.0
                all_y.extend([m - sm, m + sm])
        all_y.extend([ar_xs_value, ar_s_value])
        if all_y:
            lo, hi = min(all_y), max(all_y)
            pad = max(0.3, (hi - lo) * 0.15)
            ax.set_ylim(lo - pad, hi + pad)

    _draw(ax_cer, xs_cer_m, xs_cer_s, s_cer_m, s_cer_s,
          ar_xs_cer, ar_s_cer, "CER (%)")
    _draw(ax_wer, xs_wer_m, xs_wer_s, s_wer_m, s_wer_s,
          ar_xs_wer, ar_s_wer, "WER (%)")

    ax_cer.set_title(r"Downstream AR CER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)
    ax_wer.set_title(r"Downstream AR WER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=XS_COLOR, marker="o", markersize=7, linewidth=2,
               label=r"HWRFormer hybrid sweep, 5-fold mean $\pm$ SEM"),
        Line2D([0], [0], color=S_COLOR, marker="s", markersize=7, linewidth=2,
               label=r"HWRFormer-L hybrid sweep, 5-fold mean $\pm$ SEM"),
        Line2D([0], [0], color=XS_COLOR, linestyle=":", linewidth=1.2,
               label=r"HWRFormer AR-only reference (elementwise-gated)"),
        Line2D([0], [0], color=S_COLOR, linestyle=":", linewidth=1.2,
               label=r"HWRFormer-L AR-only reference (elementwise-gated)"),
        Line2D([0], [0], color=XS_COLOR, linestyle="--", linewidth=1.0,
               label=r"HWRFormer selected $\lambda_{\mathrm{ctc}}{=}0.1$"),
        Line2D([0], [0], color=S_COLOR, linestyle="--", linewidth=1.0,
               label=r"HWRFormer-L selected $\lambda_{\mathrm{ctc}}{=}0.6$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        fontsize=9,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
