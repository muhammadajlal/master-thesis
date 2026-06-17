#!/usr/bin/env python3
"""Generate thesis/figures/ctc_lambda_sweep_cer_wer.pdf.

Lambda_ctc sweep on OnHW WI word, dual-variant (HWRFormer baseline xs +
scaled s anchor) visualisation.

Layout: 1x2 panel grid (left CER, right WER) on OnHW WI word.
  - Blue solid line + circle markers + SEM band: HWRFormer (baseline)
    full 10-point sweep over lambda in {0.1, ..., 1.0} from
    xs_numbers.json key figure_lambda_sweep.
  - Orange star with SEM error bar at lambda = 0.6: HWRFormer (scaled)
    single anchor point from s_numbers.json key figure_lambda_sweep_s.
  - Horizontal grey dashed line: AR-only reference. Sourced from
    xs_numbers.json table1_onhw_migration.HWRFormer_AR_elementwise
    if available; falls back to 6.53 % CER otherwise (per
    tab:onhw-wi-consolidated for elementwise-gated AR).
  - Vertical dashed lines: lambda=0.1 (blue, baseline-selected),
    lambda=0.6 (orange, scaled-selected); selected markers highlighted
    (filled circle for xs at 0.1, filled star for s at 0.6).

Run from anywhere:
    python plot_lambda_sweep_dual.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

XS_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/xs_numbers.json"
)
S_JSON = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/publications/paper2_lncs_overleaf/s_numbers.json"
)
OUT_PDF = Path(
    "/home/woody/iwso/iwso214h/imu-hwr/thesis/figures/ctc_lambda_sweep_cer_wer.pdf"
)

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTED_XS = 0.1
SELECTED_S = 0.6
DATASET = "onhw_wi_word_rh"

XS_COLOR = "#1f77b4"
S_COLOR = "#ff7f0e"
AR_FALLBACK_CER = 6.53  # OnHW WI elementwise-gated AR, tab:onhw-wi-consolidated
AR_FALLBACK_WER = 9.95


def _std_to_sem(std: float | None, n: int | None) -> float:
    if std is None or n is None or n <= 1:
        return 0.0
    try:
        return float(std) / math.sqrt(int(n))
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    with open(XS_JSON) as f:
        xs = json.load(f)
    with open(S_JSON) as f:
        s = json.load(f)

    # -- HWRFormer (baseline) sweep curve from xs_numbers ---------------------
    sweep = xs.get("figure_lambda_sweep", {})
    xs_cer_means: list[float | None] = []
    xs_cer_sems: list[float | None] = []
    xs_wer_means: list[float | None] = []
    xs_wer_sems: list[float | None] = []
    for lam in LAMBDAS:
        cell = sweep.get(f"lambda_{lam:.1f}")
        if cell is None or (cell.get("status") and cell.get("status") != "ready"):
            xs_cer_means.append(None)
            xs_cer_sems.append(None)
            xs_wer_means.append(None)
            xs_wer_sems.append(None)
            continue
        n = cell.get("n_folds")
        xs_cer_means.append(cell.get("cer_mean"))
        xs_cer_sems.append(_std_to_sem(cell.get("cer_std"), n))
        xs_wer_means.append(cell.get("wer_mean"))
        xs_wer_sems.append(_std_to_sem(cell.get("wer_std"), n))

    # -- HWRFormer (scaled) anchor at lambda=0.6 from s_numbers --------------
    s_anchor = (
        s.get("figure_lambda_sweep_s", {})
        .get(DATASET, {})
        .get(f"{SELECTED_S:.1f}")
    )
    if s_anchor is not None:
        s_n = s_anchor.get("n_folds")
        s_cer_mean = s_anchor.get("cer_mean")
        s_cer_sem = _std_to_sem(s_anchor.get("cer_std"), s_n)
        s_wer_mean = s_anchor.get("wer_mean")
        s_wer_sem = _std_to_sem(s_anchor.get("wer_std"), s_n)
    else:
        s_cer_mean = s_cer_sem = s_wer_mean = s_wer_sem = None

    # -- AR-only reference from table1 ---------------------------------------
    ar_ref = (
        xs.get("table1_onhw_migration", {})
        .get("HWRFormer_AR_elementwise")
    )
    if ar_ref is not None and ar_ref.get("status") in (None, "ready"):
        ar_cer = ar_ref.get("cer_mean", AR_FALLBACK_CER)
        ar_wer = ar_ref.get("wer_mean", AR_FALLBACK_WER)
    else:
        ar_cer = AR_FALLBACK_CER
        ar_wer = AR_FALLBACK_WER

    # ------------------------------------------------------------------------
    fig, (ax_cer, ax_wer) = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)

    def _draw(ax, means, sems, ar_value, s_mean, s_sem, selected_y_label):
        # baseline curve
        xs_p = [(lam, m, s_) for lam, m, s_ in zip(LAMBDAS, means, sems) if m is not None]
        if xs_p:
            lams = [p[0] for p in xs_p]
            ms = [float(p[1]) for p in xs_p]
            ss = [float(p[2]) if p[2] is not None else 0.0 for p in xs_p]
            ax.plot(
                lams, ms, color=XS_COLOR, marker="o", linewidth=2.0, markersize=6,
                linestyle="-", zorder=5,
            )
            ax.fill_between(
                lams,
                [m - sm for m, sm in zip(ms, ss)],
                [m + sm for m, sm in zip(ms, ss)],
                color=XS_COLOR, alpha=0.18, linewidth=0, zorder=3,
            )
            # highlight selected xs marker at lambda=0.1
            if SELECTED_XS in lams:
                idx = lams.index(SELECTED_XS)
                ax.plot(
                    [SELECTED_XS], [ms[idx]],
                    marker="o", markersize=10, color=XS_COLOR,
                    markerfacecolor=XS_COLOR, markeredgecolor="black",
                    markeredgewidth=1.0, linestyle="None", zorder=7,
                )

        # AR-only reference line
        ax.axhline(ar_value, color="0.4", linestyle="--", linewidth=1.4, zorder=2)

        # vertical selection markers
        ax.axvline(SELECTED_XS, color=XS_COLOR, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
        ax.axvline(SELECTED_S, color=S_COLOR, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

        # scaled anchor (single point)
        if s_mean is not None:
            ax.errorbar(
                [SELECTED_S], [float(s_mean)],
                yerr=[[float(s_sem or 0.0)], [float(s_sem or 0.0)]],
                fmt="*", markersize=18, color=S_COLOR,
                markerfacecolor=S_COLOR, markeredgecolor="black",
                markeredgewidth=1.0, ecolor=S_COLOR, elinewidth=1.4,
                capsize=4, capthick=1.2, zorder=8,
            )
        else:
            # faded fallback marker if data missing
            ax.plot(
                [SELECTED_S], [ar_value],
                marker="*", markersize=14, color=S_COLOR,
                alpha=0.25, linestyle="None", zorder=4,
            )

        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.set_ylabel(selected_y_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAMBDAS)

        # Adapt y-limits so AR reference is comfortably visible.
        all_y = []
        if xs_p:
            for m, sm in zip(ms, ss):
                all_y.extend([m - sm, m + sm])
        if s_mean is not None:
            sm = float(s_sem or 0.0)
            all_y.extend([float(s_mean) - sm, float(s_mean) + sm])
        all_y.append(ar_value)
        if all_y:
            lo, hi = min(all_y), max(all_y)
            pad = max(0.4, (hi - lo) * 0.15)
            ax.set_ylim(lo - pad, hi + pad)

    _draw(ax_cer, xs_cer_means, xs_cer_sems, ar_cer, s_cer_mean, s_cer_sem, "CER (%)")
    _draw(ax_wer, xs_wer_means, xs_wer_sems, ar_wer, s_wer_mean, s_wer_sem, "WER (%)")

    ax_cer.set_title(r"Downstream AR CER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)
    ax_wer.set_title(r"Downstream AR WER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=XS_COLOR, marker="o", markersize=7, linewidth=2,
               label="HWRFormer (baseline), 5-fold mean $\\pm$ SEM"),
        Line2D([0], [0], color=S_COLOR, marker="*", markersize=14, linewidth=0,
               markerfacecolor=S_COLOR, markeredgecolor="black",
               label=r"HWRFormer (scaled) anchor at $\lambda_{\mathrm{ctc}}{=}0.6$"),
        Line2D([0], [0], color="0.4", linestyle="--", linewidth=1.4,
               label="AR-only baseline (xs, no CTC head)"),
        Line2D([0], [0], color=XS_COLOR, linestyle="--", linewidth=1.2,
               label=r"baseline-selected $\lambda_{\mathrm{ctc}}{=}0.1$"),
        Line2D([0], [0], color=S_COLOR, linestyle="--", linewidth=1.2,
               label=r"scaled-selected $\lambda_{\mathrm{ctc}}{=}0.6$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=9.5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
