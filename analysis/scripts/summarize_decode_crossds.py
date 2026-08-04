#!/usr/bin/env python3
"""Aggregate the cross-dataset decoding study into the thesis tables.

Single source of truth for:
  * Table tab:decode-study-crossds      (plain HWRFormer x 4 datasets x 4 methods)
  * Table tab:decode-study-cross-model  (3 recognizers on OnHW-WI, matched N=4 rescoring)
and for every paired contrast quoted in thesis section 5.8.

Reads per-fold corpus-level CER/WER from each run's metrics.json (decode_study.py
computes them with jiwer over the whole fold); table values are the unweighted
mean over folds 0-4, in percent. Nothing in the thesis is typed by hand: the
emitted row fragments are diffed against the .tex table bodies.

Evidence gates (all fail hard):
  * exactly 5 folds per cell, each with config.json + metrics.json + predictions.json
  * per-cell config assertions (method string, beam/alpha/lm-weight; nbest_size==4
    on every rescoring cell, i.e. the N=8 grid point is rejected); dual_head.enabled
    is read from config.json (the reduced config block inside metrics.json lacks it)
  * checkpoint path must contain the expected model family string
  * within each (dataset, fold): identical n_samples and identical sorted
    ground-truth hash across methods (and across WI recognizers)

Outputs (analysis/decode_crossds_out/):
  decode_crossds_summary.csv   per-cell fold values + means
  decode_crossds_deltas.csv    paired contrasts: mean pp, descriptive 95% t
                               half-width/interval, sign counts, both relative
                               conventions (rel_of_means / rel_fold_paired)
  table_rows_decode_5_13.tex   exact LaTeX rows, cross-dataset table
  table_rows_decode_5_12.tex   exact LaTeX rows, WI cross-model table

Deterministic: sorted iteration, fixed formatting, no timestamps; running twice
must produce byte-identical files.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path("/home/woody/iwso/iwso214h/imu-hwr")
RESULTS = REPO / "work/REWI_work/results/hwr2"
OUTDIR_DEFAULT = REPO / "work/REWI_work/analysis/decode_crossds_out"

# Two-sided 95% Student-t half-width factor for df=4 (five folds). The folds are
# fixed and dependent, so the resulting interval is descriptive, not confirmatory.
T95_DF4 = 2.776

FOLDS = (0, 1, 2, 3, 4)

# Stage tags (verbatim directory prefixes).
GREEDY = "stageA0_ar_greedy"
BEAM = "stageA1_ar_beam_B4"
LENORM04 = "stageA2_ar_lenorm_a0.4"
FUSION = "stageC2_ar_kenlm_shallow_B4_lw0p2_a0"
RESCORE_N4 = "stageC1_ar_kenlm_rescore_N4_lw0p1_a0"

# Expected config per method key (asserted only where the decoder uses the value).
METHOD_EXPECT = {
    "greedy": {"method": "greedy"},
    "beam": {"method": "beam", "beam_size": 4, "alpha": 0.0},
    "lenorm04": {"method": "beam", "beam_size": 4, "alpha": 0.4},
    "fusion": {"method": "beam_lm", "beam_size": 4, "alpha": 0.0, "lm_weight": 0.2},
    "rescore": {"method": "beam_rescore", "beam_size": 4, "nbest_size": 4,
                "alpha": 0.0, "lm_weight": 0.1},
}

# (study_dir, stage_tag) per cell; ckpt_family is asserted as substring of the
# checkpoint path; lm_dir asserted as substring of lm_path for LM methods.
DATASETS = {
    "wi": dict(label="OnHW-WI", lm_dir="lm_noleak/",
               plain=dict(greedy=("decode_study_xs_full_ar", GREEDY),
                          beam=("decode_study_xs_full_ar", BEAM),
                          fusion=("decode_study_xs_full_ar_noleak_a0", FUSION),
                          rescore=("decode_rescore_n4_wi", RESCORE_N4))),
    "wd": dict(label="OnHW-WD", lm_dir="lm_noleak_wd/",
               plain=dict(greedy=("decode_study_xs_full_ar_wd", GREEDY),
                          beam=("decode_study_xs_full_ar_wd", BEAM),
                          fusion=("decode_study_xs_full_ar_wd", FUSION),
                          rescore=("decode_rescore_n4_wd", RESCORE_N4))),
    "privword": dict(label="Priv. Word", lm_dir="lm_noleak_privword/",
                     plain=dict(greedy=("decode_study_xs_full_ar_privword", GREEDY),
                                beam=("decode_study_xs_full_ar_privword", BEAM),
                                fusion=("decode_study_xs_full_ar_privword", FUSION),
                                rescore=("decode_rescore_n4_privword", RESCORE_N4))),
    "privsent": dict(label="Priv. Sent.", lm_dir="lm_noleak_privsent/",
                     plain=dict(greedy=("decode_study_xs_full_ar_privsent", GREEDY),
                                beam=("decode_study_xs_full_ar_privsent", BEAM),
                                fusion=("decode_study_xs_full_ar_privsent", FUSION),
                                rescore=("decode_rescore_n4_privsent", RESCORE_N4))),
}
DATASET_ORDER = ("wi", "wd", "privword", "privsent")

WI_MODELS = {
    "plain": dict(label="HWRFormer", ckpt_family="Baseline-AR-XS-blconv_b",
                  dual_head=False, cells=DATASETS["wi"]["plain"]),
    "noise": dict(label="noise-trained HWRFormer",
                  ckpt_family="Baseline-AR-XS-InputCorruption-uniform",
                  dual_head=False,
                  cells=dict(greedy=("decode_study_xs_full_noise", GREEDY),
                             beam=("decode_study_xs_full_noise", BEAM),
                             lenorm04=("decode_study_xs_full_noise", LENORM04),
                             fusion=("decode_study_xs_full_noise", FUSION),
                             rescore=("decode_rescore_n4_wi_noise", RESCORE_N4))),
    "hybrid": dict(label="hybrid HWRFormer",
                   ckpt_family="train_element_word_hybrid_01_xs_onhw_wi",
                   dual_head=True,
                   cells=dict(greedy=("decode_study_xs_full_hybrid", GREEDY),
                              beam=("decode_study_xs_full_hybrid", BEAM),
                              fusion=("decode_study_xs_full_hybrid_noleak_a0", FUSION),
                              rescore=("decode_rescore_n4_wi_hybrid", RESCORE_N4))),
}
MODEL_ORDER = ("plain", "noise", "hybrid")
PLAIN_CKPT_FAMILY = "Baseline-AR-XS-blconv_b"

METHOD_ROWS_5_13 = (  # (key, LaTeX row label) in table order
    ("greedy", "Greedy"),
    ("beam", r"Retained beam setting ($B_{\text{beam}}{=}4$, $\alpha{=}0$)"),
    ("fusion", r"KenLM shallow fusion ($\alpha{=}0$)"),
    ("rescore", r"KenLM rescore ($N_{\text{best}}{=}B_{\text{beam}}{=}4$, $\alpha{=}0$)"),
)
METHOD_ROWS_5_12 = (  # existing Table 5.12 row order
    ("greedy", "Greedy"),
    ("beam", r"Retained beam setting ($B_{\text{beam}}{=}4$, $\alpha{=}0$)"),
    ("rescore", r"KenLM rescore ($N_{\text{best}}{=}B_{\text{beam}}{=}4$, $\alpha{=}0$)"),
    ("fusion", r"KenLM shallow fusion ($\alpha{=}0$)"),
)


def fail(msg):
    sys.exit(f"GATE FAILURE: {msg}")


def load_cell(study, tag, *, lm_dir=None, ckpt_family=None, dual_head=None,
              method_key=None, with_hash=True):
    """Load one cell (5 folds); enforce gates; return dict of per-fold data."""
    expect = METHOD_EXPECT[method_key]
    out = {"cer": [], "wer": [], "n": [], "gt_hash": []}
    for k in FOLDS:
        run = RESULTS / study / f"{tag}__fold{k}"
        for fn in ("config.json", "metrics.json", "predictions.json"):
            if not (run / fn).is_file():
                fail(f"{run}/{fn} missing")
        cfg = json.loads((run / "config.json").read_text())
        for key, want in expect.items():
            got = cfg.get(key)
            if got != want:
                fail(f"{run}: config {key}={got!r}, expected {want!r}")
        if method_key == "rescore" and cfg.get("nbest_size") != 4:
            fail(f"{run}: N=8 fallback path rejected (nbest_size={cfg.get('nbest_size')})")
        if lm_dir is not None and method_key in ("fusion", "rescore"):
            if lm_dir not in str(cfg.get("lm_path", "")):
                fail(f"{run}: lm_path {cfg.get('lm_path')!r} not under {lm_dir!r}")
        if ckpt_family is not None and ckpt_family not in str(cfg.get("checkpoint", "")):
            fail(f"{run}: checkpoint {cfg.get('checkpoint')!r} lacks family {ckpt_family!r}")
        if dual_head is not None:
            enabled = bool((cfg.get("dual_head") or {}).get("enabled", False))
            if enabled != dual_head:
                fail(f"{run}: dual_head.enabled={enabled}, expected {dual_head}")
        met = json.loads((run / "metrics.json").read_text())
        cer, wer = float(met["cer"]), float(met["wer"])
        if cer < 1.0:  # fractions -> percent
            cer, wer = cer * 100.0, wer * 100.0
        out["cer"].append(cer)
        out["wer"].append(wer)
        out["n"].append(int(met["n_samples"]))
        if with_hash:
            preds = json.loads((run / "predictions.json").read_text())
            gts = sorted(p["gt"] for p in preds)
            if len(gts) != int(met["n_samples"]):
                fail(f"{run}: predictions count {len(gts)} != n_samples {met['n_samples']}")
            out["gt_hash"].append(hashlib.sha256("\n".join(gts).encode()).hexdigest())
    return out


def check_identity(cells, context):
    """Same n_samples and gt hash across methods, per fold."""
    keys = sorted(cells)
    ref = cells[keys[0]]
    for key in keys[1:]:
        for i, k in enumerate(FOLDS):
            if cells[key]["n"][i] != ref["n"][i]:
                fail(f"{context}: fold {k} n_samples differ ({key} vs {keys[0]})")
            if cells[key]["gt_hash"][i] != ref["gt_hash"][i]:
                fail(f"{context}: fold {k} ground-truth sets differ ({key} vs {keys[0]})")


def mean(xs):
    return sum(xs) / len(xs)


def contrast(a, b):
    """Paired per-fold contrast a-b -> dict of stats (pp units)."""
    d = [x - y for x, y in zip(a, b)]
    m = mean(d)
    half = T95_DF4 * statistics.stdev(d) / math.sqrt(len(d))
    rel_folds = [100.0 * (x - y) / y for x, y in zip(a, b)]
    return dict(mean_pp=m, half=half, lo=m - half, hi=m + half,
                n_neg=sum(1 for x in d if x < 0), n_pos=sum(1 for x in d if x > 0),
                rel_of_means=100.0 * (mean(a) - mean(b)) / mean(b),
                rel_fold_paired=mean(rel_folds))


def fmt2(x):
    return f"{x:.2f}"


def bold_rows(rows):
    """rows: list of lists of floats (same length). Bold the per-column minimum
    at 2-dp rounding; rounded ties are all bolded. Returns strings."""
    ncol = len(rows[0])
    out = [[None] * ncol for _ in rows]
    for c in range(ncol):
        col = [fmt2(r[c]) for r in rows]
        lo = min(col, key=float)
        for r in range(len(rows)):
            out[r][c] = rf"\textbf{{{col[r]}}}" if col[r] == lo else col[r]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---------- load: cross-dataset (plain HWRFormer) ----------
    crossds = {}
    for ds in DATASET_ORDER:
        spec = DATASETS[ds]
        cells = {}
        for mkey, (study, tag) in sorted(spec["plain"].items()):
            cells[mkey] = load_cell(study, tag, lm_dir=spec["lm_dir"],
                                    ckpt_family=PLAIN_CKPT_FAMILY, dual_head=False,
                                    method_key=mkey)
        check_identity(cells, f"crossds/{ds}")
        crossds[ds] = cells

    # ---------- load: WI cross-model ----------
    wi = {}
    for mod in MODEL_ORDER:
        spec = WI_MODELS[mod]
        cells = {}
        for mkey, (study, tag) in sorted(spec["cells"].items()):
            cells[mkey] = load_cell(study, tag, lm_dir=DATASETS["wi"]["lm_dir"],
                                    ckpt_family=spec["ckpt_family"],
                                    dual_head=spec["dual_head"], method_key=mkey)
        check_identity(cells, f"wi/{mod}")
        wi[mod] = cells
    # same val set across the three recognizers, per fold
    check_identity({m: {"n": wi[m]["greedy"]["n"], "gt_hash": wi[m]["greedy"]["gt_hash"]}
                    for m in MODEL_ORDER}, "wi/across-models")

    # ---------- summary csv ----------
    with open(args.outdir / "decode_crossds_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "group", "method", "metric",
                    "fold0", "fold1", "fold2", "fold3", "fold4", "mean"])
        for ds in DATASET_ORDER:
            for mkey in ("greedy", "beam", "fusion", "rescore"):
                for metric in ("cer", "wer"):
                    v = crossds[ds][mkey][metric]
                    w.writerow(["crossds", ds, mkey, metric,
                                *[f"{x:.4f}" for x in v], f"{mean(v):.4f}"])
        for mod in MODEL_ORDER:
            for mkey in sorted(wi[mod]):
                for metric in ("cer", "wer"):
                    v = wi[mod][mkey][metric]
                    w.writerow(["wi_models", mod, mkey, metric,
                                *[f"{x:.4f}" for x in v], f"{mean(v):.4f}"])

    # ---------- deltas csv ----------
    def delta_rows(section, group, cells):
        pairs = [("beam", "greedy"), ("fusion", "beam"),
                 ("rescore", "beam"), ("rescore", "greedy")]
        if "lenorm04" in cells:
            pairs += [("rescore", "lenorm04"), ("fusion", "lenorm04")]
        rows = []
        for a, b in pairs:
            for metric in ("cer", "wer"):
                st = contrast(cells[a][metric], cells[b][metric])
                rows.append([section, group, f"{a}-{b}", metric,
                             f"{st['mean_pp']:.4f}", f"{st['half']:.4f}",
                             f"{st['lo']:.4f}", f"{st['hi']:.4f}",
                             st["n_neg"], st["n_pos"],
                             f"{st['rel_of_means']:.2f}", f"{st['rel_fold_paired']:.2f}"])
        return rows

    with open(args.outdir / "decode_crossds_deltas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "group", "contrast", "metric", "mean_pp",
                    "t95_half", "lo", "hi", "n_folds_neg", "n_folds_pos",
                    "rel_of_means_pct", "rel_fold_paired_pct"])
        for ds in DATASET_ORDER:
            w.writerows(delta_rows("crossds", ds, crossds[ds]))
        for mod in MODEL_ORDER:
            w.writerows(delta_rows("wi_models", mod, wi[mod]))

    # ---------- LaTeX fragments ----------
    # Table 5.13: rows = methods, columns = dataset x (CER, WER); bold per column.
    vals = [[mean(crossds[ds][mkey][metric])
             for ds in DATASET_ORDER for metric in ("cer", "wer")]
            for mkey, _ in METHOD_ROWS_5_13]
    cells = bold_rows(vals)
    with open(args.outdir / "table_rows_decode_5_13.tex", "w") as f:
        for (mkey, label), row in zip(METHOD_ROWS_5_13, cells):
            f.write(label + " & " + " & ".join(row) + r" \\" + "\n")

    # Table 5.12: rows = methods, columns = model x (CER, WER); bold per column.
    vals = [[mean(wi[mod][mkey][metric])
             for mod in MODEL_ORDER for metric in ("cer", "wer")]
            for mkey, _ in METHOD_ROWS_5_12]
    cells = bold_rows(vals)
    with open(args.outdir / "table_rows_decode_5_12.tex", "w") as f:
        for (mkey, label), row in zip(METHOD_ROWS_5_12, cells):
            f.write(label + " & " + " & ".join(row) + r" \\" + "\n")

    print("OK: gates passed; wrote 4 files to", args.outdir)
    for name in ("decode_crossds_summary.csv", "decode_crossds_deltas.csv",
                 "table_rows_decode_5_13.tex", "table_rows_decode_5_12.tex"):
        digest = hashlib.sha256((args.outdir / name).read_bytes()).hexdigest()[:16]
        print(f"  {name}  sha256:{digest}")


if __name__ == "__main__":
    main()
