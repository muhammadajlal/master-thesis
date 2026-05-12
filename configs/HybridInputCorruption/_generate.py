#!/usr/bin/env python3
"""
Generate Hybrid-CTC-AR + Input-Corruption configs by merging
- AR-InputCorruption* configs (per mode, per dataset)  -> base
- hybrid/train_element_word_06.yaml `dual_head` block  -> grafted

Result family: HybridInputCorruption/train-hyb-<mode>-<dataset>.yaml
Output dir:    results/hwr2/HybridInputCorruption_<mode>/ar_transformer_s__<dataset>/

Grid: 5 modes x 6 datasets = 30 configs. All at p_replace=0.15, lambda_ctc=0.6.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]      # configs/
OUT = ROOT / "HybridInputCorruption"
OUT.mkdir(parents=True, exist_ok=True)

# (mode, dataset_short) -> source AR config (relative to configs/)
SOURCES = {
    # mode key in input_corruption.mode -> (family, file template), keyed by ds slug
    "uniform": {
        "onhw_wi_word_rh": "AR-InputCorruption/train-ar-corrupt-onhw-word.yaml",
        "onhw_wd_word_rh": "AR-InputCorruption-WD/train-ar-uniform-onhw-wd.yaml",
        "wi_word_hw6_meta": "AR-InputCorruption/train-ar-corrupt-stabilo-word.yaml",
        "wi_sent_hw6_meta": "AR-InputCorruption/train-ar-corrupt-stabilo-sent.yaml",
        "onhw_equations_wi_word_rh": "AR-InputCorruption-Equations/train-ar-uniform-equations-wi.yaml",
        "onhw_equations_wd_word_rh": "AR-InputCorruption-Equations/train-ar-uniform-equations-wd.yaml",
    },
    "bigram_right": {
        "onhw_wi_word_rh": "AR-InputCorruption-Bigram/train-ar-bigram-onhw-word.yaml",
        "onhw_wd_word_rh": "AR-InputCorruption-WD/train-ar-bigramright-onhw-wd.yaml",
        "wi_word_hw6_meta": "AR-InputCorruption-Bigram/train-ar-bigram-stabilo-word.yaml",
        "wi_sent_hw6_meta": "AR-InputCorruption-Bigram/train-ar-bigram-stabilo-sent.yaml",
        "onhw_equations_wi_word_rh": "AR-InputCorruption-Equations/train-ar-bigramright-equations-wi.yaml",
        "onhw_equations_wd_word_rh": "AR-InputCorruption-Equations/train-ar-bigramright-equations-wd.yaml",
    },
    "bigram_left": {
        "onhw_wi_word_rh": "AR-InputCorruption-BigramLeft/train-ar-bigramleft-onhw-word.yaml",
        "onhw_wd_word_rh": "AR-InputCorruption-WD/train-ar-bigramleft-onhw-wd.yaml",
        "wi_word_hw6_meta": "AR-InputCorruption-BigramLeft/train-ar-bigramleft-stabilo-word.yaml",
        "wi_sent_hw6_meta": "AR-InputCorruption-BigramLeft/train-ar-bigramleft-stabilo-sent.yaml",
        "onhw_equations_wi_word_rh": "AR-InputCorruption-Equations/train-ar-bigramleft-equations-wi.yaml",
        "onhw_equations_wd_word_rh": "AR-InputCorruption-Equations/train-ar-bigramleft-equations-wd.yaml",
    },
    "self_confusion": {
        "onhw_wi_word_rh": "AR-InputCorruption-SelfConf/train-ar-selfconf-onhw-word.yaml",
        "onhw_wd_word_rh": "AR-InputCorruption-WD/train-ar-selfconf-onhw-wd.yaml",
        "wi_word_hw6_meta": "AR-InputCorruption-SelfConf/train-ar-selfconf-stabilo-word.yaml",
        "wi_sent_hw6_meta": "AR-InputCorruption-SelfConf/train-ar-selfconf-stabilo-sent.yaml",
        "onhw_equations_wi_word_rh": "AR-InputCorruption-Equations/train-ar-selfconf-equations-wi.yaml",
        "onhw_equations_wd_word_rh": "AR-InputCorruption-Equations/train-ar-selfconf-equations-wd.yaml",
    },
    "adjacent_swap": {
        "onhw_wi_word_rh": "AR-InputCorruption-AdjacentSwap/train-ar-adjacentswap-onhw-word.yaml",
        "onhw_wd_word_rh": "AR-InputCorruption-WD/train-ar-adjacentswap-onhw-wd.yaml",
        "wi_word_hw6_meta": "AR-InputCorruption-AdjacentSwap/train-ar-adjacentswap-stabilo-word.yaml",
        "wi_sent_hw6_meta": "AR-InputCorruption-AdjacentSwap/train-ar-adjacentswap-stabilo-sent.yaml",
        "onhw_equations_wi_word_rh": "AR-InputCorruption-Equations/train-ar-adjacentswap-equations-wi.yaml",
        "onhw_equations_wd_word_rh": "AR-InputCorruption-Equations/train-ar-adjacentswap-equations-wd.yaml",
    },
}

# dual_head block grafted on top of every AR config.
DUAL_HEAD = {
    "enabled": True,
    "arch_ctc": "linear",
    "lambda_ar": 1.0,
    "lambda_ctc": 0.6,
    "tie": {
        "ctc_to_ar_outproj": False,
        "ar_emb_outproj": False,
        "ctc_input_space": "dec",
    },
    "lambda_ctc_schedule": {
        "enabled": False,
        "type": "linear",
        "warmup_epochs": 0,
        "max": 0.6,
        "decay_start": 50,
        "min": 0.1,
        "decay_epochs": 100,
    },
    "loss_balance": "sum",
}

# Short slugs for filenames.
MODE_SLUG = {
    "uniform": "uniform",
    "bigram_right": "bigramright",
    "bigram_left": "bigramleft",
    "self_confusion": "selfconf",
    "adjacent_swap": "adjacentswap",
}

DS_SLUG = {
    "onhw_wi_word_rh": "onhw-wi-word",
    "onhw_wd_word_rh": "onhw-wd-word",
    "wi_word_hw6_meta": "stabilo-word",
    "wi_sent_hw6_meta": "stabilo-sent",
    "onhw_equations_wi_word_rh": "equations-wi",
    "onhw_equations_wd_word_rh": "equations-wd",
}


def _ordered_cfg(base: dict, mode: str, dataset: str) -> dict:
    """Insert dual_head at top, ensure input_corruption.mode/p_replace/smoothing are set, retarget dir_work."""
    cfg = {}
    # dual_head first (cosmetic but matches hybrid configs)
    cfg["dual_head"] = DUAL_HEAD
    # then the rest of the base in its original order
    for k, v in base.items():
        if k == "dual_head":
            continue
        cfg[k] = v

    # Normalize input_corruption
    ic = dict(cfg.get("input_corruption", {}))
    ic["enabled"] = True
    ic["mode"] = mode
    ic["p_replace"] = 0.15
    if mode in ("bigram_right", "bigram_left", "self_confusion"):
        ic.setdefault("smoothing", 1.0)
    if mode == "self_confusion":
        # confusion path expected to be set in the source; keep as-is
        assert "confusion_path" in ic, f"self_confusion missing confusion_path for {dataset}"
    cfg["input_corruption"] = ic

    # Retarget dir_work
    out_root = "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2"
    cfg["dir_work"] = f"{out_root}/HybridInputCorruption_{mode}/ar_transformer_s__{dataset}"

    # Belt-and-suspenders: ensure dual_head/AR settings consistent
    cfg["arch_de"] = "ar_transformer_s"
    cfg["arch_en"] = "blconv_b"
    cfg["use_gated_attention"] = True
    cfg["gating_type"] = "elementwise"
    return cfg


def main():
    n = 0
    for mode, ds_map in SOURCES.items():
        for ds, src_rel in ds_map.items():
            src = ROOT / src_rel
            assert src.is_file(), f"missing source: {src}"
            with src.open() as f:
                base = yaml.safe_load(f)
            cfg = _ordered_cfg(base, mode, ds)
            out_name = f"train-hyb-{MODE_SLUG[mode]}-{DS_SLUG[ds]}.yaml"
            out_path = OUT / out_name
            header = (
                f"# =========================\n"
                f"# Hybrid CTC-AR + Input Corruption ({mode}): blconv_b + ar_transformer_s + elementwise gating\n"
                f"# Dataset: {ds}\n"
                f"# Derived from {src_rel} with `dual_head` block grafted on top.\n"
                f"# lambda_ctc=0.6, p_replace=0.15.\n"
                f"# =========================\n"
            )
            with out_path.open("w") as f:
                f.write(header)
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            n += 1
            print(f"wrote {out_name}")
    print(f"\nTotal: {n} configs")


if __name__ == "__main__":
    main()
