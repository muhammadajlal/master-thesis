#!/usr/bin/env python3
"""
Generate the 4 missing hybrid-CTC-AR baselines (no input corruption) to match the
6-dataset coverage of HybridInputCorruption/ runs.

Existing hybrid baselines:
  train_element_word_06.yaml         -> onhw_wi_word_rh
  train_element_word_06_stabilo.yaml -> wi_word_hw6_meta

Generated here:
  train_element_word_06_onhw_wd.yaml      -> onhw_wd_word_rh
  train_element_word_06_stabilo_sent.yaml -> wi_sent_hw6_meta
  train_element_word_06_equations_wi.yaml -> onhw_equations_wi_word_rh
  train_element_word_06_equations_wd.yaml -> onhw_equations_wd_word_rh

Config is byte-identical to train_element_word_06_stabilo.yaml except for:
  dir_dataset, dir_work, categories (where needed).
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "train_element_word_06_stabilo.yaml"

DATA_ROOT = "/home/woody/iwso/iwso214h/imu-hwr/data"
OUT_ROOT = "/home/woody/iwso/iwso214h/imu-hwr/results/hwr2"

# Categories from existing AR-InputCorruption configs per dataset.
CATS_LATIN = [
    "", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R",
    "S","T","U","V","W","X","Y","Z","Ä","Ö","Ü",
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r",
    "s","t","u","v","w","x","y","z","ä","ö","ü","ß",
]
CATS_SENT = [
    "", "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r",
    "s","t","u","v","w","x","y","z",
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R",
    "S","T","U","V","W","X","Y","Z",
    "ä","ö","ü","Ä","Ö","Ü","ß",
    "0","1","2","3","4","5","6","7","8","9",
    ".",",","(",")","'","?","!","+","=","-","/",";",":","·"," ",
]
CATS_EQ = ["", "+","-","0","1","2","3","4","5","6","7","8","9",":","=","·"]

TARGETS = [
    # (filename, dataset, run_name_suffix, categories)
    ("train_element_word_06_onhw_wd.yaml",      "onhw_wd_word_rh",           "onhw_wd",      CATS_LATIN),
    ("train_element_word_06_stabilo_sent.yaml", "wi_sent_hw6_meta",          "stabilo_sent", CATS_SENT),
    ("train_element_word_06_equations_wi.yaml", "onhw_equations_wi_word_rh", "equations_wi", CATS_EQ),
    ("train_element_word_06_equations_wd.yaml", "onhw_equations_wd_word_rh", "equations_wd", CATS_EQ),
]


def main():
    with BASE.open() as f:
        base = yaml.safe_load(f)
    for fname, ds, suffix, cats in TARGETS:
        cfg = dict(base)  # shallow copy is fine (we replace fields below)
        cfg["dir_dataset"] = f"{DATA_ROOT}/{ds}"
        cfg["dir_work"] = f"{OUT_ROOT}/train_element_word_hybrid_06_{suffix}/ar_transformer_s__{ds}"
        cfg["categories"] = cats
        out = HERE / fname
        header = (
            f"# =========================\n"
            f"# Hybrid CTC-AR (no-corruption baseline, clone of train_element_word_06_stabilo): \n"
            f"# ar_transformer_s + blconv_b + elementwise gating\n"
            f"# Dataset: {ds}\n"
            f"# Matched no-corruption reference for HybridInputCorruption/* runs.\n"
            f"# =========================\n"
        )
        with out.open("w") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
