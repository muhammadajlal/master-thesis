# Reproducibility / Replication Guide

This repository contains the full training and evaluation code used in our experiments.
The most replication-friendly way to reproduce results is to run the **exact YAML configs** and generate `results.json` via `evaluate.py`.

## Data availability

- **STABILO / `wi_sent_hw6_meta` (private):** This dataset is internal/private and cannot be published. Results on this dataset are **not directly reproducible** outside our environment.
- **OnHW500 (public in this repo layout):** All other experiments in the thesis can be replicated using the OnHW500 datasets shipped in this repository under `data/`:
  - `data/onhw_wi_word_rh` (WI split)
  - `data/onhw_wd_word_rh` (WD split)

## What “replication” means here

- Training is performed per-fold (cross validation), and results are aggregated.
- Model selection is **per-fold best validation CER**, and final numbers are reported as mean ± std across folds.
- Aggregation + MACs/params export is handled by `evaluate.py`, writing `<dir_work>/results.json`.

## Environment

Run commands from `work/REWI_work`:

```bash
cd work/REWI_work
```

Install dependencies (choose one):

- Use your existing environment
- Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- Determinism: we set `seed: 42` in configs, but GPU training can still be non-deterministic depending on CUDA/cuDNN.
- Hardware: results in the thesis were obtained on GPUs (cluster + workstation). Exact wallclock can differ.

## One-command reproduction (OnHW500 tables)

We provide a helper script that:
1) patches dataset/work paths into a config,
2) runs sequential cross-validation training,
3) runs `evaluate.py` to write `results.json`.

From `work/REWI_work`:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root ../../data \
  --out-root ../../results/repro
```

This reproduces **OnHW500 WI** and **OnHW500 WD** experiments for:
- CNN–ARTransformer (scratch)
- CNN–t5-small (decoder-only)

## Getting `t5-small` weights (offline-first)

Our code defaults to offline loading for HuggingFace models (`lm_local_files_only: true`).

To download `t5-small` once (when internet is available) and store it in the repo under `assets/hf_models/`:

```bash
cd work/REWI_work
bash scripts/repro/download_t5_small.sh
```

This downloads into:
- `work/REWI_work/assets/hf_models/t5-small`

The corresponding YAML fields are:

```yaml
lm_name: work/REWI_work/assets/hf_models/t5-small
lm_local_files_only: true
```

If you prefer online loading via the HuggingFace API, set:

```yaml
lm_name: t5-small
lm_local_files_only: false
```

Model page (reference): https://huggingface.co/t5-small

### Pretrained AR decoder condition

The “pretrained decoder” condition additionally requires a decoder-only pretraining checkpoint (`best_loss.pth`).
That checkpoint depends on an external text corpus; the corpus is not shipped here.

If you have a compatible `best_loss.pth`, you can include the pretrained-decoder runs:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root ../../data \
  --out-root ../../results/repro \
  --pretrained-decoder-ckpt /abs/path/to/best_loss.pth
```

## Configs used

Replication configs live in:
- `configs/experiments/`

They are intended to be **small and explicit** (paths are patched by the reproduction script).

## Producing the same artifacts as the thesis tables

- Training (per fold): `python main.py -c <config.yaml>`
- Sequential CV wrapper: `python scripts/others/train_cv.py -c <config.yaml> -m main.py` (requires `idx_fold: -1`)
- Aggregation + MACs/params: `python evaluate.py -c <config.yaml>` -> writes `<dir_work>/results.json`
