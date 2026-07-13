# Reproducibility / Replication Guide

This document is the **replication guide** for the thesis experiments: it records data availability, canonical configuration and result locations, statistical analysis, and a public-data sanity-check workflow.

For the project description, repo layout, and general usage of `main.py` / `evaluate.py`, see [README.md](README.md).

The most replication-friendly way to reproduce results is to run the **exact YAML configs** from the repository root and generate `results.json` via `evaluate.py`.

## Data availability

- **STABILO / `wi_word_hw6_meta` and `wi_sent_hw6_meta` (private):** For commercial reasons, these datasets are not published. Their annotations and sensor files must be included in the restricted submission archive made available to the supervisor. The corresponding thesis results are not independently reproducible without that archive.
- **OnHW500 (public):** The public counterpart uses the right-handed writer-independent and writer-dependent subsets of OnHW-words500. Download instructions are provided by Fraunhofer IIS: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

We use a MSCOCO-like structure for training and evaluation. After downloading OnHW, convert it with `scripts/onhw.ipynb`; set `dir_raw`, `dir_out`, and `writer_indep`/`writer_dep` in the notebook before executing all cells.

### Data Layout

After running the notebook, the MSCOCO-like dataset structure is available under `<dir_out>/`:
```
data/
└── onhw_wd_word_rh/
    ├── train.json          # Training annotations (by fold)
    ├── val.json            # Validation annotations (by fold)
    └── data/               # Sensor CSV files
```

## What “replication” means here

- Training is performed per-fold (cross validation), and results are aggregated.
- Model selection is **per-fold best validation CER**, and final numbers are reported as mean ± std across folds.
- Aggregation + MACs/params export is handled by `evaluate.py`, writing `<dir_work>/results.json`.

## Environment

Run commands from the repository root (the folder created by `git clone`).

Install dependencies (choose one):

- Use your existing environment
- Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- Determinism: we set `seed: 42` in configs, but GPU training can still be non-deterministic depending on CUDA/cuDNN.
- Hardware: results in the thesis were obtained on GPUs (cluster + workstation). Exact wallclock can differ.

## Partial one-command public replication

We provide a helper script that:
1) patches dataset/work paths into a config,
2) runs sequential cross-validation training,
3) runs `evaluate.py` to write `results.json`.

From the repository root:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root data \
  --out-root results/repro
```

This helper sanity-checks the public data and training pipeline by reproducing **OnHW500 WI** and **OnHW500 WD** experiments for:

- CNN–ARTransformer (Decoder not-pretrained)
- CNN–t5-small (decoder-only)


It does not reproduce every thesis table. The exact Chapter 6 configuration and result mapping is recorded below.
## Getting `t5-small` weights (offline-first)

Our code defaults to offline loading for HuggingFace models (`lm_local_files_only: true`).

To download `t5-small` once (when internet is available) and store it in the repo under `assets/hf_models/`:

```bash
bash scripts/repro/download_t5_small.sh
```

This downloads into:
- `assets/hf_models/t5-small`

The corresponding YAML fields are:

```yaml
lm_name: assets/hf_models/t5-small
lm_local_files_only: true
```

If you prefer online loading via the HuggingFace API, set:

```yaml
lm_name: t5-small
lm_local_files_only: false
```

Model page (reference): https://huggingface.co/t5-small

### Legacy pretrained AR decoder helper

The helper's legacy “pretrained decoder” condition additionally requires a decoder-only pretraining checkpoint (`best_loss.pth`). This is separate from the GPT-2 experiments in Chapter 6.
The checkpoint depends on an external text corpus; the following procedure can be applied to word- or sentence-level pretraining by selecting the corresponding inputs.

### Download the source sentence/word lists (News 2024, 1M)
The input files come from the Leipzig Corpora Collection / Wortschatz Leipzig downloads:
https://wortschatz-leipzig.de/de/download/

We use the **News** corpora for year **2024** at size **1M** (measured in number of sentences on the website). Download both:
- English: https://wortschatz-leipzig.de/de/download/eng  → section **News** → row **2024** → column **Download 1M**
- German: https://wortschatz-leipzig.de/de/download/deu → section **News** → row **2024** → column **Download 1M**

After downloading, extract the archive (format depends on the download; typically a `.tar.gz`):

```bash
tar -xzf eng_news_2024_1M.tar.gz
tar -xzf deu_news_2024_1M.tar.gz
```

Then locate the `*-sentences.txt` file inside each extracted folder and place/rename it to match the expected inputs below:
- `eng_news_2024_1M-sentences.txt`
- `deu_news_2024_1M-sentences.txt`

Note: These downloads are subject to the provider's terms of use (see the download page).

## Inputs
- `eng_news_2024_1M-sentences.txt`
- `deu_news_2024_1M-sentences.txt`

These files are sentence-per-line, optionally prefixed with a numeric rank:

```
1  $0.07 of every dollar for public officers' pensions.
2  $100 million boost for new pro league.
...
```

Note: Some lines have a leading `$` character which is stripped during processing.

## What was filtered
1. Read all unique sentence labels from:
   - `../../../../data/wi_sent_hw6_meta/train.json`
   - `../../../../data/wi_sent_hw6_meta/val.json`

   In this dataset, the label is stored in the annotation field `label`.

2. Normalize everything the same way (by default):
   - Unicode NFKC normalization
   - lowercase
   - collapse whitespace runs

3. Build a mixed set of sentences from EN∪DE, then remove anything that matches a dataset label.

## Outputs
- `mixed_en_de_no_wi_sent_hw6_meta.txt`
  - Final mixed dictionary (**1,999,971 sentences**, one per line)
  - **Already has wi_sent_hw6_meta label-sentences removed**

- `removed_due_to_leakage.txt`
  - Sentences that were removed because they matched a dataset label
  - **1 sentence** removed: `"three people were injured."`
  - Sorted for easy inspection

## How to reproduce
From the repository root:

```bash
python3 scripts/tools/build_mixed_dictionary.py \
  --kind sent \
  --en assets/dictionaries/sent/eng_news_2024_1M-sentences.txt \
  --de assets/dictionaries/sent/deu_news_2024_1M-sentences.txt \
  --dataset data/wi_sent_hw6_meta \
  --out assets/dictionaries/sent/mixed_en_de_no_wi_sent_hw6_meta.txt \
  --out-removed assets/dictionaries/sent/removed_due_to_leakage.txt
```

The script used is: `scripts/tools/build_mixed_dictionary.py`.


If you have a compatible `best_loss.pth`, you can include the pretrained-decoder runs:

```bash
bash scripts/repro/reproduce_tables.sh \
  --data-root data \
  --out-root results/repro \
  --pretrained-decoder-ckpt /abs/path/to/best_loss.pth
```

## Thesis experiment manifest

All paths below are relative to the repository root. In result patterns, `DATA` denotes either `onhw_wi_word_rh` or `wi_word_hw6_meta`. A directory ending in `_noctc` denotes auxiliary CTC weight λ = 0, `_lam02` denotes λ = 0.2, and no suffix denotes λ = 0.6, unless the row states otherwise. Each terminal result directory contains `results.json`; the per-fold subdirectories preserve the generated training YAML and selected-checkpoint metadata.

Set the archived result root as:

```bash
RESULTS_ROOT=../../results/hwr2
```

### Chapter 6 configurations and results

| Thesis condition | Canonical configuration | Result family below `$RESULTS_ROOT` |
|---|---|---|
| MLP, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-A1-mlp-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A1_mlp_pretrained/vlm__DATA` |
| MLP, random GPT-2 | `configs/vlm_ablation/train-vlm-A2-mlp-random.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A2_mlp_random/vlm__DATA` |
| Linear, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-A3-linear-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_A3_linear_pretrained/vlm__DATA` |
| Pool-MLP, pretrained GPT-2 | `configs/vlm_ablation/train-vlm-B1-pooling-pretrained.yaml` | `Ablations-MMLM/GPT-2/AR-only/vlm_ablation_B1_pooling_pretrained/vlm__DATA` |
| Conv-Pool, pretrained GPT-2 | `configs/G1_minimal_connector/train-G1b-conv-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/G1b_conv_pool/vlm__DATA` |
| Frozen mini-Q-Former encoder diagnostics | `configs/_f1f2_ablation/train-F{1,2}-qenc-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/{F1_frozen_enc_mlp,F2_vlm_enc_ar}/*_qenc_ablation` |
| Frozen HWRFormer encoder diagnostics | `configs/_f1f2_ablation/train-F{1,2}-hwenc-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/AR-only/{F1_frozen_enc_mlp,F2_vlm_enc_ar}/*_hwenc_ablation` |
| MLP + auxiliary CTC | `configs/H1_hybrid_ctc_vlm/train-H1-mlp-{onhw,word}.yaml`; OnHW λ = 0.2: `configs/H1_hybrid_ctc_vlm/lambda_sweep/train-H1-mlp-onhw-lam02.yaml`; private λ = 0.2: `configs/_lam02_ch6/H1_hybrid_ctc_vlm/train-H1-mlp-word_lam02.yaml` | `Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__DATA{,_lam02}`; the canonical OnHW λ = 0.2 result is archived under `H1_LambdaSweep/H1_hybrid_mlp_lam02__onhw_wi_word_rh` |
| Pool-MLP + auxiliary CTC | `configs/H1_hybrid_ctc_vlm_pooling/train-H1-pooling-{onhw,word}.yaml`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_pooling/vlm__DATA{,_lam02}` |
| Sequence-level contrastive MLP/Pool | `configs/J1_contrastive_{mlp,pooling}/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_{mlp,pooling}/vlm__DATA{,_lam02}` |
| Argmax compression + MSE | `configs/K1_ctc_mse/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse/vlm__DATA{,_lam02}` |
| Posterior reconstruction | `configs/K2_ctc_posterior/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior/vlm__DATA{,_lam02}` |
| Argmax-segment contrastive alignment | `configs/K4_sea_contrastive/`; corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid-Contrastive/K4_sea_contrastive/vlm__DATA{,_lam02}` |
| Lightweight Q-Former | `configs/L1_mini_qformer/`, `configs/_noctc_ch6/L1_mini_qformer/`, and corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer/vlm__DATA{,_noctc,_lam02}` |
| Gated query connector | `configs/L2_kv_slim/`, `configs/_noctc_ch6/L2_kv_slim/`, and corresponding `_lam02_ch6` configs | `Ablations-MMLM/GPT-2/Hybrid/L2_kv_slim/vlm__DATA{,_noctc,_lam02}` |
| ByT5 control | `configs/M1_byt5_hybrid_mlp/train-M1-byt5-{onhw,word}.yaml` | `Ablations-MMLM/byt5-small/Hybrid/M1_byt5_hybrid_mlp/vlm__DATA` |
| Shallow-Conformer control | `configs/N1_conformer_hybrid_mlp/train-N1-conformer-{onhw,word}.yaml` | `Ablations-MMLM/GPT-2/Hybrid/N1_conformer_hybrid_mlp/vlm__DATA` |

Brace notation in the table is shorthand for the listed literal alternatives; it is not a shell command. The exact central-condition result paths used for thesis statistics are also encoded in `scripts/chapter6_analysis.py`, preventing the prose manifest from becoming the only source of truth.

### Chapter 5 configuration families

| Evidence family | Configuration family | Result family below `$RESULTS_ROOT` |
|---|---|---|
| REWI reference | `configs/Baseline-REWI/` | `Baseline-REWI/` |
| HWRFormer and gating variants | `configs/AR-Baseline/` | `Baseline-AR-*` |
| Input-noise study | `configs/AR-InputCorruption*/` | `Baseline-AR-InputCorruption*/` |
| Auxiliary-CTC study | `configs/hybrid/` | `Baseline-Hybrid/` |
| HWRFormer-L capacity controls | `configs/AR-Baseline-XS*/` | `Baseline-AR-XS*/` |
| Hybrid-noise controls | `configs/HybridInputCorruption*/` | `Baseline-Hybrid-InputCorruption*/` |
| Inference-time decoding study | `configs/decode_study/` | archived decoding outputs referenced by the Chapter 5 analysis scripts |

The Chapter 5 families contain the condition-specific YAML files used by the corresponding thesis tables. Preserve those YAMLs together with generated fold configurations and `results.json`; directory names alone are not a substitute for the archived parameters.

## Statistical reproduction

The deterministic Chapter 6 analysis validates five folds per central condition and reproduces means, paired fold differences, fold signs, and two-sided 95% Student t intervals:

```bash
python scripts/chapter6_analysis.py --results-root ../../results/hwr2
python scripts/chapter6_analysis.py --results-root ../../results/hwr2 --json
```

Run the command twice and compare the output before submission. The intervals are exploratory because operating-point selection and reporting use the same validation folds.

## Producing the thesis result artifacts

- Training (per fold): `python main.py -c <config.yaml>`
- Sequential CV wrapper: `python scripts/others/train_cv.py -c <config.yaml> -m main.py` (requires `idx_fold: -1`)
- Aggregation and MAC/parameter export: `python evaluate.py -c <config.yaml>`; this writes `<dir_work>/results.json`

For an auditable submission, record the code revision with `git rev-parse HEAD` and preserve any intentional uncommitted patch. The restricted handover must include the private inputs, exact YAMLs, generated per-fold configurations, selected-checkpoint metadata, all canonical `results.json` files, the analysis scripts, and package/environment information. Public-data replication and private-data verification should produce the same table-level means up to documented hardware non-determinism.
