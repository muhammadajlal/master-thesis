# Reproducibility / Replication Guide

This repository contains the full training and evaluation code used in our experiments.
The most replication-friendly way to reproduce results is to run the **exact YAML configs** from the repository root and generate `results.json` via `evaluate.py`.

## Data availability

- **STABILO / `wi_sent_hw6_meta` (private):** For commercial reasons, our datasets will not be published. Alternatively, you can use the OnHW public dataset for training and evaluation. Results on this dataset are **not directly reproducible** outside our environment.
- **OnHW500:** All other experiments in the thesis can be replicated using the OnHW500 datasets. In the thesis, we use the right-handed (WI/WD Split) subset of the OnHW-words500 dataset. To download the dataset, please visit: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

We use a MSCOCO-like structure for the training and evaluation of our dataset. After the OnHW dataset is downloaded, please convert the original dataset to the desired structure with the notebook `onhw.ipynb`. Please adjust the variables `dir_raw`, `dir_out`, and `writer_indep`/`writer_dep` accordingly. You can open and run the notebook from the repository root.

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

## One-command reproduction (OnHW500 tables)

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

This reproduces **OnHW500 WI** and **OnHW500 WD** experiments for:
- CNN–ARTransformer (Decoder not-pretrained)
- CNN–t5-small (decoder-only)

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

### Pretrained AR decoder condition

The “pretrained decoder” condition additionally requires a decoder-only pretraining checkpoint (`best_loss.pth`).
That checkpoint depends on an external text corpus; the corpus could be reproduced for both words and sentences levell pre-training using the following instructions please replace sentences/words accordingly in the commands. 

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

## Configs used

Replication configs live in:
- `configs/experiments/`

They are intended to be **small and explicit** (paths are patched by the reproduction script).

## Producing the same artifacts as the thesis tables

- Training (per fold): `python main.py -c <config.yaml>`
- Sequential CV wrapper: `python scripts/others/train_cv.py -c <config.yaml> -m main.py` (requires `idx_fold: -1`)
- Aggregation + MACs/params: `python evaluate.py -c <config.yaml>` -> writes `<dir_work>/results.json`
