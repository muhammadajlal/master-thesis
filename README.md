# Alignment-Stable Attention for IMU Handwriting Recognition via Hybrid CTC--AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation  
**Master’s Thesis Repository — Muhammad Ajlal Khan (FAU Pattern Recognition Lab)**

This repository contains the codebase and experimental artifacts for my Master’s thesis on **writer-independent (WI) online handwriting recognition (HWR) from inertial measurement unit (IMU) time series**. The work builds on the REWI baseline introduced by my supervisor’s paper **“Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition” (arXiv:2502.20954)**. :contentReference[oaicite:0]{index=0}

In addition, the **project report is treated as a completed thesis component** and documents the initial research phase, system migration, ablations, and diagnostic analyses. :contentReference[oaicite:1]{index=1}

---

## 1) Thesis Focus

IMU-based HWR enables writing on paper (or any surface) without cameras or digitizers, but **WI generalization** is difficult due to:
- large inter-writer variability (speed, stroke dynamics, grip, sensor placement),
- sensor noise and distribution shift,
- alignment instability and decoding-driven failure modes.

**Thesis objective:** improve **both** character-level and word-level recognition quality under realistic efficiency constraints by combining:
- **alignment-stable training**,  
- **calibrated decoding**, and  
- **hybrid CTC–autoregressive (AR) learning** (CTC for monotonic alignment supervision, AR for global conditional modeling). :contentReference[oaicite:2]{index=2}

---

## 2) What’s Already Done (Project Report Component)

The project phase established a practical and extensible AR pipeline and produced the main empirical findings below:

### A. Migration: CTC → Attention-based Autoregressive Decoder
A major engineering blocker for Transformer decoding in this domain is **variable-length encoder sequences**. The project implemented **batch-wise rectangularization** (pad to the batch maximum + attention masks), enabling efficient AR training/inference without global fixed-length padding. :contentReference[oaicite:3]{index=3}

### B. Lightweight Attention Stabilization via SDPA Output Gating
Inspired by gated-attention methods, the project introduced **post-SDPA output gating** (headwise and elementwise). This delivered consistent gains across tasks with negligible overhead. :contentReference[oaicite:4]{index=4}

### C. Tokenization Study: Character vs. BPE
BPE tokenization (multiple merge settings) **did not yield consistent improvements** over character decoding in this setting. :contentReference[oaicite:5]{index=5}

### D. Beyond Aggregate Metrics: Tail Risk + Failure Diagnostics
The evaluation moved beyond CER/WER and characterized error structure via:
- **normalized per-sample Levenshtein error distributions** (bimodality + heavy tails),
- **collision analysis** (frequency bias vs. input similarity),
- **writer-stratified error concentration** (collision lift among top writers),
- **qualitative diagnostics** using cross-attention heatmaps + Grad-CAM1D. :contentReference[oaicite:6]{index=6}

---

## 3) Main Results (Headline Numbers)

All results below are on **internal STABILO WI splits** (word-level + sentence-level) as documented in the project report. :contentReference[oaicite:7]{index=7}

### Word-level Recognition
| Model | CER ↓ | WER ↓ |
|------|------:|------:|
| **REWI baseline (CNN–BiLSTM–CTC)** | 9.39 | 31.81 |
| **Ours AR (CNN–Transformer AR)** | 12.80 | 20.60 |
| **Ours AR + Elementwise Gating** | 10.37 | **17.43** |
| **Ours AR + Headwise Gating** | **10.29** | 17.54 |

**Key takeaway:** AR decoding dramatically reduces **WER** versus CTC, and **gating recovers much of the CER gap while further improving WER**. :contentReference[oaicite:8]{index=8}

### Sentence-level Recognition
| Model | CER ↓ | WER ↓ |
|------|------:|------:|
| **REWI baseline (CNN–BiLSTM–CTC)** | 6.54 | 23.51 |
| **Ours AR (CNN–Transformer AR)** | 11.35 | 15.83 |
| **Ours AR + Elementwise Gating** | 8.68 | 12.30 |
| **Ours AR + Headwise Gating** | 8.68 | **12.25** |

**Key takeaway:** The same pattern holds at sentence-level: **AR improves WER substantially**, and **gating provides strong additional gains**. :contentReference[oaicite:9]{index=9}

### Error Structure (Operationally Important)
- Exact-match mass is large (~**78–79%**), but the remaining errors are **heavy-tailed**.
- Tail risk is non-trivial, especially for sentences (e.g., **P(e > 1)** notably higher than words).
- Collisions disproportionately involve **frequent labels**, and collisions are **concentrated in subsets of writers**, consistent with frequency bias interacting with writer shift. :contentReference[oaicite:10]{index=10}

---

## 4) Repository Contents (Recommended Layout)

> Adjust folder names to match your current codebase; the structure below is a clean thesis-friendly convention.


---

## 5) Reproducibility and Evaluation Protocol

### Metrics
- **CER / WER** (micro-averaged corpus rates)
- **Normalized per-sample edit error**: `e = d_char / |y|`  
  used to quantify bimodality + tail risk and enable cross-task comparability. :contentReference[oaicite:11]{index=11}

### Diagnostic Analyses
- **Collision analysis:** when a wrong prediction exactly matches another ground-truth label.
- **Writer-stratified reporting:** error and collision concentration (lift) across writers.
- **Qualitative inspection:** cross-attention alignment patterns + Grad-CAM1D. :contentReference[oaicite:12]{index=12}

---

## 6) Thesis Roadmap (Next Milestones)

The thesis plan formalizes the next steps beyond the project phase:

1. **Hybrid multitask training (CTC + AR)**
   - joint objective: `L = L_AR + λ · L_CTC`
   - goal: keep AR’s WER advantage while using CTC to stabilize monotonic alignment and improve CER. :contentReference[oaicite:13]{index=13}

2. **Calibrated decoding (beam search + EOS/length control)**
   - beam search, length normalization, EOS bias/min-length constraints
   - optional coverage/skip penalties to reduce attention collapse and catastrophic tails. :contentReference[oaicite:14]{index=14}

3. **Optional LM-based decoding extensions**
   - lightweight external LM fusion/rescoring (if sufficient transcripts exist)
   - or pretrained LM decoder adaptation under strict latency/parameter budgets. :contentReference[oaicite:15]{index=15}

4. **Robustness workstream**
   - writer-balanced sampling / reweighting
   - tail-risk–aware evaluation as a first-class objective
   - systematic linkage between qualitative failure signatures and recurring error categories. 

---

## 7) Primary References

- **REWI baseline paper (foundation):**  
  J. Li, T. Hamann, J. Barth, et al. *Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition.* arXiv:2502.20954, 2025. :contentReference[oaicite:17]{index=17}

- **Project report (part of thesis, completed phase):**  
  *Improving IMU-Based Online Handwriting Recognition: Upgrading from CTC to Attention-based Autoregressive Decoder.* :contentReference[oaicite:18]{index=18}

- **Thesis proposal / scope:**  
  *Alignment-Stable Attention for Writer-Independent IMU Handwriting Recognition via Hybrid CTC–AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation.* :contentReference[oaicite:19]{index=19}

---

## 8) Citation (BibTeX)

```bibtex
@misc{li2025rewi,
  title        = {Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition},
  author       = {Li, Jindong and Hamann, Tim and Barth, Jens and Kaempf, Peter and Zanca, Dario and Eskofier, Bjoern},
  year         = {2025},
  eprint       = {2502.20954},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG}
}

# IMU Handwriting Recognition (REWI / imu-hwr)

This workspace contains a handwriting-recognition codebase for IMU sensor sequences.

Important: the **active codebase and scripts live under** this folder:

- `work/REWI_work/` (this README is for that folder)

It supports multiple decoding setups:

- **CTC** models (e.g. encoder + CTC decoder)
- **CNN → autoregressive Transformer decoder** (AR)
- **CNN → HuggingFace LM decoder** (e.g. `t5-small`, `byt5-*`) for multimodal decoding

## Quickstart

### 1) Environment

This repo is commonly used with a local Conda env already present under `envs/`.

On the cluster (example):

```bash
module load python/3.12-conda
source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate /home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
```

Local (if you create a fresh env):

```bash
conda create -n rewi python=3.10
conda activate rewi
pip install -r work/REWI_work/requirements.txt
```

### 2) Run from the correct working directory

Most scripts assume you run from `work/REWI_work`:

```bash
cd work/REWI_work
python3 main.py -c configs/train_element_word.yaml
```

If you run from the repo root instead, set:

```bash
export PYTHONPATH=work/REWI_work
python3 work/REWI_work/main.py -c work/REWI_work/configs/train_element_word.yaml
```

## Data layout

Datasets live under `data/` and use an MSCOCO-like structure:

- `train.json` / `val.json`
- `data/<fold>/...` with sensor CSVs (semicolon-separated)

Examples:

- `data/onhw_wi_word_rh/`
- `data/onhw_wd_word_rh/`

You select the dataset path via the YAML key `dir_dataset`.

## Core training entrypoints

### Train / evaluate one fold

```bash
cd work/REWI_work
python3 main.py -c configs/train_element_word.yaml
```

Key YAML knobs:

- `idx_fold`: fold index (0–4)
- `dir_dataset`: dataset folder
- `dir_work`: output folder for logs/checkpoints
- `arch_en`, `arch_de`: encoder/decoder architectures

Outputs (per fold) go to:

- `<dir_work>/<idx_fold>/checkpoints/`
- `<dir_work>/<idx_fold>/train_<timestamp>.json` (metrics)
- `<dir_work>/<idx_fold>/train_<timestamp>.log` (logs)

### Cross-validation aggregation

After all folds finish (or after a partial set), aggregate results:

```bash
cd work/REWI_work
python3 evaluate.py -c configs/train_element_word.yaml
```

`evaluate.py` discovers `train_*.json` under `dir_work` recursively.

## Experiments

### A) Multimodal LM decoding (T5 / ByT5)

These runs use `arch_de: "t5-small"` (or `byt5-*`) and train encoder/projection and optionally the LM decoder.

Important knobs:

- `lm_train_lm`: whether LM decoder weights are trainable from epoch 0
- `lm_unfreeze_epoch`: epoch at which the LM decoder becomes trainable
- `lr_enc`, `lr_proj`, `lr_lm`: discriminative learning rates

Example configs (see `configs/`):

- `configs/train-t5-small-0-600.yaml` (unfreeze at 0)
- `configs/train-t5-small-60-300.yaml` (unfreeze at 60)
- `configs/train-t5-small-80-600.yaml` (unfreeze at 80)
- `configs/train-t5-small-200-600.yaml` (unfreeze at 200)
- `configs/train-t5-small_freeze.yaml` (keep LM frozen)

### B) Plot WI curves + summary table (T5-small)

This script reads `train_*.json` under the five WI experiment folders and produces:

- CER/WER curves (mean±std over folds)
- a summary table (Markdown + CSV)

Run:

```bash
python3 tools/plot_wi_t5small_curves.py
```

Outputs:

- `results/hwr2/plots_wi_t5small/wi_t5small_cer_curves.png`
- `results/hwr2/plots_wi_t5small/wi_t5small_wer_curves.png`
- `results/hwr2/plots_wi_t5small/wi_t5small_summary.md`

### C) Text-only pretraining for AR decoder (no-tokenizer / categories vocab)

You can pretrain the AR decoder on a wordlist (teacher-forced next-token prediction):

```bash
cd work/REWI_work
python3 pretrain_decoder.py -c configs/pretrain_decoder.yaml
```

The recommended checkpoint for transfer is typically:

- `<dir_work>/<idx_fold>/checkpoints/best_loss.pth`

### D) CNN → ARDecoder fine-tuning with optional decoder init + freeze ablation

In the AR pipeline, you can initialize `model.decoder` from a pretrained checkpoint:

- `pretrained_decoder_checkpoint: /path/to/best_loss.pth`

And you can freeze the AR decoder for the first N epochs:

- `freeze_decoder_epochs: 0` (train end-to-end)
- `freeze_decoder_epochs: 100` (freeze decoder epochs 0..99)

See example configs:

- `configs/train_element_word.yaml`
- `configs/train_element_word_freeze_100.yaml`
- `configs/train_element_word_freeze_complete.yaml`

## Slurm / cluster workflow

For 5-fold training as an array job, use:

- `slurm/train.sbatch` (CV training)
- `slurm/train_pretrain.sbatch` (decoder-only pretraining)

These scripts copy a base YAML into `$SLURM_TMPDIR` and patch keys like `idx_fold`, `dir_work`, and `dir_dataset`.

## Notes on metrics and reporting

- For plots, prefer `train_*.json` (structured metrics) over parsing `.log`.
- When comparing ablations, ensure the dataset is the same (e.g., WI vs WD).


<!--

ORIGINAL README (kept for reference)

# Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition

This repository is the official implementation of "Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition".

## Introduction

The paper introduces a handwriting recognition model for IMU-based data, leveraging a CNN-BiLSTM architecture. The model is designed to enhance recognition accuracy for unseen writers.

![architecture](figures/architecture.png)

### OnHW Word500 Right-Handed Dataset

- Righthanded writer-dependent/writer-independent OnHW-word500 dataset (WD: writer-dependent; WI: writer-independent)

| model                | WD CER    | WD WER    | WI CER    | WI WER    | #Params   | MACs    |
| -------------------- | --------- | --------- | --------- | --------- | --------- | ------- |
| CLDNN                | 16.18     | 50.98     | 15.62     | 36.71     | 0.75M     | 291M    |
| CNN+BiLSTM *(orig.)* | 17.16     | 51.95     | 27.80     | 60.91     | **0.40M** | 152M    |
| CNN+BiLSTM           | **15.47** | 51.55     | 17.66     | 43.45     | **0.40M** | 152M    |
| Ours-S               | 15.73     | **50.64** | **10.55** | **24.94** | 0.53M     | **79M** |

| model              | WD CER    | WD WER    | WI CER   | WI WER    | #Params   | MACs     |
| ------------------ | --------- | --------- | -------- | --------- | --------- | -------- |
| ResNet *(enc.)*    | **12.72** | **40.76** | 8.22     | 18.35     | 3.97M     | 591M     |
| MLP-Mixer *(enc.)* | 14.74     | 46.89     | 9.74     | 21.87     | 3.90M     | 802M     |
| ViT *(enc.)*       | 17.86     | 52.00     | 10.49    | 22.71     | **3.71M** | **477M** |
| ConvNeXt *(enc.)*  | 14.66     | 45.42     | 8.23     | 18.46     | 3.86M     | 600M     |
| SwinV2 *(enc.)*    | 13.23     | 43.72     | 8.62     | 19.60     | 3.88M     | 601M     |
| Ours               | 14.43     | 43.90     | **7.37** | **15.12** | 3.89M     | 600M     |

## Installation

Create a new conda virtual environment

```bash
conda create -n rewi python
conda activate rewi
```

Clone this repo and install required packages

```bash
git clone https://github.com/jindongli24/REWI.git
pip install -r requirements.txt
```

## Dataset

For commercial reasons, our datasets will not be published. Alternatively, you can use the OnHW public dataset for training and evaluation. In the paper, we use the right-handed writer-independent subset of the OnHW-words500 dataset. To download the dataset, please visit: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

We use a MSCOCO-like structure for the training and evaluation of our dataset. After the OnHW dataset is downloaded, please convert the original dataset to the desired structure with the notebook `notebooks/onhw.ipynb`. Please adjust the variables `dir_raw`, `dir_out`, and `writer_indep` accordingly.

## Training

In the paper, models are trained in a 5-fold cross validation style, which can be done using the `main.py` to train each fold individually. Please adjust the configurations in the `train.yaml` configuation file accordingly. Additionally, as competitor CLDNN is trained with different strategy, please always use the `*_cldnn.*` files for training and evaluation.

```bash
python main.py -c configs/train.yaml
python main_mohamad.py -c configs/train_mohamad.yaml # CLDNN only
```

Alternatively, you can also train all folds at once sequentially with `scripts/others/train_cv.py` (legacy helper). The script will generate configuration files for all folds in a `temp*` directory and run `main.py` with these configuration files sequentially. After the training is finished, the `temp*` directory will be deleted automatically.

```bash
python scripts/others/train_cv.py -c configs/train.yaml -m main.py
```

## Evaluation

As we are using cross validation, the results are already given in the output files of training. However, you can always re-evaluate the model with the configuration and weight you want. In the case, please ajust the `test.yaml` file accordingly and run `main.py` with it.

```bash
python main.py -c configs/test.yaml
```

After you get all results of all folds, you can summarize the results and also calculate the #Params and MACs with `evaluate.py`.

```bash
python evaluate.py -c configs/train.yaml
# or
python evaluate.py -c path_to_config_in_work_dir
```

## License

This project is released under the MIT license. Please see the `LICENSE` file for more information.

-->
