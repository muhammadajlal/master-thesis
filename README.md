# Alignment-Stable Attention for IMU Handwriting Recognition via Hybrid CTC--AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation  
**Master’s Thesis Repository — Muhammad Ajlal Khan (FAU Pattern Recognition Lab)**

This repository contains the codebase and experimental artifacts for my Master’s thesis on **writer-independent (WI) online handwriting recognition (HWR) from inertial measurement unit (IMU) time series**. The work builds on the REWI baseline introduced by my supervisor’s paper **“Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition” (arXiv:2502.20954)**.

This codebase supports multiple decoding setups:

- **CNN → CTC decoder** (e.g. encoder + BiLSTM)
- **CNN → autoregressive Transformer decoder** (encoder + Transformer)
- **CNN → HuggingFace LM decoder** (e.g. encoder + `t5-small`, `byt5-small` etc.) for multimodal decoding


---

## Quickstart (Clone → Train → Evaluate)

### 0) Clone

```bash
git clone https://github.com/muhammadajlal/master-thesis.git
cd work/REWI_work
```

### 1) Environment

```bash
conda create -n rewi python=3.10
conda activate rewi
pip install -r work/REWI_work/requirements.txt
```

### 2) Run from the right working directory

Most scripts assume your CWD is `work/REWI_work`:

```bash
cd work/REWI_work
```

If you want to run from the repo root instead, set:

```bash
export PYTHONPATH=work/REWI_work
python3 work/REWI_work/main.py -c work/REWI_work/configs/train_element_word.yaml
```

### 3) Data layout + config

Datasets live under `data/` and use an MSCOCO-like structure:

- `train.json` / `val.json`
- `data/<fold>/...` with sensor CSVs (semicolon-separated)

You select the dataset path via the YAML key `dir_dataset`.

Key YAML knobs:

- `idx_fold`: fold index (0–4)
- `dir_dataset`: dataset folder
- `dir_work`: output folder for logs/checkpoints
- `arch_en`, `arch_de`: encoder/decoder architectures

### 4) Train / evaluate one fold

```bash
cd work/REWI_work
python3 main.py -c configs/train_element_word.yaml
```

Outputs (per fold) go to:

- `<dir_work>/<idx_fold>/checkpoints/`
- `<dir_work>/<idx_fold>/train_<timestamp>.json` (metrics)
- `<dir_work>/<idx_fold>/train_<timestamp>.log` (logs)

### 5) Aggregate cross-validation results

After all folds finish (or after a partial set), aggregate results:

```bash
cd work/REWI_work
python3 evaluate.py -c configs/train_element_word.yaml
```

`evaluate.py` discovers `train_*.json` under `dir_work` recursively.

### 6) Common scripts / experiments

- **Multimodal LM decoding (T5 / ByT5)**
  - Example configs: `configs/train-t5-small-0-600.yaml`, `configs/train-t5-small-60-300.yaml`, ...
  - Key knobs: `lm_train_lm`, `lm_unfreeze_epoch`, `lr_enc`, `lr_proj`, `lr_lm`

- **Plot WI curves + summary table (T5-small)**

```bash
cd work/REWI_work
python3 scripts/tools/plot_wi_t5small_curves.py
```

- **Text-only pretraining for AR decoder (categories vocab)**

```bash
cd work/REWI_work
python3 scripts/others/pretrain_decoder.py -c configs/pretrain_decoder.yaml
```

### 7) Slurm / cluster workflow

For 5-fold training as an array job, use:

- `slurm/train.sbatch` (CV training)
- `slurm/train_pretrain.sbatch` (decoder-only pretraining)

These scripts copy a base YAML into `$SLURM_TMPDIR` and patch keys like `idx_fold`, `dir_work`, and `dir_dataset`.

---

## Reproducibility

- Replication-grade instructions and configs live in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and `configs/experiments/`.
- Download helper for HuggingFace weights: `bash scripts/repro/download_t5_small.sh` (stores under `assets/hf_models/t5-small`).
- **Private data note:** the Stabilo internal dataset (`wi_sent_hw6_meta`) is private and cannot be redistributed; experiments on it are not directly reproducible outside this environment.
- All other reported experiments can be replicated using the OnHW500 datasets (`data/onhw_wi_word_rh`, `data/onhw_wd_word_rh`).

---

## 1) Thesis Focus

IMU-based HWR enables writing on paper (or any surface) without cameras or digitizers, but **WI generalization** is difficult due to:
- large inter-writer variability (speed, stroke dynamics, grip, sensor placement),
- sensor noise and distribution shift,
- alignment instability and decoding-driven failure modes.

**Thesis objective:** improve **both** character-level and word-level recognition quality under realistic efficiency constraints by combining:
- **alignment-stable training**,  
- **calibrated decoding**, and  
- **hybrid CTC–autoregressive (AR) learning** (CTC for monotonic alignment supervision, AR for global conditional modeling).
- **Multi-modal learning** (Pre-trained LM e.g., t5-small to fine tune on our downstream task).

---


## 2) What’s Already Done

In addition, the **Master's project report is treated as a pre-requisite thesis component** and documents the initial research phase, system migration, ablations, and diagnostic analyses.
The project phase established a practical and extensible AR pipeline and produced the main empirical findings below:

### A. Migration: CTC → Attention-based Autoregressive Decoder
A major engineering blocker for Transformer decoding in this domain is **variable-length encoder sequences**. The project implemented **batch-wise rectangularization** (pad to the batch maximum + attention masks), enabling efficient AR training/inference without global fixed-length padding.

### B. Lightweight Attention Stabilization via SDPA Output Gating
Inspired by gated-attention methods, the project introduced **post-SDPA output gating** (headwise and elementwise). This delivered consistent gains across tasks with negligible overhead.

### C. Tokenization Study: Character vs. BPE
BPE tokenization (multiple merge settings) **did not yield consistent improvements** over character decoding in this setting.

### D. Beyond Aggregate Metrics: Tail Risk + Failure Diagnostics
The evaluation moved beyond CER/WER and characterized error structure via:
- **normalized per-sample Levenshtein error distributions** (bimodality + heavy tails),
- **collision analysis** (frequency bias vs. input similarity),
- **writer-stratified error concentration** (collision lift among top writers),
- **qualitative diagnostics** using cross-attention heatmaps + Grad-CAM1D.

### Main Results from Project

All results below are on **internal STABILO WI splits** (word-level + sentence-level) as documented in the project report.

### Word-level Recognition
| Model | CER ↓ | WER ↓ |
|------|------:|------:|
| **REWI baseline (CNN–BiLSTM–CTC)** | 9.39 | 31.81 |
| **Ours AR (CNN–Transformer AR)** | 12.80 | 20.60 |
| **Ours AR + Elementwise Gating** | 10.37 | **17.43** |
| **Ours AR + Headwise Gating** | **10.29** | 17.54 |

**Key takeaway:** AR decoding dramatically reduces **WER** versus CTC, and **gating recovers much of the CER gap while further improving WER**.

### Sentence-level Recognition
| Model | CER ↓ | WER ↓ |
|------|------:|------:|
| **REWI baseline (CNN–BiLSTM–CTC)** | 6.54 | 23.51 |
| **Ours AR (CNN–Transformer AR)** | 11.35 | 15.83 |
| **Ours AR + Elementwise Gating** | 8.68 | 12.30 |
| **Ours AR + Headwise Gating** | 8.68 | **12.25** |

**Key takeaway:** The same pattern holds at sentence-level: **AR improves WER substantially**, and **gating provides strong additional gains**.

### Error Structure (Operationally Important)
- Exact-match mass is large (~**78–79%**), but the remaining errors are **heavy-tailed**.
- Tail risk is non-trivial, especially for sentences (e.g., **P(e > 1)** notably higher than words).
- Collisions disproportionately involve **frequent labels**, and collisions are **concentrated in subsets of writers**, consistent with frequency bias interacting with writer shift.

## Multimodal LM-Decoder Adaptation Experiments (Pretrained Seq2Seq)

### A. Goal: Replace Scratch AR Decoder with a Pretrained LM Decoder
This experiment evaluates whether a **pretrained seq2seq language model decoder** can improve recognition quality (CER/WER) when adapted as a decoder conditioned on IMU-derived encoder tokens—while tracking **accuracy vs. compute** (MACs/Params). The core question is whether pretraining helps **sequence generation** beyond the scratch CNN–ARTransformer decoder.

### B. Conditioning Mechanism: Lightweight Adapter / Projection
The setup conditions a pretrained **T5-small** decoder on the CNN encoder output using a **lightweight projection/adapter**, and fine-tunes the system jointly **from epoch 0**. This keeps the interface minimal and makes results interpretable: any gains should come from leveraging pretrained decoding capacity.

### C. Baseline: My CNN–ARTransformer (Reference System)
All comparisons are made against my in-house baseline:
- **CNN encoder + autoregressive Transformer decoder**
- **batch-wise rectangularization + masking** to handle variable-length IMU sequences

This baseline is also the platform used for thesis work on **hybrid CTC–AR multitask learning** and **decoder calibration**.

### D. Evaluation Protocol: Multiple Domains + Compute Accounting
Experiments are reported on:
- **OnHW500** (Writer-Independent, Writer-Dependent)
- **Internal STABILO sentence-level** split

Metrics:
- **CER/WER**
- **MACs** (compute)
- **Params** (model size)

---

### E. Results: Multimodal LM-Decoder Adaptation

#### OnHW500 (Writer-Independent, WI)
| Model | CER ↓ | WER ↓ | MACs ↓ | Params ↓ |
|------|------:|------:|------:|---------:|
| **CNN–ARTransformer (baseline, no pretrain)** | **6.75** | **10.21** | **430M** | **4.69M** |
| CNN–ARTransformer (pretrained; fine-tuned from epoch 0) | 6.82 | 10.40 | 430M | 4.69M |
| CNN–T5-small decoder (fine-tuned from epoch 0) | 9.32 | 15.38 | 1.16B | 44.34M |

**Key takeaway:** Under the current conditioning setup, **T5-small underperforms** the scratch CNN–ARTransformer baseline while requiring **~2.7× MACs** and **~9.5× parameters**.

---

#### OnHW500 (Writer-Dependent, WD)
| Model | CER ↓ | WER ↓ | MACs ↓ | Params ↓ |
|------|------:|------:|------:|---------:|
| **CNN–ARTransformer (baseline, no pretrain)** | **16.52** | **32.85** | **430M** | **4.69M** |
| CNN–ARTransformer (pretrained; fine-tuned from epoch 0) | 17.03 | 33.74 | 430M | 4.69M |
| CNN–T5-small decoder (fine-tuned from epoch 0) | 32.16 | 57.37 | 1.16B | 44.34M |

**Key takeaway:** The performance gap widens on WD: pretrained LM adaptation **does not transfer effectively** and is substantially worse than the baseline.

---

#### Internal STABILO (Sentence-Level)
| Model | CER ↓ | WER ↓ | MACs ↓ | Params ↓ |
|------|------:|------:|------:|---------:|
| **CNN–ARTransformer (baseline, no pretrain)** | **8.68** | **12.30** | **1.7B** | **4.69M** |
| CNN–ARTransformer (pretrained; fine-tuned from epoch 0) | 10.68 | 14.71 | 1.7B | 4.69M |
| CNN–T5-small decoder (lr = $10^{-3}$; fine-tuned from epoch 0) | 15.57 | 22.57 | 33.74B | 44.34M |

**Key takeaway:** For sentence-level recognition, T5-small adaptation is **both less accurate and significantly more expensive** (MACs increase by an order of magnitude), suggesting the current multimodal interface is not yet leveraging LM pretraining.

---

## Interpretation / What This Suggests
- **Pretrained decoder capacity alone is not sufficient**: without stronger grounding/alignment signals, the LM decoder can become fluent-but-wrong and fails to exploit IMU evidence effectively.
- The **scratch CNN–ARTransformer remains the best accuracy–efficiency tradeoff** under the current setup.
- These results motivate the thesis direction:
  - **Hybrid CTC–AR multitask training** (CTC provides alignment supervision; AR provides context modeling),
  - **Decoder calibration** (beam search, EOS/length control),
  - Potentially **LM fusion / rescoring** rather than full LM-decoder replacement.

---

## Reproducibility
- Configurations, logs, and result tables are documented in this repository (see experiment folders and result summaries).

### Next Steps
- Improve grounding/alignment for LM adaptation via hybrid CTC–AR supervision.
- Evaluate beam search + length/EOS calibration for AR and LM decoders.
- Explore LM fusion/rescoring as a lower-risk way to leverage pretraining.

---

## 3) Repository Contents (Recommended Layout)

This repo is organized so the “active” codebase lives under `work/REWI_work/`.

- **Main entrypoints**
  - `main.py`: train/eval one fold (writes `train_*.json`, checkpoints)
  - `evaluate.py`: aggregates CV runs by discovering `train_*.json` under `dir_work`
  - `pretrain_decoder.py`: decoder-only pretraining (writes `train_*.json`, checkpoints)
- **Core package**
  - `rewi/`: datasets, models, training/manager utilities
- **Experiment configuration**
  - `configs/`: YAML configs for training, testing, and decoder pretraining
- **Utilities and analyses**
  - `scripts/tools/`: plotting + dictionary building (e.g., WI T5-small curves)
- **Cluster workflow**
  - `slurm/`: sbatch scripts (arrays over folds, YAML patching via `$SLURM_TMPDIR`)
- **Assets / docs**
  - `assets/`: dictionaries and other static artifacts
  - `docs/`: thesis/project PDFs

At the repository root:

- `data/`: MSCOCO-like datasets (`train.json`, `val.json`, `data/<fold>/...`)
- `results/`: experiment outputs (as configured by `dir_work`)


---

## 4) Reproducibility and Evaluation Protocol

### Metrics
- **CER / WER** (micro-averaged corpus rates)
- **Normalized per-sample edit error**: `e = d_char / |y|`  
  used to quantify bimodality + tail risk and enable cross-task comparability.

### Diagnostic Analyses
- **Collision analysis:** when a wrong prediction exactly matches another ground-truth label.
- **Writer-stratified reporting:** error and collision concentration (lift) across writers.
- **Qualitative inspection:** cross-attention alignment patterns + Grad-CAM1D.

---

## 5) Thesis Roadmap (Next Milestones)

The thesis plan formalizes the next steps beyond the project phase:

1. **Hybrid multitask training (CTC + AR)**
    - joint objective: `L = L_AR + λ · L_CTC`
    - goal: keep AR’s WER advantage while using CTC to stabilize monotonic alignment and improve CER.

2. **Multi-modal training (Encoder + Pre-trained LM Decoder)**
    - domain adaption
    - goal: use the language prior of the pre-trained LM and fine tune it our IMU based task.

3. **Calibrated decoding (beam search + EOS/length control)**
    - beam search, length normalization, EOS bias/min-length constraints
    - optional coverage/skip penalties to reduce attention collapse and catastrophic tails.

4. **Optional LM-based decoding extensions**
    - lightweight external LM fusion/rescoring (if sufficient transcripts exist)
    - or pretrained LM decoder adaptation under strict latency/parameter budgets.

5. **Robustness workstream**
    - writer-balanced sampling / reweighting
    - tail-risk–aware evaluation as a first-class objective
    - systematic linkage between qualitative failure signatures and recurring error categories.

---

## 7) Primary References

- **REWI baseline paper (foundation):**  
Li, J., Hamann, T., Barth, J., Kämpf, P., Zanca, D., Eskofier, B. (2026). Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition. In: Durmaz Incel, Ö., Qin, J., Bieber, G., Kuijper, A. (eds) Sensor-Based Activity Recognition and Artificial Intelligence. iWOAR 2025. Lecture Notes in Computer Science, vol 16292. Springer, Cham. https://doi.org/10.1007/978-3-032-13312-0_16

- **Master's Project report (part of thesis, completed phase):**  
  *Improving IMU-Based Online Handwriting Recognition: Upgrading from CTC to Attention-based Autoregressive Decoder.* [docs/Master-Project-Report.pdf](docs/Master-Project-Report.pdf)

- **Thesis proposal / scope:**  
  *Alignment-Stable Attention for Writer-Independent IMU Handwriting Recognition via Hybrid CTC–AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation.* [docs/Master-Thesis-Proposal.pdf](docs/Master-Thesis-Proposal.pdf)

---


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
