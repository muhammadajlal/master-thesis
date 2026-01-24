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
cd master-thesis
```


### 1) Environment

```bash
conda create -n rewi python=3.10
conda activate rewi
pip install -r requirements.txt
```

### 2) Run from the right working directory

Most scripts assume your CWD is the repository root (this folder).

If you want to run from outside this folder, set `PYTHONPATH` to point to the repo root:

```bash
export PYTHONPATH=/path/to/master-thesis
python3 /path/to/master-thesis/main.py -c /path/to/master-thesis/configs/train_element_word.yaml
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
python3 main.py -c configs/train_element_word.yaml
```

Outputs (per fold) go to:

- `<dir_work>/<idx_fold>/checkpoints/`
- `<dir_work>/<idx_fold>/train_<timestamp>.json` (metrics)
- `<dir_work>/<idx_fold>/train_<timestamp>.log` (logs)

### 5) Aggregate cross-validation results

After all folds finish (or after a partial set), aggregate results:

```bash
python3 evaluate.py -c configs/train_element_word.yaml
```

`evaluate.py` discovers `train_*.json` under `dir_work` recursively.

### 6) Common scripts / experiments

- **Multimodal LM decoding (T5 / ByT5)**
  - Example configs: `configs/train-t5-small-0-600.yaml`, `configs/train-t5-small-60-300.yaml`, ...
  - Key knobs: `lm_train_lm`, `lm_unfreeze_epoch`, `lr_enc`, `lr_proj`, `lr_lm`

- **Plot WI curves + summary table (T5-small)**

```bash
python3 scripts/tools/plot_wi_t5small_curves.py
```

- **Text-only pretraining for AR decoder (categories vocab)**

```bash
python3 scripts/others/pretrain_decoder.py -c configs/pretrain_decoder.yaml
```

### 7) Slurm / cluster workflow

For 5-fold training as an array job, use:

- `slurm/train.sbatch` (CV training)
- `slurm/train_pretrain.sbatch` (decoder-only pretraining)

These scripts copy a base YAML into `$SLURM_TMPDIR` and patch keys like `idx_fold`, `dir_work`, and `dir_dataset`.

---

## Reproducibility

Replication-grade instructions live in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
That document covers dataset availability, exact YAMLs, the one-command reproduction script, and offline-first HuggingFace weight handling.

Quick notes:
- **Private data note:** the Stabilo internal dataset (`wi_sent_hw6_meta`) is private and cannot be redistributed.
- **Public replication:** OnHW500 experiments can be replicated using `data/onhw_wi_word_rh` and `data/onhw_wd_word_rh`.


---

## Thesis context (short)

This thesis targets writer-independent IMU handwriting recognition and studies:
- alignment-stable attention / decoding,
- hybrid CTC–AR training,
- adapting pretrained LM decoders (e.g., T5/ByT5) for multimodal decoding.

For thesis-ready tables/figures and protocol text, see:
- [docs/results.tex](docs/results.tex)
- [docs/Master-Project-Report.pdf](docs/Master-Project-Report.pdf)
- [docs/Master-Thesis-Proposal.pdf](docs/Master-Thesis-Proposal.pdf)

## Repository layout

- Entrypoints: `main.py` (train/eval one fold), `evaluate.py` (aggregate CV), `scripts/others/pretrain_decoder.py` (decoder-only pretraining)
- Core package: `rewi/` (datasets, models, training utilities)
- Configs: `configs/` (training/testing/pretraining YAMLs)
- Utilities: `scripts/tools/` (plots, dictionary builders)
- Cluster: `slurm/` (sbatch scripts)

## References

- **REWI baseline paper (foundation):**  
Li, J., Hamann, T., Barth, J., Kämpf, P., Zanca, D., Eskofier, B. (2026). Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition. In: Durmaz Incel, Ö., Qin, J., Bieber, G., Kuijper, A. (eds) Sensor-Based Activity Recognition and Artificial Intelligence. iWOAR 2025. Lecture Notes in Computer Science, vol 16292. Springer, Cham. https://doi.org/10.1007/978-3-032-13312-0_16

- **Master's Project report (part of thesis, completed phase):**  
  *Improving IMU-Based Online Handwriting Recognition: Upgrading from CTC to Attention-based Autoregressive Decoder.* [docs/Master-Project-Report.pdf](docs/Master-Project-Report.pdf)

- **Thesis proposal / scope:**  
  *Alignment-Stable Attention for Writer-Independent IMU Handwriting Recognition via Hybrid CTC–AR Training, Calibrated Decoding, and Pretrained LM Decoder Adaptation.* [docs/Master-Thesis-Proposal.pdf](docs/Master-Thesis-Proposal.pdf)

---
