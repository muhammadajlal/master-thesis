# IMU Handwriting Recognition (IMU-HWR)

**Writer-Independent Online Handwriting Recognition from Inertial Measurement Unit (IMU) Time Series**

This repository implements multiple neural network architectures for handwriting recognition from IMU sensor data, supporting:
- **CTC decoding** (CNN encoder + BiLSTM/Transformer)
- **Autoregressive (AR) decoding** (CNN encoder + Transformer decoder)
- **Multimodal LM decoding** (CNN encoder + HuggingFace T5/ByT5)

---

## 📁 Project Structure

```
work/REWI_work/
├── main.py                 # Main training/evaluation entry point
├── evaluate.py             # Cross-validation aggregation
├── pretrain_decoder.py     # Text-only decoder pretraining
│
├── configs/                # YAML configuration files
│   ├── train.yaml          # Base training config
│   ├── test.yaml           # Evaluation config
│   └── ...
│
├── rewi/                   # Core library
│   ├── model/              # Neural network architectures
│   │   ├── __init__.py     # BaseModel, build_encoder
│   │   ├── conv.py         # CNN encoders (ConvNeXt, etc.)
│   │   ├── transformer.py  # Transformer decoder
│   │   ├── ARDecoder.py    # Autoregressive decoder
│   │   └── multimodal_lm_model.py  # HuggingFace LM integration
│   │
│   ├── dataset/            # Data loading
│   │   ├── __init__.py     # HRDataset
│   │   ├── utils.py        # Collate functions
│   │   └── lm_collate.py   # LM-specific collation
│   │
│   ├── training/           # Training loops and utilities
│   │   ├── __init__.py
│   │   ├── loops.py        # train_one_epoch, test functions
│   │   └── utils.py        # Freeze/unfreeze, optimizer helpers
│   │
│   ├── analysis/           # Qualitative & quantitative analysis
│   │   ├── __init__.py
│   │   ├── attention.py    # Cross-attention visualization
│   │   ├── gradcam.py      # Grad-CAM 1D for encoder
│   │   ├── selection.py    # Sample selection by quantiles
│   │   └── metrics.py      # Levenshtein distance, CER
│   │
│   ├── tokenizer/          # Text tokenization
│   │   ├── __init__.py
│   │   ├── base.py         # BaseTokenizer interface
│   │   ├── bpe.py          # SentencePiece BPE tokenizer
│   │   ├── char.py         # Character-level tokenizer
│   │   └── utils.py        # Text normalization
│   │
│   ├── evaluate.py         # Evaluation metrics (CER, WER)
│   ├── visualize.py        # Result visualization
│   ├── manager.py          # RunManager for logging/checkpointing
│   ├── loss.py             # CTC loss wrapper
│   └── ctc_decoder.py      # CTC best-path decoder
│
├── analysis/               # Analysis scripts and outputs
│   ├── scripts/            # Quantitative analysis scripts
│   └── notebooks/          # Jupyter notebooks
│
├── scripts/                # Utility scripts
│   └── tools/              # Plotting, preprocessing tools
│
└── slurm/                  # Cluster job scripts
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n rewi python=3.10
conda activate rewi
pip install -r requirements.txt
```

### 2. Data Layout

Datasets follow MSCOCO-like structure under `data/`:
```
data/
└── wi_word_hw6_meta/
    ├── train.json          # Training annotations (by fold)
    ├── val.json            # Validation annotations (by fold)
    └── data/               # Sensor CSV files
```

### 3. Training

```bash
cd work/REWI_work
python main.py -c configs/train.yaml
```

### 4. Evaluation

```bash
# Single fold evaluation
python main.py -c configs/test.yaml

# Aggregate cross-validation results
python evaluate.py -c configs/train.yaml
```

---

## ⚙️ Configuration

Key YAML parameters:

| Parameter | Description |
|-----------|-------------|
| `idx_fold` | Fold index (0-4, or -1 for all) |
| `dir_dataset` | Path to dataset folder |
| `dir_work` | Output directory for checkpoints/logs |
| `arch_en` | Encoder architecture (e.g., `blconv_b`) |
| `arch_de` | Decoder architecture (e.g., `transformer_s`, `ar_transformer_s`, `t5-small`) |
| `lr` | Learning rate |
| `epoch` | Number of training epochs |
| `size_batch` | Batch size |

### Training Modes

**CTC Mode** (`arch_de: transformer_s`):
```yaml
arch_en: blconv_b
arch_de: transformer_s
```

**AR Mode** (`arch_de: ar_transformer_*`):
```yaml
arch_en: blconv_b
arch_de: ar_transformer_s
use_gated_attention: true
gating_type: elementwise
```

**LM Mode** (`arch_de: t5-small` or `byt5_small`):
```yaml
arch_en: blconv_b
arch_de: t5-small
lm_name: google/t5-v1_1-small
lm_train_lm: false
lm_unfreeze_epoch: 60
lr_enc: 1e-4
lr_proj: 1e-4
lr_lm: 1e-5
```

---

## 📊 Analysis Tools

### Qualitative Analysis

Enable in test config:
```yaml
test: true
qualitative: true
qual_csv: analysis/quant_all_val_predictions.csv
qual_outdir: qualitative_outputs
qual_use_gradcam: true
```

This generates:
- Cross-attention heatmaps
- Grad-CAM 1D visualizations
- Sample selection by error quantiles (correct, near-miss, catastrophic)

### Quantitative Analysis

```bash
# Generate unified predictions CSV
python analysis/scripts/quant_analysis.py

# Analyze by error quantiles
python analysis/scripts/quant_analysis_2.py
```

---

## 🔧 Extending the Codebase

### Adding a New Encoder

1. Create `rewi/model/my_encoder.py`
2. Register in `rewi/model/__init__.py`:
   ```python
   ENCODERS['my_encoder'] = MyEncoder
   ```

### Adding a New Tokenizer

1. Inherit from `rewi.tokenizer.BaseTokenizer`
2. Implement `encode()`, `decode()`, and special token properties
3. Register in `rewi/tokenizer/__init__.py`

### Adding Analysis Methods

1. Add to appropriate module in `rewi/analysis/`
2. Export in `rewi/analysis/__init__.py`
3. Integrate in `rewi/training/loops.py` if needed during evaluation

---

## 📚 Module Overview

### `rewi.model`
Neural network architectures including CNN encoders (ConvNeXt-style blocks), Transformer decoders, autoregressive decoders with gated attention, and multimodal LM integration.

### `rewi.dataset`
Data loading utilities for IMU time series with support for augmentation, sequence padding, and CTC length constraints.

### `rewi.training`
Training and evaluation loops for CTC, AR, and LM modes with utilities for parameter freezing, optimizer debugging, and checkpoint management.

### `rewi.analysis`
Tools for model interpretability including attention visualization, Grad-CAM for temporal saliency, and error analysis by quantiles.

### `rewi.tokenizer`
Text tokenization with consistent API for BPE (SentencePiece) and character-level encoding. Supports vocabulary building and serialization.

---

## 📝 Citation

If you use this code, please cite:
```bibtex
@article{rewi2025,
  title={Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition},
  author={...},
  journal={arXiv preprint arXiv:2502.20954},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.
