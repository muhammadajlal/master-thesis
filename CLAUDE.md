# CLAUDE.md — REWI_work (Main Codebase)

## Quick Orientation

IMU handwriting recognition codebase. CNN encoder extracts features from 13-channel IMU time series, then one of several decoder heads produces text.

**Entry points:** `main.py` (train/eval), `evaluate.py` (CV aggregation), `pretrain_decoder.py` (text-only AR pretraining)

## Architecture Overview

```
IMU CSV (T x 13) → CNN Encoder → [Decoder Head] → Text
```

### Supported Decoder Modes

| Mode | `arch_de` value | Model class | Training loop | Loss |
|------|----------------|-------------|---------------|------|
| CTC | `bilstm_wide`, `linear` | `BaseModel` | `train_one_epoch` / `test` | CTC |
| AR | `ar_transformer_s/m` | `BaseModel` | `train_one_epoch` / `test` | CrossEntropy |
| Hybrid CTC-AR | any AR + `dual_head.enabled: true` | `DualHeadModel` | `train_one_epoch_hybrid` / `test_hybrid` | CTC + CE |
| LM (T5/ByT5) | `t5-small`, `byt5-small` | `MultimodalLMModel` | `train_one_epoch_lm` / `test_lm` | LM loss |
| VLM (GPT-2) | `vlm` + `vlm_enabled: true` | `VLMModel` | `train_one_epoch_lm` / `test_lm` | LM loss |
| Hybrid CTC-VLM | `vlm` + `dual_head` + `vlm_enabled` | `DualHeadModel` wrapping VLM | `train_one_epoch_lm_hybrid` / `test_lm_hybrid` | CTC + LM |

### Key Model Files

- `rewi/model/builders.py` — **factory functions** `build_encoder()`, `build_decoder()` that map `arch_en`/`arch_de` strings to classes
- `rewi/model/base_model.py` — `BaseModel`: encoder + decoder pipeline
- `rewi/model/dual_head.py` — `DualHeadModel`: adds CTC head to any AR/LM model
- `rewi/model/conv.py` — CNN encoders (`blconv_a`, `blconv_b`, etc.)
- `rewi/model/ARDecoder.py` — Transformer AR decoder with cross-attention
- `rewi/model/multimodal_lm_model.py` — encoder → linear projection → HF LM
- `rewi/model/vlm_model.py` — encoder → Q-Former → soft prompt → GPT-2 (BLIP-2 style)
- `rewi/model/qformer.py` — Q-Former cross-attention connector
- `rewi/model/projectors.py` — projection/connector layer variants

## Configuration System

**YAML-driven.** Config loaded as `argparse.Namespace` via:
```python
cfgs = argparse.Namespace(**yaml.safe_load(open(args.config)))
```

### Critical Config Keys

```yaml
# Architecture
arch_en: blconv_b                    # Encoder
arch_de: ar_transformer_s            # Decoder

# Paths
dir_dataset: /path/to/data/dataset   # Dataset root
dir_work: /path/to/results/exp_name  # Output root

# Training
idx_fold: 0          # Fold (0-4, or -1 for all)
epoch: 300           # Total epochs
lr: 0.001            # Learning rate
size_batch: 64       # Batch size
num_channel: 13      # IMU channels
seed: 42

# Hybrid CTC-AR
dual_head:
  enabled: true
  lambda_ar: 1.0
  lambda_ctc: 0.5

# LM mode
lm_name: /path/to/hf_models/t5-small
lm_train_lm: false
lm_unfreeze_epoch: 60

# VLM mode
vlm_enabled: true
vlm:
  lm_name: /path/to/hf_models/gpt2
  num_queries: 16
  qformer_layers: 2
  num_soft_tokens: 20
```

## Data Format

```
data/<dataset>/
├── train.json   # {"annotations": {"0": [{label, filename, id_writer}, ...], ...}}
├── val.json
└── data/*.csv   # Semicolon-delimited, 13 columns, variable rows (timesteps)
```

- `HRDataset` in `rewi/dataset/__init__.py` handles loading
- Augmentations in `rewi/dataset/transforms.py`: AddNoise, Drift, Dropout, TimeWarp (25% each)
- Collation: `rewi/dataset/utils.py` (CTC/AR), `rewi/dataset/lm_collate.py` (LM/VLM)

## Training Flow (main.py)

1. Load YAML config → `argparse.Namespace`
2. Build tokenizer (char or BPE)
3. Build model via factory (`build_encoder` + `build_decoder` or specialized model class)
4. Create `RunManager` for logging/checkpointing
5. Training loop dispatched by mode:
   - CTC/AR → `train_one_epoch()` + `test()`
   - Hybrid → `train_one_epoch_hybrid()` + `test_hybrid()`
   - LM/VLM → `train_one_epoch_lm()` + `test_lm()`
   - Hybrid+LM → `train_one_epoch_lm_hybrid()` + `test_lm_hybrid()`
6. Best model tracked by CER on validation set
7. Results saved to `train_<timestamp>.json`

## How to Add / Modify Things

### New encoder
1. Add class in `rewi/model/` (or existing file)
2. Register in `rewi/model/builders.py` → `build_encoder()` factory

### New decoder
1. Add class in `rewi/model/`
2. Register in `rewi/model/builders.py` → `build_decoder()` factory

### New training mode
1. Add loop functions in `rewi/training/loops.py`
2. Wire dispatch in `main.py`

### New config option
1. Add to YAML with a sensible default
2. Access via `cfgs.new_option` (use `getattr(cfgs, 'new_option', default)` for backward compat)

## Tokenization

Two tokenizer backends with a common API (`encode`, `decode`, special token properties):
- **CharTokenizer** (`rewi/tokenizer/char.py`): uses `categories` list from YAML. Index 0 = CTC blank.
- **BPETokenizer** (`rewi/tokenizer/bpe.py`): SentencePiece model at `tokenizer/bpe100.model`

Special tokens: PAD (0), BOS, EOS — indices vary by tokenizer.

## Evaluation & Metrics

- **CER/WER** computed in `rewi/evaluate.py` and `rewi/analysis/metrics.py`
- `evaluate.py` (top-level) globs `**/train_*.json` recursively under `dir_work` to aggregate folds
- Best epoch per fold: `result['best']['character_error_rate'][0]` in JSON

## SLURM Workflow

Sbatch scripts in `slurm/` handle:
1. Copy YAML to `$SLURM_TMPDIR`
2. Patch `idx_fold` from `$SLURM_ARRAY_TASK_ID`
3. Patch `dir_work` to include `fold_<k>` subdirectory
4. Set `HF_HUB_OFFLINE=1`
5. Run `main.py`

## Important Patterns

- **Offline HuggingFace:** Models in `assets/hf_models/` — always use local paths on cluster
- **`getattr` for optional configs:** Many newer features use `getattr(cfgs, 'key', default)` for backward compatibility
- **Gradient clipping:** Applied in training loops
- **Deterministic seeds:** `seed_everything(42)` + `seed_worker` for DataLoader
- **Logging:** `loguru` throughout — `from loguru import logger`
- **No test suite:** Validation is via training loop eval and `evaluate.py` aggregation
