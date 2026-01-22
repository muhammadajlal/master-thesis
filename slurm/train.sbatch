#!/bin/bash
#SBATCH -J rewi-cv
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4


set -euo pipefail
set -x

# ---- Params you can tweak ----
# Specify the YAML config file to use (path relative to configs/)
TRAIN_YAML=${TRAIN_YAML:-train_element_sent.yaml}

# Dataset name (can still be overridden)
DATASET=${DATASET:-wi_sent_hw6_meta}  # onhw_wi_word_rh | wi_word_hw6_meta etc.

# Toggle tokenizer/BPE usage: "true"/"false" or "1"/"0"
USE_BPE=${USE_BPE:-false}

# BPE size only matters if USE_BPE=true
BPE_SIZE=${BPE_SIZE:-500}                # 100|200|500 etc.

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${DATASET}

# Setup conda before reading config
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

# Read metadata from the specified YAML
YAML_PATH="$PROJ/configs/$TRAIN_YAML"
if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: Config file not found: $YAML_PATH"
  exit 1
fi

# Extract arch_de and dir_work from YAML
read -r ARCH_DE BASE_DIR_WORK <<< $(
$PY_BIN - "$YAML_PATH" <<'PY'
import yaml, os, sys
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
arch_de = cfg.get("arch_de", "unknown")
dir_work = cfg.get("dir_work", "")
# Extract base output name from dir_work if present
# e.g., /path/to/results/hwr2/multimodal_t5_small/... -> multimodal_t5_small
if dir_work:
    parts = dir_work.split("/")
    # Find 'hwr2' and take next part
    try:
        idx = parts.index("hwr2")
        base_name = parts[idx + 1] if idx + 1 < len(parts) else "train_run"
    except (ValueError, IndexError):
        base_name = "train_run"
else:
    base_name = "train_run"
print(arch_de, base_name)
PY
)

# Results folder: encode whether tokenizer is used or not + arch_de from YAML
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  BASE_OUT=$WORK/imu-hwr/results/hwr2/blconv_${ARCH_DE}_tokenizer_${BPE_SIZE}/${ARCH_DE}__${DATASET}
else
  BASE_OUT=$WORK/imu-hwr/results/hwr2/${BASE_DIR_WORK}/${ARCH_DE}__${DATASET}
fi

FOLD=${SLURM_ARRAY_TASK_ID}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

# Always define BPE_MODEL so set -u doesn't complain
BPE_MODEL=""

# --- Only define / check BPE model if we actually use it ---
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  BPE_DIR="$PROJ/tokenizer"
  BPE_PREFIX="bpe${BPE_SIZE}_fold_${FOLD}"
  BPE_MODEL="${BPE_DIR}/${BPE_PREFIX}.model"

  if [[ ! -f "$BPE_MODEL" ]]; then
    echo "ERROR: Missing tokenizer model: $BPE_MODEL"
    exit 1
  fi
fi

TMPCFG="${SLURM_TMPDIR:-/tmp}/train_${DATASET}_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"

use_bpe = "$USE_BPE".lower() in ("1", "true", "yes")
if use_bpe:
    tok = cfg.get("tokenizer", {})
    tok["type"]       = "sentencepiece"
    tok["model"]      = "$BPE_MODEL"
    tok["vocab_size"] = int("$BPE_SIZE")
    cfg["tokenizer"]  = tok
    cfg["use_bpe"]    = True

# Force local path for ByT5 (offline)
if str(cfg.get("arch_de","")).startswith("byt5"):
    cfg["lm_name"] = "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/assets/hf_models/byt5-small"
# Force local path for T5 (offline)
if str(cfg.get("arch_de","")) == "t5-small":
    cfg["lm_name"] = "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/assets/hf_models/t5-small"

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== RUN INFO =========="
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "Config YAML:         $TRAIN_YAML"
echo "Patched config:      $TMPCFG"
echo "Work dir (out):      $OUTDIR"
echo "Decoder arch (from YAML): $ARCH_DE"
echo "USE_BPE:             $USE_BPE"
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  echo "SentencePiece BPE:   ENABLED (size=${BPE_SIZE})"
  echo "BPE model path:      ${BPE_MODEL}"
else
  echo "SentencePiece BPE:   DISABLED (LM mode uses pretrained tokenizer internally if arch_de=byt5*/t5-small)"
fi
echo "=============================="

cd "$PROJ"
which $PY_BIN
$PY_BIN -V
nvidia-smi -L || true

export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running fold ${FOLD} with config: $TMPCFG"
$PY_BIN main.py -c "$TMPCFG"