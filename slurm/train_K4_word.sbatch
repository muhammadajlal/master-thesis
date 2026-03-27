#!/bin/bash
#SBATCH -J K4-sea-word
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
TRAIN_YAML="K4_sea_contrastive/train-K4-sea-word.yaml"
DATASET="wi_word_hw6_meta"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${DATASET}

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/configs/$TRAIN_YAML"
if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: Config file not found: $YAML_PATH"
  exit 1
fi

# Extract arch_de and dir_work from YAML
read -r ARCH_DE BASE_DIR_WORK <<< $(
$PY_BIN - "$YAML_PATH" <<'PY'
import yaml, sys
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
arch_de = cfg.get("arch_de", "unknown")
dir_work = cfg.get("dir_work", "")
if dir_work:
    parts = dir_work.split("/")
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

BASE_OUT=$WORK/imu-hwr/results/hwr2/${BASE_DIR_WORK}/${ARCH_DE}__${DATASET}
FOLD=${SLURM_ARRAY_TASK_ID}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/train_${DATASET}_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# ---- Auto-resume: check for last.pth ----
CKPT_PATH="${OUTDIR}/${FOLD}/checkpoints/last.pth"
if [[ -f "$CKPT_PATH" ]]; then
  echo "RESUME: Found checkpoint at $CKPT_PATH"
  RESUME_CKPT="$CKPT_PATH"
else
  echo "FRESH START: No checkpoint found at $CKPT_PATH"
  RESUME_CKPT=""
fi

# Patch config
$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"

# Auto-resume
resume_ckpt = "$RESUME_CKPT"
if resume_ckpt:
    cfg["checkpoint"] = resume_ckpt
else:
    cfg["checkpoint"] = None

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
echo "Resume checkpoint:   ${RESUME_CKPT:-NONE}"
echo "=============================="

cd "$PROJ"
which $PY_BIN
$PY_BIN -V

# ========== GPU VERIFICATION ==========
echo "Verifying GPU allocation..."
nvidia-smi -L
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$GPU_COUNT" -eq 0 ]]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi
echo "GPU verification passed. Found $GPU_COUNT GPU(s)."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "======================================"

export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF mode: OFFLINE"

echo "Running fold ${FOLD} with config: $TMPCFG"
$PY_BIN main.py -c "$TMPCFG"
