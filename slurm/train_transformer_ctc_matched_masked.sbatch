#!/bin/bash
#SBATCH -J ctc-xs-masked
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
TRAIN_YAML="AR-Baseline/train-transformer-xs-ctc-onhw-word-matched-masked.yaml"
DATASET="onhw_wi_word_rh"

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

BASE_OUT=$WORK/imu-hwr/results/hwr2/Baseline-Transformer-XS-CTC-Matched-Masked/transformer_xs__${DATASET}
FOLD=${SLURM_ARRAY_TASK_ID}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/train_ctc_masked_${DATASET}_fold${FOLD}.yaml"
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
cfg["num_worker"]  = 8

resume_ckpt = "$RESUME_CKPT"
cfg["checkpoint"] = resume_ckpt if resume_ckpt else None

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== RUN INFO =========="
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "Config YAML:         $TRAIN_YAML"
echo "Work dir (out):      $OUTDIR"
echo "Resume checkpoint:   ${RESUME_CKPT:-NONE}"
echo "=============================="

cd "$PROJ"
nvidia-smi -L

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

$PY_BIN main.py -c "$TMPCFG"
