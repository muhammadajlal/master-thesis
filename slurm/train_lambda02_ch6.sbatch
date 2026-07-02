#!/bin/bash
#SBATCH -J ch6-lam02
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p a100,v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-114
# ═══════════════════════════════════════════════════════════════════
# Chapter 6 multimodal sweep at lambda_ctc = 0.2
# Array layout: 115 tasks = 23 configs x 5 folds
#   config_idx = task_id / 5  (1-indexed line in manifest)
#   fold       = task_id % 5
# The 23 cloned configs (already at hybrid_lambda_ctc: 0.2) are listed
# in scripts/_lam02_ch6_manifest.txt (one path per line, relative to
# the REWI_work root).
#
# Auto-resume: each fold checks for a last.pth checkpoint and resumes
# if found, so re-queueing on timeout is safe.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
set -x

TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$((TASK_ID / 5))
FOLD=$((TASK_ID % 5))

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

MANIFEST="$PROJ/scripts/_lam02_ch6_manifest.txt"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest not found: $MANIFEST"
  exit 1
fi

# Manifest is one path per line; pick line (CONFIG_IDX + 1).
TRAIN_YAML_REL=$(sed -n "$((CONFIG_IDX + 1))p" "$MANIFEST")
if [[ -z "$TRAIN_YAML_REL" ]]; then
  echo "ERROR: Manifest has no line $((CONFIG_IDX + 1)) for task $TASK_ID"
  exit 1
fi
TRAIN_YAML="$PROJ/$TRAIN_YAML_REL"
if [[ ! -f "$TRAIN_YAML" ]]; then
  echo "ERROR: Config file not found: $TRAIN_YAML"
  exit 1
fi

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

# Read base dir_work from the cloned config; append fold subdirectory.
BASE_OUT=$($PY_BIN -c "
import yaml, sys
with open('$TRAIN_YAML') as f:
    cfg = yaml.safe_load(f)
print(cfg['dir_work'])
")
if [[ -z "$BASE_OUT" ]]; then
  echo "ERROR: Failed to read dir_work from $TRAIN_YAML"
  exit 1
fi
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/ch6_lam02_t${TASK_ID}.yaml"
cp "$TRAIN_YAML" "$TMPCFG"

# Auto-resume from last.pth if it exists. RunManager writes to
# <dir_work>/<idx>/checkpoints/last.pth where <idx>==FOLD per convention.
CKPT_PATH="${OUTDIR}/${FOLD}/checkpoints/last.pth"
if [[ -f "$CKPT_PATH" ]]; then
  echo "RESUME: found checkpoint at $CKPT_PATH"
  RESUME_CKPT="$CKPT_PATH"
else
  echo "FRESH START: no checkpoint at $CKPT_PATH"
  RESUME_CKPT=""
fi

# Patch idx_fold + dir_work + (optional) resume checkpoint into the
# task-specific YAML. dir_dataset stays as set in the cloned config.
$PY_BIN - <<PY
import yaml

with open("$TMPCFG", "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"] = int("$FOLD")
cfg["dir_work"] = "$OUTDIR"

resume = "$RESUME_CKPT"
cfg["checkpoint"] = resume if resume else None

with open("$TMPCFG", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", "$TMPCFG")
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========== RUN INFO =========="
echo "Task id:             $TASK_ID"
echo "Config idx:          $CONFIG_IDX"
echo "Fold:                $FOLD"
echo "Config YAML:         $TRAIN_YAML_REL"
echo "Patched config:      $TMPCFG"
echo "Work dir (out):      $OUTDIR"
echo "Resume checkpoint:   ${RESUME_CKPT:-NONE}"
echo "=============================="

cd "$PROJ"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "Running task ${TASK_ID} (config_idx=${CONFIG_IDX}, fold=${FOLD})"
$PY_BIN main.py -c "$TMPCFG"

echo "═══ Task $TASK_ID complete ═══"
