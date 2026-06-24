#!/bin/bash
#SBATCH -J H1-lam-sweep
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p a100,v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-44

# H1 Hybrid MLP lambda_ctc sweep on OnHW WI word.
# Array layout: 45 tasks = 9 lambda values x 5 folds.
#   lambda_idx = task_id / 5  (in {0..8} -> LAMBDA_LIST)
#   fold       = task_id % 5
# lambda=0.6 is intentionally excluded; reuse existing run under
# results/hwr2/Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/...

set -euo pipefail
set -x

LAMBDA_LIST=(01 02 03 04 05 07 08 09 10)

TASK_ID=${SLURM_ARRAY_TASK_ID}
LAM_IDX=$((TASK_ID / 5))
FOLD=$((TASK_ID % 5))
LAM_TAG=${LAMBDA_LIST[$LAM_IDX]}

DATASET=onhw_wi_word_rh
TRAIN_YAML="configs/H1_hybrid_ctc_vlm/lambda_sweep/train-H1-mlp-onhw-lam${LAM_TAG}.yaml"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${DATASET}

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/${TRAIN_YAML}"
if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: Config file not found: $YAML_PATH"
  exit 1
fi

# Base output: results/hwr2/H1_LambdaSweep/H1_hybrid_mlp_lamNN__<dataset>/fold_K
BASE_OUT=$WORK/imu-hwr/results/hwr2/H1_LambdaSweep/H1_hybrid_mlp_lam${LAM_TAG}__${DATASET}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/H1_lam${LAM_TAG}_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Auto-resume
CKPT_PATH="${OUTDIR}/${FOLD}/checkpoints/last.pth"
if [[ -f "$CKPT_PATH" ]]; then
  echo "RESUME: Found checkpoint at $CKPT_PATH"
  RESUME_CKPT="$CKPT_PATH"
else
  echo "FRESH START: No checkpoint found at $CKPT_PATH"
  RESUME_CKPT=""
fi

# Patch fold + dir_work (+ resume) into the lambda-specific YAML
$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"

resume_ckpt = "$RESUME_CKPT"
cfg["checkpoint"] = resume_ckpt if resume_ckpt else None

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== RUN INFO =========="
echo "Task id:             $TASK_ID"
echo "Lambda tag:          $LAM_TAG"
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

echo "Running lambda=${LAM_TAG} fold=${FOLD} with config: $TMPCFG"
$PY_BIN main.py -c "$TMPCFG"
