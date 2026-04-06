#!/bin/bash
#SBATCH -J O1-H1-sent
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
TRAIN_YAML="O1_sentence_eval/test-O1-H1-sent.yaml"
DATASET="wi_sent_hw6_meta"

# H1 word-level best checkpoints (best_cer.pth per fold)
H1_BASE="/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp/vlm__wi_word_hw6_meta"

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

FOLD=${SLURM_ARRAY_TASK_ID}

# Find best checkpoint for this fold
CKPT="${H1_BASE}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT"
  exit 1
fi

OUTDIR="$WORK/imu-hwr/results/hwr2/Ablations-MMLM/GPT-2/Hybrid/O1_sent_eval_H1/vlm__${DATASET}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/test_${DATASET}_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Patch config
$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"
cfg["checkpoint"]  = "$CKPT"
cfg["test"]        = True

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== O1 EVAL INFO =========="
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "H1 checkpoint:       $CKPT"
echo "Config YAML:         $TRAIN_YAML"
echo "Patched config:      $TMPCFG"
echo "Work dir (out):      $OUTDIR"
echo "==================================="

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

echo "Running sentence-level eval fold ${FOLD} with config: $TMPCFG"
$PY_BIN main.py -c "$TMPCFG"
