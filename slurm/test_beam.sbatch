#!/bin/bash
#SBATCH -J rewi-beam
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
# Which trained model results to evaluate (point to your training output)
# Options: multimodal_t5_small_FreezeLM, multimodal_t5_small_half, multimodal_t5_small
TRAIN_RUN=${TRAIN_RUN:-multimodal_t5_small_FreezeLM}

# Which beam config to use: test-t5-small_beam4.yaml or test-t5-small_beam4_lp08.yaml
BEAM_CFG=${BEAM_CFG:-test-t5-small_beam4.yaml}

# Length penalty suffix for output folder (derived from BEAM_CFG)
if [[ "$BEAM_CFG" == *"lp08"* ]]; then
  LP_SUFFIX="_lp08"
else
  LP_SUFFIX=""
fi

DATASET=${DATASET:-onhw_wi_word_rh}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
DATA=$WORK/imu-hwr/data/${DATASET}

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

FOLD=${SLURM_ARRAY_TASK_ID}

# Path to training results (where checkpoints are)
TRAIN_BASE=$WORK/imu-hwr/results/hwr2/${TRAIN_RUN}/t5-small__${DATASET}
TRAIN_FOLD_DIR="${TRAIN_BASE}/fold_${FOLD}/${FOLD}"

# Find best checkpoint (prefer best_cer.pth, fallback to best_wer.pth or last epoch)
if [[ -f "${TRAIN_FOLD_DIR}/checkpoints/best_cer.pth" ]]; then
  CKPT="${TRAIN_FOLD_DIR}/checkpoints/best_cer.pth"
elif [[ -f "${TRAIN_FOLD_DIR}/checkpoints/best_wer.pth" ]]; then
  CKPT="${TRAIN_FOLD_DIR}/checkpoints/best_wer.pth"
else
  # Find the highest epoch checkpoint
  CKPT=$(ls -v "${TRAIN_FOLD_DIR}/checkpoints/"*.pth 2>/dev/null | tail -1 || echo "")
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: No checkpoint found for fold $FOLD in ${TRAIN_FOLD_DIR}/checkpoints/"
  exit 1
fi

# Output directory for beam search evaluation
OUTDIR=$WORK/imu-hwr/results/hwr2/${TRAIN_RUN}_beam4${LP_SUFFIX}_eval/t5-small__${DATASET}/fold_${FOLD}
mkdir -p "$OUTDIR"

# Create temp config
TMPCFG="${SLURM_TMPDIR:-/tmp}/test_beam_${DATASET}_fold${FOLD}.yaml"
cp "$PROJ/configs/${BEAM_CFG}" "$TMPCFG"

$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"
cfg["checkpoint"]  = "$CKPT"

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========== BEAM SEARCH EVAL =========="
echo "Fold:           $FOLD"
echo "Train run:      $TRAIN_RUN"
echo "Beam config:    $BEAM_CFG"
echo "Checkpoint:     $CKPT"
echo "Output dir:     $OUTDIR"
echo "======================================="

cd "$PROJ"
$PY_BIN -V
nvidia-smi -L || true

echo "Running beam search evaluation for fold ${FOLD}"
$PY_BIN main.py -c "$TMPCFG"

echo "Done. Results in: $OUTDIR"
