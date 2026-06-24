#!/bin/bash
#SBATCH -J single-corruption
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p a100,v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --array=0-39

# Single-corruption recovery profile on the XS HWRFormer checkpoints.
# Array layout: 40 tasks = 2 models x 4 datasets x 5 folds.
#   model_idx = task_id / 20       (0=AR-only, 1=noise)
#   dataset_idx = (task_id % 20) / 5
#   fold = task_id % 5
# Outputs land beside each fold's checkpoint as
#   <fold_dir>/eval_single_corruption.json
# Aggregated downstream by analysis/scripts/plot_single_corruption_recovery.py.

set -uo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RES=$WORK/imu-hwr/results/hwr2
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

MODELS=(
  "AR-only|Baseline-AR-XS-blconv_b"
  "noise|Baseline-AR-XS-InputCorruption-uniform"
)
DATASETS=(
  "ar_transformer_xs__onhw_wi_word_rh"
  "ar_transformer_xs__onhw_wd_word_rh"
  "ar_transformer_xs__wi_word_hw6_meta"
  "ar_transformer_xs__wi_sent_hw6_meta"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 20))
DATASET_IDX=$(( (TASK_ID % 20) / 5 ))
FOLD=$((TASK_ID % 5))

MODEL_ENTRY="${MODELS[$MODEL_IDX]}"
MODEL_LABEL="${MODEL_ENTRY%%|*}"
MODEL_DIR="${MODEL_ENTRY##*|}"
ARCH_DATASET="${DATASETS[$DATASET_IDX]}"

IDX_DIR="$RES/$MODEL_DIR/$ARCH_DATASET/fold_${FOLD}/${FOLD}"
CKPT="$IDX_DIR/checkpoints/best_cer.pth"
YAML=$(ls -1 "$IDX_DIR"/train_*.yaml 2>/dev/null | head -1)
OUT_DIR="$RES/$MODEL_DIR/$ARCH_DATASET/fold_${FOLD}"
OUT_PATH="$OUT_DIR/eval_single_corruption.json"

echo "========== RUN INFO =========="
echo "Task id:     $TASK_ID"
echo "Model:       $MODEL_LABEL  ($MODEL_DIR)"
echo "Dataset:     $ARCH_DATASET"
echo "Fold:        $FOLD"
echo "YAML:        $YAML"
echo "Checkpoint:  $CKPT"
echo "Out:         $OUT_PATH"
echo "=============================="

if [[ -z "$YAML" ]]; then
  echo "ERROR: no train_*.yaml in $IDX_DIR"
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint $CKPT"
  exit 2
fi
if [[ -f "$OUT_PATH" ]]; then
  echo "[done] $OUT_PATH already exists; skipping."
  exit 0
fi

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L

SEED=$((FOLD * 100 + MODEL_IDX * 10000))

"$PY_BIN" eval_single_corruption.py \
    -c "$YAML" \
    --checkpoint "$CKPT" \
    --out "$OUT_PATH" \
    --n_offsets 6 \
    --n_corruptions_per_sample 1 \
    --min_eligibility_offset 1 \
    --seed "$SEED"
