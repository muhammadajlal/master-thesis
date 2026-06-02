#!/bin/bash
#SBATCH -J tfgap
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH --array=0-39
# ═══════════════════════════════════════════════════════════════════
# Teacher-forcing vs free-running AR gap (4 cells × 2 datasets × 5 folds)
# ═══════════════════════════════════════════════════════════════════
# Submit:
#   cd work/REWI_work
#   sbatch -p v100,a100,rtx3080,work slurm/eval_tf_gap.sbatch
# Each array task evaluates ONE (cell, fold). Results land under
#   results/hwr2/<cell-dir>/eval_tf_gap/fold_<k>.json
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

IDX=${SLURM_ARRAY_TASK_ID}

# 8 cells × 5 folds = 40 array tasks. CELL_IDX = IDX/5, FOLD = IDX%5.
CELL_IDX=$(( IDX / 5 ))
FOLD=$(( IDX % 5 ))

CELLS=(
  "AR-only/OnHW-WI|Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh"
  "AR-only/Priv-S|Baseline-AR-XS-blconv_b/ar_transformer_xs__wi_sent_hw6_meta"
  "+Noise/OnHW-WI|Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__onhw_wi_word_rh"
  "+Noise/Priv-S|Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__wi_sent_hw6_meta"
  "+Hybrid/OnHW-WI|train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh"
  "+Hybrid/Priv-S|train_element_word_hybrid_01_xs_stabilo_sent/ar_transformer_xs__wi_sent_hw6_meta"
  "+Hybrid+Noise/OnHW-WI|HybridInputCorruption-XS-L01_uniform/ar_transformer_xs__onhw_wi_word_rh"
  "+Hybrid+Noise/Priv-S|HybridInputCorruption-XS-L01_uniform/ar_transformer_xs__wi_sent_hw6_meta"
)
ENTRY="${CELLS[$CELL_IDX]}"
LABEL="${ENTRY%%|*}"
SUBDIR="${ENTRY##*|}"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS_BASE=$WORK/imu-hwr/results/hwr2
CELL_DIR=$RESULTS_BASE/$SUBDIR
FOLD_DIR=$CELL_DIR/fold_${FOLD}/${FOLD}

# Locate inputs
YAML_PATH=$(ls -1 ${FOLD_DIR}/train_*.yaml | head -1)
CKPT=${FOLD_DIR}/checkpoints/best_cer.pth
OUT_DIR=${CELL_DIR}/eval_tf_gap
OUT_JSON=${OUT_DIR}/fold_${FOLD}.json

echo "═══ TF-gap probe ═══"
echo "Cell:       $LABEL ($SUBDIR)"
echo "Fold:       $FOLD"
echo "YAML:       $YAML_PATH"
echo "Checkpoint: $CKPT"
echo "Output:     $OUT_JSON"
echo "Node:       $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

mkdir -p "$OUT_DIR"

# Conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh 2>/dev/null || \
  source /home/hpc/iwso/iwso214h/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PY_BIN" eval_tf_gap.py \
  -c "$YAML_PATH" \
  --checkpoint "$CKPT" \
  --out "$OUT_JSON" \
  --device cuda

echo "═══ Done ═══"
ls -l "$OUT_JSON"
