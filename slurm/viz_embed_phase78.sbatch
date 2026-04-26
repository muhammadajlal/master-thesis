#!/bin/bash
#SBATCH -J viz-phase78
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

# Embedding visualizations for L1, L2, M1, N1 (Phase 7-8 experiments).
# Mirrors viz_embeddings.sbatch but handles the deeper Ablations-MMLM/<family>/<regime>/ layout.

set -euo pipefail
set -x

FOLD=${FOLD:-0}
MAX_SAMPLES=${MAX_SAMPLES:-500}

# (experiment_dir_name, full_subpath_under_results/hwr2)
# Format below uses bash array of "name:subpath" tuples
EXPERIMENTS=(
  "L1_mini_qformer:Ablations-MMLM/GPT-2/Hybrid"
  "L2_kv_slim:Ablations-MMLM/GPT-2/Hybrid"
  "M1_byt5_hybrid_mlp:Ablations-MMLM/byt5-small/Hybrid"
  "N1_conformer_hybrid_mlp:Ablations-MMLM/GPT-2/Hybrid"
)

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS=$WORK/imu-hwr/results/hwr2

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

HAS_UMAP=$($PY_BIN -c "import umap" 2>/dev/null && echo 1 || echo 0)

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"

for ENTRY in "${EXPERIMENTS[@]}"; do
  EXPERIMENT="${ENTRY%%:*}"
  SUBPATH="${ENTRY##*:}"
  EXP_BASE="${RESULTS}/${SUBPATH}/${EXPERIMENT}"

  echo ""
  echo "========== ${EXPERIMENT} (${SUBPATH}) =========="

  CKPT_ONHW="${EXP_BASE}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CKPT_HW6="${EXP_BASE}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CONFIG_ONHW=$(ls "${EXP_BASE}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)
  CONFIG_HW6=$(ls "${EXP_BASE}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)

  ARGS=""
  if [[ -f "$CKPT_ONHW" && -n "$CONFIG_ONHW" ]]; then
    ARGS="$ARGS --ckpt_onhw $CKPT_ONHW --config_onhw $CONFIG_ONHW"
    echo "OnHW: $CKPT_ONHW"
  else
    echo "OnHW: NOT AVAILABLE"
  fi

  if [[ -f "$CKPT_HW6" && -n "$CONFIG_HW6" ]]; then
    ARGS="$ARGS --ckpt_hw6 $CKPT_HW6 --config_hw6 $CONFIG_HW6"
    echo "HW6:  $CKPT_HW6"
  else
    echo "HW6:  NOT AVAILABLE"
  fi

  if [[ -z "$ARGS" ]]; then
    echo "SKIP: $EXPERIMENT (no checkpoints found for fold $FOLD)"
    continue
  fi

  OUT_DIR="${WORK}/imu-hwr/analysis/embedding_viz/${EXPERIMENT}"
  mkdir -p "$OUT_DIR"
  echo "Output: $OUT_DIR"

  $PY_BIN scripts/visualize_embeddings.py \
      $ARGS \
      --out_dir "$OUT_DIR" \
      --method pca \
      --max_samples "$MAX_SAMPLES" \
      --fold "$FOLD" \
      --device cuda

  if [[ "$HAS_UMAP" -eq 1 ]]; then
    $PY_BIN scripts/visualize_embeddings.py \
        $ARGS \
        --out_dir "$OUT_DIR" \
        --method umap \
        --max_samples "$MAX_SAMPLES" \
        --fold "$FOLD" \
        --device cuda || echo "UMAP failed for $EXPERIMENT"
  else
    echo "SKIP UMAP: umap-learn not installed"
  fi

  echo "Done: $EXPERIMENT → $OUT_DIR"
done

echo ""
echo "========== ALL DONE =========="
echo "Results in: ${WORK}/imu-hwr/analysis/embedding_viz/"
