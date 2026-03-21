#!/bin/bash
#SBATCH -J viz-embed
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00

set -euo pipefail
set -x

# ---- Configuration (override via environment) ----
FOLD=${FOLD:-0}
MAX_SAMPLES=${MAX_SAMPLES:-500}

# Key experiments to visualize (covers all phases)
EXPERIMENTS=${EXPERIMENTS:-"\
vlm_Qformer_gpt2_word_v2 \
vlm_ablation_A1_mlp_pretrained \
vlm_ablation_B1_pooling_pretrained \
vlm_ablation_A3_linear_pretrained \
G1a_linear \
G1b_conv_pool \
F1_frozen_enc_mlp \
H1_hybrid_mlp \
H1_hybrid_pooling \
H2_hybrid_prompt_aug_pooling \
H3_hybrid_two_step_prompt_pooling \
"}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS=$WORK/imu-hwr/results/hwr2

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

# Check if umap is available (no pip on cluster)
HAS_UMAP=$($PY_BIN -c "import umap" 2>/dev/null && echo 1 || echo 0)

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"

for EXPERIMENT in $EXPERIMENTS; do
  echo ""
  echo "========== $EXPERIMENT =========="

  # Resolve paths — either or both datasets may be available
  CKPT_ONHW="${RESULTS}/${EXPERIMENT}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CKPT_HW6="${RESULTS}/${EXPERIMENT}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CONFIG_ONHW=$(ls "${RESULTS}/${EXPERIMENT}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)
  CONFIG_HW6=$(ls "${RESULTS}/${EXPERIMENT}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)

  # Build args dynamically — at least one dataset must exist
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
  echo "Output: $OUT_DIR"

  # PCA
  $PY_BIN scripts/visualize_embeddings.py \
      $ARGS \
      --out_dir "$OUT_DIR" \
      --method pca \
      --max_samples "$MAX_SAMPLES" \
      --fold "$FOLD" \
      --device cuda

  # UMAP (only if installed)
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
