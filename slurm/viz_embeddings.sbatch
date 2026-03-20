#!/bin/bash
#SBATCH -J viz-embed
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail
set -x

# ---- Configuration (override via environment) ----
FOLD=${FOLD:-0}
MAX_SAMPLES=${MAX_SAMPLES:-500}

# Experiments to visualize (space-separated, override with EXPERIMENTS env var)
EXPERIMENTS=${EXPERIMENTS:-"vlm_Qformer_gpt2_word_v2 vlm_ablation_A1_mlp_pretrained vlm_ablation_B1_pooling_pretrained H1_hybrid_mlp F1_frozen_enc_mlp G1a_linear G1b_conv_pool vlm_ablation_A3_linear_pretrained"}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS=$WORK/imu-hwr/results/hwr2

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

# Install umap-learn if missing
$PY_BIN -c "import umap" 2>/dev/null || pip install umap-learn --quiet

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"

for EXPERIMENT in $EXPERIMENTS; do
  echo ""
  echo "========== $EXPERIMENT =========="

  # Resolve paths
  CKPT_ONHW="${RESULTS}/${EXPERIMENT}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CKPT_HW6="${RESULTS}/${EXPERIMENT}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CONFIG_ONHW=$(ls "${RESULTS}/${EXPERIMENT}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1)
  CONFIG_HW6=$(ls "${RESULTS}/${EXPERIMENT}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1)

  # Skip if missing
  if [[ ! -f "$CKPT_ONHW" || ! -f "$CKPT_HW6" || -z "$CONFIG_ONHW" || -z "$CONFIG_HW6" ]]; then
    echo "SKIP: $EXPERIMENT (missing checkpoints or configs for fold $FOLD)"
    continue
  fi

  OUT_DIR="${WORK}/imu-hwr/analysis/embedding_viz/${EXPERIMENT}"

  echo "CKPT OnHW:   $CKPT_ONHW"
  echo "CKPT HW6:    $CKPT_HW6"
  echo "Config OnHW: $CONFIG_ONHW"
  echo "Config HW6:  $CONFIG_HW6"
  echo "Output:      $OUT_DIR"

  # PCA
  $PY_BIN scripts/visualize_embeddings.py \
      --ckpt_onhw "$CKPT_ONHW" \
      --ckpt_hw6 "$CKPT_HW6" \
      --config_onhw "$CONFIG_ONHW" \
      --config_hw6 "$CONFIG_HW6" \
      --out_dir "$OUT_DIR" \
      --method pca \
      --max_samples "$MAX_SAMPLES" \
      --fold "$FOLD" \
      --device cuda

  # UMAP
  $PY_BIN scripts/visualize_embeddings.py \
      --ckpt_onhw "$CKPT_ONHW" \
      --ckpt_hw6 "$CKPT_HW6" \
      --config_onhw "$CONFIG_ONHW" \
      --config_hw6 "$CONFIG_HW6" \
      --out_dir "$OUT_DIR" \
      --method umap \
      --max_samples "$MAX_SAMPLES" \
      --fold "$FOLD" \
      --device cuda || echo "UMAP failed for $EXPERIMENT"

  echo "Done: $EXPERIMENT → $OUT_DIR"
done

echo ""
echo "========== ALL DONE =========="
echo "Results in: ${WORK}/imu-hwr/analysis/embedding_viz/"
