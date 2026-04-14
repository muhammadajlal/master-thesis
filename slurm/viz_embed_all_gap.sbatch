#!/bin/bash
#SBATCH -J viz-gap-all
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00

set -euo pipefail
set -x

# ---- Configuration ----
FOLD=${FOLD:-0}
MAX_SAMPLES=${MAX_SAMPLES:-500}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS=$WORK/imu-hwr/results/hwr2
VIZ_OUT=$WORK/imu-hwr/analysis/embedding_viz

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

# ---- Experiment list ----
# Format: "RESULT_SUBDIR|OUTPUT_NAME"
# RESULT_SUBDIR: path under results/hwr2/ where vlm__onhw_wi_word_rh/ and vlm__wi_word_hw6_meta/ live
# OUTPUT_NAME: short name for the output directory under analysis/embedding_viz/

declare -a EXPERIMENTS=(
  # H1 baseline (already have, but re-run for consistency)
  "Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp|H1_hybrid_mlp"
  # J-series: global contrastive
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_mlp|J1_contrastive_mlp"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_pooling|J1_contrastive_pooling"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_mlp|J2_contrastive_mlp"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_pooling|J2_contrastive_pooling"
  # K-series: modality gap strategies
  "Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse|K1_ctc_mse"
  "Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior|K2_ctc_posterior"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/K3_ec_loss|K3_ec_loss"
  "K4_sea_contrastive|K4_sea_contrastive"
  "Ablations-MMLM/GPT-2/Hybrid/K5_kv_multiview|K5_kv_multiview"
)

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r RESULT_SUBDIR OUTPUT_NAME <<< "$ENTRY"

  echo ""
  echo "========== $OUTPUT_NAME =========="

  CKPT_ONHW="${RESULTS}/${RESULT_SUBDIR}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CKPT_HW6="${RESULTS}/${RESULT_SUBDIR}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  CONFIG_ONHW=$(ls "${RESULTS}/${RESULT_SUBDIR}/vlm__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)
  CONFIG_HW6=$(ls "${RESULTS}/${RESULT_SUBDIR}/vlm__wi_word_hw6_meta/fold_${FOLD}/${FOLD}"/train_*.yaml 2>/dev/null | head -1 || true)

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
    echo "SKIP: $OUTPUT_NAME (no checkpoints found for fold $FOLD)"
    continue
  fi

  OUT_DIR="${VIZ_OUT}/${OUTPUT_NAME}"
  echo "Output: $OUT_DIR"

  # PCA
  $PY_BIN scripts/visualize_embeddings.py \
      $ARGS \
      --out_dir "$OUT_DIR" \
      --method pca \
      --max_samples "$MAX_SAMPLES" \
      --fold "$FOLD" \
      --device cuda

  # UMAP
  if [[ "$HAS_UMAP" -eq 1 ]]; then
    $PY_BIN scripts/visualize_embeddings.py \
        $ARGS \
        --out_dir "$OUT_DIR" \
        --method umap \
        --max_samples "$MAX_SAMPLES" \
        --fold "$FOLD" \
        --device cuda || echo "UMAP failed for $OUTPUT_NAME"
  else
    echo "SKIP UMAP: umap-learn not installed"
  fi

  echo "Done: $OUTPUT_NAME → $OUT_DIR"
done

echo ""
echo "========== ALL DONE =========="
echo "Results in: ${VIZ_OUT}/"
