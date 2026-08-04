#!/bin/bash
#SBATCH -J dec_resc_n4_wi_mdl
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# KenLM N-best rescoring at the CALIBRATED beam (N=4, B=4) for the
# noise-trained and hybrid recognizers on OnHW-WI.
#
# Why: Table tab:decode-study-cross-model reports rescoring from the
# N=8 grid point, which forces beam 8 and therefore silently overrides
# the calibrated search point (B=4, a=0) used by the search-only
# control and by shallow fusion. decode_rescore_n4.sbatch closed that
# gap for plain HWRFormer on all four datasets; this run closes it for
# the two remaining WI recognizers. Requested 2026-08-04.
#
#   MODEL=noise | hybrid
#   sbatch -p v100 --gres=gpu:v100:1 --export=ALL,MODEL=noise  slurm/decode_rescore_n4_models_wi.sbatch
#   sbatch -p v100 --gres=gpu:v100:1 --export=ALL,MODEL=hybrid slurm/decode_rescore_n4_models_wi.sbatch
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-noise}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

case "$MODEL" in
  noise)  CKPT="${RESULTS_BASE}/Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
          BASE_CFG="configs/decode_study/base_ar_xs.yaml" ;;
  hybrid) CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
          BASE_CFG="configs/decode_study/base_hybrid_xs.yaml" ;;
  *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
OUTDIR="results/hwr2/decode_rescore_n4_wi_${MODEL}"
LM_PATH="lm_noleak/fold_${FOLD}/char_5gram.binary"
[[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
mkdir -p "$OUTDIR"
echo "=== rescore N4/B4 model=$MODEL wi fold=$FOLD ==="
echo "ckpt: $CKPT"
echo "lm:   $LM_PATH"

$PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "stageC1_ar_kenlm_rescore_N4_lw0p1_a0__fold${FOLD}" \
    --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda \
    --decoder ar --method beam_rescore --nbest_size 4 --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0

echo "=== $MODEL wi fold=$FOLD done ==="
