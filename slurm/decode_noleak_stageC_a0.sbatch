#!/bin/bash
#SBATCH -J dec_noleakC_a0
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Stage C (AR + KenLM rescoring / shallow fusion) re-run at ALPHA=0,
# the calibrated beam operating point, instead of the transferred
# alpha=0.6. Everything else (leak-free per-fold KenLM, checkpoints,
# lambda, beam sizes) matches decode_noleak_stageC.sbatch. This makes
# the KenLM comparison against calibrated beam use the SAME length
# normalisation (audit F2). Inference-only; no retraining.
#   MODEL=hybrid_xs | ar_xs
#   sbatch --export=ALL,MODEL=ar_xs     slurm/decode_noleak_stageC_a0.sbatch
#   sbatch --export=ALL,MODEL=hybrid_xs slurm/decode_noleak_stageC_a0.sbatch
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-ar_xs}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

LM_PATH="lm_noleak/fold_${FOLD}/char_5gram.binary"
[[ -f "$LM_PATH" ]] || { echo "ERROR: clean LM missing: $LM_PATH"; exit 1; }

if [[ "$MODEL" == "hybrid_xs" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_hybrid_xs.yaml"; OUTDIR="results/hwr2/decode_study_xs_full_hybrid_noleak_a0"
else
    CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_ar_xs.yaml"; OUTDIR="results/hwr2/decode_study_xs_full_ar_noleak_a0"
fi
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
mkdir -p "$OUTDIR"

run() { local tag="$1"; shift; echo "-- $tag --"; $PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"; }

run "stageC1_ar_kenlm_rescore_N8_lw0p1_a0" \
    --decoder ar --method beam_rescore --nbest_size 8 --beam_size 8 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
run "stageC2_ar_kenlm_shallow_B4_lw0p2_a0" \
    --decoder ar --method beam_lm --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.2
echo "=== fold $FOLD ($MODEL) Stage C noleak alpha=0 done ==="
