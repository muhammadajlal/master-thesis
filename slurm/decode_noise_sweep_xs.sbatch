#!/bin/bash
#SBATCH -J dec_noise_sweep
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Decoding calibration sweep for HWRFormer + uniform noise injection
# (XS, p_ni = 0.15) on OnHW-WI -- mirrors the stage tags of
# decode_study_xs_full_{ar,hybrid} so the same analysis applies.
# Adds beam-size and length-normalization sweeps plus the matched
# alpha=0 KenLM stages (leak-free per-fold LMs from lm_noleak/).
# Requested 2026-07-30 to extend the defense decoding figure/table,
# pending supervisor approval.
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

CKPT="${RESULTS_BASE}/Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
BASE_CFG="configs/decode_study/base_ar_xs.yaml"
OUTDIR="results/hwr2/decode_study_xs_full_noise"
LM_PATH="lm_noleak/fold_${FOLD}/char_5gram.binary"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
mkdir -p "$OUTDIR"
echo "=== noise sweep fold=$FOLD ckpt=$CKPT ==="

run() { local tag="$1"; shift; echo "-- $tag --"; $PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"; }

# Stage A0: greedy
run "stageA0_ar_greedy" --decoder ar --method greedy
# Stage A1: beam-size sweep at alpha=0
for B in 1 2 4 8 16; do
    run "stageA1_ar_beam_B${B}" --decoder ar --method beam --beam_size "$B" --alpha 0.0
done
# Stage A2: length-normalization sweep at B=4
for A in 0.0 0.2 0.4 0.6 0.8; do
    run "stageA2_ar_lenorm_a${A}" --decoder ar --method beam --beam_size 4 --alpha "$A"
done
# Stage C (matched alpha=0, leak-free KenLM)
run "stageC1_ar_kenlm_rescore_N8_lw0p1_a0" \
    --decoder ar --method beam_rescore --nbest_size 8 --beam_size 8 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
run "stageC2_ar_kenlm_shallow_B4_lw0p2_a0" \
    --decoder ar --method beam_lm --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.2
echo "=== fold $FOLD noise sweep done ==="
