#!/bin/bash
#SBATCH -J dec_ar_crossds
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Decoding study for plain HWRFormer (XS, elementwise) on the three
# datasets missing from the OnHW-WI-only study: OnHW-WD, private
# words, private sentences. Canonical stages only: greedy, calibrated
# beam (B=4, alpha=0), and matched-alpha KenLM rescoring + shallow
# fusion with leak-free per-fold, per-dataset 5-gram LMs.
#   DATASET=wd | privword | privsent
#   sbatch -p rtx3080 --gres=gpu:rtx3080:1 --export=ALL,DATASET=wd       slurm/decode_ar_crossds.sbatch
#   sbatch -p v100    --gres=gpu:v100:1    --export=ALL,DATASET=privword slurm/decode_ar_crossds.sbatch
#   sbatch -p v100    --gres=gpu:v100:1    --export=ALL,DATASET=privsent slurm/decode_ar_crossds.sbatch
# Requested 2026-07-30 to fill the cumulative-table blanks, pending
# supervisor approval.
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
DATASET=${DATASET:-wd}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

case "$DATASET" in
  wd)       DSDIR="onhw_wd_word_rh" ;;
  privword) DSDIR="wi_word_hw6_meta" ;;
  privsent) DSDIR="wi_sent_hw6_meta" ;;
  *) echo "ERROR: unknown DATASET=$DATASET"; exit 1 ;;
esac
CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__${DSDIR}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
BASE_CFG="configs/decode_study/base_ar_xs_${DATASET}.yaml"
OUTDIR="results/hwr2/decode_study_xs_full_ar_${DATASET}"
LM_PATH="lm_noleak_${DATASET}/fold_${FOLD}/char_5gram.binary"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
mkdir -p "$OUTDIR"
echo "=== crossds decode dataset=$DATASET fold=$FOLD ==="
echo "ckpt: $CKPT"

run() { local tag="$1"; shift; echo "-- $tag --"; $PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"; }

run "stageA0_ar_greedy" --decoder ar --method greedy
run "stageA1_ar_beam_B4" --decoder ar --method beam --beam_size 4 --alpha 0.0
run "stageC1_ar_kenlm_rescore_N8_lw0p1_a0" \
    --decoder ar --method beam_rescore --nbest_size 8 --beam_size 8 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
run "stageC2_ar_kenlm_shallow_B4_lw0p2_a0" \
    --decoder ar --method beam_lm --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.2
echo "=== dataset=$DATASET fold=$FOLD done ==="
