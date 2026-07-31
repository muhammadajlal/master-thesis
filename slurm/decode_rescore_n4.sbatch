#!/bin/bash
#SBATCH -J dec_resc_n4
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# KenLM N-best rescoring at the CALIBRATED beam (N=4, B=4) for plain
# HWRFormer (XS, elementwise) on all four datasets.
#
# Why: the canonical rescoring stage uses N=8, which forces beam 8 and
# therefore silently overrides the calibrated search point (B=4, a=0)
# used by the search-only control and by shallow fusion. This run closes
# that gap so rescoring can be reported at the same beam as everything
# else. Requested 2026-07-31.
#
#   DATASET=wi | wd | privword | privsent
#   sbatch -p v100 --gres=gpu:v100:1 --export=ALL,DATASET=wi slurm/decode_rescore_n4.sbatch
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
DATASET=${DATASET:-wi}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

case "$DATASET" in
  wi)       DSDIR="onhw_wi_word_rh"; BASE_CFG="configs/decode_study/base_ar_xs.yaml";          LM_DIR="lm_noleak" ;;
  wd)       DSDIR="onhw_wd_word_rh"; BASE_CFG="configs/decode_study/base_ar_xs_wd.yaml";       LM_DIR="lm_noleak_wd" ;;
  privword) DSDIR="wi_word_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privword.yaml"; LM_DIR="lm_noleak_privword" ;;
  privsent) DSDIR="wi_sent_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privsent.yaml"; LM_DIR="lm_noleak_privsent" ;;
  *) echo "ERROR: unknown DATASET=$DATASET"; exit 1 ;;
esac
CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__${DSDIR}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
OUTDIR="results/hwr2/decode_rescore_n4_${DATASET}"
LM_PATH="${LM_DIR}/fold_${FOLD}/char_5gram.binary"
[[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
mkdir -p "$OUTDIR"
echo "=== rescore N4/B4 dataset=$DATASET fold=$FOLD ==="
echo "ckpt: $CKPT"
echo "lm:   $LM_PATH"

$PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "stageC1_ar_kenlm_rescore_N4_lw0p1_a0__fold${FOLD}" \
    --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda \
    --decoder ar --method beam_rescore --nbest_size 4 --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0

echo "=== dataset=$DATASET fold=$FOLD done ==="
