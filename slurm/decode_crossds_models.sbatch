#!/bin/bash
#SBATCH -J dec_xds_mdl
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Completes the decoding matrix: the noise-trained and hybrid HWRFormer
# recognizers on the three datasets beyond OnHW-WI. Canonical stages
# only, all at the calibrated search point (B=4, alpha=0):
#   greedy, calibrated beam, KenLM shallow fusion, KenLM rescoring N=4.
# Backup material for the defense. Requested 2026-07-31.
#
#   MODEL=noise|hybrid   DATASET=wd|privword|privsent
#   sbatch -p v100 --gres=gpu:v100:1 --export=ALL,MODEL=noise,DATASET=wd slurm/decode_crossds_models.sbatch
# ------------------------------------------------------------------
set -euo pipefail
FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-noise}
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
  wd)       DSDIR="onhw_wd_word_rh";  BASE_CFG="configs/decode_study/base_ar_xs_wd.yaml";       LM_DIR="lm_noleak_wd";       HYB_ROOT="train_element_word_hybrid_01_xs_onhw_wd" ;;
  privword) DSDIR="wi_word_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privword.yaml"; LM_DIR="lm_noleak_privword"; HYB_ROOT="train_element_word_hybrid_01_xs_stabilo" ;;
  privsent) DSDIR="wi_sent_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privsent.yaml"; LM_DIR="lm_noleak_privsent"; HYB_ROOT="train_element_word_hybrid_01_xs_stabilo_sent" ;;
  *) echo "ERROR: unknown DATASET=$DATASET"; exit 1 ;;
esac
case "$MODEL" in
  noise)  CKPT="${RESULTS_BASE}/Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__${DSDIR}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth" ;;
  hybrid) CKPT="${RESULTS_BASE}/${HYB_ROOT}/ar_transformer_xs__${DSDIR}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth" ;;
  *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
OUTDIR="results/hwr2/decode_crossds_${MODEL}_${DATASET}"
LM_PATH="${LM_DIR}/fold_${FOLD}/char_5gram.binary"
[[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
mkdir -p "$OUTDIR"
echo "=== $MODEL / $DATASET fold=$FOLD ==="
echo "ckpt: $CKPT"

run() { local tag="$1"; shift; echo "-- $tag --"; $PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"; }

run "stageA0_ar_greedy"    --decoder ar --method greedy
run "stageA1_ar_beam_B4"   --decoder ar --method beam --beam_size 4 --alpha 0.0
run "stageC2_ar_kenlm_shallow_B4_lw0p2_a0" \
    --decoder ar --method beam_lm --beam_size 4 --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.2
run "stageC1_ar_kenlm_rescore_N4_lw0p1_a0" \
    --decoder ar --method beam_rescore --nbest_size 4 --beam_size 4 \
    --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
echo "=== $MODEL / $DATASET fold=$FOLD done ==="
