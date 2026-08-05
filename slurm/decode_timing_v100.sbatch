#!/bin/bash
#SBATCH -J dec_timing
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Dedicated single-hardware TIMING pass for the decoding study
# (plain HWRFormer, all four datasets x four canonical methods).
#
# Why: the published decode runs are correct for CER/WER but were
# distributed across v100/rtx3080/a100 partitions (quota dodging), so
# their runtime_per_sample_ms values are not comparable across cells.
# The lab guidelines (drichtl_en.pdf 1.6.2) ask for per-algorithm
# runtime, so this job re-decodes every Table-5.13 cell back-to-back
# on ONE GPU type. Decoding is deterministic: the aggregator asserts
# CER equality against the canonical cells as a free regression check;
# only the runtime fields of these runs are ever reported.
# Requested 2026-08-04.
#
#   sbatch -p v100 --gres=gpu:v100:1 slurm/decode_timing_v100.sbatch
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

for DATASET in wi wd privword privsent; do
  case "$DATASET" in
    wi)       DSDIR="onhw_wi_word_rh";  BASE_CFG="configs/decode_study/base_ar_xs.yaml";          LM_DIR="lm_noleak" ;;
    wd)       DSDIR="onhw_wd_word_rh";  BASE_CFG="configs/decode_study/base_ar_xs_wd.yaml";       LM_DIR="lm_noleak_wd" ;;
    privword) DSDIR="wi_word_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privword.yaml"; LM_DIR="lm_noleak_privword" ;;
    privsent) DSDIR="wi_sent_hw6_meta"; BASE_CFG="configs/decode_study/base_ar_xs_privsent.yaml"; LM_DIR="lm_noleak_privsent" ;;
  esac
  CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__${DSDIR}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
  OUTDIR="results/hwr2/decode_timing_v100_${DATASET}"
  LM_PATH="${LM_DIR}/fold_${FOLD}/char_5gram.binary"
  [[ -f "$CKPT" ]]    || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
  [[ -f "$LM_PATH" ]] || { echo "ERROR: LM missing: $LM_PATH"; exit 1; }
  mkdir -p "$OUTDIR"
  echo "=== timing dataset=$DATASET fold=$FOLD ==="

  run() { local tag="$1"; shift; echo "-- $tag --"; $PY_BIN decode_study.py -c "$BASE_CFG" \
      --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"; }

  run "stageA0_ar_greedy"    --decoder ar --method greedy
  run "stageA1_ar_beam_B4"   --decoder ar --method beam --beam_size 4 --alpha 0.0
  run "stageC2_ar_kenlm_shallow_B4_lw0p2_a0" \
      --decoder ar --method beam_lm --beam_size 4 --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.2
  run "stageC1_ar_kenlm_rescore_N4_lw0p1_a0" \
      --decoder ar --method beam_rescore --nbest_size 4 --beam_size 4 \
      --alpha 0.0 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
  echo "=== timing dataset=$DATASET fold=$FOLD done ==="
done
echo "=== all datasets fold=$FOLD done ==="
