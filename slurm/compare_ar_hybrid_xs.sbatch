#!/bin/bash
#SBATCH -J ar_hyb_xs
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
PAPER=$WORK/imu-hwr/publications/paper2_lncs_overleaf
export PROJ

AR_CKPT=$WORK/imu-hwr/results/hwr2/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth
HYB_CKPT=$WORK/imu-hwr/results/hwr2/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth
DATASET=onhw_wi_word_rh
FOLD=0
OUTDIR=$PAPER/figures

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

$PY_BIN analysis/scripts/compare_ar_hybrid.py \
  --ar_ckpt   "$AR_CKPT" \
  --hyb_ckpt  "$HYB_CKPT" \
  --dataset   "data/$DATASET" \
  --fold      "$FOLD" \
  --outdir    "$OUTDIR" \
  --arch_en   blconv_b \
  --arch_de   ar_transformer_xs \
  --use_gated_attention \
  --gating_type elementwise \
  --difficulty_mode random \
  --device    cuda \
  --max_samples 1000
