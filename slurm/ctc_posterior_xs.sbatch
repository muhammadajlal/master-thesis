#!/bin/bash
#SBATCH -J ctc_post_xs
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
PAPER=$WORK/imu-hwr/publications/paper2_lncs_overleaf

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

# Template each ckpt path with {fold}; the script substitutes per fold.
CKPT_BASE='results/hwr2/train_element_word_hybrid_XX_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth'

CKPTS=()
for k in 01 02 03 04 05 06 07 08 09 10; do
  CKPTS+=("${CKPT_BASE/XX/$k}")
done

$PY_BIN analysis/scripts/ctc_posterior_lambda_analysis.py \
  --lambdas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --ckpts "${CKPTS[@]}" \
  --folds 0,1,2,3,4 \
  --dataset data/onhw_wi_word_rh \
  --arch_en blconv_b \
  --arch_de ar_transformer_xs \
  --use_gated_attention \
  --gating_type elementwise \
  --device cuda \
  --outdir "$PAPER/figures/ctc_posterior_lambda_xs"

# Copy the foldavg figure to the file name expected by supplementary.tex.
cp "$PAPER/figures/ctc_posterior_lambda_xs/metrics_vs_lambda_foldavg.pdf" \
   "$PAPER/figures/ctc_metrics_vs_lambda_foldavg.pdf"
echo "Done. Final figure at: $PAPER/figures/ctc_metrics_vs_lambda_foldavg.pdf"
