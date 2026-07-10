#!/bin/bash
#SBATCH -J ctc_post_xs
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
set -euo pipefail
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1
mkdir -p "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/ctc_post_xs_out"
$ENV_PREFIX/bin/python3 analysis/scripts/ctc_posterior_lambda_analysis.py \
  --lambdas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --ckpts results/hwr2/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_02_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_03_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_04_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_05_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_06_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_07_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_08_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_09_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth results/hwr2/train_element_word_hybrid_10_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_{fold}/{fold}/checkpoints/best_cer.pth \
  --dataset data/onhw_wi_word_rh --folds 0,1,2,3,4 --qual_fold 0 \
  --arch_de ar_transformer_xs --arch_en blconv_b --gating_type elementwise \
  --n_quant_samples 500 --device cuda \
  --outdir "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/ctc_post_xs_out"
echo "=== DONE; outputs in /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/ctc_post_xs_out ==="
ls "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/ctc_post_xs_out"
