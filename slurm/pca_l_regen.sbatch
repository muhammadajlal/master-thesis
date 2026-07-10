#!/bin/bash
#SBATCH -J pca_l_regen
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
set -euo pipefail
WORK=/home/woody/iwso/iwso214h; PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1
mkdir -p "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/pca_l_out"
$ENV_PREFIX/bin/python3 analysis/scripts/compare_ar_hybrid.py \
  --ar_ckpt  "$WORK/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth" \
  --hyb_ckpt "$WORK/imu-hwr/results/hwr2/Baseline-Hybrid/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth" \
  --dataset data/onhw_wi_word_rh --fold 0 --outdir "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/pca_l_out" \
  --arch_en blconv_b --arch_de ar_transformer_s \
  --use_gated_attention --gating_type elementwise \
  --difficulty_mode random --device cuda --max_samples 1000
echo "=== DONE ==="; ls "/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/pca_l_out"/pca_*.pdf
