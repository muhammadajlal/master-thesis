#!/bin/bash
#SBATCH -J prefix_perturbation_sweep
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
# ═══════════════════════════════════════════════════════════════════
# Sweep the prefix-perturbation rate p_replace in
# {0.00, 0.05, 0.10, 0.15, 0.20} for the AR-only and AR + noise
# injection HWRFormer xs checkpoints on OnHW WI word, 5 folds each.
# 50 short evaluations sequentially; one GPU suffices.
# ═══════════════════════════════════════════════════════════════════

set -uo pipefail

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash analysis/scripts/_run_prefix_perturbation_sweep.sh
