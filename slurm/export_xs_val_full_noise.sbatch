#!/bin/bash
#SBATCH -J export_xs_val_full_noise
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
# ═══════════════════════════════════════════════════════════════════
# Drive analysis/scripts/_export_xs_val_full_noise.sh on a GPU node.
# The underlying script runs 10 short main.py test=True invocations
# sequentially. The AR-only exports already exist on disk and are
# skipped; only the 5 noise-injection folds are decoded.
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

echo "=== GPU presence check ==="
nvidia-smi -L 2>&1 || echo "nvidia-smi -L failed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "=========================="

bash analysis/scripts/_export_xs_val_full_noise.sh
