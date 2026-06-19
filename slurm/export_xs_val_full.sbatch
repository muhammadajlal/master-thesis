#!/bin/bash
#SBATCH -J export_xs_val_full
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
# ═══════════════════════════════════════════════════════════════════
# Drive analysis/scripts/_export_xs_val_full.sh on a GPU node. The
# underlying script runs 20 short main.py test=True invocations
# sequentially; one GPU is sufficient.
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

bash analysis/scripts/_export_xs_val_full.sh
