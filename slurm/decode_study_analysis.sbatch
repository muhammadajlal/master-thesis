#!/bin/bash
#SBATCH -J decode-analysis
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

echo "Running decode study thesis analysis..."
$PY_BIN scripts/decode_study_thesis_analysis.py

echo "Done. Results in: $PROJ/results/hwr2/decode_study_thesis_analysis/"
