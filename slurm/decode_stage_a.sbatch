#!/bin/bash
#SBATCH -J rewi-cv-a
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage A only (AR calibration)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=${MODEL:-hybrid}  # "hybrid" | "ar"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$MODEL" != "hybrid" && "$MODEL" != "ar" ]]; then
    echo "ERROR: Stage A supports MODEL=hybrid or MODEL=ar only."
    exit 1
fi

export FOLD MODEL
PYTHON="$PY_BIN" bash scripts/run_decode_stage_a.sh

echo "═══ Fold $FOLD stage A complete ═══"
