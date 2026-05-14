#!/bin/bash
#SBATCH -J extract-confusion-xs
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH -t 03:00:00

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
PY_BIN=$ENV_PREFIX/bin/python3

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"
nvidia-smi -L

OUT_ROOT=$WORK/imu-hwr/results/hwr2/confusion_matrices_xs

for DATASET in onhw_wi_word_rh onhw_wd_word_rh wi_word_hw6_meta wi_sent_hw6_meta onhw_equations_wi_word_rh onhw_equations_wd_word_rh; do
  echo "========== ${DATASET} (xs variant) =========="
  $PY_BIN scripts/extract_confusion_matrix.py \
    --dataset "${DATASET}" \
    --folds 0 1 2 3 4 \
    --out-root "${OUT_ROOT}" \
    --variant xs \
    --device cuda
done

echo "========== ALL DONE =========="
ls -la "$OUT_ROOT"
