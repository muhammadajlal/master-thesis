#!/bin/bash
#SBATCH -J rescore-fixedcap
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-14

# Camera-ready sensitivity check: re-evaluate retained best-CER AR-XS
# checkpoints (OnHW-WI) with the label-independent decode cap
# (decode_max_len=32 + EOS early stop) instead of the old
# len_y.max()+2 horizon. Eval-only; writes test_*.json under
# results/hwr2/CameraReady-RescoreFixedCap/.

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

MANIFEST=$WORK/imu-hwr/results/hwr2/CameraReady-RescoreFixedCap/manifest.txt
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$CFG" || ! -f "$CFG" ]]; then
  echo "ERROR: no config for array index $SLURM_ARRAY_TASK_ID"
  exit 1
fi

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"
nvidia-smi -L
echo "Rescoring with config: $CFG"
$PY_BIN main.py -c "$CFG"
