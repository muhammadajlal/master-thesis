#!/bin/bash
#SBATCH -J rewi-ew-single
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail
set -x

# ---- user knobs (override at submit time) ----
# Example:
#   sbatch --export=ALL,FOLD=2 train_element_word_single_fold.sbatch
FOLD=${FOLD:-2}
DATASET=${DATASET:-onhw_wd_word_rh}
TRAIN_YAML=${TRAIN_YAML:-train_element_word.yaml}
# --------------------------------------------

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
DATA=$WORK/imu-hwr/data/${DATASET}

# Keep the output folder consistent with the existing CV run
BASE_OUT=$WORK/imu-hwr/results/hwr2/train_element_word_complete/ar_transformer_s__${DATASET}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate $WORK/imu-hwr/envs/rewi26

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

TMPCFG="${SLURM_TMPDIR:-/tmp}/train_${DATASET}_fold${FOLD}.yaml"

# Patch fold + paths into the YAML
sed -E \
  -e "s|^[[:space:]]*idx_fold:.*|idx_fold: ${FOLD}|" \
  -e "s|^[[:space:]]*dir_work:.*|dir_work: ${OUTDIR}|" \
  -e "s|^[[:space:]]*dir_dataset:.*|dir_dataset: ${DATA}|" \
  "$PROJ/configs/${TRAIN_YAML}" > "$TMPCFG"

# If there is a half-written run directory (common when cancelling), archive it.
RUN_DIR="$OUTDIR/$FOLD"
if [[ -d "$RUN_DIR" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  mv "$RUN_DIR" "${RUN_DIR}_old_${TS}" || true
fi

cd "$PROJ"
which python3
python3 -V
nvidia-smi -L || true

echo "Running fold ${FOLD} with config: $TMPCFG"
python3 main.py -c "$TMPCFG"
