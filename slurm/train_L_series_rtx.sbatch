#!/bin/bash
#SBATCH -J rewi-L-rtx
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
# TRAIN_YAML must be passed as environment variable
# e.g.: TRAIN_YAML=L1_mini_qformer/train-L1-mini-qformer-onhw.yaml sbatch slurm/train_L_series.sbatch
TRAIN_YAML=${TRAIN_YAML:?ERROR: Set TRAIN_YAML to the config path relative to configs/}
DATASET=${DATASET:-onhw_wi_word_rh}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
DATA=$WORK/imu-hwr/data/${DATASET}

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/configs/$TRAIN_YAML"
if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: Config file not found: $YAML_PATH"
  exit 1
fi

FOLD=${SLURM_ARRAY_TASK_ID}

# Extract dir_work from YAML and append fold
DIR_WORK=$($PY_BIN -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get('dir_work', ''))
" "$YAML_PATH")

OUTDIR="${DIR_WORK}/fold_${FOLD}"
mkdir -p "$OUTDIR"

# Patch config for this fold
TMPCFG="${SLURM_TMPDIR:-/tmp}/train_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

$PY_BIN -c "
import yaml
with open('$TMPCFG') as f:
    cfg = yaml.safe_load(f)
cfg['idx_fold'] = $FOLD
cfg['dir_work'] = '$OUTDIR'
cfg['dir_dataset'] = '$DATA'
with open('$TMPCFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print('Patched config:', '$TMPCFG')
"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJ"

echo "========== RUN INFO =========="
echo "Fold:       $FOLD"
echo "Config:     $TRAIN_YAML"
echo "Dataset:    $DATASET"
echo "Output:     $OUTDIR"
echo "=============================="

nvidia-smi -L
$PY_BIN main.py -c "$TMPCFG"
