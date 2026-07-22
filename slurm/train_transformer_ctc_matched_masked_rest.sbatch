#!/bin/bash
#SBATCH -J ctc-xs-masked-rest
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100,rtx3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-14

# Camera-ready: fill the remaining CNN-Transformer-CTC (matched, masked)
# columns — OnHW WD, private words, private sentences — 5 folds each.
# Array mapping: dataset = id / 5, fold = id % 5.

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

DS_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))

case $DS_IDX in
  0) TRAIN_YAML="AR-Baseline/train-transformer-xs-ctc-onhw-wd-matched-masked.yaml";      DATASET="onhw_wd_word_rh" ;;
  1) TRAIN_YAML="AR-Baseline/train-transformer-xs-ctc-stabilo-matched-masked.yaml";      DATASET="wi_word_hw6_meta" ;;
  2) TRAIN_YAML="AR-Baseline/train-transformer-xs-ctc-stabilo-sent-matched-masked.yaml"; DATASET="wi_sent_hw6_meta" ;;
  *) echo "bad DS_IDX"; exit 1 ;;
esac

DATA=$WORK/imu-hwr/data/${DATASET}

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/configs/$TRAIN_YAML"
[[ -f "$YAML_PATH" ]] || { echo "ERROR: missing $YAML_PATH"; exit 1; }

BASE_OUT=$WORK/imu-hwr/results/hwr2/Baseline-Transformer-XS-CTC-Matched-Masked/transformer_xs__${DATASET}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/train_ctc_masked_${DATASET}_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

CKPT_PATH="${OUTDIR}/${FOLD}/checkpoints/last.pth"
if [[ -f "$CKPT_PATH" ]]; then RESUME_CKPT="$CKPT_PATH"; else RESUME_CKPT=""; fi

$PY_BIN - <<PY
import yaml
cfg_path = "$TMPCFG"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"
cfg["num_worker"]  = 8
resume = "$RESUME_CKPT"
cfg["checkpoint"] = resume if resume else None
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "RUN: dataset=$DATASET fold=$FOLD yaml=$TRAIN_YAML resume=${RESUME_CKPT:-NONE}"
cd "$PROJ"
nvidia-smi -L
$PY_BIN main.py -c "$TMPCFG"
