#!/bin/bash
#SBATCH -J hyb-corrupt
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

# 5-fold array launcher for Hybrid CTC-AR + Input-Corruption experiments.
# Usage:
#   TRAIN_YAML=HybridInputCorruption/train-hyb-uniform-onhw-wi-word.yaml \
#   DATASET=onhw_wi_word_rh sbatch slurm/train_hyb_corrupt.sbatch

set -euo pipefail
set -x

: "${TRAIN_YAML:?Set TRAIN_YAML (path relative to configs/, e.g. HybridInputCorruption/train-hyb-uniform-onhw-wi-word.yaml)}"
: "${DATASET:?Set DATASET (e.g. onhw_wi_word_rh)}"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${DATASET}

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

# Pull arch_de and a base name from dir_work
read -r ARCH_DE BASE_DIR_WORK <<< $(
$PY_BIN - "$YAML_PATH" <<'PY'
import yaml, sys
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
arch_de = cfg.get("arch_de", "unknown")
dir_work = cfg.get("dir_work", "")
if dir_work:
    parts = dir_work.split("/")
    try:
        idx = parts.index("hwr2")
        base_name = parts[idx + 1] if idx + 1 < len(parts) else "train_run"
    except (ValueError, IndexError):
        base_name = "train_run"
else:
    base_name = "train_run"
print(arch_de, base_name)
PY
)

BASE_OUT=$WORK/imu-hwr/results/hwr2/${BASE_DIR_WORK}/${ARCH_DE}__${DATASET}
FOLD=${SLURM_ARRAY_TASK_ID}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/$(basename "${TRAIN_YAML%.yaml}")_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Auto-resume: check for last.pth in the conventional location
CKPT_PATH="${OUTDIR}/${FOLD}/checkpoints/last.pth"
if [[ -f "$CKPT_PATH" ]]; then
  echo "RESUME: Found checkpoint at $CKPT_PATH"
  RESUME_CKPT="$CKPT_PATH"
else
  echo "FRESH START: No checkpoint found at $CKPT_PATH"
  RESUME_CKPT=""
fi

$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"

# Patch self-confusion confusion_path to use the actual fold
ic = cfg.get("input_corruption", {})
if ic.get("mode") == "self_confusion":
    cp = ic.get("confusion_path", "")
    if "{fold}" in cp:
        ic["confusion_path"] = cp.replace("{fold}", str(int("$FOLD")))
    cfg["input_corruption"] = ic

resume = "$RESUME_CKPT"
cfg["checkpoint"] = resume if resume else None

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== RUN INFO =========="
echo "TRAIN_YAML:          $TRAIN_YAML"
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "Patched config:      $TMPCFG"
echo "Work dir (out):      $OUTDIR"
echo "Resume checkpoint:   ${RESUME_CKPT:-NONE}"
echo "Decoder (from YAML): $ARCH_DE"
echo "=============================="

cd "$PROJ"
$PY_BIN -V
nvidia-smi -L || true

export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running fold ${FOLD} with config: $TMPCFG"
$PY_BIN main.py -c "$TMPCFG"
