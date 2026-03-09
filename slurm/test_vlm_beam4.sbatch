#!/bin/bash
#SBATCH -J vlm-beam4
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Configuration ----
TRAIN_YAML=${TRAIN_YAML:-test-vlm-beam4.yaml}
DATASET=${DATASET:-onhw_wi_word_rh}

# Source model results (v2 checkpoints)
SRC_RESULTS=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2/vlm_gpt2_word_v2/vlm__onhw_wi_word_rh

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
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

# Extract dir_work base from YAML
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
        base_name = parts[idx + 1] if idx + 1 < len(parts) else "eval_run"
    except (ValueError, IndexError):
        base_name = "eval_run"
else:
    base_name = "eval_run"
print(arch_de, base_name)
PY
)

FOLD=${SLURM_ARRAY_TASK_ID}
BASE_OUT=$WORK/imu-hwr/results/hwr2/${BASE_DIR_WORK}/${ARCH_DE}__${DATASET}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

# Find best checkpoint from v2 training
CKPT_PATH="${SRC_RESULTS}/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT_PATH"
  exit 1
fi
echo "Using checkpoint: $CKPT_PATH"

TMPCFG="${SLURM_TMPDIR:-/tmp}/test_beam4_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Patch config
$PY_BIN - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"
cfg["checkpoint"]  = "$CKPT_PATH"

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== BEAM EVAL INFO =========="
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "Checkpoint:          $CKPT_PATH"
echo "Output dir:          $OUTDIR"
echo "Beams:               4"
echo "======================================"

cd "$PROJ"

# GPU verification
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"

export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Running beam4 eval fold ${FOLD}"
$PY_BIN main.py -c "$TMPCFG"
