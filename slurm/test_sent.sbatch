#!/bin/bash
#SBATCH -J rewi-infer-sent
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# -----------------------------
# User-tunable parameters
# -----------------------------
DATASET=${DATASET:-wi_sent_hw6_meta}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

DATA=$WORK/imu-hwr/data/${DATASET}

# Base inference YAML (must exist)
BASE_YAML="$PROJ/configs/test_sent.yaml"

# Output root for inference artifacts
OUT_ROOT="$PROJ/inference_results_sent"
mkdir -p "$OUT_ROOT"

# -----------------------------
# Best checkpoints per fold (SENT-level)
# -----------------------------
CKPTS=(
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_0/0/checkpoints/269.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_1/1/checkpoints/299.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_2/2/checkpoints/279.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_3/3/checkpoints/254.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_sent_hw6_meta/fold_4/4/checkpoints/274.pth"
)

FOLD=${SLURM_ARRAY_TASK_ID}
CKPT="${CKPTS[$FOLD]}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

# Fold-specific work directory (prevents collisions across array tasks)
FOLD_WORKDIR="${OUT_ROOT}/fold_${FOLD}"
mkdir -p "$FOLD_WORKDIR"

# -----------------------------
# Environment setup (mirrors your training script)
# -----------------------------
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh

ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

# -----------------------------
# Create per-fold temp config (patch test.yaml)
# -----------------------------
TMPCFG="${SLURM_TMPDIR:-/tmp}/test_${DATASET}_fold${FOLD}.yaml"
cp "$BASE_YAML" "$TMPCFG"

python3 - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Force inference mode
cfg["test"] = True
cfg["epoch"] = 1
cfg["freq_eval"] = 1

# Fold + checkpoint
cfg["idx_fold"] = int("$FOLD")
cfg["checkpoint"] = "$CKPT"

# Dataset + output (fold-specific directory)
cfg["dir_dataset"] = "$DATA"
cfg["dir_work"] = "$FOLD_WORKDIR"

# Ensure GPU inference
cfg["device"] = "cuda"

# Practical speed/stability knobs
cfg["size_batch"] = 8
cfg["num_worker"] = 16

# Disable qualitative pipeline
cfg["qualitative"] = False
cfg["qual_use_gradcam"] = False

# OPTIONAL: ensure you use the sentence categories (if your test.yaml still has word categories)
# If you already maintain separate test.yaml files for word vs sent, remove this block.
# Otherwise, point to a sent-specific categories list or keep as-is if your dataset loader ignores it.
#
# cfg["categories"] = [...]

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote patched config:", cfg_path)
PY

# -----------------------------
# Logging summary
# -----------------------------
echo "========== INFER RUN INFO =========="
echo "Fold:           $FOLD"
echo "Dataset:        $DATASET"
echo "dir_dataset:    $DATA"
echo "Checkpoint:     $CKPT"
echo "Base YAML:      $BASE_YAML"
echo "Patched YAML:   $TMPCFG"
echo "dir_work:       $FOLD_WORKDIR"
echo "Conda env:      $ENV_PREFIX"
echo "===================================="

cd "$PROJ"
which python3
python3 -V
nvidia-smi -L || true

echo "Running inference for fold ${FOLD} ..."
python3 main.py -c "$TMPCFG"

echo "Fold ${FOLD} finished. Searching for export JSONs..."
find "$FOLD_WORKDIR" -maxdepth 8 -type f -path "*exports*" -name "*.json" || true
