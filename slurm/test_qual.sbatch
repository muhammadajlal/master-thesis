#!/bin/bash
#SBATCH -J rewi-infer-word
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# -----------------------------
# User-tunable parameters
# -----------------------------
DATASET=${DATASET:-wi_word_hw6_meta}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

DATA=$WORK/imu-hwr/data/${DATASET}

# Base YAML (your provided test.yaml)
BASE_YAML="$PROJ/configs/test.yaml"

# Output roots
OUT_ROOT="$PROJ/quant_results_word"
QUAL_ROOT="$PROJ/qualitative_outputs"
mkdir -p "$OUT_ROOT" "$QUAL_ROOT"

# -----------------------------
# Best checkpoints per fold (WORD-level)
# -----------------------------
CKPTS=(
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_word_hw6_meta/fold_0/0/checkpoints/299.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_word_hw6_meta/fold_1/1/checkpoints/294.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_word_hw6_meta/fold_2/2/checkpoints/284.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_word_hw6_meta/fold_3/3/checkpoints/269.pth"
  "$WORK/imu-hwr/results/hwr2/blconv_ARDecoder_no_tokenizer/ar_transformer_s__wi_word_hw6_meta/fold_4/4/checkpoints/244.pth"
)

FOLD=${SLURM_ARRAY_TASK_ID}
CKPT="${CKPTS[$FOLD]}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

# Fold-specific work directories (prevents collisions across array tasks)
FOLD_WORKDIR="${OUT_ROOT}/fold_${FOLD}"
FOLD_QUALDIR="${QUAL_ROOT}/fold_${FOLD}"
mkdir -p "$FOLD_WORKDIR" "$FOLD_QUALDIR"

# -----------------------------
# Environment setup
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

# Keep inference mode as per YAML, but enforce safe values
cfg["test"] = True
cfg["epoch"] = 1
cfg["freq_eval"] = 1

# Fold + checkpoint
cfg["idx_fold"] = int("$FOLD")
cfg["checkpoint"] = "$CKPT"

# Dataset + outputs
cfg["dir_dataset"] = "$DATA"
cfg["dir_work"] = "$FOLD_WORKDIR"

# IMPORTANT: your YAML sets device: cpu; for AR decoding you want GPU
cfg["device"] = "cuda"

# Optional: keep YAML batch size unless you want to override
# If 64 is too large for A100 in your model, reduce here.
# cfg["size_batch"] = 64

# Workers: match SLURM
cfg["num_worker"] = 16

# Qualitative: keep enabled, but make outdir fold-specific
if cfg.get("qualitative", False):
    cfg["qual_outdir"] = "$FOLD_QUALDIR"
    # Keep qual_csv from YAML as-is unless you want per-fold CSV

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
echo "qual_outdir:    $FOLD_QUALDIR"
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

echo "Fold ${FOLD} finished. Searching for qualitative outputs..."
find "$FOLD_QUALDIR" -maxdepth 3 -type f \( -name "*.png" -o -name "*.csv" \) || true
