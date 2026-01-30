#!/bin/bash
#SBATCH -J rewi-rescore-ctc
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# -----------------------------
# User-tunable parameters
# -----------------------------
DATASET=${DATASET:-onhw_wi_word_rh}

# Option A (recommended): point to a training run folder that contains fold_*/<fold>/checkpoints
# Example:
#   export TRAIN_RUN=train_element_word_complete
#   export MODEL_TAG=ar_transformer_s__wi_word_hw6_meta
# Defaults prefilled for this run (override via --export=... if needed):
#   /home/woody/iwso/iwso214h/imu-hwr/results/hwr2/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh
TRAIN_RUN=${TRAIN_RUN:-train_element_word_hybrid_06}
MODEL_TAG=${MODEL_TAG:-ar_transformer_s__onhw_wi_word_rh}

# Option B: provide a space-separated list of 5 checkpoints (fold0..fold4)
# Example:
#   export CKPT_LIST="/path/fold0.pth /path/fold1.pth /path/fold2.pth /path/fold3.pth /path/fold4.pth"
CKPT_LIST=${CKPT_LIST:-}

ARCH_CTC=${ARCH_CTC:-linear}
BEAM_SIZE=${BEAM_SIZE:-4}
BETAS_CTC=${BETAS_CTC:-"[0.0, 0.3, 0.6, 1.0]"}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

DATA=$WORK/imu-hwr/data/${DATASET}

# Base inference YAML
BASE_YAML="$PROJ/configs/test_rescore_ctc.yaml"

# Output root (separate directory)
# Default includes run name to avoid mixing outputs across runs.
OUT_ROOT=${OUT_ROOT:-"$PROJ/inference_results_word_rescore_ctc/${TRAIN_RUN}__${MODEL_TAG}__beam${BEAM_SIZE}"}
mkdir -p "$OUT_ROOT"

# -----------------------------
# Resolve checkpoint for this fold
# -----------------------------
FOLD=${SLURM_ARRAY_TASK_ID}
CKPT=""

if [[ -n "$CKPT_LIST" ]]; then
  read -r -a CKPTS <<< "$CKPT_LIST"
  CKPT="${CKPTS[$FOLD]:-}"
elif [[ -n "$TRAIN_RUN" ]]; then
  TRAIN_BASE=$WORK/imu-hwr/results/hwr2/${TRAIN_RUN}/${MODEL_TAG}
  TRAIN_FOLD_DIR="${TRAIN_BASE}/fold_${FOLD}/${FOLD}"

  if [[ -f "${TRAIN_FOLD_DIR}/checkpoints/best_cer.pth" ]]; then
    CKPT="${TRAIN_FOLD_DIR}/checkpoints/best_cer.pth"
  elif [[ -f "${TRAIN_FOLD_DIR}/checkpoints/best_wer.pth" ]]; then
    CKPT="${TRAIN_FOLD_DIR}/checkpoints/best_wer.pth"
  else
    CKPT=$(ls -v "${TRAIN_FOLD_DIR}/checkpoints/"*.pth 2>/dev/null | tail -1 || echo "")
  fi
else
  echo "ERROR: Provide either CKPT_LIST or TRAIN_RUN (and optionally MODEL_TAG)."
  exit 2
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found for fold ${FOLD}: ${CKPT}"
  exit 1
fi

# Fold-specific work directory
FOLD_WORKDIR="${OUT_ROOT}/fold_${FOLD}"
mkdir -p "$FOLD_WORKDIR"

# -----------------------------
# Environment setup
# -----------------------------
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh

# Pick one of your envs; default matches word inference script.
ENV_PREFIX=${ENV_PREFIX:-/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26}
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

# -----------------------------
# Patch YAML into $SLURM_TMPDIR
# -----------------------------
TMPCFG="${SLURM_TMPDIR:-/tmp}/test_rescore_ctc_${DATASET}_fold${FOLD}.yaml"
cp "$BASE_YAML" "$TMPCFG"

python3 - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["test"] = True
cfg["epoch"] = 1
cfg["freq_eval"] = 1

cfg["idx_fold"] = int("$FOLD")
cfg["checkpoint"] = "$CKPT"

cfg["dir_dataset"] = "$DATA"
cfg["dir_work"] = "$FOLD_WORKDIR"

cfg["device"] = "cuda"
cfg["size_batch"] = 8
cfg["num_worker"] = 16

cfg["qualitative"] = False
cfg["qual_use_gradcam"] = False

# Ensure dual-head is enabled for rescoring (must match trained model)
dh = cfg.get("dual_head", {}) or {}
dh["enabled"] = True
dh["arch_ctc"] = "$ARCH_CTC"
cfg["dual_head"] = dh

rc = cfg.get("rescore_ctc", {}) or {}
rc["enabled"] = True
rc["beam_size"] = int("$BEAM_SIZE")
rc["betas_ctc"] = yaml.safe_load("$BETAS_CTC")
cfg["rescore_ctc"] = rc

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote patched config:", cfg_path)
print("Checkpoint:", cfg["checkpoint"])
print("dir_work:", cfg["dir_work"])
print("dual_head:", cfg.get("dual_head"))
print("rescore_ctc:", cfg.get("rescore_ctc"))
PY

# -----------------------------
# Run
# -----------------------------
echo "========== RESCORE RUN INFO =========="
echo "Fold:           $FOLD"
echo "Dataset:        $DATASET"
echo "dir_dataset:    $DATA"
echo "Checkpoint:     $CKPT"
echo "Base YAML:      $BASE_YAML"
echo "Patched YAML:   $TMPCFG"
echo "dir_work:       $FOLD_WORKDIR"
echo "Conda env:      $ENV_PREFIX"
echo "====================================="

cd "$PROJ"
python3 -V
nvidia-smi -L || true

python3 main.py -c "$TMPCFG"

echo "Fold ${FOLD} finished. Exports:" 
find "$FOLD_WORKDIR" -maxdepth 4 -type f -name "*.json" | grep -E "exports_rescore_ctc|train_|test_" || true
