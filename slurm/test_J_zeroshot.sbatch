#!/bin/bash
# ==================================================================
# Zero-Shot evaluation of a J-series checkpoint on OnHW equations.
# Invoke with env vars:
#   VARIANT   = J1_mlp | J1_pooling | J2_mlp | J2_pooling
#   TARGET    = equations_wi | equations_wd
# e.g. VARIANT=J1_mlp TARGET=equations_wi sbatch test_J_zeroshot.sbatch
# ==================================================================
#SBATCH -J J-zshot
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

VARIANT=${VARIANT:?"Set VARIANT=J1_mlp|J1_pooling|J2_mlp|J2_pooling"}
TARGET=${TARGET:?"Set TARGET=equations_wi|equations_wd"}

case "$VARIANT" in
  J1_mlp)      SRC_GROUP=J1_contrastive_mlp     ; SRC_DATA=onhw_wi_word_rh   ; CONNECTOR=mlp     ;;
  J1_pooling)  SRC_GROUP=J1_contrastive_pooling ; SRC_DATA=onhw_wi_word_rh   ; CONNECTOR=pooling ;;
  J2_mlp)      SRC_GROUP=J2_contrastive_mlp     ; SRC_DATA=wi_word_hw6_meta  ; CONNECTOR=mlp     ;;
  J2_pooling)  SRC_GROUP=J2_contrastive_pooling ; SRC_DATA=wi_word_hw6_meta  ; CONNECTOR=pooling ;;
  *) echo "Unknown VARIANT=$VARIANT"; exit 2 ;;
esac

case "$TARGET" in
  equations_wi) TARGET_DATASET=onhw_equations_wi_word_rh ;;
  equations_wd) TARGET_DATASET=onhw_equations_wd_word_rh ;;
  *) echo "Unknown TARGET=$TARGET"; exit 2 ;;
esac

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${TARGET_DATASET}

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

NOLORA_SUFFIX=""
if [[ "${NO_LORA:-0}" == "1" ]]; then
  NOLORA_SUFFIX="-nolora"
fi
BASE_YAML="$PROJ/configs/zero_shot/test-J-${CONNECTOR}-zeroshot${NOLORA_SUFFIX}.yaml"
if [[ ! -f "$BASE_YAML" ]]; then
  echo "ERROR: Base config not found: $BASE_YAML"
  exit 1
fi

SRC_RESULTS="$WORK/imu-hwr/results/hwr2/Ablations-MMLM/GPT-2/Hybrid-Contrastive/${SRC_GROUP}/vlm__${SRC_DATA}"

FOLD=${SLURM_ARRAY_TASK_ID}
OUT_ROOT="$WORK/imu-hwr/results/hwr2/ZeroShot/${VARIANT}__${TARGET}${NOLORA_SUFFIX}"
OUTDIR="${OUT_ROOT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

CKPT_PATH="${SRC_RESULTS}/fold_${FOLD}/${FOLD}/checkpoints/best_ar_cer.pth"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT_PATH"
  exit 1
fi

TMPCFG="${SLURM_TMPDIR:-/tmp}/zshot_${VARIANT}_${TARGET}_fold${FOLD}.yaml"
cp "$BASE_YAML" "$TMPCFG"

$PY_BIN - <<PY
import yaml
with open("$TMPCFG", "r") as f:
    cfg = yaml.safe_load(f)
cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"
cfg["checkpoint"]  = "$CKPT_PATH"
with open("$TMPCFG", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("Patched config:", "$TMPCFG")
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========== ZERO-SHOT EVAL =========="
echo "Variant:        $VARIANT"
echo "Source:         $SRC_GROUP (trained on $SRC_DATA)"
echo "Target dataset: $TARGET_DATASET  (unseen character classes)"
echo "Fold:           $FOLD"
echo "Checkpoint:     $CKPT_PATH"
echo "Output dir:     $OUTDIR"
echo "NOTE: CTC-head metrics on this target are INVALID (char-set mismatch)."
echo "NOTE: Only AR (GPT-2) CER/WER are valid zero-shot results."
echo "===================================="

cd "$PROJ"
nvidia-smi -L
$PY_BIN main.py -c "$TMPCFG"
