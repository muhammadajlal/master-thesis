#!/bin/bash
#SBATCH -J rewi-cv-a
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage A only (AR calibration)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=${MODEL:-hybrid}  # "hybrid" | "ar"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$MODEL" != "hybrid" && "$MODEL" != "ar" ]]; then
    echo "ERROR: Stage A supports MODEL=hybrid or MODEL=ar only."
    exit 1
fi

# ── Resolve checkpoint per fold ──────────────────────────────────
RESULTS_BASE=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2

if [[ "$MODEL" == "hybrid" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    OUTDIR="results/hwr2/decode_study_hybrid"
else
    AR_DIR="${RESULTS_BASE}/element_word_onhw500_new/ar_transformer_s__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
    if [[ -f "${AR_DIR}/checkpoints/best_cer.pth" ]]; then
        CKPT="${AR_DIR}/checkpoints/best_cer.pth"
    else
        BEST_EPOCH=$($PY_BIN -c "
import json, glob
files = sorted(glob.glob('${AR_DIR}/train_*.json'))
if not files:
    print('264')
else:
    with open(files[-1]) as f:
        d = json.load(f)
    print(d['best']['character_error_rate'][0])
")
        CKPT="${AR_DIR}/checkpoints/${BEST_EPOCH}.pth"
    fi
    OUTDIR="results/hwr2/decode_study_ar"
fi

echo "Checkpoint: $CKPT"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

export FOLD MODEL CHECKPOINT="$CKPT" OUTDIR
PYTHON="$PY_BIN" bash scripts/run_decode_stage_a.sh

echo "═══ Fold $FOLD stage A complete ═══"
