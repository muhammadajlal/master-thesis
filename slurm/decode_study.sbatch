#!/bin/bash
#SBATCH -J rewi-cv
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Slurm Array Job (one fold per array task)
# ═══════════════════════════════════════════════════════════════════
#
# Submit:
#   cd work/REWI_work
#   export MODEL=hybrid   # "hybrid" | "ar" | "rewi_ctc"
#   sbatch slurm/decode_study.sbatch
#
# This runs the full sweep for each fold (0-4) in parallel.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=${MODEL:-hybrid}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

# Setup conda before reading config
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

echo "═══ Decode Study: fold=$FOLD model=$MODEL ═══"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Environment ──────────────────────────────────────────────────
cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Resolve checkpoint per fold ──────────────────────────────────
RESULTS_BASE=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2
OUTDIR_BASE=${OUTDIR_BASE:-results/hwr2}

if [[ "$MODEL" == "hybrid" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_hybrid.yaml"
    OUTDIR="${OUTDIR_BASE}/decode_study_hybrid"
elif [[ "$MODEL" == "rewi_ctc" ]]; then
    CTC_DIR="${RESULTS_BASE}/blconv_bilstm_wide_no_tokenizer/bilstm_wide__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
    if [[ -f "${CTC_DIR}/checkpoints/best_cer.pth" ]]; then
        CKPT="${CTC_DIR}/checkpoints/best_cer.pth"
    else
        echo "ERROR: REWI CTC checkpoint not found: ${CTC_DIR}/checkpoints/best_cer.pth"
        exit 1
    fi
    BASE_CFG="configs/decode_study/base_rewi_ctc.yaml"
    OUTDIR="${OUTDIR_BASE}/decode_study_rewi_ctc"
else
    # AR-only: need to find best epoch per fold from JSON results
    # Use best_cer checkpoint if available
    AR_DIR="${RESULTS_BASE}/element_word_onhw500_new/ar_transformer_s__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
    if [[ -f "${AR_DIR}/checkpoints/best_cer.pth" ]]; then
        CKPT="${AR_DIR}/checkpoints/best_cer.pth"
    else
        # Find best epoch from result JSON
        BEST_EPOCH=$($PY_BIN -c "
import json, glob
files = sorted(glob.glob('${AR_DIR}/train_*.json'))
if not files:
    print('264')  # fallback
else:
    with open(files[-1]) as f:
        d = json.load(f)
    print(d['best']['character_error_rate'][0])
")
        CKPT="${AR_DIR}/checkpoints/${BEST_EPOCH}.pth"
    fi
    BASE_CFG="configs/decode_study/base_ar.yaml"
    OUTDIR="${OUTDIR_BASE}/decode_study_ar"
fi

echo "Checkpoint: $CKPT"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# ── LM path (per-fold for proper CV) ─────────────────────────────
LM_PATH=""
if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
elif [[ -f "lm/char_5gram.binary" ]]; then
    LM_PATH="lm/char_5gram.binary"
    echo "WARNING: Using global LM for fold $FOLD (not per-fold)"
fi

# ── Run sweep ────────────────────────────────────────────────────
export FOLD MODEL LM_PATH CHECKPOINT="$CKPT" OUTDIR
echo "Output dir: $OUTDIR"
bash scripts/run_decode_study.sh \
    --fold "$FOLD" \
    --model "$MODEL" \
    --lm "$LM_PATH" \
    --checkpoint "$CKPT"

echo "═══ Fold $FOLD complete ═══"
