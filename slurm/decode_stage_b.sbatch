#!/bin/bash
#SBATCH -J rewi-cv-b
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage B only (CTC + KenLM)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=${MODEL:-hybrid}  # "hybrid" | "rewi_ctc"

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

if [[ "$MODEL" != "hybrid" && "$MODEL" != "rewi_ctc" ]]; then
    echo "ERROR: Stage B supports MODEL=hybrid or MODEL=rewi_ctc only."
    exit 1
fi

# ── Resolve checkpoint per fold ──────────────────────────────────
RESULTS_BASE=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2

if [[ "$MODEL" == "hybrid" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    OUTDIR="results/hwr2/decode_study_hybrid"
else
    CTC_DIR="${RESULTS_BASE}/blconv_bilstm_wide_no_tokenizer/bilstm_wide__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
    if [[ -f "${CTC_DIR}/checkpoints/best_cer.pth" ]]; then
        CKPT="${CTC_DIR}/checkpoints/best_cer.pth"
    else
        echo "ERROR: REWI CTC checkpoint not found: ${CTC_DIR}/checkpoints/best_cer.pth"
        exit 1
    fi
    OUTDIR="results/hwr2/decode_study_rewi_ctc"
fi

echo "Checkpoint: $CKPT"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# Resolve per-fold KenLM path
LM_PATH=""
if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
elif [[ -f "lm/char_5gram.binary" ]]; then
    LM_PATH="lm/char_5gram.binary"
    echo "WARNING: Using global LM for fold $FOLD (not per-fold)"
fi

export FOLD MODEL LM_PATH CHECKPOINT="$CKPT" OUTDIR
PYTHON="$PY_BIN" bash scripts/run_decode_stage_b.sh

echo "═══ Fold $FOLD stage B complete ═══"
