#!/bin/bash
#SBATCH -J dec_xs_xchk
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — XS Baseline Cross-Check (one fold per array task)
# ═══════════════════════════════════════════════════════════════════
#
# Mirror of `decode_study.sbatch` but for the **XS** AR baseline
# (`ar_transformer_xs` + `Baseline-AR-XS-blconv_b`). Used to cross-check
# the S-scale decode study under the XS swap (paper2/thesis).
#
# Submit:
#   cd work/REWI_work
#   sbatch slurm/decode_baseline_xs_cross_check.sbatch
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-ar}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

echo "═══ Decode XS Cross-Check: fold=$FOLD model=$MODEL ═══"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RESULTS_BASE=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2
OUTDIR_BASE=${OUTDIR_BASE:-results/hwr2}

# ── Resolve XS checkpoint per fold ───────────────────────────────
AR_XS_DIR="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
if [[ -f "${AR_XS_DIR}/checkpoints/best_cer.pth" ]]; then
    CKPT="${AR_XS_DIR}/checkpoints/best_cer.pth"
else
    BEST_EPOCH=$($PY_BIN -c "
import json, glob
files = sorted(glob.glob('${AR_XS_DIR}/train_*.json'))
if not files:
    print('264')
else:
    with open(files[-1]) as f:
        d = json.load(f)
    print(d['best']['character_error_rate'][0])
")
    CKPT="${AR_XS_DIR}/checkpoints/${BEST_EPOCH}.pth"
fi

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"

# Override BASE_CFG so run_decode_study.sh uses XS base config.
BASE_CFG="configs/decode_study/base_ar_xs.yaml"
OUTDIR="${OUTDIR_BASE}/decode_study_ar_xs"

# ── Per-fold KenLM ───────────────────────────────────────────────
LM_PATH=""
if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
elif [[ -f "lm/char_5gram.binary" ]]; then
    LM_PATH="lm/char_5gram.binary"
    echo "WARNING: Using global LM for fold $FOLD"
fi

export FOLD MODEL LM_PATH CHECKPOINT="$CKPT" OUTDIR BASE_CFG
echo "Output dir: $OUTDIR"
echo "Base config: $BASE_CFG"

bash scripts/run_decode_study.sh \
    --fold "$FOLD" \
    --model "$MODEL" \
    --lm "$LM_PATH" \
    --checkpoint "$CKPT"

echo "═══ Fold $FOLD XS cross-check complete ═══"
