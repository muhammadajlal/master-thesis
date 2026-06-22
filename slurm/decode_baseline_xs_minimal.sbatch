#!/bin/bash
#SBATCH -J dec_xs_min
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — XS Baseline MINIMAL cross-check
# ═══════════════════════════════════════════════════════════════════
# Companion to the full decode_baseline_xs_cross_check.sbatch (timed out
# at 4h running the full sweep). This minimal version runs the THREE
# configs needed to validate §5.15 finding #1 ("AR decoding is near-
# optimal with greedy search; LM doesn't help AR") on baseline:
#   1. AR greedy (baseline reference)
#   2. AR calibrated beam B=4 alpha=0.6 (canonical winning beam config)
#   3. AR + KenLM shallow fusion B=4 alpha=0.6 lw=0.2 (LM saturation test)
# Per fold task: 3 inference runs × ~15 min = ~45 min, well within 2h.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=ar

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

echo "═══ Decode XS Minimal Cross-Check: fold=$FOLD ═══"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_BASE=$WORK/imu-hwr/results/hwr2
AR_XS_DIR="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}"
CKPT="${AR_XS_DIR}/checkpoints/best_cer.pth"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"

BASE_CFG="configs/decode_study/base_ar_xs.yaml"
OUTDIR="results/hwr2/decode_study_ar_xs_minimal"
mkdir -p "$OUTDIR"

# Resolve per-fold KenLM (5-gram character LM) for the LM-saturation test
LM_PATH=""
if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
elif [[ -f "lm/char_5gram.binary" ]]; then
    LM_PATH="lm/char_5gram.binary"
    echo "WARNING: Using global LM for fold $FOLD"
fi
echo "KenLM: $LM_PATH"

# Helper — invokes decode_study.py with the right boilerplate
run() {
    local tag="$1"; shift
    echo ""
    echo "── Running: $tag ──"
    echo "   Args: $@"
    $PY_BIN decode_study.py -c "$BASE_CFG" \
        --idx_fold "$FOLD" \
        --tag "${tag}__fold${FOLD}" \
        --checkpoint "$CKPT" \
        --outdir "$OUTDIR" \
        "$@"
}

# ═════════════════════════════════════════════════════════════════
# CONFIG 1: AR greedy (reference)
# ═════════════════════════════════════════════════════════════════
run "stageA0_ar_greedy" --decoder ar --method greedy

# ═════════════════════════════════════════════════════════════════
# CONFIG 2: AR calibrated beam B=4 alpha=0.6 (canonical winning
# config per tab:decode-study-cross-model; matches scaled rows for
# Calibrated beam B=4)
# ═════════════════════════════════════════════════════════════════
run "stageA_calibrated_beam_B4_alpha0p6" \
    --decoder ar --method beam --beam_size 4 --alpha 0.6

# ═════════════════════════════════════════════════════════════════
# CONFIG 3: AR + KenLM shallow fusion (LM saturation test)
# Direct cross-check of §5.15 finding #1: LM should NOT improve
# baseline AR CER, mirroring the scaled-config result.
# ═════════════════════════════════════════════════════════════════
if [[ -n "$LM_PATH" && -f "$LM_PATH" ]]; then
    run "stageC_ar_kenlm_shallow_lw0p2" \
        --decoder ar --method beam_lm --beam_size 4 --alpha 0.6 \
        --lm_path "$LM_PATH" --lm_weight 0.2
else
    echo "WARNING: KenLM not available — skipping LM saturation test"
fi

echo "═══ Fold $FOLD minimal cross-check complete ═══"
