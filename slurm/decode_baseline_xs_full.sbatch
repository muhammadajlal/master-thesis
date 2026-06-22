#!/bin/bash
#SBATCH -J dec_xs_full
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — XS BASELINE full canonical cross-check
# ═══════════════════════════════════════════════════════════════════
# Reproduces the full §5.15 tab:decode-study-cross-model table for the
# baseline (xs) configuration. Runs ONE canonical config per method
# (the winning hyperparams from the scaled sweep) instead of the full
# parameter sweep — that fits in 4h per fold on rtx3080.
#
# MODEL env var: "ar" (default, AR-only) or "hybrid" (Hybrid CTC+AR).
# Set per submission:
#   sbatch slurm/decode_baseline_xs_full.sbatch               # AR-only
#   sbatch --export=ALL,MODEL=hybrid slurm/decode_baseline_xs_full.sbatch
#
# Reuses the existing scaled neural-LM at run_1512332 — text-only LM
# pretrain is architecture-independent so the same LM can be used to
# cross-check both configurations.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-ar}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

echo "═══ Decode XS FULL Cross-Check: fold=$FOLD model=$MODEL ═══"
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

if [[ "$MODEL" == "hybrid" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_hybrid_xs.yaml"
    OUTDIR="results/hwr2/decode_study_xs_full_hybrid"
else
    CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_ar_xs.yaml"
    OUTDIR="results/hwr2/decode_study_xs_full_ar"
fi

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"; exit 1
fi
echo "Checkpoint: $CKPT"
echo "Base config: $BASE_CFG"
mkdir -p "$OUTDIR"

LM_PATH=""
if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
    LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
elif [[ -f "lm/char_5gram.binary" ]]; then
    LM_PATH="lm/char_5gram.binary"
fi
echo "KenLM: $LM_PATH"

NEURAL_LM_PATH="${RESULTS_BASE}/pretrain_ar_decoder_word/run_1512332/0/checkpoints/best_loss.pth"
if [[ ! -f "$NEURAL_LM_PATH" ]]; then
    echo "WARNING: Neural LM not found — Stage D will be skipped"
    NEURAL_LM_PATH=""
else
    echo "Neural LM: $NEURAL_LM_PATH"
fi

run() {
    local tag="$1"; shift
    echo ""
    echo "── Running: $tag ──"
    $PY_BIN decode_study.py -c "$BASE_CFG" \
        --idx_fold "$FOLD" \
        --tag "${tag}__fold${FOLD}" \
        --checkpoint "$CKPT" \
        --outdir "$OUTDIR" \
        "$@"
}

# Stage A: AR decoding (greedy + canonical beam)
run "stageA0_ar_greedy"           --decoder ar --method greedy
run "stageA2_ar_beam_B4_a0p6"     --decoder ar --method beam --beam_size 4 --alpha 0.6

# Stage B: CTC decoding (Hybrid only)
if [[ "$MODEL" == "hybrid" ]]; then
    run "stageB1_ctc_greedy"     --decoder ctc --method greedy
    run "stageB2_ctc_beam_B50"   --decoder ctc --method beam --beam_size 50
    if [[ -n "$LM_PATH" && -f "$LM_PATH" ]]; then
        run "stageB3_ctc_kenlm_lw0p4" \
            --decoder ctc --method beam_lm --beam_size 25 \
            --lm_path "$LM_PATH" --lm_weight 0.4 --insertion_bonus 0.0
    fi
fi

# Stage C: AR + KenLM (canonical rescore + shallow)
if [[ -n "$LM_PATH" && -f "$LM_PATH" ]]; then
    run "stageC1_ar_kenlm_rescore_N8_lw0p1" \
        --decoder ar --method beam_rescore --nbest_size 8 --beam_size 8 \
        --alpha 0.6 --lm_path "$LM_PATH" --lm_weight 0.1 --length_bonus 0.0
    run "stageC2_ar_kenlm_shallow_B4_lw0p2" \
        --decoder ar --method beam_lm --beam_size 4 \
        --alpha 0.6 --lm_path "$LM_PATH" --lm_weight 0.2
fi

# Stage D: AR + Neural LM (canonical rescore + shallow)
if [[ -n "$NEURAL_LM_PATH" ]]; then
    run "stageD1_ar_neural_rescore_N8_lw0p2" \
        --decoder ar --method beam_rescore --nbest_size 8 --beam_size 8 \
        --alpha 0.6 --neural_lm_path "$NEURAL_LM_PATH" --lm_weight 0.2 --length_bonus 0.0
    run "stageD2_ar_neural_shallow_B4_lw0p1" \
        --decoder ar --method beam_lm --beam_size 4 \
        --alpha 0.6 --neural_lm_path "$NEURAL_LM_PATH" --lm_weight 0.1
fi

echo "═══ Fold $FOLD model=$MODEL FULL cross-check complete ═══"
