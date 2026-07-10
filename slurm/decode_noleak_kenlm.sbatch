#!/bin/bash
#SBATCH -J dec_noleak
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Re-decode the LM-dependent CTC+KenLM rows with the LEAK-FREE per-fold
# KenLM (built from the disjoint val.json partition; see
# scripts/train_char_lm.py). Only these rows change when the LM changes;
# greedy/beam-only rows are LM-independent and are reused as-is.
#
#   MODEL=rewi      -> full REWI lambda x beta grid (heatmap + curve + row)
#   MODEL=hybrid_xs -> hybrid xs CTC head, lambda sweep at beta=0
#
#   sbatch --export=ALL,MODEL=rewi      slurm/decode_noleak_kenlm.sbatch
#   sbatch --export=ALL,MODEL=hybrid_xs slurm/decode_noleak_kenlm.sbatch
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODEL:-hybrid_xs}
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX" 2>/dev/null || true
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESULTS_BASE=$WORK/imu-hwr/results/hwr2

# Leak-free per-fold KenLM (built to lm_noleak/ by train_char_lm.py --per_fold)
LM_PATH="lm_noleak/fold_${FOLD}/char_5gram.binary"
[[ -f "$LM_PATH" ]] || { echo "ERROR: clean LM missing: $LM_PATH"; exit 1; }
echo "Clean KenLM: $LM_PATH"

if [[ "$MODEL" == "rewi" ]]; then
    CKPT="${RESULTS_BASE}/Baseline-REWI/bilstm_wide__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_rewi_ctc.yaml"
    OUTDIR="results/hwr2/decode_study_rewi_ctc_noleak"
elif [[ "$MODEL" == "hybrid_xs" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    BASE_CFG="configs/decode_study/base_hybrid_xs.yaml"
    OUTDIR="results/hwr2/decode_study_xs_full_hybrid_noleak"
else
    echo "ERROR: MODEL must be rewi or hybrid_xs"; exit 1
fi
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
mkdir -p "$OUTDIR"
echo "Checkpoint: $CKPT"

run() {
    local tag="$1"; shift
    echo ""; echo "── $tag ──"
    $PY_BIN decode_study.py -c "$BASE_CFG" \
        --idx_fold "$FOLD" --tag "${tag}__fold${FOLD}" \
        --checkpoint "$CKPT" --outdir "$OUTDIR" --device cuda "$@"
}

if [[ "$MODEL" == "rewi" ]]; then
    # Full lambda x beta grid at B=25 (reproduces the heatmap grid on the clean LM)
    for LW in 0.1 0.2 0.4 0.8 1.2; do
        for IB in -0.5 0.0 0.5; do
            run "stageB3_ctc_beam_lm_lw${LW}_ib${IB}" \
                --decoder ctc --method beam_lm --beam_size 25 \
                --lm_path "$LM_PATH" --lm_weight "$LW" --insertion_bonus "$IB"
        done
    done
else
    # Hybrid xs CTC head: lambda sweep at beta=0 to reconfirm the argmin + operating point
    for LW in 0.2 0.4 0.8; do
        run "stageB3_ctc_kenlm_lw${LW}" \
            --decoder ctc --method beam_lm --beam_size 25 \
            --lm_path "$LM_PATH" --lm_weight "$LW" --insertion_bonus 0.0
    done
fi
echo "═══ fold $FOLD ($MODEL) noleak re-decode complete ═══"
