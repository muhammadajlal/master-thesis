#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage D only (Neural LM)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────
FOLD=${FOLD:-0}
MODEL=${MODEL:-hybrid}  # "ar" or "hybrid" or "rewi_ctc"
CHECKPOINT=${CHECKPOINT:-}
NEURAL_LM_PATH=${NEURAL_LM_PATH:-results/hwr2/pretrain_ar_decoder_word/run_1512332/0/checkpoints/best_loss.pth}
OUTDIR=${OUTDIR:-results/hwr2/decode_study}
PYTHON=${PYTHON:-python3}
SKIP_DONE=${SKIP_DONE:-1}

if [[ "$MODEL" == "hybrid" ]]; then
    BASE_CFG="configs/decode_study/base_hybrid.yaml"
elif [[ "$MODEL" == "rewi_ctc" ]]; then
    BASE_CFG="configs/decode_study/base_rewi_ctc.yaml"
else
    BASE_CFG="configs/decode_study/base_ar.yaml"
fi

run() {
    local tag="$1"; shift
    local outpath="${OUTDIR}/${tag}__fold${FOLD}/metrics.json"
    if [[ "$SKIP_DONE" == "1" && -f "$outpath" ]]; then
        echo "── Skipping (exists): $tag"
        return 0
    fi
    echo ""
    echo "── Running: $tag ──"
    echo "   Args: $@"
    $PYTHON decode_study.py -c "$BASE_CFG" \
        --idx_fold "$FOLD" \
        --outdir "$OUTDIR" \
        --tag "${tag}__fold${FOLD}" \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        "$@"
}

if [[ ! -f "$NEURAL_LM_PATH" ]]; then
    echo "ERROR: Neural LM not found at $NEURAL_LM_PATH"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════
# Stage D1/D2: AR + Neural LM (only if AR decoder exists)
# ═════════════════════════════════════════════════════════════════
if [[ "$MODEL" != "rewi_ctc" ]]; then
    for N in 8 16 32; do
        for LW in 0.1 0.2 0.4 0.8; do
            for LB in -0.2 0.0 0.2; do
                run "stageD1_ar_rescore_neural_N${N}_lw${LW}_lb${LB}" \
                    --decoder ar --method beam_rescore \
                    --nbest_size "$N" --beam_size "$N" \
                    --alpha 0.6 \
                    --neural_lm_path "$NEURAL_LM_PATH" --lm_weight "$LW" --length_bonus "$LB"
            done
        done
    done

    for LW in 0.05 0.1 0.2 0.4; do
        for B in 2 4 8; do
            run "stageD2_ar_shallow_neural_B${B}_lw${LW}" \
                --decoder ar --method beam_lm --beam_size "$B" \
                --alpha 0.6 \
                --neural_lm_path "$NEURAL_LM_PATH" --lm_weight "$LW"
        done
    done
else
    echo "Skipping D1/D2 (rewi_ctc has no AR decoder)."
fi

# ═════════════════════════════════════════════════════════════════
# Stage D3: CTC + Neural LM (hybrid + rewi_ctc)
# ═════════════════════════════════════════════════════════════════
if [[ "$MODEL" == "hybrid" || "$MODEL" == "rewi_ctc" ]]; then
    for LW in 0.1 0.2 0.4 0.8 1.2; do
        for IB in -0.5 0.0 0.5; do
            run "stageD3_ctc_beam_neural_lw${LW}_ib${IB}" \
                --decoder ctc --method beam_lm --beam_size 25 \
                --neural_lm_path "$NEURAL_LM_PATH" --lm_weight "$LW" --insertion_bonus "$IB"
        done
    done
fi

echo ""
echo "Stage D sweep complete for fold ${FOLD}."
