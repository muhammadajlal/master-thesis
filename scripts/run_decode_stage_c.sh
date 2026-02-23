#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage C only (AR + KenLM)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────
FOLD=${FOLD:-0}
MODEL=${MODEL:-hybrid}  # "ar" or "hybrid"
CHECKPOINT=${CHECKPOINT:-}
LM_PATH=${LM_PATH:-}
OUTDIR=${OUTDIR:-results/hwr2/decode_study}
PYTHON=${PYTHON:-python3}
SKIP_DONE=${SKIP_DONE:-1}

# Resolve LM path: per-fold first, then global
if [[ -z "$LM_PATH" ]]; then
    if [[ -f "lm/fold_${FOLD}/char_5gram.binary" ]]; then
        LM_PATH="lm/fold_${FOLD}/char_5gram.binary"
    elif [[ -f "lm/fold_${FOLD}/char_5gram.arpa" ]]; then
        LM_PATH="lm/fold_${FOLD}/char_5gram.arpa"
    elif [[ -f "lm/char_5gram.binary" ]]; then
        LM_PATH="lm/char_5gram.binary"
        echo "WARNING: Using global LM (not per-fold). Risk of label leakage."
    elif [[ -f "lm/char_5gram.arpa" ]]; then
        LM_PATH="lm/char_5gram.arpa"
        echo "WARNING: Using global LM (not per-fold). Risk of label leakage."
    else
        LM_PATH=""
    fi
fi

if [[ "$MODEL" == "hybrid" ]]; then
    BASE_CFG="configs/decode_study/base_hybrid.yaml"
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

if [[ -z "$LM_PATH" ]]; then
    echo "ERROR: KenLM not found. Set LM_PATH or train per-fold LMs."
    exit 1
fi

# ═════════════════════════════════════════════════════════════════
# Stage C1: AR N-best rescoring with KenLM
# ═════════════════════════════════════════════════════════════════
for N in 8 16 32; do
    for LW in 0.1 0.2 0.4 0.8; do
        for LB in -0.2 0.0 0.2; do
            run "stageC1_ar_rescore_N${N}_lw${LW}_lb${LB}" \
                --decoder ar --method beam_rescore \
                --nbest_size "$N" --beam_size "$N" \
                --alpha 0.6 \
                --lm_path "$LM_PATH" --lm_weight "$LW" --length_bonus "$LB"
        done
    done
done

# ═════════════════════════════════════════════════════════════════
# Stage C2: AR shallow fusion with KenLM
# ═════════════════════════════════════════════════════════════════
for B in 2 4 8; do
    for LW in 0.05 0.1 0.2 0.4; do
        run "stageC2_ar_shallow_B${B}_lw${LW}" \
            --decoder ar --method beam_lm --beam_size "$B" \
            --alpha 0.6 \
            --lm_path "$LM_PATH" --lm_weight "$LW"
    done
done

echo ""
echo "Stage C sweep complete for fold ${FOLD}."
