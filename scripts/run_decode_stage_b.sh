#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage B only (CTC + KenLM)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${FOLD:-0}
MODEL=${MODEL:-hybrid}  # "hybrid" or "rewi_ctc"
CHECKPOINT=${CHECKPOINT:-}
LM_PATH=${LM_PATH:-}
OUTDIR=${OUTDIR:-results/hwr2/decode_study}
PYTHON=${PYTHON:-python3}
SKIP_DONE=${SKIP_DONE:-1}

if [[ "$MODEL" == "hybrid" ]]; then
    BASE_CFG="configs/decode_study/base_hybrid.yaml"
elif [[ "$MODEL" == "rewi_ctc" ]]; then
    BASE_CFG="configs/decode_study/base_rewi_ctc.yaml"
else
    echo "ERROR: Stage B is CTC-only. Use MODEL=hybrid or MODEL=rewi_ctc."
    exit 1
fi

# Resolve LM path
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
    fi
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

# B1) CTC greedy
run "stageB1_ctc_greedy" --decoder ctc --method greedy

# B2) CTC beam (no LM)
for B in 10 25 50; do
    run "stageB2_ctc_beam_B${B}" \
        --decoder ctc --method beam --beam_size "$B"
done

# B3) CTC beam + KenLM
if [[ -n "$LM_PATH" && -f "$LM_PATH" ]]; then
    for LW in 0.1 0.2 0.4 0.8 1.2; do
        for IB in -0.5 0.0 0.5; do
            run "stageB3_ctc_beam_lm_lw${LW}_ib${IB}" \
                --decoder ctc --method beam_lm --beam_size 25 \
                --lm_path "$LM_PATH" --lm_weight "$LW" --insertion_bonus "$IB"
        done
    done
else
    echo "WARNING: KenLM not found for fold ${FOLD}; skipping B3"
fi

echo ""
echo "Stage B sweep complete for fold ${FOLD}."
