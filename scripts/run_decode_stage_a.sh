#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage A only (AR calibration)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${FOLD:-0}
MODEL=${MODEL:-hybrid}  # "hybrid" or "ar"
CHECKPOINT=${CHECKPOINT:-}
OUTDIR=${OUTDIR:-results/hwr2/decode_study}
PYTHON=${PYTHON:-python3}
SKIP_DONE=${SKIP_DONE:-1}

if [[ "$MODEL" == "hybrid" ]]; then
    BASE_CFG="configs/decode_study/base_hybrid.yaml"
elif [[ "$MODEL" == "ar" ]]; then
    BASE_CFG="configs/decode_study/base_ar.yaml"
else
    echo "ERROR: Stage A is AR-only. Use MODEL=hybrid or MODEL=ar."
    exit 1
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

# A1) Beam size sweep
for B in 1 2 4 8 16; do
    run "stageA1_ar_beam_B${B}" \
        --decoder ar --method beam --beam_size "$B" --alpha 0.0
done

# A2) Length-norm sweep
for ALPHA in 0.0 0.2 0.4 0.6 0.8; do
    run "stageA2_ar_lenorm_a${ALPHA}" \
        --decoder ar --method beam --beam_size 4 --alpha "$ALPHA"
done

# A3) EOS control sweep
for GAMMA in -2.0 -1.0 0.0 1.0; do
    run "stageA3_ar_eos_g${GAMMA}" \
        --decoder ar --method beam --beam_size 4 --alpha 0.6 --eos_bias "$GAMMA"
done

# A3b) Min-length ratio sweep
for R in 0.6 0.8; do
    run "stageA3_ar_minlen_r${R}" \
        --decoder ar --method beam --beam_size 4 --alpha 0.6 --min_len_ratio "$R"
done

# A0) Greedy baseline
run "stageA0_ar_greedy" --decoder ar --method greedy

echo ""
echo "Stage A sweep complete for fold ${FOLD}."
