#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Full Sweep Runner
# ═══════════════════════════════════════════════════════════════════
#
# Runs all stages of the decoding study on the validation set.
# Results are written to $OUTDIR/<run_tag>/
#
# Usage (local):
#   cd work/REWI_work
#   bash scripts/run_decode_study.sh [--fold 0] [--model hybrid]
#
# Usage (Slurm):
#   sbatch slurm/decode_study.sbatch
#
# Prerequisites:
#   1. Train the KenLM character LM first:
#      python scripts/train_char_lm.py --outdir lm
#   2. Ensure checkpoints exist (see configs/decode_study/)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────
FOLD=${FOLD:-0}
MODEL=${MODEL:-hybrid}  # "ar" or "hybrid" or "rewi_ctc"
# Per-fold LM by default (if available), fall back to single LM
LM_PATH=${LM_PATH:-}
NEURAL_LM_PATH=${NEURAL_LM_PATH:-results/hwr2/pretrain_ar_decoder_word/run_1512332/0/checkpoints/best_loss.pth}
PYTHON=${PYTHON:-python3}

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

# Parse arguments
CHECKPOINT=${CHECKPOINT:-}
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold) FOLD="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --lm) LM_PATH="$2"; shift 2;;
        --neural-lm) NEURAL_LM_PATH="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Allow caller to override BASE_CFG via env var (used by XS cross-check sbatch).
if [[ -z "${BASE_CFG:-}" ]]; then
    if [[ "$MODEL" == "hybrid" ]]; then
        BASE_CFG="configs/decode_study/base_hybrid.yaml"
    elif [[ "$MODEL" == "rewi_ctc" ]]; then
        BASE_CFG="configs/decode_study/base_rewi_ctc.yaml"
    else
        BASE_CFG="configs/decode_study/base_ar.yaml"
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Decoding Study — fold=$FOLD model=$MODEL"
echo "  Config: $BASE_CFG"
echo "  Checkpoint: ${CHECKPOINT:-<from YAML>}"
echo "  KenLM: $LM_PATH"
echo "  Neural LM: $NEURAL_LM_PATH"
echo "═══════════════════════════════════════════════════════════"

# Helper function
run() {
    local tag="$1"; shift
    echo ""
    echo "── Running: $tag ──"
    echo "   Args: $@"
    $PYTHON decode_study.py -c "$BASE_CFG" \
        --idx_fold "$FOLD" \
        --tag "${tag}__fold${FOLD}" \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        ${OUTDIR:+--outdir "$OUTDIR"} \
        "$@"
}

# ═════════════════════════════════════════════════════════════════
# Stage A: AR decoding calibration  (primary decoding study)
# ═════════════════════════════════════════════════════════════════
if [[ "$MODEL" != "rewi_ctc" ]]; then
    echo ""
    echo "═══ STAGE A: AR Decoding Calibration ═══"

    # A1) Beam size sweep (no LM, alpha=0)
    for B in 1 2 4 8 16; do
        run "stageA1_ar_beam_B${B}" \
            --decoder ar --method beam --beam_size "$B" --alpha 0.0
    done

    # A2) Length-norm sweep (fix B=4)
    for ALPHA in 0.0 0.2 0.4 0.6 0.8; do
        run "stageA2_ar_lenorm_a${ALPHA}" \
            --decoder ar --method beam --beam_size 4 --alpha "$ALPHA"
    done

    # A3) EOS control sweep (fix B=4, alpha=0.6)
    for GAMMA in -2.0 -1.0 0.0 1.0; do
        run "stageA3_ar_eos_g${GAMMA}" \
            --decoder ar --method beam --beam_size 4 --alpha 0.6 --eos_bias "$GAMMA"
    done

    # A3b) Min-length ratio sweep
    for R in 0.6 0.8; do
        run "stageA3_ar_minlen_r${R}" \
            --decoder ar --method beam --beam_size 4 --alpha 0.6 --min_len_ratio "$R"
    done

    # AR greedy baseline
    run "stageA0_ar_greedy" --decoder ar --method greedy
else
    echo ""
    echo "═══ STAGE A: Skipped (rewi_ctc has no AR decoder) ═══"
fi


# ═════════════════════════════════════════════════════════════════
# Stage B: CTC + external LM  (comparison arm)
# ═════════════════════════════════════════════════════════════════
# NOTE: CTC decoding works for hybrid and rewi_ctc modes
if [[ "$MODEL" == "hybrid" || "$MODEL" == "rewi_ctc" ]]; then
    echo ""
    echo "═══ STAGE B: CTC + External LM ═══"

    # B1) CTC greedy
    run "stageB1_ctc_greedy" --decoder ctc --method greedy

    # B2) CTC beam (no LM)
    for B in 10 25 50; do
        run "stageB2_ctc_beam_B${B}" \
            --decoder ctc --method beam --beam_size "$B"
    done

    # B3) CTC beam + LM shallow fusion
    if [[ -f "$LM_PATH" ]]; then
        for LW in 0.1 0.2 0.4 0.8 1.2; do
            for IB in -0.5 0.0 0.5; do
                run "stageB3_ctc_beam_lm_lw${LW}_ib${IB}" \
                    --decoder ctc --method beam_lm --beam_size 25 \
                    --lm_path "$LM_PATH" --lm_weight "$LW" --insertion_bonus "$IB"
            done
        done
    else
        echo "WARNING: LM not found at $LM_PATH — skipping Stage B3"
        echo "  Run: python scripts/train_char_lm.py --outdir lm"
    fi
else
    echo ""
    echo "═══ STAGE B: Skipped (AR-only model has no CTC decoder) ═══"
fi


# ═════════════════════════════════════════════════════════════════
# Stage C: AR + external LM  (new requirement)
# ═════════════════════════════════════════════════════════════════
if [[ "$MODEL" != "rewi_ctc" && -f "$LM_PATH" ]]; then
    echo ""
    echo "═══ STAGE C: AR + External LM ═══"

    # C1) AR N-best rescoring
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

    # C2) AR shallow fusion (optional, smaller lambda)
    for LW in 0.05 0.1 0.2 0.4; do
        run "stageC2_ar_shallow_lw${LW}" \
            --decoder ar --method beam_lm --beam_size 4 \
            --alpha 0.6 \
            --lm_path "$LM_PATH" --lm_weight "$LW"
    done
elif [[ "$MODEL" == "rewi_ctc" ]]; then
    echo ""
    echo "═══ STAGE C: Skipped (rewi_ctc has no AR decoder) ═══"
else
    echo ""
    echo "═══ STAGE C: Skipped (KenLM not found at $LM_PATH) ═══"
fi


# ═════════════════════════════════════════════════════════════════
# Stage D: Neural LM (pretrained AR decoder as external LM)
# ═════════════════════════════════════════════════════════════════
if [[ -f "$NEURAL_LM_PATH" ]]; then
    echo ""
    if [[ "$MODEL" != "rewi_ctc" ]]; then
        echo "═══ STAGE D: AR + Neural LM (pretrained AR decoder) ═══"

        # D1) AR N-best rescoring with neural LM
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

        # D2) AR shallow fusion with neural LM
        for LW in 0.05 0.1 0.2 0.4; do
            run "stageD2_ar_shallow_neural_lw${LW}" \
                --decoder ar --method beam_lm --beam_size 4 \
                --alpha 0.6 \
                --neural_lm_path "$NEURAL_LM_PATH" --lm_weight "$LW"
        done
    else
        echo "═══ STAGE D: AR sub-stages skipped (rewi_ctc has no AR decoder) ═══"
    fi

    # D3) CTC beam + neural LM (hybrid + rewi_ctc)
    if [[ "$MODEL" == "hybrid" || "$MODEL" == "rewi_ctc" ]]; then
        for LW in 0.1 0.2 0.4 0.8 1.2; do
            for IB in -0.5 0.0 0.5; do
                run "stageD3_ctc_beam_neural_lw${LW}_ib${IB}" \
                    --decoder ctc --method beam_lm --beam_size 25 \
                    --neural_lm_path "$NEURAL_LM_PATH" --lm_weight "$LW" --insertion_bonus "$IB"
            done
        done
    fi
else
    echo ""
    echo "═══ STAGE D: Skipped (Neural LM not found at $NEURAL_LM_PATH) ═══"
fi


echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Decoding study sweep complete!"
echo "  Results in: results/hwr2/decode_study/"
echo "═══════════════════════════════════════════════════════════"
