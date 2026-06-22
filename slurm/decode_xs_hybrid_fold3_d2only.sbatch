#!/bin/bash
#SBATCH -J dec_d2_f3
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
# ═══════════════════════════════════════════════════════════════════
# Hybrid baseline decode-study — fold 3 / Stage D2 only
# ═══════════════════════════════════════════════════════════════════
# 1709547_3 timed out on v100 after completing 8 of 9 configs; only the
# last config (stageD2_ar_neural_shallow_B4_lw0p1) was missed. This
# sbatch runs just that one config for fold 3 — ~15 min, not the full
# ~3.5h re-run. Output drops into the same dir as decode_baseline_xs_full
# (decode_study_xs_full_hybrid/stageD2_ar_neural_shallow_B4_lw0p1__fold3/).
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

FOLD=3

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_BASE=$WORK/imu-hwr/results/hwr2
CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
BASE_CFG="configs/decode_study/base_hybrid_xs.yaml"
OUTDIR="results/hwr2/decode_study_xs_full_hybrid"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"; exit 1
fi

NEURAL_LM_PATH="${RESULTS_BASE}/pretrain_ar_decoder_word/run_1512332/0/checkpoints/best_loss.pth"
if [[ ! -f "$NEURAL_LM_PATH" ]]; then
    echo "ERROR: Neural LM not found: $NEURAL_LM_PATH"; exit 1
fi

echo "═══ fold=$FOLD, single config (stageD2) ═══"
echo "Node: $(hostname)"
echo "Checkpoint: $CKPT"
echo "Neural LM:  $NEURAL_LM_PATH"

$PY_BIN decode_study.py -c "$BASE_CFG" \
    --idx_fold "$FOLD" \
    --tag "stageD2_ar_neural_shallow_B4_lw0p1__fold${FOLD}" \
    --checkpoint "$CKPT" \
    --outdir "$OUTDIR" \
    --decoder ar --method beam_lm --beam_size 4 \
    --alpha 0.6 --neural_lm_path "$NEURAL_LM_PATH" --lm_weight 0.1

echo "═══ Done ═══"
