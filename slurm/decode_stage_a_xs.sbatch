#!/bin/bash
#SBATCH -J rewi-cv-a-xs
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ═══════════════════════════════════════════════════════════════════
# Decoding Study — Stage A (AR calibration) — HWRFormer (xs) variant
# Targets ar_transformer_xs checkpoints with lambda_ctc=0.1 hybrid.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=${MODEL:-hybrid}  # "hybrid" | "ar"

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ "$MODEL" != "hybrid" && "$MODEL" != "ar" ]]; then
    echo "ERROR: Stage A supports MODEL=hybrid or MODEL=ar only."
    exit 1
fi

# ── Resolve checkpoint per fold (HWRFormer xs) ──────────────────
RESULTS_BASE=/home/woody/iwso/iwso214h/imu-hwr/results/hwr2

if [[ "$MODEL" == "hybrid" ]]; then
    CKPT="${RESULTS_BASE}/train_element_word_hybrid_01_xs_onhw_wi/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    OUTDIR="results/hwr2/decode_study_xs_full_hybrid"
else
    CKPT="${RESULTS_BASE}/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_${FOLD}/${FOLD}/checkpoints/best_cer.pth"
    OUTDIR="results/hwr2/decode_study_xs_full_ar"
fi

echo "Checkpoint: $CKPT"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# Force the driver to pick the xs base config (overrides default in run_decode_stage_a.sh)
export BASE_CFG="configs/decode_study/base_${MODEL}_xs.yaml"

export FOLD MODEL CHECKPOINT="$CKPT" OUTDIR
PYTHON="$PY_BIN" bash scripts/run_decode_stage_a.sh

echo "═══ Fold $FOLD stage A (xs) complete ═══"
