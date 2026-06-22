#!/bin/bash
#SBATCH -J ar_hyb_qual
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
# NOTE: This cluster forbids --mem with GPU jobs. Remove/adjust if CPU-only.
#SBATCH -t 24:00:00

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

# ---- Params you can tweak ----
AR_CKPT=${AR_CKPT:-$WORK/imu-hwr/results/hwr2/Baseline-AR-ElementwiseGating/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth}
HYB_CKPT=${HYB_CKPT:-$WORK/imu-hwr/results/hwr2/Baseline-Hybrid/train_element_word_hybrid_06/ar_transformer_s__onhw_wi_word_rh/fold_0/0/checkpoints/best_cer.pth}
DATASET=${DATASET:-onhw_wi_word_rh}
FOLD=${FOLD:-0}
OUTDIR=${OUTDIR:-figures/ar_vs_hybrid_new_new}
DIFFICULTY_MODE=${DIFFICULTY_MODE:-difficulty}
DEVICE=${DEVICE:-cuda}

# Optional CSV-based difficulty selection
DIFFICULTY_CSV=${DIFFICULTY_CSV:-}
# Set a sensible default task name from your unified CSV.
DIFFICULTY_TASK=${DIFFICULTY_TASK:-AR-only}
DIFFICULTY_SEP=${DIFFICULTY_SEP:-;}
DIFFICULTY_QUANTILES=${DIFFICULTY_QUANTILES:-0.50,0.50,0.90,0.90,0.99,0.99}
DIFFICULTY_N=${DIFFICULTY_N:-6}

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

cd "$PROJ"

DEFAULT_DIFFICULTY_CSV="$PROJ/analysis/quant_all_val_predictions_ar_vs_hybrid.csv"
if [[ -z "$DIFFICULTY_CSV" && -f "$DEFAULT_DIFFICULTY_CSV" ]]; then
  DIFFICULTY_CSV="$DEFAULT_DIFFICULTY_CSV"
fi

if [[ "$DIFFICULTY_MODE" == "difficulty" ]]; then
  if [[ -z "$DIFFICULTY_CSV" || ! -f "$DIFFICULTY_CSV" ]]; then
    echo "ERROR: DIFFICULTY_MODE=difficulty but DIFFICULTY_CSV is missing or not found."
    echo "Set DIFFICULTY_CSV to a valid CSV path, or use DIFFICULTY_MODE=random."
    exit 1
  fi
fi

ARGS=(
  analysis/scripts/compare_ar_hybrid.py
  --ar_ckpt "$AR_CKPT"
  --hyb_ckpt "$HYB_CKPT"
  --dataset "data/$DATASET"
  --fold "$FOLD"
  --outdir "$OUTDIR"
  --difficulty_mode "$DIFFICULTY_MODE"
  --device "$DEVICE"
)

if [[ -n "$DIFFICULTY_CSV" ]]; then
  ARGS+=(--difficulty_csv "$DIFFICULTY_CSV")
fi
if [[ -n "$DIFFICULTY_TASK" ]]; then
  ARGS+=(--difficulty_task "$DIFFICULTY_TASK")
fi
if [[ -n "$DIFFICULTY_SEP" ]]; then
  ARGS+=(--difficulty_sep "$DIFFICULTY_SEP")
fi
if [[ -n "$DIFFICULTY_QUANTILES" ]]; then
  ARGS+=(--difficulty_quantiles "$DIFFICULTY_QUANTILES")
fi
if [[ -n "$DIFFICULTY_N" ]]; then
  ARGS+=(--difficulty_n "$DIFFICULTY_N")
fi

$PY_BIN "${ARGS[@]}"
