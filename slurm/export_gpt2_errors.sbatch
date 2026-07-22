#!/bin/bash
#SBATCH -J gpt2_err_export
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH --array=0-9

# Regenerate per-sample validation predictions for the best-reported HWR-GPT
# config (Q-Former + auxiliary CTC = L1_mini_qformer) so its errors can be
# printed. Predictions were never exported during training; only checkpoints
# were retained. This runs one eval pass per (dataset, fold) with
# export_val_full=true and writes val_full_fold{k}_*_lm.json.
#
# Array 0-4 = OnHW WI (lam02, reported 6.86); 5-9 = private word (lam0.6, 16.33).

set -euo pipefail
set -x

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RES=$WORK/imu-hwr/results/hwr2/Ablations-MMLM/GPT-2/Hybrid/L1_mini_qformer
# Outside the imu-hwr git tree: private-word references must never enter a repo.
OUTROOT=$WORK/gpt2_error_export

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

TID=${SLURM_ARRAY_TASK_ID}
if (( TID < 5 )); then
  SRC="$RES/vlm__onhw_wi_word_rh_lam02"
  DSET=onhw_wi_word_rh
  TAG=onhw_wi
  FOLD=$TID
else
  SRC="$RES/vlm__wi_word_hw6_meta"
  DSET=wi_word_hw6_meta
  TAG=priv_word
  FOLD=$(( TID - 5 ))
fi

RUNDIR="$SRC/fold_${FOLD}/${FOLD}"
CKPT="$RUNDIR/checkpoints/best_cer.pth"
SRCYAML=$(ls -t "$RUNDIR"/train_*.yaml | head -1)
DATA=$WORK/imu-hwr/data/${DSET}
OUTDIR="$OUTROOT/${TAG}/fold_${FOLD}"
mkdir -p "$OUTDIR"

[[ -f "$CKPT" ]] || { echo "MISSING CKPT $CKPT"; exit 1; }
[[ -f "$SRCYAML" ]] || { echo "MISSING YAML in $RUNDIR"; exit 1; }

TMPCFG="${SLURM_TMPDIR:-/tmp}/gpt2err_${TAG}_fold${FOLD}.yaml"
cp "$SRCYAML" "$TMPCFG"

$PY_BIN - "$TMPCFG" "$FOLD" "$OUTDIR" "$DATA" "$CKPT" <<'PY'
import sys, yaml
cfg_path, fold, outdir, data, ckpt = sys.argv[1:6]
cfg = yaml.safe_load(open(cfg_path))
cfg["idx_fold"]        = int(fold)
cfg["dir_work"]        = outdir
cfg["dir_dataset"]     = data
cfg["checkpoint"]      = ckpt
cfg["test"]            = True     # eval-only path (_run_test_epoch), the actual trigger
cfg["epoch"]           = 1
cfg["epoch_warmup"]    = 0
cfg["freq_eval"]       = 1
cfg["export_val_full"] = True     # dump per-sample predictions on the eval pass
cfg["aug"]             = False    # deterministic eval, no train-time augmentation
yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
print("patched", cfg_path, "fold", fold)
PY

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJ"
nvidia-smi -L || true
echo "=== export $TAG fold $FOLD | ckpt $CKPT ==="
$PY_BIN main.py -c "$TMPCFG"
echo "=== done; exports in $OUTDIR/exports ==="
ls -la "$OUTDIR"/exports 2>/dev/null || true
