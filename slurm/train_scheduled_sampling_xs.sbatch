#!/bin/bash
#SBATCH -J ss_xs
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4
# ------------------------------------------------------------------
# Scheduled-sampling probes on HWRFormer (ar_transformer_xs) + blconv_b,
# elementwise gating, 300 epochs / 30 warm-up, five-fold CV.
#
# Four variants (CONFIG selects one YAML):
#   AR-NoTeacherForcing/      ramp p 0->1 over epochs 0-150   (delay_epochs=0)
#   AR-ScheduledSamplingFixed/fixed p=0.15 throughout          (delay_epochs=0)
#   AR-SS-DelayedRamp/        TF 0-149, then ramp 0->1 over 150-300  (variant A)
#   AR-SS-DelayedAbrupt/      TF 0-149, then p=1 for 150-300         (variant B)
#
# The partition/gres are passed at submit time so the sweep can be spread
# across v100/rtx3080/a100, e.g.:
#   sbatch -p v100    --gres=gpu:v100:1    --export=ALL,CONFIG=... this.sbatch
#
# NOTE for analysis: the delayed variants are teacher-forced for epochs 0-149,
# so their global best-CER checkpoint predates any self-feedback. Select the
# best CER from epochs >= 150 only; otherwise the run just reports the
# teacher-forced baseline.
# ------------------------------------------------------------------
set -euo pipefail
set -x

CONFIG=${CONFIG:?set CONFIG=<path relative to configs/>}
FOLD=${SLURM_ARRAY_TASK_ID:-0}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh 2>/dev/null || true
ENV_PREFIX=$WORK/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/configs/$CONFIG"
[[ -f "$YAML_PATH" ]] || { echo "ERROR: config not found: $YAML_PATH"; exit 1; }

cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -gt 0 ]] || { echo "ERROR: no GPU"; exit 1; }

# Patch idx_fold and append fold_<k> to dir_work, mirroring the repo convention.
TMPCFG="${SLURM_TMPDIR:-/tmp}/$(basename "$CONFIG" .yaml)_fold${FOLD}.yaml"
cp "$YAML_PATH" "$TMPCFG"
$PY_BIN - "$TMPCFG" "$FOLD" <<'PY'
import sys, yaml
cfg_path, fold = sys.argv[1], int(sys.argv[2])
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["idx_fold"] = fold
cfg["dir_work"] = f"{cfg['dir_work'].rstrip('/')}/fold_{fold}"
ss = cfg.get("scheduled_sampling", {})
assert ss.get("enabled") is True, "scheduled_sampling must be enabled"
assert cfg["arch_de"] == "ar_transformer_xs", f"expected xs decoder, got {cfg['arch_de']}"
assert cfg["epoch"] == 300 and cfg["epoch_warmup"] == 30, "expected 300 epochs / 30 warm-up"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("patched:", cfg_path)
print("  dir_work:", cfg["dir_work"])
print("  scheduled_sampling:", ss)
PY

$PY_BIN main.py -c "$TMPCFG"
echo "=== $CONFIG fold ${FOLD} done ==="
