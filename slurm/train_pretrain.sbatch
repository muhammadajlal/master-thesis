#!/bin/bash
#SBATCH -J rewi-pretrain-decoder
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00

set -euo pipefail
set -x

# ---- Params you can tweak ----
PRETRAIN_YAML=${PRETRAIN_YAML:-pretrain_decoder_sent.yaml}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

YAML_PATH="$PROJ/configs/$PRETRAIN_YAML"
if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: Config file not found: $YAML_PATH"
  exit 1
fi

OUTDIR_BASE=$WORK/imu-hwr/results/hwr2/${PRETRAIN_YAML%.yaml}

# Optional: set this to an existing run directory to resume there.
# Example:
#   sbatch --export=ALL,PRETRAIN_YAML=pretrain_decoder_sent.yaml,RESUME_RUNDIR=/path/to/run_1514217 slurm/train_pretrain.sbatch
RESUME_RUNDIR=${RESUME_RUNDIR:-}

if [[ -n "$RESUME_RUNDIR" ]]; then
  if [[ ! -f "$RESUME_RUNDIR/0/checkpoints/last.pth" ]]; then
    echo "ERROR: RESUME_RUNDIR is set but checkpoint is missing: $RESUME_RUNDIR/0/checkpoints/last.pth"
    exit 1
  fi
fi

TMPCFG="${SLURM_TMPDIR:-/tmp}/pretrain_decoder_${SLURM_JOB_ID}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Patch config and choose OUTDIR.
# If resume is enabled and YAML already points at an existing run (with last.pth), reuse it.
OUTDIR="$($PY_BIN - "$TMPCFG" "$OUTDIR_BASE" "${SLURM_JOB_ID:-local}" "$RESUME_RUNDIR" <<'PY'
from pathlib import Path
import sys

import yaml

cfg_path = sys.argv[1]
outdir_base = sys.argv[2]
job_id = sys.argv[3]
resume_rundir = sys.argv[4] if len(sys.argv) > 4 else ""

with open(cfg_path, "r", encoding="utf-8") as f:
  cfg = yaml.safe_load(f) or {}

cfg["idx_fold"] = 0
cfg["device"] = "cuda"
cfg["max_train_iters"] = None

resume = bool(cfg.get("resume", False))
idx_fold = int(cfg.get("idx_fold", 0))

def has_last_pth(dir_work: str) -> bool:
  try:
    p = Path(dir_work) / str(idx_fold) / "checkpoints" / "last.pth"
    return p.exists()
  except Exception:
    return False

dir_work_cfg = cfg.get("dir_work")
if resume_rundir:
  outdir = resume_rundir
  cfg["resume"] = True
elif resume and isinstance(dir_work_cfg, str) and dir_work_cfg and has_last_pth(dir_work_cfg):
  outdir = dir_work_cfg
else:
  outdir = str(Path(outdir_base) / f"run_{job_id}")

cfg["dir_work"] = outdir

if resume and not cfg.get("resume_checkpoint"):
  cfg["resume_checkpoint"] = str(Path(outdir) / str(idx_fold) / "checkpoints" / "last.pth")

if cfg.get("resume") and not cfg.get("resume_checkpoint"):
  cfg["resume_checkpoint"] = str(Path(outdir) / str(idx_fold) / "checkpoints" / "last.pth")

with open(cfg_path, "w", encoding="utf-8") as f:
  yaml.safe_dump(cfg, f, sort_keys=False)

print(outdir)
PY
)"

mkdir -p "$OUTDIR"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "========== RUN INFO =========="
echo "Config YAML:     $PRETRAIN_YAML"
echo "Patched config:  $TMPCFG"
echo "Work dir (out):  $OUTDIR"
echo "=============================="

cd "$PROJ"
$PY_BIN -V
nvidia-smi -L || true

echo "Running pretraining with config: $TMPCFG"
$PY_BIN pretrain_decoder.py -c "$TMPCFG"
