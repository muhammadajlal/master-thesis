#!/bin/bash
#SBATCH -J rewi-pretrain-decoder
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00

set -euo pipefail
set -x

# ---- Params you can tweak ----
PRETRAIN_YAML=${PRETRAIN_YAML:-pretrain_decoder.yaml}

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

# Output directory for this run
OUTDIR_BASE=$WORK/imu-hwr/results/hwr2/${PRETRAIN_YAML%.yaml}
OUTDIR="$OUTDIR_BASE/run_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

TMPCFG="${SLURM_TMPDIR:-/tmp}/pretrain_decoder_${SLURM_JOB_ID}.yaml"
cp "$YAML_PATH" "$TMPCFG"

# Patch config to write into OUTDIR, keep idx_fold=0
$PY_BIN - <<PY
import yaml
cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
cfg["idx_fold"] = 0
cfg["dir_work"] = "$OUTDIR"
cfg["device"] = "cuda"
cfg["max_train_iters"] = None
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

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
