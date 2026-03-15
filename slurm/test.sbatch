#!/bin/bash
#SBATCH -J rewi-infer-word
#SBATCH -o logs/slurm/%x_%A_%a.out
#SBATCH -e logs/slurm/%x_%A_%a.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# -----------------------------
# User-tunable parameters
# -----------------------------
DATASET=${DATASET:-}

# YAML to use for inference.
# - If relative, it is resolved under $PROJ/configs/
# - If absolute, it is used as-is
# Examples:
#   export TEST_YAML=hybrid/train_element_word_06.yaml
#   export TEST_YAML=experiments/onhw_wi_ar_pretrained.yaml
TEST_YAML=${TEST_YAML:-train_element_word.yaml}

# Optional: override which checkpoint filename to use if the YAML's checkpoint is null
CKPT_NAME=${CKPT_NAME:-best_cer.pth}

# Optional speed knobs
SIZE_BATCH=${SIZE_BATCH:-}
NUM_WORKER=${NUM_WORKER:-}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ

if [[ -n "${DATASET}" ]]; then
  DATA=$WORK/imu-hwr/data/${DATASET}
else
  DATA=""
fi

FOLD=${SLURM_ARRAY_TASK_ID}

# Resolve YAML path
if [[ -f "$TEST_YAML" ]]; then
  YAML_PATH="$TEST_YAML"
else
  YAML_PATH="$PROJ/configs/$TEST_YAML"
fi

if [[ ! -f "$YAML_PATH" ]]; then
  echo "ERROR: YAML not found: $YAML_PATH"
  echo "Hint: set TEST_YAML to a path relative to configs/ or an absolute path."
  exit 1
fi

# -----------------------------
# Environment setup (mirrors your training script)
# -----------------------------
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh

ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

# -----------------------------
# Create per-fold temp config (patch YAML)
# -----------------------------
TMPCFG=$(mktemp "${SLURM_TMPDIR:-/tmp}/test_${DATASET:-dataset}_fold${FOLD}_job${SLURM_JOB_ID:-manual}_XXXXXX.yaml")
cp "$YAML_PATH" "$TMPCFG"

python3 - <<PY
import yaml
import os
import re
from glob import glob

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Force inference mode
cfg["test"] = True
cfg["epoch"] = 1
cfg["freq_eval"] = 1

fold = int("$FOLD")
cfg["idx_fold"] = fold

def pick_checkpoint(cfg: dict, fold: int) -> str:
  # Priority 1: environment override
  env_ckpt = os.environ.get("CKPT", "")
  if env_ckpt:
    return env_ckpt

  ckpt0 = cfg.get("checkpoint", None)
  if isinstance(ckpt0, str) and ckpt0.strip():
    # If ckpt0 already points to fold_0/0/... we rewrite fold/idx to current fold.
    ckpt = ckpt0
    ckpt = re.sub(r"/fold_\d+/\d+/checkpoints/", f"/fold_{fold}/{fold}/checkpoints/", ckpt)
    ckpt = re.sub(r"/fold_\d+/", f"/fold_{fold}/", ckpt)
    # If it exists, use it.
    if os.path.isfile(ckpt):
      return ckpt

    # Try same directory but with configured ckpt name
    ckpt_dir = os.path.dirname(ckpt)
    cand = os.path.join(ckpt_dir, os.environ.get("CKPT_NAME", "best_cer.pth"))
    if os.path.isfile(cand):
      return cand

  # Priority 2: infer from dir_work + conventional fold layout
  dir_work = cfg.get("dir_work", "")
  if isinstance(dir_work, str) and dir_work.strip():
    # Common training layout is: <dir_work>/<arch_de>__<dataset>/fold_k/k/checkpoints/
    arch_de = str(cfg.get("arch_de", ""))
    dir_dataset = str(cfg.get("dir_dataset", ""))
    dataset_name = os.path.basename(dir_dataset.rstrip("/")) if dir_dataset else ""
    run_root = os.path.join(dir_work, f"{arch_de}__{dataset_name}") if arch_de and dataset_name else dir_work
    ckpt_dir = os.path.join(run_root, f"fold_{fold}", str(fold), "checkpoints")
    ckpt_name = os.environ.get("CKPT_NAME", "best_cer.pth")
    cand = os.path.join(ckpt_dir, ckpt_name)
    if os.path.isfile(cand):
      return cand

    # Fallback: take the latest epoch-like checkpoint if present
    paths = glob(os.path.join(ckpt_dir, "*.pth"))
    if paths:
      def _score(p: str) -> int:
        b = os.path.basename(p)
        m = re.match(r"^(\d+)\.pth$", b)
        return int(m.group(1)) if m else -1
      paths.sort(key=_score)
      best = paths[-1]
      if os.path.isfile(best):
        return best

  raise FileNotFoundError(
    f"Could not infer checkpoint for fold={fold}. "
    "Set CKPT=/abs/path/to/checkpoint.pth or set cfg.checkpoint in YAML."
  )

ckpt = pick_checkpoint(cfg, fold)
if not os.path.isfile(ckpt):
  raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
cfg["checkpoint"] = ckpt

def infer_fold_workdir_from_ckpt(ckpt: str) -> str:
  # ckpt: .../fold_k/k/checkpoints/<file>.pth  -> fold dir is .../fold_k
  ckpt_dir = os.path.dirname(ckpt)          # .../checkpoints
  idx_dir = os.path.dirname(ckpt_dir)       # .../<k>
  fold_dir = os.path.dirname(idx_dir)       # .../fold_<k>
  return fold_dir

# Dataset override (optional)
data_override = "$DATA"
if data_override:
  cfg["dir_dataset"] = data_override

# Write artifacts back into the original training fold directory for cleanliness
cfg["dir_work"] = infer_fold_workdir_from_ckpt(ckpt)

# Ensure GPU inference (AR greedy decoding on CPU is extremely slow)
cfg["device"] = "cuda"

# Practical speed/stability knobs (allow override via env)
sb = os.environ.get("SIZE_BATCH", "").strip()
if sb:
  cfg["size_batch"] = int(sb)
else:
  cfg["size_batch"] = int(cfg.get("size_batch", 8) or 8)

nw = os.environ.get("NUM_WORKER", "").strip()
if nw:
  cfg["num_worker"] = int(nw)
else:
  # default to SLURM cpus-per-task if available
  slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
  cfg["num_worker"] = int(slurm_cpus) if slurm_cpus else int(cfg.get("num_worker", 8) or 8)

# Disable qualitative pipeline to avoid missing CSV + extra overhead
cfg["qualitative"] = False
cfg["qual_use_gradcam"] = False

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote patched config:", cfg_path)
print("Resolved checkpoint:", ckpt)
print("Resolved dir_work:", cfg.get("dir_work"))
PY

# -----------------------------
# Logging summary (helps debugging)
# -----------------------------
echo "========== INFER RUN INFO =========="
echo "Fold:           $FOLD"
echo "Dataset:        ${DATASET:-<from-yaml>}"
if [[ -n "${DATA}" ]]; then
  echo "dir_dataset:    $DATA (override)"
else
  echo "dir_dataset:    <from-yaml>"
fi
echo "YAML:           $YAML_PATH"
echo "Patched YAML:   $TMPCFG"
echo "dir_work:       <inferred from checkpoint fold dir>"
echo "Conda env:      $ENV_PREFIX"
echo "===================================="

cd "$PROJ"
which python3
python3 -V
nvidia-smi -L || true

echo "Running inference for fold ${FOLD} ..."
python3 main.py -c "$TMPCFG"

echo "Fold ${FOLD} finished. Searching for export JSONs..."
DIR_WORK=$(python3 - "$TMPCFG" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

print(cfg.get("dir_work", ""))
PY
)

if [[ -z "${DIR_WORK}" ]]; then
  echo "ERROR: Could not read dir_work from patched YAML: $TMPCFG"
  exit 1
fi
find "$DIR_WORK" -maxdepth 8 -type f -path "*exports*" -name "*.json" || true
find "$DIR_WORK" -maxdepth 8 -type f -path "*visualization*" -name "mat_se_*.pdf" || true
