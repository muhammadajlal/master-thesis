#!/bin/bash
#SBATCH -J rewi-cv
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err
#SBATCH -p a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --array=0-4

set -euo pipefail
set -x

# ---- Params you can tweak ----
DATASET=${DATASET:-wi_word_hw6_meta}

# NEW: which base YAML to use (in configs/)
# e.g. TRAIN_CFG=train_element_word.yaml
TRAIN_CFG=${TRAIN_CFG:-train.yaml}

# Toggle tokenizer/BPE usage: "true"/"false" or "1"/"0"
USE_BPE=${USE_BPE:-false}

# BPE size only matters if USE_BPE=true
BPE_SIZE=${BPE_SIZE:-500}                # 100|200|500 etc.

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
export PROJ
DATA=$WORK/imu-hwr/data/${DATASET}

# Map TRAIN_CFG -> experiment tag (for result folder)
case "$TRAIN_CFG" in
  train_element_word.yaml) EXP_TAG="element_word" ;;
  train_element_sent.yaml) EXP_TAG="element_sent" ;;
  train_head_word.yaml)    EXP_TAG="head_word" ;;
  train_head_sent.yaml)    EXP_TAG="head_sent" ;;
  *)                       EXP_TAG="${TRAIN_CFG%.*}" ;;   # fallback
esac

# Read arch_de from the selected base train yaml (do NOT override it)
ARCH_DE=$(
python3 - "$PROJ" "$TRAIN_CFG" <<'PY'
import yaml, os, sys
proj = sys.argv[1]
cfg_name = sys.argv[2]
p = os.path.join(proj, "configs", cfg_name)
with open(p, "r") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("arch_de", "unknown"))
PY
)

# Results folder: encode experiment + tokenizer usage + arch_de + dataset
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  BASE_OUT=$WORK/imu-hwr/results/hwr2/${EXP_TAG}_tokenizer_${BPE_SIZE}/${ARCH_DE}__${DATASET}
else
  BASE_OUT=$WORK/imu-hwr/results/hwr2/${EXP_TAG}_no_tokenizer_init/${ARCH_DE}__${DATASET}
fi

FOLD=${SLURM_ARRAY_TASK_ID}
OUTDIR="${BASE_OUT}/fold_${FOLD}"
mkdir -p "$OUTDIR"

# Always define BPE_MODEL so set -u doesn't complain
BPE_MODEL=""

# --- Only define / check BPE model if we actually use it ---
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  BPE_DIR="$PROJ/tokenizer"
  BPE_PREFIX="bpe${BPE_SIZE}_fold_${FOLD}"
  BPE_MODEL="${BPE_DIR}/${BPE_PREFIX}.model"

  if [[ ! -f "$BPE_MODEL" ]]; then
    echo "ERROR: Missing tokenizer model: $BPE_MODEL"
    exit 1
  fi
fi

module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh

ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

# Make a per-fold temp config and patch it safely with Python (PyYAML)
TMPCFG="${SLURM_TMPDIR:-/tmp}/train_${DATASET}_fold${FOLD}.yaml"
cp "$PROJ/configs/$TRAIN_CFG" "$TMPCFG"

python3 - <<PY
import yaml

cfg_path = "$TMPCFG"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Core run-specific overrides (do NOT touch arch_de)
cfg["idx_fold"]    = int("$FOLD")
cfg["dir_work"]    = "$OUTDIR"
cfg["dir_dataset"] = "$DATA"

# Optional tokenizer block
use_bpe = "$USE_BPE".lower() in ("1", "true", "yes")
if use_bpe:
    tok = cfg.get("tokenizer", {})
    tok["type"]       = "sentencepiece"
    tok["model"]      = "$BPE_MODEL"
    tok["vocab_size"] = int("$BPE_SIZE")
    cfg["tokenizer"]  = tok
    cfg["use_bpe"]    = True
# else: leave any existing tokenizer/use_bpe fields as-is

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote patched config:", cfg_path)
PY

# >>> logging <<<
echo "========== RUN INFO =========="
echo "Fold:                $FOLD"
echo "Dataset:             $DATASET"
echo "Base config (TRAIN_CFG): configs/$TRAIN_CFG"
echo "Experiment tag:      $EXP_TAG"
echo "Patched config:      $TMPCFG"
echo "Work dir (out):      $OUTDIR"
echo "Decoder arch (from YAML): $ARCH_DE"
echo "USE_BPE:             $USE_BPE"
if [[ "${USE_BPE,,}" == "true" || "${USE_BPE}" == "1" ]]; then
  echo "BPE size:            ${BPE_SIZE}"
  echo "BPE model path:      ${BPE_MODEL}"
else
  echo "BPE / tokenizer:     DISABLED — CHARACTER-BASED (categories from YAML)."
fi
echo "=============================="
echo "Tokenizer block from YAML (if any):"
awk '
  /(^tokenizer:)/ {p=1}
  p && NF==0 {p=0}
  p
' "$TMPCFG" || true
echo "=============================="

# Extra explicit mode summary (BPE vs char) with counts:
python3 - "$TMPCFG" <<'PY'
import yaml, sys
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
tok = cfg.get("tokenizer")
use_bpe = cfg.get("use_bpe", False)
if not use_bpe or not tok:
    nchar = len(cfg.get("categories", []))
    print(f"Decoding mode: CHARACTER-BASED (no tokenizer). Categories={nchar}.")
else:
    print(f"Decoding mode: BPE tokenizer @ {tok.get('model')} (vocab={tok.get('vocab_size')}).")
PY
echo "=============================="

cd "$PROJ"
which python3
python3 -V
nvidia-smi -L || true
echo "Running fold ${FOLD} with config: $TMPCFG"
python3 main.py -c "$TMPCFG"
