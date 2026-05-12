#!/bin/bash
# Submit the full Hybrid-CTC-AR + Input-Corruption matrix:
#   5 noise modes x 6 datasets = 30 array jobs (5 folds each = 150 fold trainings).
#
# Usage:
#   bash submit_hyb_corrupt_all.sh                  # everything
#   MODES="uniform bigramright" bash submit_hyb_corrupt_all.sh
#   DATASETS="onhw_wi_word_rh" bash submit_hyb_corrupt_all.sh
#   DRY_RUN=1 bash submit_hyb_corrupt_all.sh        # print without submitting
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$HERE/train_hyb_corrupt.sbatch"

MODES=${MODES:-"uniform bigramright bigramleft selfconf adjacentswap"}

# Dataset slug used in filenames -> actual dataset dir name
declare -A DSMAP=(
  ["onhw-wi-word"]="onhw_wi_word_rh"
  ["onhw-wd-word"]="onhw_wd_word_rh"
  ["stabilo-word"]="wi_word_hw6_meta"
  ["stabilo-sent"]="wi_sent_hw6_meta"
  ["equations-wi"]="onhw_equations_wi_word_rh"
  ["equations-wd"]="onhw_equations_wd_word_rh"
)
DATASET_SLUGS=${DATASET_SLUGS:-"onhw-wi-word onhw-wd-word stabilo-word stabilo-sent equations-wi equations-wd"}

DRY_RUN=${DRY_RUN:-0}

for MODE in $MODES; do
  for DS_SLUG in $DATASET_SLUGS; do
    DATASET=${DSMAP[$DS_SLUG]:-}
    if [[ -z "$DATASET" ]]; then
      echo "WARN: unknown dataset slug '$DS_SLUG', skipping" >&2
      continue
    fi
    YAML_REL="HybridInputCorruption/train-hyb-${MODE}-${DS_SLUG}.yaml"
    YAML_PATH="$HERE/../configs/$YAML_REL"
    if [[ ! -f "$YAML_PATH" ]]; then
      echo "WARN: missing config $YAML_REL, skipping" >&2
      continue
    fi
    JOBNAME="hyb-${MODE}-${DS_SLUG}"
    echo ">>> Submit JOB=$JOBNAME  YAML=$YAML_REL  DATASET=$DATASET"
    if [[ "$DRY_RUN" == "1" ]]; then
      continue
    fi
    sbatch -J "$JOBNAME" \
      --export=ALL,TRAIN_YAML="$YAML_REL",DATASET="$DATASET" \
      "$SBATCH_FILE"
  done
done
