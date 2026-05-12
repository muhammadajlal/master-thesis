#!/bin/bash
# Submit the 4 missing hybrid (no-corruption) baselines that complete the matched
# reference set for the HybridInputCorruption matrix.
#
# Usage:
#   bash submit_hyb_baselines_missing.sh                      # all 4
#   DRY_RUN=1 bash submit_hyb_baselines_missing.sh            # print without submitting
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="$HERE/train_hyb_corrupt.sbatch"  # generic; works for non-corruption too

declare -a JOBS=(
  "hyb-bl-onhw-wd-word    hybrid/train_element_word_06_onhw_wd.yaml      onhw_wd_word_rh"
  "hyb-bl-stabilo-sent    hybrid/train_element_word_06_stabilo_sent.yaml wi_sent_hw6_meta"
  "hyb-bl-equations-wi    hybrid/train_element_word_06_equations_wi.yaml onhw_equations_wi_word_rh"
  "hyb-bl-equations-wd    hybrid/train_element_word_06_equations_wd.yaml onhw_equations_wd_word_rh"
)

DRY_RUN=${DRY_RUN:-0}

for ROW in "${JOBS[@]}"; do
  read -r JOBNAME YAML DATASET <<< "$ROW"
  YAML_PATH="$HERE/../configs/$YAML"
  if [[ ! -f "$YAML_PATH" ]]; then
    echo "WARN: missing config $YAML, skipping" >&2
    continue
  fi
  echo ">>> Submit JOB=$JOBNAME  YAML=$YAML  DATASET=$DATASET"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  sbatch -J "$JOBNAME" \
    --export=ALL,TRAIN_YAML="$YAML",DATASET="$DATASET" \
    "$SBATCH_FILE"
done
