#!/bin/bash
# Submit all 8 zero-shot (variant x target) pairs for the J-series.
# Usage:
#   bash submit_J_zeroshot_all.sh           # all 8
#   VARIANTS="J1_mlp J2_mlp" bash submit_J_zeroshot_all.sh
#   TARGETS="equations_wi"   bash submit_J_zeroshot_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VARIANTS=${VARIANTS:-"J1_mlp J1_pooling J2_mlp J2_pooling"}
TARGETS=${TARGETS:-"equations_wi equations_wd"}

for V in $VARIANTS; do
  for T in $TARGETS; do
    echo ">>> Submitting VARIANT=$V TARGET=$T"
    VARIANT=$V TARGET=$T sbatch --export=ALL,VARIANT=$V,TARGET=$T \
      "$HERE/test_J_zeroshot.sbatch"
  done
done
