#!/usr/bin/env bash
set -euo pipefail

# Reproduce OnHW500 experiments (CV training + evaluate.py)
# Run from: work/REWI_work

DATA_ROOT="../../data"
OUT_ROOT="../../results/repro"
PRETRAIN_DECODER_CKPT=""

usage() {
  cat <<'USAGE'
Usage: bash scripts/repro/reproduce_tables.sh [options]

Options:
  --data-root <path>                 Path to repo data/ (default: ../../data)
  --out-root <path>                  Output root for runs (default: ../../results/repro)
  --pretrained-decoder-ckpt <path>   Path to best_loss.pth for pretrained-decoder runs (optional)

Example:
  bash scripts/repro/reproduce_tables.sh --data-root ../../data --out-root ../../results/repro
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --pretrained-decoder-ckpt) PRETRAIN_DECODER_CKPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

PATCH_AND_RUN() {
  local template_cfg="$1"
  local dataset_dir="$2"
  local work_dir="$3"
  local pretrain_ckpt="$4"  # empty => keep as-is / possibly placeholder

  mkdir -p "$OUT_ROOT/_patched"
  local patched_cfg="$OUT_ROOT/_patched/$(basename "$template_cfg")"

  python - "$template_cfg" "$patched_cfg" "$dataset_dir" "$work_dir" "$pretrain_ckpt" <<'PY'
import sys
import yaml

src, dst, dataset_dir, work_dir, pretrain_ckpt = sys.argv[1:6]
with open(src, 'r') as f:
    cfg = yaml.safe_load(f)

# Patch paths
cfg['dir_dataset'] = dataset_dir
cfg['dir_work'] = work_dir
cfg['idx_fold'] = -1

# Patch pretrained-decoder checkpoint if requested
if 'pretrained_decoder_checkpoint' in cfg:
    val = cfg['pretrained_decoder_checkpoint']
    if isinstance(val, str) and val.strip() == '__PRETRAIN_DECODER_CKPT__':
        if pretrain_ckpt:
            cfg['pretrained_decoder_checkpoint'] = pretrain_ckpt
        else:
            # Keep placeholder; caller decides whether to skip
            cfg['pretrained_decoder_checkpoint'] = '__PRETRAIN_DECODER_CKPT__'

with open(dst, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  # Skip pretrained-decoder runs unless a checkpoint is provided
  if python - "$patched_cfg" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r'))
val = cfg.get('pretrained_decoder_checkpoint', None)
print('SKIP' if val == '__PRETRAIN_DECODER_CKPT__' else 'RUN')
PY
  then :; fi
  local decision
  decision=$(python - "$patched_cfg" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r'))
val = cfg.get('pretrained_decoder_checkpoint', None)
print('SKIP' if val == '__PRETRAIN_DECODER_CKPT__' else 'RUN')
PY
)

  if [[ "$decision" == "SKIP" ]]; then
    echo "[SKIP] $(basename "$template_cfg") (missing --pretrained-decoder-ckpt)"
    return 0
  fi

  echo "[TRAIN] $patched_cfg"
  python scripts/others/train_cv.py -c "$patched_cfg" -m main.py

  echo "[EVAL]  $patched_cfg"
  python evaluate.py -c "$patched_cfg"
}

# --- OnHW500 WI ---
PATCH_AND_RUN \
  configs/experiments/onhw_wi_ar_scratch.yaml \
  "$DATA_ROOT/onhw_wi_word_rh" \
  "$OUT_ROOT/onhw_wi/ar_scratch" \
  "$PRETRAIN_DECODER_CKPT"

PATCH_AND_RUN \
  configs/experiments/onhw_wi_ar_pretrained.yaml \
  "$DATA_ROOT/onhw_wi_word_rh" \
  "$OUT_ROOT/onhw_wi/ar_pretrained" \
  "$PRETRAIN_DECODER_CKPT"

PATCH_AND_RUN \
  configs/experiments/onhw_wi_t5.yaml \
  "$DATA_ROOT/onhw_wi_word_rh" \
  "$OUT_ROOT/onhw_wi/t5_small" \
  "$PRETRAIN_DECODER_CKPT"

# --- OnHW500 WD ---
PATCH_AND_RUN \
  configs/experiments/onhw_wd_ar_scratch.yaml \
  "$DATA_ROOT/onhw_wd_word_rh" \
  "$OUT_ROOT/onhw_wd/ar_scratch" \
  "$PRETRAIN_DECODER_CKPT"

PATCH_AND_RUN \
  configs/experiments/onhw_wd_ar_pretrained.yaml \
  "$DATA_ROOT/onhw_wd_word_rh" \
  "$OUT_ROOT/onhw_wd/ar_pretrained" \
  "$PRETRAIN_DECODER_CKPT"

PATCH_AND_RUN \
  configs/experiments/onhw_wd_t5.yaml \
  "$DATA_ROOT/onhw_wd_word_rh" \
  "$OUT_ROOT/onhw_wd/t5_small" \
  "$PRETRAIN_DECODER_CKPT"

echo "Done. Aggregated metrics are in: $OUT_ROOT/**/results.json"
