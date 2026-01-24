#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper. Run from work/REWI_work.
python3 scripts/repro/download_hf_model.py --model t5-small --out-dir assets/hf_models/t5-small
