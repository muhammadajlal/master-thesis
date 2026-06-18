#!/bin/bash
# Submit HWRFormer (scaled, ar_transformer_s) p_replace sweep on wi_word_hw6_meta (Private word).
#
# Purpose: fill the 4 missing scaled p_replace cells on the Private word row of
# thesis tab:p-sweep (line ~195 of classical_results.tex). The p=0.15 anchor
# already exists (from Baseline-AR-InputCorruption-blconv_b/ar_transformer_s__wi_word_hw6_meta,
# CER 6.97); this script submits the remaining 4 p values.
#
# Missing values (this script submits):
#   p_replace in {0.05, 0.10, 0.20, 0.30}, uniform-mode noise injection.
#
# Each submission triggers a 5-fold array job (slurm/train.sbatch has --array=0-4).
# Total: 4 configs x 5 folds = 20 fold-runs.
# Estimated wall time per fold: ~3-5 h on rtx3080 (priv-word is the smallest of
# the four datasets; folds finish faster than on OnHW WI).
# Result paths: results/hwr2/Baseline-AR-InputCorruption-Sweep-blconv_b/
#               ar_transformer_s__wi_word_hw6_meta__p0p{05,10,20,30}/fold_{0..4}/
#
# Usage (run from a Woody / Fritz / Alex login node, not csnhr jumphost):
#   cd /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work
#   bash slurm/submit_scaled_pwsweep_priv_word.sh
#
# Routes to rtx3080 because the v100 partition is currently saturated by the
# in-flight lambda_ctc sweep (1705973-1706001). a100 is unusable (fairshare
# wait ~6 weeks).

set -e
cd /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work

# hyb-s-pw-p05 (p_replace=0.05, scaled, priv word)
TRAIN_YAML=AR-InputCorruption-Sweep/train-ar-corrupt-stabilo-word-p0p05.yaml DATASET=wi_word_hw6_meta \
  sbatch --job-name=hyb-s-pw-p05 -p rtx3080 --gres=gpu:rtx3080:1 slurm/train.sbatch

# hyb-s-pw-p10 (p_replace=0.10)
TRAIN_YAML=AR-InputCorruption-Sweep/train-ar-corrupt-stabilo-word-p0p10.yaml DATASET=wi_word_hw6_meta \
  sbatch --job-name=hyb-s-pw-p10 -p rtx3080 --gres=gpu:rtx3080:1 slurm/train.sbatch

# hyb-s-pw-p20 (p_replace=0.20)
TRAIN_YAML=AR-InputCorruption-Sweep/train-ar-corrupt-stabilo-word-p0p20.yaml DATASET=wi_word_hw6_meta \
  sbatch --job-name=hyb-s-pw-p20 -p rtx3080 --gres=gpu:rtx3080:1 slurm/train.sbatch

# hyb-s-pw-p30 (p_replace=0.30)
TRAIN_YAML=AR-InputCorruption-Sweep/train-ar-corrupt-stabilo-word-p0p30.yaml DATASET=wi_word_hw6_meta \
  sbatch --job-name=hyb-s-pw-p30 -p rtx3080 --gres=gpu:rtx3080:1 slurm/train.sbatch

cat <<'EOF'

==============================================================
All 4 array jobs submitted (5 folds each = 20 fold-runs total).
==============================================================

Watch progress:
  squeue -u $USER --format='%.10i %.9P %.20j %.2t %.10M %R' | grep hyb-s-pw

Result paths (after completion):
  results/hwr2/Baseline-AR-InputCorruption-Sweep-blconv_b/
    ar_transformer_s__wi_word_hw6_meta__p0p{05,10,20,30}/fold_{0..4}/

After all jobs finish, refresh the aggregator + figure:
  envs/rewi26/bin/python work/REWI_work/analysis/scripts/collect_s_numbers.py
  envs/rewi26/bin/python work/REWI_work/analysis/scripts/plot_corruption_p_sweep_dual.py

The new s_numbers.json figure_pic_sweep_s.wi_word_hw6_meta key will then
contain all 5 p values (was: only p=0.15). The Private word row of
thesis tab:p-sweep can be backfilled from these means.
EOF
