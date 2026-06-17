#!/bin/bash
# Submit HWRFormer (scaled, ar_transformer_s) lambda_ctc gap-fill sweep on OnHW WI word.
#
# Purpose: fill the 7 missing lambda_ctc values needed to draw a full symmetric
# scaled curve on thesis figure fig:ctc-lambda-cer-wer (currently a single star
# at lambda=0.6, with baseline curve over the full 10-point sweep).
#
# Already trained for scaled on OnHW WI:
#   train_element_word_hybrid_{01,06,09}/ar_transformer_s__onhw_wi_word_rh (5 folds each)
# Missing (this script submits):
#   lambda_ctc in {0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0}
#
# Each submission triggers a 5-fold array job (slurm/train.sbatch has --array=0-4).
# Total: 7 configs x 5 folds = 35 fold-runs.
# Estimated wall time per fold: ~6-12 h on v100 partition.
# Result paths: results/hwr2/train_element_word_hybrid_{02,03,04,05,07,08,10}/
#               ar_transformer_s__onhw_wi_word_rh/fold_{0..4}/
#
# Usage (run from a Woody / Fritz / Alex login node, not csnhr jumphost):
#   cd /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work
#   bash slurm/submit_scaled_lambda_sweep.sh

set -e
cd /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work

# hyb-s-l02 (lambda_ctc=0.2)
TRAIN_YAML=hybrid/train_element_word_02.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l02 slurm/train.sbatch

# hyb-s-l03 (lambda_ctc=0.3)
TRAIN_YAML=hybrid/train_element_word_03.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l03 slurm/train.sbatch

# hyb-s-l04 (lambda_ctc=0.4)
TRAIN_YAML=hybrid/train_element_word_04.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l04 slurm/train.sbatch

# hyb-s-l05 (lambda_ctc=0.5)
TRAIN_YAML=hybrid/train_element_word_05.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l05 slurm/train.sbatch

# hyb-s-l07 (lambda_ctc=0.7)
TRAIN_YAML=hybrid/train_element_word_07.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l07 slurm/train.sbatch

# hyb-s-l08 (lambda_ctc=0.8)
TRAIN_YAML=hybrid/train_element_word_08.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l08 slurm/train.sbatch

# hyb-s-l10 (lambda_ctc=1.0)
TRAIN_YAML=hybrid/train_element_word_10.yaml DATASET=onhw_wi_word_rh \
  sbatch --job-name=hyb-s-l10 slurm/train.sbatch

cat <<'EOF'

==============================================================
All 7 array jobs submitted (5 folds each = 35 fold-runs total).
==============================================================

Watch progress:
  squeue -u $USER --format='%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'

Result paths (after completion):
  results/hwr2/train_element_word_hybrid_{02,03,04,05,07,08,10}/
    ar_transformer_s__onhw_wi_word_rh/fold_{0..4}/

After all jobs finish, refresh the aggregator and figure:
  envs/rewi26/bin/python work/REWI_work/analysis/scripts/collect_s_numbers.py
  envs/rewi26/bin/python work/REWI_work/analysis/scripts/plot_lambda_sweep_dual.py

The new s_numbers.json figure_lambda_sweep_s key should then contain
10 lambda values (0.1, 0.2, ..., 1.0) instead of just 0.6, so the
thesis figure fig:ctc-lambda-cer-wer will show a full symmetric
scaled curve overlaid on the baseline curve.

After updating the figure, also revisit:
  - thesis/classical_results.tex sec:ctc-diagnostics — the prose
    currently says 'the corresponding 10-point sweep was not
    performed; only the operating point lambda_ctc=0.6 was trained
    (marked as an orange star ...)'. Update to describe the full
    sweep and to confirm whether lambda=0.6 is also the CER/WER
    optimum for the scaled config (or whether some other value is).
  - thesis/methodology.tex sec:meth-hybrid — the dual-operating-point
    statement may need an update if the scaled optimum is not 0.6.
EOF
