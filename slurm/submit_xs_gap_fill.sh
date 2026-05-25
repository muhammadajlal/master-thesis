#!/bin/bash
# Auto-generated SLURM submission script for XS gap-filling jobs.
# Each line submits one 5-fold array job.

set -e
cd /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work

# xs-hw-onhw-wd
TRAIN_YAML=AR-Baseline/train-ar-xs-headwise-onhw-wd.yaml DATASET=onhw_wd_word_rh sbatch --job-name=xs-hw-onhw-wd slurm/train.sbatch

# xs-hw-equations-wi
TRAIN_YAML=AR-Baseline/train-ar-xs-headwise-equations-wi.yaml DATASET=onhw_equations_wi_word_rh sbatch --job-name=xs-hw-equations-wi slurm/train.sbatch

# xs-hw-equations-wd
TRAIN_YAML=AR-Baseline/train-ar-xs-headwise-equations-wd.yaml DATASET=onhw_equations_wd_word_rh sbatch --job-name=xs-hw-equations-wd slurm/train.sbatch

# xs-hw-stabilo
TRAIN_YAML=AR-Baseline/train-ar-xs-headwise-stabilo.yaml DATASET=wi_word_hw6_meta sbatch --job-name=xs-hw-stabilo slurm/train.sbatch

# xs-hw-stabilo-sent
TRAIN_YAML=AR-Baseline/train-ar-xs-headwise-stabilo-sent.yaml DATASET=wi_sent_hw6_meta sbatch --job-name=xs-hw-stabilo-sent slurm/train.sbatch

# hyb-xs-l01
TRAIN_YAML=hybrid-xs/train_element_word_01_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l01 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l02
TRAIN_YAML=hybrid-xs/train_element_word_02_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l02 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l03
TRAIN_YAML=hybrid-xs/train_element_word_03_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l03 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l04
TRAIN_YAML=hybrid-xs/train_element_word_04_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l04 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l05
TRAIN_YAML=hybrid-xs/train_element_word_05_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l05 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l07
TRAIN_YAML=hybrid-xs/train_element_word_07_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l07 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l08
TRAIN_YAML=hybrid-xs/train_element_word_08_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l08 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l09
TRAIN_YAML=hybrid-xs/train_element_word_09_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l09 slurm/train_hyb_corrupt.sbatch

# hyb-xs-l10
TRAIN_YAML=hybrid-xs/train_element_word_10_xs_onhw_wi.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-l10 slurm/train_hyb_corrupt.sbatch

# xs-sw-onhw-word-p0p05
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-onhw-word-p0p05.yaml DATASET=onhw_wi_word_rh sbatch --job-name=xs-sw-onhw-word-p0p05 slurm/train.sbatch

# xs-sw-onhw-word-p0p10
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-onhw-word-p0p10.yaml DATASET=onhw_wi_word_rh sbatch --job-name=xs-sw-onhw-word-p0p10 slurm/train.sbatch

# xs-sw-onhw-word-p0p20
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-onhw-word-p0p20.yaml DATASET=onhw_wi_word_rh sbatch --job-name=xs-sw-onhw-word-p0p20 slurm/train.sbatch

# xs-sw-onhw-word-p0p30
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-onhw-word-p0p30.yaml DATASET=onhw_wi_word_rh sbatch --job-name=xs-sw-onhw-word-p0p30 slurm/train.sbatch

# xs-sw-stabilo-sent-p0p05
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-stabilo-sent-p0p05.yaml DATASET=wi_sent_hw6_meta sbatch --job-name=xs-sw-stabilo-sent-p0p05 slurm/train.sbatch

# xs-sw-stabilo-sent-p0p10
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-stabilo-sent-p0p10.yaml DATASET=wi_sent_hw6_meta sbatch --job-name=xs-sw-stabilo-sent-p0p10 slurm/train.sbatch

# xs-sw-stabilo-sent-p0p20
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-stabilo-sent-p0p20.yaml DATASET=wi_sent_hw6_meta sbatch --job-name=xs-sw-stabilo-sent-p0p20 slurm/train.sbatch

# xs-sw-stabilo-sent-p0p30
TRAIN_YAML=AR-InputCorruption-Sweep-XS/train-ar-corrupt-xs-stabilo-sent-p0p30.yaml DATASET=wi_sent_hw6_meta sbatch --job-name=xs-sw-stabilo-sent-p0p30 slurm/train.sbatch

# hyb-xs-tied
TRAIN_YAML=hybrid-xs/train_element_word_06_xs_onhw_wi_ctc_to_ar_outproj.yaml DATASET=onhw_wi_word_rh sbatch --job-name=hyb-xs-tied slurm/train_hyb_corrupt.sbatch

