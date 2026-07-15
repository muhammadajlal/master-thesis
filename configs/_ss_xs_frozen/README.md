# Frozen XS scheduled-sampling configs (2026-07-15)

Immutable snapshots for the supplementary HWRFormer (ar_transformer_xs) scheduled-sampling
runs. The canonical configs under AR-NoTeacherForcing/ and AR-ScheduledSamplingFixed/ were
reverted to ar_transformer_s (the retained thesis runs) on 2026-07-14; jobs read configs at
START time, so queued XS jobs must point HERE to be immune to further edits of the canonical
files. The three train-ar-* files are the exact XS versions from commit c035fbf; the
train-ar-ssdelay* files are copies of configs/AR-SS-Delayed{Ramp,Abrupt}/.
Do not edit; the training sbatch asserts arch_de == ar_transformer_xs at job start.
