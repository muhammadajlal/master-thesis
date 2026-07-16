# Frozen HWRFormer scheduled-sampling configs (2026-07-15)

Immutable snapshots for the HWRFormer (`ar_transformer_xs`) scheduled-sampling runs reported
in thesis Table 5.4. The six `train-ar-{noTF,ssfixed}-*.yaml` files cover the immediate
`p: 0 -> 1` ramp and fixed `p = 0.15` schedules on OnHW WI, private words, and private
sentences. They are the exact configurations used by the matching
`Baseline-AR-XS-{NoTeacherForcing,ScheduledSamplingFixed}-blconv_b` result families.

The `train-ar-ssdelay*` files define separate delayed-ramp and delayed-abrupt curriculum
controls and are not reported in the thesis. The canonical configs under
`AR-NoTeacherForcing/` and `AR-ScheduledSamplingFixed/` remain historical HWRFormer-L
