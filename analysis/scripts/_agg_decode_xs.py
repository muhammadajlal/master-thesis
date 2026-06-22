"""Aggregate 5-fold CER/WER per config from decode_study metrics.json files for the baseline xs cross-check."""
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT_AR  = Path('/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/results/hwr2/decode_study_xs_full_ar')
ROOT_HYB = Path('/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/results/hwr2/decode_study_xs_full_hybrid')


def agg5(root, tag):
    cers, wers = [], []
    for f in range(5):
        d = root / f'{tag}__fold{f}' / 'metrics.json'
        if not d.exists():
            continue
        r = json.loads(d.read_text())
        cer = r.get('cer')
        wer = r.get('wer')
        if cer is None:
            continue
        if cer < 1: cer *= 100
        if wer < 1: wer *= 100
        cers.append(cer)
        wers.append(wer)
    if not cers:
        return None
    return (mean(cers), pstdev(cers) if len(cers) > 1 else 0.0,
            mean(wers), pstdev(wers) if len(wers) > 1 else 0.0, len(cers))


AR = [
    ('stageA0_ar_greedy', 'Greedy'),
    ('stageA2_ar_beam_B4_a0p6', 'Cal beam B=4'),
    ('stageC1_ar_kenlm_rescore_N8_lw0p1', '+ KenLM rescore'),
    ('stageC2_ar_kenlm_shallow_B4_lw0p2', '+ KenLM shallow'),
    ('stageD1_ar_neural_rescore_N8_lw0p2', '+ Pretr LM rescore'),
    ('stageD2_ar_neural_shallow_B4_lw0p1', '+ Pretr LM shallow'),
]

print('=== BASELINE XS  AR-only ===')
for t, l in AR:
    r = agg5(ROOT_AR, t)
    if r:
        print(f'  {l:<22} CER {r[0]:5.2f}+/-{r[1]:4.2f}   WER {r[2]:5.2f}+/-{r[3]:4.2f}   (n={r[4]})')

print()
print('=== BASELINE XS  Hybrid (AR-decoded) ===')
for t, l in AR:
    r = agg5(ROOT_HYB, t)
    if r:
        print(f'  {l:<22} CER {r[0]:5.2f}+/-{r[1]:4.2f}   WER {r[2]:5.2f}+/-{r[3]:4.2f}   (n={r[4]})')

print()
print('=== BASELINE XS  Hybrid (CTC-decoded) ===')
for t, l in [
    ('stageB1_ctc_greedy', 'CTC Greedy'),
    ('stageB2_ctc_beam_B50', 'CTC Beam B=50'),
    ('stageB3_ctc_kenlm_lw0p4', 'CTC + KenLM'),
]:
    r = agg5(ROOT_HYB, t)
    if r:
        print(f'  {l:<22} CER {r[0]:5.2f}+/-{r[1]:4.2f}   WER {r[2]:5.2f}+/-{r[3]:4.2f}   (n={r[4]})')
