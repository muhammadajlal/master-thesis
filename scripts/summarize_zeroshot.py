"""
Aggregate zero-shot eval results from results/hwr2/ZeroShot/*/fold_*/*/test_*.json
into a CSV + LaTeX table.

Reports AR (GPT-2) CER/WER only — CTC-head metrics on equations are invalid
because the training char set does not contain digits/operators.

Usage:
    python scripts/summarize_zeroshot.py \
        --root /home/woody/iwso/iwso214h/imu-hwr/results/hwr2/ZeroShot \
        --out  /home/woody/iwso/iwso214h/imu-hwr/results/hwr2/ZeroShot/summary
"""
import argparse
import csv
import glob
import json
import os
import statistics


def collect_run(run_dir):
    """Read every test_*.json under run_dir/fold_*/ and return per-fold AR CER/WER.
    Picks the newest test_*.json per fold; resolves the winning epoch via
    result['best']['character_error_rate'][0] and reads result[str(epoch)]['evaluation'].
    """
    per_fold = {}
    for path in sorted(glob.glob(os.path.join(run_dir, 'fold_*', '*', 'test_*.json'))):
        fold = int(path.split('/fold_')[1].split('/')[0])
        with open(path, 'r') as f:
            data = json.load(f)
        best = data.get('best') or {}
        ce_entry = best.get('character_error_rate')
        if not ce_entry:
            continue
        epoch_key = str(ce_entry[0])
        ev = data.get(epoch_key, {}).get('evaluation')
        if not ev:
            continue
        cer = ev.get('character_error_rate')
        wer = ev.get('word_error_rate')
        if cer is None or wer is None:
            continue
        # keep newest timestamp per fold
        per_fold[fold] = (float(cer), float(wer))
    return per_fold


def mean_std(xs):
    if not xs:
        return (float('nan'), float('nan'))
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (statistics.mean(xs), statistics.stdev(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.root, '*__*'))):
        name = os.path.basename(run_dir)
        variant, target = name.split('__', 1)
        per_fold = collect_run(run_dir)
        if not per_fold:
            continue
        cers = [c for c, _ in per_fold.values()]
        wers = [w for _, w in per_fold.values()]
        cer_m, cer_s = mean_std(cers)
        wer_m, wer_s = mean_std(wers)
        rows.append({
            'variant': variant,
            'target': target,
            'folds': len(per_fold),
            'cer_mean': cer_m,
            'cer_std': cer_s,
            'wer_mean': wer_m,
            'wer_std': wer_s,
            'per_fold_cer': ';'.join(f'{per_fold[k][0]:.4f}' for k in sorted(per_fold)),
        })

    csv_path = os.path.join(args.out, 'zeroshot_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['variant', 'target', 'folds', 'cer_mean', 'cer_std',
                            'wer_mean', 'wer_std', 'per_fold_cer'])
        w.writeheader()
        w.writerows(rows)

    tex_path = os.path.join(args.out, 'zeroshot_table.tex')
    with open(tex_path, 'w') as f:
        f.write('% Zero-shot AR-head CER/WER. CTC-head scores are invalid on equations.\n')
        f.write('\\begin{tabular}{llcc}\n\\toprule\n')
        f.write('Model & Target & CER (\\%) & WER (\\%) \\\\\n\\midrule\n')
        for r in rows:
            f.write(f"{r['variant'].replace('_', r'\\_')} & "
                    f"{r['target'].replace('_', r'\\_')} & "
                    f"{100*r['cer_mean']:.2f} $\\pm$ {100*r['cer_std']:.2f} & "
                    f"{100*r['wer_mean']:.2f} $\\pm$ {100*r['wer_std']:.2f} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')

    print(f'Wrote {csv_path}')
    print(f'Wrote {tex_path}')
    print(f'Runs aggregated: {len(rows)}')


if __name__ == '__main__':
    main()
