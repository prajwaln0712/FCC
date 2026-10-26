"""
Correlate FCC (TinyImageNet -> CIFAR10) with CIFAR10 transfer-learning accuracy across
augment.py seed runs.

Each seed run (dataaugmentation/aug_<seed>/results.json, written by augment.py) contributes
one (accuracy, fcc) point per FCC metric. FCC is reported for two feature extractors
(Inception-v3, DINOv3), each with two aggregators:
  * mean  - mean of the per-class Frechet distances (primary)
  * ratio - mean / max of the per-class Frechet distances (coverage ratio)

Statistics per metric: Spearman rho, Pearson r (with p-values) and
scipy.stats.weightedtau (scipy returns no p-value for it, so it is reported as n/a).
Mirrors the convention already used by nist_analysis.py for the *NIST FCC-vs-PG table.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr, weightedtau

METRICS = {
    ('inception', 'mean'): 'FCC-inception mean',
    ('inception', 'ratio'): 'FCC-inception ratio',
    ('dino', 'mean'): 'FCC-dino mean',
    ('dino', 'ratio'): 'FCC-dino ratio',
}


def load_runs(parent_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(parent_dir, 'aug_*', 'results.json'))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def points_for(runs, network, aggregator):
    acc, fcc = [], []
    for r in runs:
        acc.append(r['cifar10_acc_adapt'])
        fcc.append(r['fcc'][network][aggregator])
    return np.array(acc), np.array(fcc)


def correlations(acc, fcc):
    rho, rho_p = spearmanr(fcc, acc)
    r, r_p = pearsonr(fcc, acc)
    wt = weightedtau(fcc, acc)
    return {
        'spearman': float(rho), 'spearman_p': float(rho_p),
        'pearson': float(r), 'pearson_p': float(r_p),
        'weighted_kendall': float(wt.statistic),
        'weighted_kendall_p': None,  # scipy.stats.weightedtau provides no p-value
    }


def compute_all(runs):
    rows = []
    n = len(runs)
    for (network, aggregator), label in METRICS.items():
        row = {'method': label, 'n_seeds': n}
        if n < 2:
            rows.append(row)
            continue
        acc, fcc = points_for(runs, network, aggregator)
        row.update(correlations(acc, fcc))
        rows.append(row)
    return rows


def _cell(row, key):
    v = row.get(key)
    if v is None:
        return '—'
    p = row.get(f'{key}_p')
    return f'{v:.3f} (p = {p:.3g})' if p is not None else f'{v:.3f}'


def _cell_with_na(row, key):
    if row.get(key) is not None and key == 'weighted_kendall':
        return f'{row[key]:.3f} (p n/a)'
    return _cell(row, key)


def render_table(rows, n_seeds):
    header = ['Method', 'Spearman', 'Pearson', 'Weighted Kendall']
    lines = []
    for r in rows:
        if r['n_seeds'] < 2:
            lines.append([r['method'], '—', '—', '—'])
        else:
            lines.append([r['method'], _cell(r, 'spearman'), _cell(r, 'pearson'), _cell_with_na(r, 'weighted_kendall')])
    widths = [max(len(header[i]), *(len(l[i]) for l in lines)) for i in range(4)]
    fmt = lambda cells: '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'
    md = [fmt(header), '|' + '|'.join('-' * (w + 2) for w in widths) + '|'] + [fmt(l) for l in lines]
    md.append('')
    if n_seeds < 2:
        md.append(f'Not enough seed runs yet (n={n_seeds}); need at least 2 to compute a correlation.')
    else:
        md.append(f'n = {n_seeds} seed runs. scipy.stats.weightedtau returns no p-value.')
    return '\n'.join(md)


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # — prints correctly on Windows consoles
    ap = argparse.ArgumentParser(description='Correlate FCC with CIFAR10 transfer accuracy across augment.py seeds')
    ap.add_argument('--parent_dir', type=str, default='dataaugmentation', help='Same --parent_dir used by augment.py')
    ap.add_argument('--json-out', type=str, default='dataaugmentation/aug_correlations.json')
    ap.add_argument('--md-out', type=str, default='dataaugmentation/aug_correlations.md')
    args = ap.parse_args()

    runs = load_runs(args.parent_dir)
    print(f'Loaded {len(runs)} seed run(s) from {args.parent_dir}/aug_*/results.json')
    if runs:
        print('Seeds:', ', '.join(str(r['seed']) for r in runs))

    rows = compute_all(runs)
    table = render_table(rows, len(runs))
    print(f'\n=== FCC vs CIFAR10 transfer accuracy (n = {len(runs)} seeds) ===')
    print(table)

    with open(args.json_out, 'w') as f:
        json.dump({'n_seeds': len(runs), 'rows': rows}, f, indent=2)
    with open(args.md_out, 'w', encoding='utf-8') as f:
        f.write('# FCC vs CIFAR10 transfer accuracy: data augmentation experiment\n\n' + table + '\n')
    print(f'\nSaved {args.json_out} and {args.md_out}')
    return rows


if __name__ == '__main__':
    main()
