"""
Stage 5: merge Stage 2 (training results) and Stage 4 (FCC) outputs into imagenet1k_results.csv,
and compute the aggregate Spearman/Pearson/weighted-Kendall correlations of Gain vs. each FCC
column, modeled on nist/nist_analysis.py's table+correlation pattern
(Design_choices_imagenet1k.txt item 4). n=7, so every correlation statistic here is
explicitly caveated as high-variance in the rendered output -- same spirit as the "1 seed for
now" decision (item 6), doubly true with only 7 data points.
"""
import csv
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr, weightedtau

from imagenet1k.data_targets import TARGETS
from imagenet1k.fcc_experiment import load_cache as load_fcc_cache
from imagenet1k.imagenet_source_stats import NETWORKS

RESULTS_DIR = 'imagenet1k/results'
TABLES_DIR = 'imagenet1k/tables'
CSV_OUT = f'{TABLES_DIR}/imagenet1k_results.csv'
JSON_OUT = f'{TABLES_DIR}/imagenet1k_correlations.json'
MD_OUT = f'{TABLES_DIR}/imagenet1k_correlations.md'

FCC_COLUMNS = {net: (f'fcc_{net}_mean', f'fcc_{net}_ratio') for net in NETWORKS}

# 2026-11-10, user request: SVHN stays a full row in imagenet1k_results.csv (the raw record), but
# is excluded from the correlation statistics and (in plots.py) from the scatter plot points.
EXCLUDE_FROM_CORRELATION = {'svhn'}


def load_training_results(target):
    rows = {}
    for arm in ('frozen', 'scratch'):
        with open(f'{RESULTS_DIR}/{target}_{arm}.json') as f:
            rows[arm] = json.load(f)
    return rows


def build_rows(targets):
    fcc_cache = load_fcc_cache()
    rows = []
    for target in targets:
        tr = load_training_results(target)
        row = {
            'target': target,
            'num_classes': tr['frozen']['num_classes'],
            'n_train_native': tr['frozen']['n_train_native'],
            'n_test_native': tr['frozen']['n_test_native'],
            'frozen_final_acc': round(tr['frozen']['final_acc'], 2),
            'frozen_best_val_acc': round(tr['frozen']['best_val_acc'], 2),
            'frozen_epochs_run': tr['frozen']['epochs_run'],
            'scratch_final_acc': round(tr['scratch']['final_acc'], 2),
            'scratch_best_val_acc': round(tr['scratch']['best_val_acc'], 2),
            'scratch_epochs_run': tr['scratch']['epochs_run'],
        }
        # Gain(T) = acc(ImageNet->T) - acc(T alone), matching the original spec's single
        # definition (no baselineA/baselineB split this time -- only one baseline exists here).
        row['gain'] = round(row['frozen_final_acc'] - row['scratch_final_acc'], 2)
        for net in NETWORKS:
            entry = fcc_cache[(target, net)]
            mean_col, ratio_col = FCC_COLUMNS[net]
            row[mean_col] = round(entry['mean'], 4)
            row[ratio_col] = round(entry['ratio'], 4)
        rows.append(row)
    return rows


def correlations(fcc, gain):
    rho, rho_p = spearmanr(fcc, gain)
    r, r_p = pearsonr(fcc, gain)
    wt = weightedtau(fcc, gain)
    return {'spearman': float(rho), 'spearman_p': float(rho_p), 'pearson': float(r),
            'pearson_p': float(r_p), 'weighted_kendall': float(wt.statistic)}


def build_correlation_rows(rows):
    rows = [r for r in rows if r['target'] not in EXCLUDE_FROM_CORRELATION]
    gain = np.array([r['gain'] for r in rows])
    out = []
    for net in NETWORKS:
        for agg, col in zip(('mean', 'ratio'), FCC_COLUMNS[net]):
            fcc = np.array([r[col] for r in rows])
            out.append({'method': f'FCC-{net}-{agg}', 'n_pairs': len(rows), **correlations(fcc, gain)})
    return out


def render_table(corr_rows):
    header = ['Method', 'Spearman', 'Pearson', 'Weighted Kendall']
    lines = []
    for r in corr_rows:
        lines.append([r['method'], f'{r["spearman"]:.3f} (p={r["spearman_p"]:.3g})',
                      f'{r["pearson"]:.3f} (p={r["pearson_p"]:.3g})', f'{r["weighted_kendall"]:.3f}'])
    widths = [max(len(header[i]), *(len(l[i]) for l in lines)) for i in range(4)]
    fmt = lambda cells: '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'
    md = [fmt(header), '|' + '|'.join('-' * (w + 2) for w in widths) + '|'] + [fmt(l) for l in lines]
    md.append('')
    md.append(f'n = {corr_rows[0]["n_pairs"]} targets ({sorted(EXCLUDE_FROM_CORRELATION)} excluded '
              f'from these statistics by request, though still present as full rows in '
              f'imagenet1k_results.csv). With this few points, every correlation statistic above is '
              f'high-variance -- treat as suggestive, not conclusive, and read alongside '
              f'imagenet1k_results.csv directly rather than in isolation.')
    return '\n'.join(md)


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    rows = build_rows(TARGETS)
    fieldnames = list(rows[0].keys())

    # Summary rows (appended to the CSV) are computed excluding SVHN; the data rows above are
    # not -- SVHN keeps its full row in the raw CSV, it's just excluded from these statistics.
    corr_rows_data = [r for r in rows if r['target'] not in EXCLUDE_FROM_CORRELATION]
    gain = np.array([r['gain'] for r in corr_rows_data])

    summary_rows = []
    for stat_name, stat_fn in (('pearson_r', pearsonr), ('spearman_rho', spearmanr)):
        summary = {k: '' for k in fieldnames}
        summary['target'] = f'{stat_name}_vs_gain'
        for net in NETWORKS:
            for col in FCC_COLUMNS[net]:
                vals = np.array([r[col] for r in corr_rows_data])
                stat_value, _p = stat_fn(vals, gain)
                summary[col] = round(float(stat_value), 4)
        summary_rows.append(summary)

    with open(CSV_OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(summary_rows)
    print(f'Saved {CSV_OUT} ({len(rows)} data rows + {len(summary_rows)} summary rows)')

    corr_rows = build_correlation_rows(rows)
    table = render_table(corr_rows)
    print(f'\n=== FCC vs Gain (n = {len(corr_rows_data)} targets, {sorted(EXCLUDE_FROM_CORRELATION)} excluded) ===')
    print(table)
    with open(JSON_OUT, 'w') as f:
        json.dump(corr_rows, f, indent=2)
    with open(MD_OUT, 'w') as f:
        f.write('# FCC vs Gain: ImageNet-1K -> 7 target datasets\n\n' + table + '\n')
    print(f'\nSaved {JSON_OUT} and {MD_OUT}')


if __name__ == '__main__':
    main()
