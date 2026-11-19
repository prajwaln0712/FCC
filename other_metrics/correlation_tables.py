"""
Correlation tables: FCC (Inception), FCC (DINO), LogME and LEEP versus the performance measure,
one table per experiment.

  Augmentation (TinyImageNet -> CIFAR-10, one point per seed):
      measure = acc_adapt;  FCC columns = ratio  (both from tables/augmentation/aug_ratio.csv)
      LogME / LEEP        = tables/other_metrics/aug_scores.csv  (other_metrics/aug_scores.py)
  Imagenet1k (ImageNet-1K -> targets, one point per target):
      measure = gain;       FCC columns = ratio  (tables/imagenet1k/inception2048/imagenet1k_results.csv)
      LogME / LEEP        = tables/other_metrics/imagenet1k_scores.csv (other_metrics/imagenet1k_scores.py)

Each table has one row per metric and the columns Spearman rho, its p-value, Pearson r, its p-value.
Writes <out-dir>/{aug,imagenet1k}_correlation_table.{csv,md}.

Expected signs if a metric works: FCC is a distance, so it should correlate NEGATIVELY with
accuracy/gain; LogME and LEEP are higher-is-better scores, so they should correlate POSITIVELY.

Usage (from the repo root):  python -m other_metrics.correlation_tables [--exclude-imagenet1k svhn]
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

AUG_FCC_CSV = 'tables/augmentation/aug_ratio.csv'            # seed_no, inception, dino, acc_adapt
BIG_RESULTS_CSV = 'tables/imagenet1k/inception2048/imagenet1k_results.csv'
AUG_SCORES_CSV = 'tables/other_metrics/aug_scores.csv'       # seed, logme, leep, ...
BIG_SCORES_CSV = 'tables/other_metrics/imagenet1k_scores.csv'   # target, logme, leep, ...
BIG_FCC_COLUMNS = {'FCC (Inception)': 'fcc_inception2048_ratio', 'FCC (DINO)': 'fcc_dinov3vith16plus_ratio'}


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def correlation_row(name, x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    return {'Metric': name, 'Spearman rho': rho, 'Spearman p': rho_p, 'Pearson r': r, 'Pearson p': r_p}


def aug_table(fcc_csv, scores_csv):
    fcc = {r['seed_no']: r for r in read_csv(fcc_csv)}
    scores = {r['seed']: r for r in read_csv(scores_csv)}
    seeds = sorted(set(fcc) & set(scores), key=int)
    missing = sorted((set(fcc) ^ set(scores)), key=int)
    if missing:
        print(f'  augmentation: seeds not in both files are dropped: {missing}')
    y = [float(fcc[s]['acc_adapt']) for s in seeds]
    rows = [
        correlation_row('FCC (Inception)', [float(fcc[s]['inception']) for s in seeds], y),
        correlation_row('FCC (DINO)', [float(fcc[s]['dino']) for s in seeds], y),
        correlation_row('LogME', [float(scores[s]['logme']) for s in seeds], y),
        correlation_row('LEEP', [float(scores[s]['leep']) for s in seeds], y),
    ]
    return rows, len(seeds), 'accuracy (acc_adapt)'


def imagenet1k_table(results_csv, scores_csv, exclude):
    results = {r['target']: r for r in read_csv(results_csv)
               if r['n_train_native'] != '' and r['target'] not in exclude}
    scores = {r['target']: r for r in read_csv(scores_csv)}
    targets = [t for t in results if t in scores]
    dropped = sorted(set(results) ^ set(scores) - set(exclude))
    if dropped:
        print(f'  imagenet1k: targets not in both files are dropped: {dropped}')
    y = [float(results[t]['gain']) for t in targets]
    rows = [correlation_row(name, [float(results[t][col]) for t in targets], y)
            for name, col in BIG_FCC_COLUMNS.items()]
    rows.append(correlation_row('LogME', [float(scores[t]['logme']) for t in targets], y))
    rows.append(correlation_row('LEEP', [float(scores[t]['leep']) for t in targets], y))
    return rows, len(targets), 'performance gain (%)'


COLUMNS = ['Metric', 'Spearman rho', 'Spearman p', 'Pearson r', 'Pearson p']


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def write_table(rows, n, measure, stem, title):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([r['Metric']] + [f'{r[c]:.4f}' for c in COLUMNS[1:]])
    lines = [f'# {title}', '',
             f'Correlation with {measure}, n = {n}.', '',
             '| Metric | Spearman ρ | p-value | Pearson r | p-value |',
             '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['Metric']} | {r['Spearman rho']:.3f} | {fmt_p(r['Spearman p'])} | "
                     f"{r['Pearson r']:.3f} | {fmt_p(r['Pearson p'])} |")
    with open(stem + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines).replace('ρ', 'rho'))  # plain ASCII: Windows consoles can't print the Greek letter
    print(f'  wrote {stem}.csv and {stem}.md\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aug-fcc', default=AUG_FCC_CSV)
    ap.add_argument('--aug-scores', default=AUG_SCORES_CSV)
    ap.add_argument('--imagenet1k-results', default=BIG_RESULTS_CSV)
    ap.add_argument('--imagenet1k-scores', default=BIG_SCORES_CSV)
    ap.add_argument('--exclude-imagenet1k', nargs='*', default=[], help='targets to leave out, e.g. svhn')
    ap.add_argument('--out-dir', default='tables/other_metrics')
    args = ap.parse_args()

    rows, n, measure = aug_table(args.aug_fcc, args.aug_scores)
    write_table(rows, n, measure, f'{args.out_dir}/aug_correlation_table',
                'Augmentation experiment (Tiny-ImageNet -> CIFAR10)')
    rows, n, measure = imagenet1k_table(args.imagenet1k_results, args.imagenet1k_scores, set(args.exclude_imagenet1k))
    write_table(rows, n, measure, f'{args.out_dir}/imagenet1k_correlation_table',
                'Large-scale source transfer (ImageNet-1K)')


if __name__ == '__main__':
    main()
