"""
Print and save the aligned FCC / performance-gap table for the 20 directed *NIST pairs.

Uses nist_analysis.load_points, the same loader behind the correlations and plots, so the
table shows exactly the values those statistics are computed from.
"""
import csv
import sys

from scipy.stats import rankdata

import nist_analysis

CSV_OUT = 'nist_aligned_table.csv'
COLUMNS = ['pair', 'source', 'target', 'fcc_mean', 'fcc_ratio', 'pg',
           'rank_fcc_mean', 'rank_fcc_ratio', 'rank_pg']


def build_rows():
    pairs, mean, pg = nist_analysis.load_points('mean')
    pairs_r, ratio, pg_r = nist_analysis.load_points('ratio')
    assert pairs == pairs_r and (pg == pg_r).all(), 'aggregators are not aligned on the same pairs'

    ranks = {'mean': rankdata(mean), 'ratio': rankdata(ratio), 'pg': rankdata(pg)}
    rows = []
    for i, pair in enumerate(pairs):
        source, target = pair.split('->')
        rows.append({
            'pair': pair, 'source': source, 'target': target,
            'fcc_mean': float(mean[i]), 'fcc_ratio': float(ratio[i]), 'pg': float(pg[i]),
            'rank_fcc_mean': float(ranks['mean'][i]), 'rank_fcc_ratio': float(ranks['ratio'][i]),
            'rank_pg': float(ranks['pg'][i]),
        })
    return sorted(rows, key=lambda r: -r['pg']), (mean, ratio, pg)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rows, (mean, ratio, pg) = build_rows()

    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'{"#":>2}  {"pair":<32}{"FCC mean":>10}{"FCC ratio":>11}{"PG":>7}')
    for n, r in enumerate(rows, 1):
        print(f'{n:>2}  {r["pair"]:<32}{r["fcc_mean"]:>10.2f}{r["fcc_ratio"]:>11.4f}{r["pg"]:>7.2f}')

    print(f'\n{"aggregator":<12}{"Spearman":>10}{"Pearson":>10}{"W. Kendall":>12}')
    for name, fcc in (('mean', mean), ('ratio', ratio)):
        c = nist_analysis.correlations(fcc, pg)
        print(f'{name:<12}{c["spearman"]:>10.4f}{c["pearson"]:>10.4f}{c["weighted_kendall"]:>12.4f}')
    print(f'\nSaved {len(rows)} rows to {CSV_OUT}')


if __name__ == '__main__':
    main()
