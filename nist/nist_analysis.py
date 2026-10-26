"""
Correlate raw-pixel FCC with performance gap (PG) over the 20 directed *NIST pairs.

Both FCC aggregators are reported (neither is dropped):
  * mean   - mean of the per-class Frechet distances (primary)
  * ratio  - mean / max of the per-class Frechet distances (coverage ratio)

Statistics per aggregator: Spearman rho, Pearson r (with p-values) and
scipy.stats.weightedtau (scipy returns no p-value for it, so it is reported as n/a).
Published OTDD / s-OTDD rows are taken from the paper, not recomputed.
"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr, weightedtau

CACHE_PATH = 'nist_distances_cache.npz'
PG_PATH = 'nist_pg_vals.json'
JSON_OUT = 'nist_correlations.json'
MD_OUT = 'nist_correlations.md'

AGGREGATORS = {
    'mean': 'FCC-pixels mean',
    'ratio': 'FCC-pixels ratio',
}

# Taken from the paper; None = not reported there.
PUBLISHED = [
    {'method': 'OTDD', 'spearman': 0.41, 'pearson': None, 'weighted_kendall': None},
    {'method': 's-OTDD', 'spearman': 0.40, 'pearson': 0.43, 'weighted_kendall': None},
]


def load_points(aggregator):
    """Return (pairs, fcc, pg) aligned on the same 20 directed-pair keys."""
    with open(PG_PATH) as f:
        pg = json.load(f)
    with np.load(CACHE_PATH) as npz:
        fcc = {k.split('::')[0]: float(npz[k]) for k in npz.files if k.endswith(f'::{aggregator}')}
    assert set(fcc) == set(pg), (
        f'pair mismatch: only in FCC cache {sorted(set(fcc) - set(pg))}, only in PG {sorted(set(pg) - set(fcc))}')
    assert len(pg) == 20, f'expected 20 pairs, got {len(pg)}'
    pairs = sorted(pg)
    return pairs, np.array([fcc[p] for p in pairs]), np.array([pg[p] for p in pairs])


def correlations(fcc, pg):
    rho, rho_p = spearmanr(fcc, pg)
    r, r_p = pearsonr(fcc, pg)
    wt = weightedtau(fcc, pg)
    return {
        'spearman': float(rho), 'spearman_p': float(rho_p),
        'pearson': float(r), 'pearson_p': float(r_p),
        'weighted_kendall': float(wt.statistic),
        'weighted_kendall_p': None,  # scipy.stats.weightedtau provides no p-value
    }


def compute_all():
    rows = [dict(p, source='paper', n_pairs=None) for p in PUBLISHED]
    for agg, label in AGGREGATORS.items():
        _, fcc, pg = load_points(agg)
        rows.append({'method': label, 'source': 'computed', 'n_pairs': len(pg), **correlations(fcc, pg)})
    return rows


def _cell(row, key):
    v = row.get(key)
    if v is None:
        return '—'
    p = row.get(f'{key}_p')
    return f'{v:.3f} (p = {p:.3g})' if p is not None else f'{v:.3f}'


def _cell_with_na(row, key):
    if row['source'] == 'computed' and key == 'weighted_kendall':
        return f'{row[key]:.3f} (p n/a)'
    return _cell(row, key)


def render_table(rows):
    header = ['Method', 'Spearman', 'Pearson', 'Weighted Kendall']
    lines = []
    for r in rows:
        name = r['method'] + (' †' if r['source'] == 'paper' else '')
        lines.append([name, _cell(r, 'spearman'), _cell(r, 'pearson'), _cell_with_na(r, 'weighted_kendall')])
    widths = [max(len(header[i]), *(len(l[i]) for l in lines)) for i in range(4)]
    fmt = lambda cells: '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'
    md = [fmt(header), '|' + '|'.join('-' * (w + 2) for w in widths) + '|'] + [fmt(l) for l in lines]
    md.append('')
    md.append('† Taken from the paper (not recomputed); "—" = not reported there. '
              'scipy.stats.weightedtau returns no p-value.')
    return '\n'.join(md)


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # † and — print correctly on Windows consoles
    rows = compute_all()
    table = render_table(rows)
    print('\n=== FCC vs performance gap (n = 20 directed pairs) ===')
    print(table)

    with open(JSON_OUT, 'w') as f:
        json.dump(rows, f, indent=2)
    with open(MD_OUT, 'w', encoding='utf-8') as f:
        f.write('# FCC vs performance gap: NIST datasets\n\n' + table + '\n')
    print(f'\nSaved {JSON_OUT} and {MD_OUT}')
    return rows


if __name__ == '__main__':
    main()
