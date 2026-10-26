"""
FCC (Fréchet Class Coverage) for every directed pair of the five *NIST datasets
(MNIST, FashionMNIST, KMNIST, USPS, EMNIST-letters), on raw 784-d pixels (default) or on
Inception / DINOv3 embeddings saved by nist_extract_embeddings.py (--features inception|dino).

FCC(S -> T):
  * source pool  = the full source feature matrix; source labels are ignored
  * for each target class c, the Fréchet distance between the source pool and
    the target rows with label c  ->  per-class vector of length n_classes(T)
  * FCC scalar   = MEAN of the per-class vector   <- PRIMARY value used downstream
  * coverage     = mean / max of the per-class vector   (secondary variant)

Pooled and per-class (mean, covariance) summaries are computed once up front;
the 20-pair loop consumes only those summaries.  Pixel covariances are
rank-deficient, so RIDGE * I is added before the matrix square root.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data
import util

DATASETS = ['MNIST', 'FashionMNIST', 'KMNIST', 'USPS', 'EMNIST-letters']
RIDGE = 1e-6
MAX_IMAG = 1e-3
CACHE_PATH = 'nist_distances_cache.npz'
RUNTIMES_PATH = 'nist_runtimes.json'
EMBEDDING_DIR = 'nist_embeddings'
SELF_SPLIT_SEED = 0
CACHE_RTOL = 1e-8


def pair_key(src, tgt):
    return f'{src}->{tgt}'


def ordered_pairs():
    return [(s, t) for s in DATASETS for t in DATASETS if s != t]


LOADERS = {
    'MNIST': data.load_mnist,
    'FashionMNIST': data.load_fashionmnist,
    'KMNIST': data.load_kmnist,
    'USPS': data.load_usps,
    'EMNIST-letters': data.load_emnist_letters,
}


def load_datasets(features='pixels'):
    if features == 'pixels':
        return {name: data.dataset_to_tensors(LOADERS[name](train=True, download=True)) for name in DATASETS}
    datasets = {}
    for name in DATASETS:
        path = os.path.join(EMBEDDING_DIR, f'{name}_{features}.pt')
        payload = torch.load(path)
        assert payload['meta']['dataset'] == name and payload['meta']['network'] == features, f'unexpected contents in {path}'
        datasets[name] = (payload['features'].double(), payload['labels'])
        print(f'  {name:15s} {path}  {tuple(payload["features"].shape)}  {payload["meta"]["weights"]}')
    return datasets


def n_classes_of(y):
    return int(y.max().item()) + 1


def compute_statistics(datasets):
    start = time.time()
    pooled, per_class = {}, {}
    for name, (X, y) in datasets.items():
        pooled[name] = util.calculate_mean_covariance(X)
        per_class[name] = [util.calculate_mean_covariance(X[y == c]) for c in range(n_classes_of(y))]
        min_n = int(torch.bincount(y).min())
        note = f'  <- fewer samples than dims ({X.shape[1]}): rank-deficient class covariance' if min_n < X.shape[1] else ''
        print(f'  {name:15s} n={len(X):6d} classes={len(per_class[name])} min_class_n={min_n}{note}')
    return pooled, per_class, time.time() - start


def fcc_from_stats(src_pooled, tgt_class_stats):
    mu_s, sigma_s = src_pooled
    per_class = np.array([
        util.calculate_fid(mu_s, sigma_s, mu_c, sigma_c, ridge=RIDGE, max_imag=MAX_IMAG)
        for mu_c, sigma_c in tgt_class_stats
    ])
    assert np.all(np.isfinite(per_class)), f'non-finite Fréchet distance: {per_class}'
    assert np.all(per_class >= 0), f'negative Fréchet distance: {per_class}'
    return per_class, float(per_class.mean()), float(per_class.mean() / per_class.max())


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    nested = {}
    with np.load(CACHE_PATH) as npz:
        for key in npz.files:
            pair, field = key.split('::')
            nested.setdefault(pair, {})[field] = npz[key]
    return {
        pair: {'per_class': v['per_class'], 'mean': float(v['mean']), 'ratio': float(v['ratio'])}
        for pair, v in nested.items()
    }


def save_cache(results):
    flat = {}
    for pair, r in results.items():
        flat[f'{pair}::per_class'] = r['per_class']
        flat[f'{pair}::mean'] = np.array(r['mean'])
        flat[f'{pair}::ratio'] = np.array(r['ratio'])
    np.savez_compressed(CACHE_PATH, **flat)


def load_runtimes():
    if os.path.exists(RUNTIMES_PATH):
        with open(RUNTIMES_PATH) as f:
            return json.load(f)
    return {}


def verify_cache(pooled, per_class, results):
    src, tgt = next((s, t) for s, t in ordered_pairs() if pair_key(s, t) in results)
    key = pair_key(src, tgt)
    vec, _, _ = fcc_from_stats(pooled[src], per_class[tgt])
    np.testing.assert_allclose(vec, results[key]['per_class'], rtol=CACHE_RTOL)
    print(f'  cache check: recomputed {key} matches cached per-class vector (rtol={CACHE_RTOL:.0e})')


def compute_all_pairs(pooled, per_class, results, runtimes):
    for src, tgt in ordered_pairs():
        key = pair_key(src, tgt)
        if key in results:
            print(f'  {key:30s} cached    mean={results[key]["mean"]:.4f}')
            continue
        start = time.time()
        vec, mean, ratio = fcc_from_stats(pooled[src], per_class[tgt])
        runtimes[key] = time.time() - start
        results[key] = {'per_class': vec, 'mean': mean, 'ratio': ratio}
        print(f'  {key:30s} {runtimes[key]:6.2f}s   mean={mean:.4f} ratio={ratio:.4f}')


def print_table(results):
    w = max(len(n) for n in DATASETS) + 2
    print('\n=== FCC mean (rows = source, cols = target) ===')
    print(' ' * w + ''.join(f'{n:>{w}}' for n in DATASETS))
    for src in DATASETS:
        cells = ''.join(
            f'{"":>{w}}' if src == tgt else f'{results[pair_key(src, tgt)]["mean"]:>{w}.4f}'
            for tgt in DATASETS
        )
        print(f'{src:<{w}}{cells}')


def validate(datasets, results):
    print('\n=== Validation ===')
    means = {k: r['mean'] for k, r in results.items()}
    all_ok = True

    # 1. self-distance: MNIST split into two halves with a fixed seed
    X, y = datasets['MNIST']
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SELF_SPLIT_SEED))
    half = len(X) // 2
    a, b = perm[:half], perm[half:]
    Xb, yb = X[b], y[b]
    src_pooled = util.calculate_mean_covariance(X[a])
    tgt_stats = [util.calculate_mean_covariance(Xb[yb == c]) for c in range(n_classes_of(y))]
    _, self_fcc, _ = fcc_from_stats(src_pooled, tgt_stats)
    min_cross = min(means.values())
    ok = self_fcc < min_cross
    all_ok &= ok
    print(f'[{"PASS" if ok else "FAIL"}] self-distance: MNIST half->half FCC={self_fcc:.4f}, '
          f'smallest cross-dataset FCC={min_cross:.4f}')

    # 2. asymmetry over the 10 unordered pairs
    unordered = [(p, q) for i, p in enumerate(DATASETS) for q in DATASETS[i + 1:]]
    n_asym = sum(not np.isclose(means[pair_key(p, q)], means[pair_key(q, p)]) for p, q in unordered)
    ok = n_asym > len(unordered) / 2
    all_ok &= ok
    print(f'[{"PASS" if ok else "FAIL"}] asymmetry: FCC(A->B) != FCC(B->A) for {n_asym}/{len(unordered)} unordered pairs')

    # 3. shapes
    n_classes = {name: n_classes_of(y) for name, (_, y) in datasets.items()}
    shape_ok = all(len(results[pair_key(s, t)]['per_class']) == n_classes[t] for s, t in ordered_pairs())
    ok = shape_ok and len(results) == 20
    all_ok &= ok
    print(f'[{"PASS" if ok else "FAIL"}] shapes: per-class lengths match target class counts={shape_ok}, '
          f'cache keys={len(results)} (expected 20)')

    # 4. sanity ordering (warnings only)
    ranked = sorted(means, key=means.get)
    rank = {k: i + 1 for i, k in enumerate(ranked)}
    mnist_usps = [pair_key('MNIST', 'USPS'), pair_key('USPS', 'MNIST')]
    mu_ranks = [rank[k] for k in mnist_usps]
    if max(mu_ranks) <= 5:
        print(f'[OK]   MNIST<->USPS among the 5 smallest distances (ranks {mu_ranks} of {len(ranked)})')
    else:
        print(f'[WARN] MNIST<->USPS not among the 5 smallest distances (ranks {mu_ranks} of {len(ranked)})')
    fashion_ranks = sorted(rank[k] for k in ranked if 'FashionMNIST' in k)
    if np.median(fashion_ranks) > len(ranked) / 2:
        print(f'[OK]   FashionMNIST pairs among the largest distances (ranks {fashion_ranks} of {len(ranked)})')
    else:
        print(f'[WARN] FashionMNIST pairs not among the largest distances (ranks {fashion_ranks} of {len(ranked)})')

    assert all_ok, 'validation failed (see FAIL lines above)'


def print_timing(runtimes):
    pair_times = {k: v for k, v in runtimes.items() if '->' in k}
    print('\n=== Timing ===')
    print(f'statistics (one-time): {runtimes["statistics"]:.2f}s')
    if pair_times:
        slowest = max(pair_times, key=pair_times.get)
        print(f'pairs total:           {sum(pair_times.values()):.2f}s over {len(pair_times)} pairs '
              f'(slowest {slowest}: {pair_times[slowest]:.2f}s)')
    else:
        print('pairs total:           0.00s (all pairs loaded from cache)')


def main():
    global CACHE_PATH, RUNTIMES_PATH, RIDGE
    ap = argparse.ArgumentParser(description='FCC for all 20 directed *NIST pairs')
    ap.add_argument('--features', choices=['pixels', 'inception', 'dino'], default='pixels')
    ap.add_argument('--ridge', type=float, default=RIDGE)
    args = ap.parse_args()
    RIDGE = args.ridge
    if args.features != 'pixels':
        CACHE_PATH = f'nist_distances_cache_{args.features}.npz'
        RUNTIMES_PATH = f'nist_runtimes_{args.features}.json'
    print(f'Features: {args.features}  ridge={RIDGE:g}  cache={CACHE_PATH}')

    print('Loading datasets...')
    datasets = load_datasets(args.features)

    print('Computing pooled and per-class statistics...')
    pooled, per_class, t_stats = compute_statistics(datasets)
    print(f'  statistics stage took {t_stats:.2f}s')

    results = load_cache()
    runtimes = load_runtimes()
    runtimes['statistics'] = t_stats

    if results:
        print(f'Loaded {len(results)} cached pairs from {CACHE_PATH}')
        verify_cache(pooled, per_class, results)

    print('Computing FCC for all ordered pairs...')
    compute_all_pairs(pooled, per_class, results, runtimes)
    runtimes['pairs_total'] = sum(v for k, v in runtimes.items() if '->' in k)

    save_cache(results)
    with open(RUNTIMES_PATH, 'w') as f:
        json.dump(runtimes, f, indent=2)

    print_table(results)
    validate(datasets, results)
    print_timing(runtimes)


if __name__ == '__main__':
    main()
