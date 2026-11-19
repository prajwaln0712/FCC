"""
Stage 4: FCC(ImageNet-1K -> target) for all 7 targets x 2 embedding networks, from the cached
Stage 1 (source pooled stats) and Stage 3 (target per-class stats) outputs. Modeled on
nist/nist_experiment.py's rigor: ridge/max_imag-hardened Frechet distance, finite/non-negative
assertions, a cache self-check on load, and a shape-check validation pass.
"""
import argparse
import os

import numpy as np
import torch

import util
from imagenet1k.data_targets import TARGETS
from imagenet1k.imagenet_source_stats import NETWORKS, CACHE_DIR as SOURCE_CACHE_DIR
from imagenet1k.target_fcc_stats import CACHE_DIR as TARGET_CACHE_DIR, PER_DATASET_N

CACHE_PATH = 'imagenet1k/cache/imagenet1k_fcc_cache.npz'
RIDGE = 1e-6
MAX_IMAG = 1e-3


def fcc_from_stats(src_mean, src_cov, tgt_per_class):
    per_class = np.array([
        util.calculate_fid(src_mean, src_cov, mu_c, sigma_c, ridge=RIDGE, max_imag=MAX_IMAG)
        for mu_c, sigma_c in tgt_per_class
    ])
    assert np.all(np.isfinite(per_class)), f'non-finite Frechet distance: {per_class}'
    assert np.all(per_class >= 0), f'negative Frechet distance: {per_class}'
    return per_class, float(per_class.mean()), float(per_class.mean() / per_class.max())


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    nested = {}
    with np.load(CACHE_PATH) as npz:
        for key in npz.files:
            target, network, field = key.split('::')
            nested.setdefault((target, network), {})[field] = npz[key]
    return {k: {'per_class': v['per_class'], 'mean': float(v['mean']), 'ratio': float(v['ratio'])}
            for k, v in nested.items()}


def save_cache(results):
    flat = {}
    for (target, network), r in results.items():
        flat[f'{target}::{network}::per_class'] = r['per_class']
        flat[f'{target}::{network}::mean'] = np.array(r['mean'])
        flat[f'{target}::{network}::ratio'] = np.array(r['ratio'])
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, **flat)


def _load_source(network):
    return torch.load(f'{SOURCE_CACHE_DIR}/imagenet_source_stats_{network}.pt', weights_only=False)


def _load_target_stats(target, network):
    n = PER_DATASET_N[target]
    return torch.load(f'{TARGET_CACHE_DIR}/{target}_{network}_n{n}.pt', weights_only=False)


def verify_cache(results):
    (target, network), cached = next(iter(results.items()))
    src = _load_source(network)
    tgt = _load_target_stats(target, network)
    vec, _, _ = fcc_from_stats(src['mean'], src['cov'], tgt['per_class'])
    np.testing.assert_allclose(vec, cached['per_class'], rtol=1e-8)
    print(f'  cache check: recomputed ({target}, {network}) matches cached per-class vector')


def compute_all(targets, networks):
    results = load_cache()
    if results:
        print(f'Loaded {len(results)} cached (target, network) pairs from {CACHE_PATH}')
        verify_cache(results)

    for network in networks:
        src = _load_source(network)
        for target in targets:
            key = (target, network)
            if key in results:
                print(f'  {target:14s} {network:18s} cached    mean={results[key]["mean"]:.4f}')
                continue
            tgt = _load_target_stats(target, network)
            vec, mean, ratio = fcc_from_stats(src['mean'], src['cov'], tgt['per_class'])
            results[key] = {'per_class': vec, 'mean': mean, 'ratio': ratio}
            print(f'  {target:14s} {network:18s} mean={mean:.4f} ratio={ratio:.4f}')

    save_cache(results)
    return results


def validate(results, targets, networks):
    print('\n=== Validation ===')
    all_ok = True

    expected_n = len(targets) * len(networks)
    ok = len(results) == expected_n
    all_ok &= ok
    print(f'[{"PASS" if ok else "FAIL"}] shapes: {len(results)} cache entries (expected {expected_n})')

    shape_ok = True
    for (target, network), r in results.items():
        meta = _load_target_stats(target, network)['meta']
        if len(r['per_class']) != meta['num_classes']:
            shape_ok = False
            print(f'  mismatch: ({target}, {network}) per_class len={len(r["per_class"])} '
                  f'!= expected {meta["num_classes"]}')
    all_ok &= shape_ok
    print(f"[{'PASS' if shape_ok else 'FAIL'}] per-class length matches each target's class count")

    # WARN-only sanity print (no hard-fail ordering assertion, unlike nist_experiment.py's
    # MNIST/USPS check -- there is no equally strong prior here with only 1 source and 7
    # heterogeneous targets).
    for network in networks:
        ranked = sorted(((t, results[(t, network)]['mean']) for t in targets), key=lambda p: p[1])
        print(f'[INFO] {network}: targets ranked by FCC mean (closest to ImageNet first): '
              f'{[f"{t}={m:.2f}" for t, m in ranked]}')

    assert all_ok, 'validation failed (see FAIL lines above)'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS)
    ap.add_argument('--networks', nargs='+', choices=list(NETWORKS), default=list(NETWORKS))
    args = ap.parse_args()

    results = compute_all(args.targets, args.networks)
    validate(results, args.targets, args.networks)


if __name__ == '__main__':
    main()
