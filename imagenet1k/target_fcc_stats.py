"""
Stage 3: per-class (mean, cov) for FCC's target side, for one target dataset and one embedding
network -- independent of Stage 1 (source stats) and Stage 2 (training).

REVISED 2026-11-10: N is now per-dataset (PER_DATASET_N below), not a single global cap. Each
dataset uses its own real minimum per-class count (confirmed via the real verify_targets run on
RC), so every class uses 100% real images with zero augmentation padding -- except EuroSAT,
a deliberate exception (see PER_DATASET_N's comment). This replaces the earlier global N=2000
approach (Design_choices_imagenet1k.txt item 3); the N=2000 cached results are preserved
automatically rather than overwritten, since the cache path below now encodes N in its filename.
Padding (tinyimagenet/fcc_common.py's augmented_copy, flip p=0.5, random-resized-crop scale
0.7-1.0, ColorJitter(0.4,0.4,0.4,0.1)) is now only ever exercised for EuroSAT.

Always uses each target's FULL native training set (via data_targets.load_target, not an 80/20
split) -- this is independent of Stage 2's training-time 80/20 validation carve, per the N-cap
decoupling decision (item 2): FCC's per-class stats and the training arms' data are deliberately
separate concerns.
"""
import argparse
import datetime
import os

import numpy as np
import torch

import util
from nist.nist_extract_embeddings import build_network
from tinyimagenet.fcc_common import embed_01_images, augmented_copy
from imagenet1k.data_targets import load_target, TARGETS

CACHE_DIR = 'imagenet1k/cache/target_fcc_stats'
PAD_SEED_BASE = 20_260_200

# Per-dataset N: each target's own minimum per-class count (confirmed via the real
# verify_targets run on RC, 2026-11-07/2026-11-10), so every class uses 100% real images with
# zero augmentation -- except EuroSAT, an explicit exception: its true minimum (1600) is below
# full Inception's 2048-d, which would make its covariance drastically rank-deficient (448 dims
# short); padded up to 2100 instead (still mostly real, minimal synthetic augmentation) per
# explicit user decision rather than silently keeping the global N=2000 cap this replaces.
PER_DATASET_N = {
    'cifar10': 5000,
    'places365': 5000,
    'eurosat': 2100,  # exception -- true min is 1600, padded for Inception-2048 compatibility
    'svhn': 4659,
    'mnist': 5421,
    'fashionmnist': 6000,
    'kmnist': 6000,
}

NETWORKS = {
    'inception2048': dict(network='inception', inception_weights='fid', inception_dim=2048),
    'dinov3vith16plus': dict(network='dino', dino_model='facebook/dinov3-vith16plus-pretrain-lvd1689m'),
}


def class_images(train_full, train_targets, class_id, n_per_class, seed):
    idx = np.where(train_targets == class_id)[0]
    rng = np.random.default_rng(seed)
    if len(idx) >= n_per_class:
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        return torch.stack([train_full[int(i)][0] for i in chosen])
    # Pad with augmented copies up to n_per_class -- same recipe as
    # tinyimagenet/fcc_common.py::compute_task_stats.
    xs = torch.stack([train_full[int(i)][0] for i in idx])
    copies = [xs]
    total = len(xs)
    while total < n_per_class:
        copy = augmented_copy(xs, rng)
        copies.append(copy)
        total += len(copy)
    return torch.cat(copies)[:n_per_class]


def compute_target_stats(target_name, target_idx, net_key, net_args, n_per_class, data_root,
                          batch_size, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f'{CACHE_DIR}/{target_name}_{net_key}_n{n_per_class}.pt'
    if os.path.exists(cache_path):
        print(f'[{target_name}/{net_key}] {cache_path} already exists, skipping')
        return torch.load(cache_path, weights_only=False)

    info = load_target(target_name, variant='01', root=data_root)
    net = build_network(net_args['network'], argparse.Namespace(**net_args), device)

    per_class = []
    for c in range(info['num_classes']):
        # Deterministic, reproducible seed: Python's builtin hash() is randomized per-process
        # (PYTHONHASHSEED) and must not be used here.
        seed = PAD_SEED_BASE + target_idx * 10_000 + c
        n_real = int((info['train_targets'] == c).sum())
        xs = class_images(info['train_full'], info['train_targets'], c, n_per_class, seed)
        feats = embed_01_images(xs, net, batch_size=batch_size, device=device)
        per_class.append(util.calculate_mean_covariance(feats))
        print(f'[{target_name}/{net_key}] class {c}/{info["num_classes"] - 1} done (n_real={n_real})')
        del xs

    meta = dict(target=target_name, network=net_key, n_per_class=n_per_class,
                num_classes=info['num_classes'],
                extracted_at=datetime.datetime.now().isoformat(timespec='seconds'), **net['meta'])
    payload = {'per_class': per_class, 'meta': meta}
    torch.save(payload, cache_path + '.tmp')
    os.replace(cache_path + '.tmp', cache_path)
    print(f'[{target_name}/{net_key}] saved {cache_path}')
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True, choices=TARGETS)
    ap.add_argument('--networks', nargs='+', choices=list(NETWORKS), default=list(NETWORKS))
    ap.add_argument('--n-per-class', type=int, default=None,
                     help='Override this target\'s default N (its own entry in PER_DATASET_N); '
                          'omit to use the default.')
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    target_idx = TARGETS.index(args.target)
    n_per_class = args.n_per_class if args.n_per_class is not None else PER_DATASET_N[args.target]
    for net_key in args.networks:
        compute_target_stats(args.target, target_idx, net_key, NETWORKS[net_key],
                              n_per_class, args.data_root, args.batch_size, device)


if __name__ == '__main__':
    main()
