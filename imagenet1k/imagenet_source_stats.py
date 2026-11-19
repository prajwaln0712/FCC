"""
Stage 1: ImageNet-1K pooled (mean, covariance) for FCC's source side -- computed once per
embedding network, cached, and reused by fcc_experiment.py. Pooled means label-agnostic: all
sampled images contribute to one combined (mean, cov), matching nist_experiment.py's and
tinyimagenet/fcc_common.py's existing source-side convention.

The --n-images flag (default 500000, out of ImageNet-1K's ~1.28M training images --
Design_choices_imagenet1k.txt item 5) is the mechanism for the pending-work item of later
re-running with the full training set: rerun with a larger/omitted --n-images value and rerun
fcc_experiment.py -- no retraining needed (important_concepts.txt explains why a shared sample
size matters for cross-dataset FCC comparisons in the first place).
"""
import argparse
import datetime
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import util
from nist.nist_extract_embeddings import build_network
from tinyimagenet.fcc_common import embed_01_images

CACHE_DIR = 'imagenet1k/cache'
IMAGENET_ROOT_DEFAULT = '/shared/rc/nlagent/data/imagenet'
SAMPLE_SEED = 20_260_100

NETWORKS = {
    # Inception moved 768-d -> full 2048-d (2026-11-10, user request). DINO unchanged, so its
    # existing cached stats (keyed by this same dict's key) are reused automatically; only
    # Inception recomputes, under its new key (doesn't collide with the old 'inception768' cache).
    'inception2048': dict(network='inception', inception_weights='fid', inception_dim=2048),
    'dinov3vith16plus': dict(network='dino', dino_model='facebook/dinov3-vith16plus-pretrain-lvd1689m'),
}


def _transform():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def compute_source_stats(net_key, net_args, n_images, imagenet_root, batch_size, num_workers, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f'{CACHE_DIR}/imagenet_source_stats_{net_key}.pt'
    if os.path.exists(cache_path):
        print(f'[{net_key}] {cache_path} already exists, skipping (delete it to recompute)')
        return torch.load(cache_path, weights_only=False)

    full = torchvision.datasets.ImageFolder(f'{imagenet_root}/train', transform=_transform())
    n_total = len(full)
    rng = np.random.default_rng(SAMPLE_SEED)
    n_use = min(n_images, n_total) if n_images is not None else n_total
    indices = rng.choice(n_total, size=n_use, replace=False)
    indices.sort()  # sequential-ish disk access, friendlier to a shared filesystem than random order

    print(f'[{net_key}] embedding {n_use}/{n_total} ImageNet-1K training images...')
    net = build_network(net_args['network'], argparse.Namespace(**net_args), device)

    loader = DataLoader(Subset(full, indices.tolist()), batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    feats = []
    seen = 0
    for xs, _ in loader:
        feats.append(embed_01_images(xs, net, batch_size=len(xs), device=device))
        seen += len(xs)
        if seen % (batch_size * 50) < batch_size:
            print(f'  {seen}/{n_use}')
    feats = torch.cat(feats)

    mean, cov = util.calculate_mean_covariance(feats)
    meta = dict(network=net_key, n_images=n_use, n_total=n_total, seed=SAMPLE_SEED,
                extracted_at=datetime.datetime.now().isoformat(timespec='seconds'), **net['meta'])
    payload = {'mean': mean, 'cov': cov, 'meta': meta}
    torch.save(payload, cache_path + '.tmp')
    os.replace(cache_path + '.tmp', cache_path)
    print(f'[{net_key}] saved {cache_path} (n_images={n_use})')
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-images', type=int, default=500_000,
                     help='Subsample size out of ImageNet-1K training set (pending work: rerun '
                          'with a larger value, or pass a value >= the full dataset size, once '
                          'initial results are in -- see Design_choices_imagenet1k.txt item 5)')
    ap.add_argument('--imagenet-root', default=IMAGENET_ROOT_DEFAULT)
    ap.add_argument('--networks', nargs='+', choices=list(NETWORKS), default=list(NETWORKS))
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    for net_key in args.networks:
        compute_source_stats(net_key, NETWORKS[net_key], args.n_images, args.imagenet_root,
                              args.batch_size, args.num_workers, device)


if __name__ == '__main__':
    main()
