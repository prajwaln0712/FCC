"""
LogME and LEEP for the augmentation experiment (TinyImageNet -> CIFAR-10), one row per seed.

Per dataaugmentation/aug_<seed>/ this loads the saved SOURCE model (the ResNet50 backbone
imagenet_ft_extractor.pth and the 200-way TinyImageNet head imagenet_classifier.pth, both saved
BEFORE the CIFAR-10 adaptation), runs it in eval mode over the CIFAR-10 training images, and scores:
  * LogME on the 2048-d backbone features vs the 10 CIFAR-10 labels
  * LEEP  on the softmax over the 200 source classes vs the 10 CIFAR-10 labels

CIFAR-10 images: aug_<seed>/transformed_train_cifar10.pt if it exists (the exact tensors
augment.py trained on; CIFAR-10 itself is not augmented, only TinyImageNet is), else CIFAR-10
reloaded from --cifar-root with ToTensor only (uint8/255, no normalisation -- same as
data.load_cifar_pytorch_datasets). Models were trained on [0,1] 32x32 inputs, so no normalisation.

Output: tables/other_metrics/aug_scores.csv (resumable: seeds already present are skipped unless
--overwrite). --limit N scores only N images and prints instead of writing the CSV (smoke/timing).

Usage (from ~/fcc):  python -m other_metrics.aug_scores [--seeds 1 2 3] [--limit 2000]
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.resnet import ResNet50
from other_metrics.extract_common import (extract, limit_dataset, score_logme_leep, set_strict_fp32,
                                          timed, read_rows, upsert_row)

OUT_CSV = 'tables/other_metrics/aug_scores.csv'
FIELDS = ['seed', 'logme', 'leep', 'n_images', 'n_target_classes', 'n_source_classes',
          'feature_dim', 'data_source', 'extract_seconds', 'score_seconds']
N_SOURCE_CLASSES = 200


def discover_seeds(parent_dir):
    seeds = []
    for path in glob.glob(f'{parent_dir}/aug_*/imagenet_ft_extractor.pth'):
        d = os.path.dirname(path)
        m = re.fullmatch(r'aug_(\d+)', os.path.basename(d))
        if m and os.path.exists(f'{d}/imagenet_classifier.pth'):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load_cifar10_train(parent_dir, seed, cifar_root):
    pt = f'{parent_dir}/aug_{seed}/transformed_train_cifar10.pt'
    if os.path.exists(pt):
        x, y = torch.load(pt)
        return x, y, pt
    import torchvision
    ds = torchvision.datasets.CIFAR10(cifar_root, train=True, download=False)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0  # == ToTensor()
    return x, torch.tensor(ds.targets), f'CIFAR10@{cifar_root}'


def load_source_model(parent_dir, seed, device):
    d = f'{parent_dir}/aug_{seed}'
    backbone = ResNet50()
    backbone.load_state_dict(torch.load(f'{d}/imagenet_ft_extractor.pth', map_location='cpu'))
    head = torch.nn.Linear(backbone.latent_dims, N_SOURCE_CLASSES)
    head.load_state_dict(torch.load(f'{d}/imagenet_classifier.pth', map_location='cpu'))
    backbone.to(device).eval()
    head.to(device).eval()

    def model_fn(x):
        f = backbone(x)
        return f, head(f)
    return model_fn


def score_seed(seed, args, device):
    x, y, data_source = load_cifar10_train(args.parent_dir, seed, args.cifar_root)
    dataset = limit_dataset(TensorDataset(x, y), args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model_fn = load_source_model(args.parent_dir, seed, device)

    (features, logits, labels), t_extract = timed(extract, model_fn, loader, device)
    assert logits.shape[1] == N_SOURCE_CLASSES, logits.shape
    scores, t_score = timed(score_logme_leep, features, logits, labels)
    return {'seed': seed, **scores, 'n_images': len(labels), 'n_target_classes': int(labels.max()) + 1,
            'n_source_classes': logits.shape[1], 'feature_dim': features.shape[1],
            'data_source': data_source, 'extract_seconds': round(t_extract, 1),
            'score_seconds': round(t_score, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent-dir', default='dataaugmentation')
    ap.add_argument('--cifar-root', default='data/CIFAR10')
    ap.add_argument('--seeds', type=int, nargs='*', default=None,
                    help='default: every aug_<seed>/ folder that has both .pth source files')
    ap.add_argument('--batch-size', type=int, default=500)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=None, help='score only N images; prints, no CSV write')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    set_strict_fp32()
    device = torch.device(args.device)
    seeds = args.seeds if args.seeds else discover_seeds(args.parent_dir)
    print(f'Device {device}; seeds to score: {seeds}')
    done = {r['seed'] for r in read_rows(OUT_CSV)}

    failed = []
    for seed in seeds:
        if args.limit is None and str(seed) in done and not args.overwrite:
            print(f'[seed {seed}] already in {OUT_CSV}, skipping (use --overwrite to redo)')
            continue
        start = time.time()
        try:
            row = score_seed(seed, args, device)
        except Exception as e:  # keep going so one broken seed folder doesn't lose the others
            print(f'[seed {seed}] FAILED: {type(e).__name__}: {e}')
            failed.append(seed)
            continue
        print(f'[seed {seed}] LogME={row["logme"]:.4f} LEEP={row["leep"]:.4f} '
              f'(n={row["n_images"]}, extract {row["extract_seconds"]}s, score {row["score_seconds"]}s, '
              f'total {time.time() - start:.1f}s, data: {row["data_source"]})')
        if args.limit is None:
            upsert_row(OUT_CSV, FIELDS, 'seed', row)
    if failed:
        raise SystemExit(f'seeds failed: {failed}')


if __name__ == '__main__':
    main()
