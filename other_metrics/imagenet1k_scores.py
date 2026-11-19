"""
LogME and LEEP for the imagenet1k experiment (ImageNet-1K source -> 7 targets), one row per target.

Source model: torchvision's ImageNet-1K-pretrained ResNet18 (ResNet18_Weights.IMAGENET1K_V1) -- the
same backbone the frozen-transfer arm used -- in eval mode with its original 1000-way fc head.
For each target the images come from imagenet1k.data_targets.load_target(name, 'norm') (224x224,
ImageNet normalisation, i.e. exactly the training pipeline) over the target's full native TRAIN set
(EuroSAT: its 80% carve), and:
  * LogME is computed on the 512-d avgpool features vs the target labels
  * LEEP  is computed on the softmax over the 1000 ImageNet classes vs the target labels
All 7 targets are scored (SVHN included); drop SVHN later in the analysis if wanted.

Output: tables/other_metrics/imagenet1k_scores.csv (resumable: targets already present are skipped
unless --overwrite). --limit N scores only N images per target and prints instead of writing the CSV
(smoke/timing). A multi-target run keeps going if one target fails and exits non-zero at the end.

Usage (from ~/fcc):  python -m other_metrics.imagenet1k_scores [--targets cifar10 eurosat] [--limit 2000]
"""
import argparse
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from imagenet1k.data_targets import load_target, TARGETS
from other_metrics.extract_common import (extract, limit_dataset, score_logme_leep, set_strict_fp32,
                                          timed, read_rows, upsert_row)

OUT_CSV = 'tables/other_metrics/imagenet1k_scores.csv'
FIELDS = ['target', 'logme', 'leep', 'n_images', 'n_target_classes', 'n_source_classes',
          'feature_dim', 'extract_seconds', 'score_seconds']


def build_source_model(device):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.to(device).eval()
    trunk = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool)

    def model_fn(x):
        f = torch.flatten(trunk(x), 1)  # 512-d, identical to torchvision's own forward before fc
        return f, net.fc(f)
    return model_fn


def score_target(name, model_fn, args, device):
    info = load_target(name, variant='norm', root=args.data_root)
    dataset = limit_dataset(info['train_full'], args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    (features, logits, labels), t_extract = timed(extract, model_fn, loader, device)
    if args.limit is None:
        # loader order == dataset order (no shuffle), so labels must equal the loader's own targets
        assert np.array_equal(labels, np.asarray(info['train_targets'])), f'{name}: label order mismatch'
    assert logits.shape[1] == 1000, logits.shape
    scores, t_score = timed(score_logme_leep, features, logits, labels)
    return {'target': name, **scores, 'n_images': len(labels), 'n_target_classes': int(labels.max()) + 1,
            'n_source_classes': logits.shape[1], 'feature_dim': features.shape[1],
            'extract_seconds': round(t_extract, 1), 'score_seconds': round(t_score, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', choices=TARGETS, default=TARGETS)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=None,
                    help='score only N images per target; prints, no CSV write')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    set_strict_fp32()
    device = torch.device(args.device)
    print(f'Device {device}; targets to score: {args.targets}')
    model_fn = build_source_model(device)
    done = {r['target'] for r in read_rows(OUT_CSV)}

    failed = []
    for name in args.targets:
        if args.limit is None and name in done and not args.overwrite:
            print(f'[{name}] already in {OUT_CSV}, skipping (use --overwrite to redo)')
            continue
        start = time.time()
        try:
            row = score_target(name, model_fn, args, device)
        except Exception as e:
            print(f'[{name}] FAILED: {type(e).__name__}: {e}')
            failed.append(name)
            continue
        print(f'[{name}] LogME={row["logme"]:.4f} LEEP={row["leep"]:.4f} '
              f'(n={row["n_images"]}, extract {row["extract_seconds"]}s, score {row["score_seconds"]}s, '
              f'total {time.time() - start:.1f}s)')
        if args.limit is None:
            upsert_row(OUT_CSV, FIELDS, 'target', row)
    if failed:
        raise SystemExit(f'targets failed: {failed}')


if __name__ == '__main__':
    main()
