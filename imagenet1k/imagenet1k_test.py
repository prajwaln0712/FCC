"""
Stage 2: per-target entry point for the ImageNet-1K FCC experiment. Trains BOTH the
frozen-backbone transfer arm and the full-from-scratch arm against one target dataset
(independent of Stage 1's ImageNet-1K source stats and Stage 3's FCC target stats), writing
imagenet1k/results/{target}_{frozen,scratch}.json and imagenet1k/checkpoints/{target}_{arm}_best.pt.

Frozen-transfer arm: torchvision's ImageNet-1K-pretrained ResNet18 (NOT this repo's custom
models/resnet18.py -- confirmed incompatible state_dict naming, and we specifically want real
torchvision pretrained weights here, not a from-scratch-trained custom backbone), backbone
frozen, only a new linear head trained (Adam lr=0.01, matches tinyimagenet/transfer_eval.py).

Full-from-scratch arm: torchvision's resnet18 with random initialization (weights=None), both
backbone and head trained together (Adam lr=0.001, matches tinyimagenet/train_task.py).

Both arms: batch size 64, no training-time augmentation (Design_choices_imagenet1k.txt item
8), adaptive early-stopping loop from imagenet1k/train_common.py (80/20 validation split, 0.4pp
threshold, 5-epoch patience, 30-epoch cap).

Usage: python -m imagenet1k.imagenet1k_test --target cifar10 [--arm {frozen,scratch,both}]
"""
import argparse
import itertools
import json
import os

import torch
import torch.nn as nn
import torchvision

from models.trainer import frozen_module
from imagenet1k.data_targets import load_target, stratified_split, TARGETS
from imagenet1k.train_common import seed_everything, split_seed_for, run_seed_for, run_arm

RESULTS_DIR = 'imagenet1k/results'
CHECKPOINTS_DIR = 'imagenet1k/checkpoints'

ARM_IDX = {'frozen': 0, 'scratch': 1}


def build_frozen_arm(num_classes, device):
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    frozen_module(backbone)
    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    # ft_extractor_optimizer=None -> models/trainer.py::train() runs the backbone forward pass
    # under torch.no_grad(), i.e. the frozen-backbone / linear-probe mode.
    return backbone, classifier, None, optimizer


def build_scratch_arm(num_classes, device):
    backbone = torchvision.models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.Adam(
        itertools.chain(backbone.parameters(), classifier.parameters()), lr=0.001)
    return backbone, classifier, optimizer, None


ARM_BUILDERS = {'frozen': build_frozen_arm, 'scratch': build_scratch_arm}


def run_one_arm(target_name, target_idx, arm, info, device, overwrite):
    out_json = f'{RESULTS_DIR}/{target_name}_{arm}.json'
    if os.path.exists(out_json) and not overwrite:
        print(f'[{target_name}/{arm}] {out_json} already exists, skipping (use --overwrite to redo)')
        return

    split_seed = split_seed_for(target_idx)
    run_seed = run_seed_for(target_idx, ARM_IDX[arm])

    train_idx, val_idx = stratified_split(info['train_targets'], [0.8, 0.2], seed=split_seed)

    seed_everything(run_seed)
    backbone, classifier, ft_opt, clf_opt = ARM_BUILDERS[arm](info['num_classes'], device)

    results, best_state = run_arm(
        feature_extractor=backbone, classifier=classifier, device=device,
        train_full=info['train_full'], train_idx=train_idx, val_idx=val_idx,
        test_dataset=info['test'], ft_extractor_optimizer=ft_opt, classifier_optimizer=clf_opt,
        out_prefix=f'{target_name}/{arm}')

    results.update({
        'target': target_name, 'arm': arm, 'num_classes': info['num_classes'],
        'n_train_native': len(info['train_targets']), 'n_test_native': len(info['test']),
        'split_seed': split_seed, 'run_seed': run_seed,
    })

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    torch.save(best_state, f'{CHECKPOINTS_DIR}/{target_name}_{arm}_best.pt')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[{target_name}/{arm}] saved {out_json}: final_acc={results["final_acc"]:.2f} '
          f'best_val_acc={results["best_val_acc"]:.2f} epochs_run={results["epochs_run"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True, choices=TARGETS)
    ap.add_argument('--arm', choices=['frozen', 'scratch', 'both'], default='both')
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device)
    target_idx = TARGETS.index(args.target)
    info = load_target(args.target, variant='norm', root=args.data_root)

    arms = ['frozen', 'scratch'] if args.arm == 'both' else [args.arm]
    for arm in arms:
        run_one_arm(args.target, target_idx, arm, info, device, args.overwrite)


if __name__ == '__main__':
    main()
