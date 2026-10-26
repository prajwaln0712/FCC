"""
Combined script for: transfer learning (Step 2), the --load_pretrain 0 floor baseline, and
baselineA -- one script, three flags (--source, --target, --load_pretrain). baselineA is just the
source == target, load_pretrain=0 case of this same script; it is not a separate file.

Always freezes the backbone and trains only a fresh fc layer on the target task's data.
  --load_pretrain 1 (default), source != target: real transfer learning.
      Loads checkpoints/task_{source}/task_{source}_last.pt's backbone weights (the fully-trained
      final checkpoint -- train_task.py no longer writes a "_best.pt" file).
      Output: finetune/source_{source}/target_{target}_{best,last}.pt, results_target_{target}.txt
  --load_pretrain 0, source != target: the zero-information floor baseline.
      Backbone is randomly initialized (not loaded from any checkpoint).
      Output: baseline/source_{source}/target_{target}_{best,last}.pt, results_target_{target}.txt
  --load_pretrain 0, source == target: baselineA.
      Same random-backbone procedure as above, but target is forced equal to source.
      Output: baselineA/task_{target}_{best,last}.pt, results_task_{target}.txt

This script's own "_best.pt" here is still a periodic checkpoint (every 5 epochs -- --epochs
defaults to 20, so every 10 would only give 2 checkpoints), not the best-accuracy one. "_last.pt"
stays unconditional (final epoch, every run).

Checkpoints here only save the classifier's state dict -- the backbone is frozen throughout (never
changes during this script), so saving it again would just duplicate either Step 1's checkpoint
(load_pretrain=1) or a backbone that is deterministically reproducible from backbone_seed
(load_pretrain=0).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.resnet18 import ResNet18
from models.trainer import train, test_func, frozen_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Transfer learning / floor baseline / baselineA for TinyImageNet task splits")
    parser.add_argument("--source", type=int, required=True, help="Source task index (0-9)")
    parser.add_argument("--target", type=int, required=True, help="Target task index (0-9)")
    parser.add_argument("--load_pretrain", type=int, choices=[0, 1], default=1,
                        help="1: load checkpoints/task_{source}'s backbone (real transfer). "
                             "0: random-init backbone instead (floor baseline / baselineA).")
    parser.add_argument("--epochs", type=int, default=20, help="fc-only training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Base seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    backbone_seed = args.seed + 20_000_000 + args.source
    run_seed = args.seed + args.source * 1000 + args.target * 10 + args.load_pretrain

    if args.load_pretrain == 0:
        # Seed the random backbone init with a seed tied only to `source`, so every target run
        # under the same --source gets the identical random backbone -- a fair, fixed-features
        # comparison across that source's 9 targets, mirroring how the real transfer case reuses
        # one fixed pretrained backbone across its 9 targets.
        seed_everything(backbone_seed)
    backbone = ResNet18().to(DEVICE)

    # Everything else (fc init, loader shuffling) is seeded independently of the backbone init.
    seed_everything(run_seed)

    if args.load_pretrain == 1:
        ckpt = torch.load(f"checkpoints/task_{args.source}/task_{args.source}_last.pt", map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone"])

    frozen_module(backbone)
    classifier = nn.Linear(backbone.latent_dims, 20).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x_train, y_train = torch.load(f"saved_split_task/data/trainset_{args.target}.pt")
    x_test, y_test = torch.load(f"saved_split_task/data/test_{args.target}.pt")
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    if args.source == args.target:
        out_dir = "baselineA"
        stem = f"task_{args.target}"
    elif args.load_pretrain == 1:
        out_dir = f"finetune/source_{args.source}"
        stem = f"target_{args.target}"
    else:
        out_dir = f"baseline/source_{args.source}"
        stem = f"target_{args.target}"
    os.makedirs(out_dir, exist_ok=True)

    best_acc = -1.0
    test_acc = None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train(feature_extractor=backbone, classifier=classifier, device=DEVICE,
                            train_loader=train_loader, epoch=epoch, criterion=criterion,
                            ft_extractor_optimizer=None, classifier_optimizer=optimizer)
        test_acc = test_func(feature_extractor=backbone, classifier=classifier, device=DEVICE,
                              test_loader=test_loader)
        print(f"[source {args.source} target {args.target} load_pretrain {args.load_pretrain}] "
              f"epoch {epoch}, train loss: {train_loss}, test acc: {test_acc}")
        best_acc = max(best_acc, test_acc)  # still tracked for results.txt regardless of checkpoint cadence
        if epoch % 5 == 0:
            torch.save({"classifier": classifier.state_dict()}, f"{out_dir}/{stem}_best.pt")
    elapsed = time.time() - t_start
    print(f"[{out_dir}/{stem}] fc-only training ({args.epochs} epochs) took {elapsed:.1f}s "
          f"({elapsed / args.epochs:.2f}s/epoch)")

    torch.save({"classifier": classifier.state_dict()}, f"{out_dir}/{stem}_last.pt")

    results = {
        "source": args.source,
        "target": args.target,
        "load_pretrain": args.load_pretrain,
        "epochs": args.epochs,
        "backbone_seed": backbone_seed if args.load_pretrain == 0 else None,
        "run_seed": run_seed,
        "final_acc": test_acc,
        "best_acc": best_acc,
        "train_seconds": elapsed,
    }
    with open(f"{out_dir}/results_{stem}.txt", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_dir}/results_{stem}.txt")


if __name__ == "__main__":
    main()
