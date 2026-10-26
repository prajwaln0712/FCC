"""
Step 1: train a ResNet18 from scratch (all layers trainable) on one TinyImageNet task split.

Loads saved_split_task/data/trainset_{task_id}.pt / test_{task_id}.pt (produced by tiny_split.py),
trains for --epochs with Adam lr=0.001, and saves a periodic checkpoint every 10 epochs
(checkpoints/task_{task_id}/task_{task_id}_{epoch}epoch.pt) plus task_{task_id}_last.pt (final
epoch, unconditional), and a JSON-formatted results_task_{task_id}.txt whose "best_acc" field
still tracks the true highest test accuracy seen, independent of which epoch's weights got
checkpointed. transfer_eval.py loads the _last.pt file for real transfer learning -- there is no
"_best.pt" file anymore.
"""
import argparse
import itertools
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.resnet18 import ResNet18
from models.trainer import train, test_func

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 from scratch on one TinyImageNet task split")
    parser.add_argument("--task-id", type=int, required=True, help="Task index (0-9)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--seed", type=int, default=1, help="Base seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    seed = args.seed + args.task_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = f"checkpoints/task_{args.task_id}"
    os.makedirs(out_dir, exist_ok=True)

    x_train, y_train = torch.load(f"saved_split_task/data/trainset_{args.task_id}.pt")
    x_test, y_test = torch.load(f"saved_split_task/data/test_{args.task_id}.pt")
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    backbone = ResNet18().to(DEVICE)
    classifier = nn.Linear(backbone.latent_dims, 20).to(DEVICE)
    # One combined Adam optimizer over both modules, routed through train()'s single
    # ft_extractor_optimizer slot. Passing the same optimizer as BOTH of train()'s optimizer
    # arguments would make it call .step() on it twice per batch (it loops over both slots) --
    # wrong for Adam's momentum state -- so classifier_optimizer stays None here.
    optimizer = torch.optim.Adam(
        itertools.chain(backbone.parameters(), classifier.parameters()), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    test_acc = None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train(feature_extractor=backbone, classifier=classifier, device=DEVICE,
                            train_loader=train_loader, epoch=epoch, criterion=criterion,
                            ft_extractor_optimizer=optimizer, classifier_optimizer=None)
        test_acc = test_func(feature_extractor=backbone, classifier=classifier, device=DEVICE,
                              test_loader=test_loader)
        print(f"[task {args.task_id}] epoch {epoch}, train loss: {train_loss}, test acc: {test_acc}")
        best_acc = max(best_acc, test_acc)  # still tracked for results.txt regardless of checkpoint cadence
        if epoch % 10 == 0:
            torch.save({"backbone": backbone.state_dict(), "classifier": classifier.state_dict()},
                       f"{out_dir}/task_{args.task_id}_{epoch}epoch.pt")
    elapsed = time.time() - t_start
    print(f"[task {args.task_id}] training ({args.epochs} epochs) took {elapsed:.1f}s "
          f"({elapsed / args.epochs:.2f}s/epoch)")

    torch.save({"backbone": backbone.state_dict(), "classifier": classifier.state_dict()},
               f"{out_dir}/task_{args.task_id}_last.pt")

    results = {
        "task_id": args.task_id,
        "epochs": args.epochs,
        "seed": seed,
        "final_acc": test_acc,
        "best_acc": best_acc,
        "train_seconds": elapsed,
    }
    with open(f"{out_dir}/results_task_{args.task_id}.txt", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_dir}/results_task_{args.task_id}.txt")


if __name__ == "__main__":
    main()
