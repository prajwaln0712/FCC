"""
BaselineB: per-task "train everything from scratch for 20 epochs" floor, read off train_task.py's
existing checkpoints rather than re-running training.

train_task.py trained each task for 50 epochs in one continuous, deterministically-seeded Adam run,
saving a periodic checkpoint every 10 epochs (task_{T}_10epoch.pt, task_{T}_20epoch.pt, ...). Because
that run is fully deterministic (fixed seed, cudnn.deterministic=True), the weights in
task_{T}_20epoch.pt are identical to what a from-scratch 20-epoch run would have produced, so
evaluating that checkpoint on the task's own test set gives the true epoch-20 test accuracy without
retraining anything. This replaces the old per-(source,target) baseline/ floor (a frozen random
backbone + trained fc) with a genuine full-network-trained-on-target-alone baseline.

Output: baselineB/results_task_{T}.txt (JSON), one per task (0-9) -- reused across all 9 sources
that target T, the same way baselineA already works.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.resnet18 import ResNet18
from models.trainer import test_func

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

CHECKPOINT_EPOCH = 20


def main():
    parser = argparse.ArgumentParser(
        description="BaselineB: evaluate train_task.py's epoch-20 checkpoint on each task's own test set")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    os.makedirs("baselineB", exist_ok=True)

    for task_id in range(10):
        ckpt_path = f"checkpoints/task_{task_id}/task_{task_id}_{CHECKPOINT_EPOCH}epoch.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        backbone = ResNet18().to(DEVICE)
        classifier = nn.Linear(backbone.latent_dims, 20).to(DEVICE)
        backbone.load_state_dict(ckpt["backbone"])
        classifier.load_state_dict(ckpt["classifier"])

        x_test, y_test = torch.load(f"saved_split_task/data/test_{task_id}.pt")
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

        acc = test_func(feature_extractor=backbone, classifier=classifier, device=DEVICE, test_loader=test_loader)
        print(f"[task {task_id}] baselineB (epoch {CHECKPOINT_EPOCH} checkpoint) test acc: {acc}")

        results = {
            "task_id": task_id,
            "epoch": CHECKPOINT_EPOCH,
            "acc": acc,
            "source_checkpoint": ckpt_path,
        }
        with open(f"baselineB/results_task_{task_id}.txt", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved baselineB/results_task_{task_id}.txt")


if __name__ == "__main__":
    main()
