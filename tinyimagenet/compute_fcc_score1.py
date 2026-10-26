"""
FCC_score1: lower-dimension embeddings, no augmentation -- Inception at the 192-d block instead of
the usual 2048-d pool3, and DINOv3-ViT-S/16 (384-d) instead of the usual ViT-B/16 (768-d). Both
chosen so the real 500 images/class already comfortably exceed the embedding dimension (500 > 193
for 192-d, 500 > 385 for 384-d), avoiding the rank-deficient per-class covariance that the original
2048-d/768-d choices hit (500 samples/class < either of those dimensions). See fcc_common.py.

Computes each of the 10 tasks' embedding stats once per network (pooled -> used when the task is a
source, per-class -> used when the task is a target), cached to fcc_cache/score1/<network>/, then
the 90 source != target pairs are closed-form Frechet-distance lookups against those cached stats.

Output: tables/fcc_score1.csv
  source, target, fcc_inception192_mean, fcc_inception192_ratio, fcc_dinovits16_mean, fcc_dinovits16_ratio
"""
import argparse
import csv
import os

import torch

from nist.nist_extract_embeddings import build_network
from tinyimagenet.fcc_common import NUM_TASKS, fcc_for_pairs, load_or_compute_task_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

NETWORKS = {
    'inception192': dict(network='inception', inception_weights='fid', inception_dim=192),
    'dinovits16': dict(network='dino', dino_model='facebook/dinov3-vits16-pretrain-lvd1689m'),
}


def main():
    parser = argparse.ArgumentParser(description="FCC_score1: lower-dimension embeddings, no augmentation")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    os.makedirs("tables", exist_ok=True)
    per_network_rows = {}
    for key, net_args in NETWORKS.items():
        print(f"Computing {key} task stats...")
        ns = argparse.Namespace(**net_args)
        net = build_network(net_args['network'], ns, DEVICE)
        task_stats = {
            t: load_or_compute_task_stats(t, net, cache_dir=f"fcc_cache/score1/{key}",
                                          augment_to=None, seed=args.seed)
            for t in range(NUM_TASKS)
        }
        per_network_rows[key] = {(s, t): (m, r) for s, t, m, r in fcc_for_pairs(task_stats)}
        del net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    out_path = "tables/fcc_score1.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "fcc_inception192_mean", "fcc_inception192_ratio",
                         "fcc_dinovits16_mean", "fcc_dinovits16_ratio"])
        for source in range(NUM_TASKS):
            for target in range(NUM_TASKS):
                if source == target:
                    continue
                incep_mean, incep_ratio = per_network_rows['inception192'][(source, target)]
                dino_mean, dino_ratio = per_network_rows['dinovits16'][(source, target)]
                writer.writerow([source, target, incep_mean, incep_ratio, dino_mean, dino_ratio])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
