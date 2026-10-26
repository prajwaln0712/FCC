"""
FCC_score2: original 768-d embeddings -- Inception's block-2 (768-d, same dimension that
DINOv3-ViT-B/16 happens to use) and DINOv3-ViT-B/16 (768-d, the same model augment.py already
defaults to) -- kept at the original dimension by padding each class's real ~500 images up to
AUGMENT_TARGET_PER_CLASS samples via random flip/crop/color-jitter augmentation before embedding,
instead of switching to a lower-dimensional model/layer (that's compute_fcc_score1.py). See
fcc_common.py for why the covariance would otherwise be rank-deficient, and for the augmentation
recipe.

Output: tables/fcc_score2.csv
  source, target, fcc_inception768_mean, fcc_inception768_ratio, fcc_dinovitb16_mean, fcc_dinovitb16_ratio
"""
import argparse
import csv
import os

import torch

from nist.nist_extract_embeddings import build_network
from tinyimagenet.fcc_common import NUM_TASKS, fcc_for_pairs, load_or_compute_task_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

AUGMENT_TARGET_PER_CLASS = 800  # comfortably above the 769-sample minimum for a 768-d covariance

NETWORKS = {
    'inception768': dict(network='inception', inception_weights='fid', inception_dim=768),
    'dinovitb16': dict(network='dino', dino_model='facebook/dinov3-vitb16-pretrain-lvd1689m'),
}


def main():
    parser = argparse.ArgumentParser(description="FCC_score2: original 768-d embeddings, padded via augmentation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--augment-to", type=int, default=AUGMENT_TARGET_PER_CLASS)
    args = parser.parse_args()

    os.makedirs("tables", exist_ok=True)
    per_network_rows = {}
    for key, net_args in NETWORKS.items():
        print(f"Computing {key} task stats (padded to {args.augment_to} samples/class)...")
        ns = argparse.Namespace(**net_args)
        net = build_network(net_args['network'], ns, DEVICE)
        task_stats = {
            t: load_or_compute_task_stats(t, net, cache_dir=f"fcc_cache/score2/{key}",
                                          augment_to=args.augment_to, seed=args.seed)
            for t in range(NUM_TASKS)
        }
        per_network_rows[key] = {(s, t): (m, r) for s, t, m, r in fcc_for_pairs(task_stats)}
        del net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    out_path = "tables/fcc_score2.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "fcc_inception768_mean", "fcc_inception768_ratio",
                         "fcc_dinovitb16_mean", "fcc_dinovitb16_ratio"])
        for source in range(NUM_TASKS):
            for target in range(NUM_TASKS):
                if source == target:
                    continue
                incep_mean, incep_ratio = per_network_rows['inception768'][(source, target)]
                dino_mean, dino_ratio = per_network_rows['dinovitb16'][(source, target)]
                writer.writerow([source, target, incep_mean, incep_ratio, dino_mean, dino_ratio])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
