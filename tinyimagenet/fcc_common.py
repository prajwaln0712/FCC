"""
Shared helpers for computing FCC(source task -> target task) across the 10 TinyImageNet task
splits, used by compute_fcc_score1.py (lower-dimension embeddings, no augmentation) and
compute_fcc_score2.py (original 768-d embeddings, padded via augmentation to stay full-rank).

Both scores need the same two things per task, per network: a POOLED (mean, cov) over all 20
classes (used when the task acts as a source) and a PER-CLASS (mean, cov) x20 (used when the task
acts as a target). Computing these once per task (cached to disk) and then only doing closed-form
Frechet-distance lookups for the 90 (source, target) pairs avoids 90x redundant embedding passes.

Why a covariance problem exists at all: each task has 500 training images per class. Inception's
usual 2048-d pool3 feature and DINOv3-ViT-B/16's 768-d CLS token both exceed that per-class sample
count, so the per-class sample covariance is rank-deficient (rank <= 499) in the usual setup --
score1 avoids this by reading out a lower-dimensional layer/model (192-d Inception block, 384-d
DINOv3-ViT-S/16), score2 avoids it by padding each class up to ~800 samples via augmentation
instead, keeping the original 768-d dimensions.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import data
import util
from nist.nist_extract_embeddings import resize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TASKS = 10
CLASSES_PER_TASK = 20

_IMAGENET_MEAN, _IMAGENET_STD = data.DATASET_NORMALIZATION['ImageNet']

_FLIP = T.RandomHorizontalFlip(p=0.5)
_COLOR_JITTER = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

FCC_RIDGE = 1e-6
FCC_MAX_IMAG = 1e-3


def denormalize_imagenet(x):
    """Undo the ImageNet Normalize baked into tiny_split.py's saved tensors -> [0, 1]."""
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def embed_01_images(x_01, net, batch_size=64, device=DEVICE):
    """x_01: [N, 3, H, W] in [0, 1]. Returns [N, D] double-precision embeddings (CPU)."""
    feats = []
    mean = torch.tensor(net['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(net['std'], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for i in range(0, len(x_01), batch_size):
            x = x_01[i:i + batch_size].to(device)
            x = resize(x, net['size'])
            x = (x - mean) / std
            feats.append(net['forward'](x).float().cpu())
    return torch.cat(feats).double()


def embed_norm_images(x_norm, net, batch_size=64, device=DEVICE):
    """x_norm: [N, 3, H, W], ImageNet-normalized (as tiny_split.py saves it). Denormalizes
    per-batch instead of up front, so the full ~10,000-image task is never duplicated in memory
    (denormalize_imagenet(x) on the WHOLE task tensor at once is what caused the OOM -- for one
    task that's ~5.6 GiB for the loaded tensor plus another ~5.6 GiB for its denormalized copy,
    before anything else even runs)."""
    feats = []
    mean = torch.tensor(net['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(net['std'], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for i in range(0, len(x_norm), batch_size):
            x = denormalize_imagenet(x_norm[i:i + batch_size]).to(device)
            x = resize(x, net['size'])
            x = (x - mean) / std
            feats.append(net['forward'](x).float().cpu())
    return torch.cat(feats).double()


def augmented_copy(x_01, rng):
    """One randomly-augmented copy of a batch of [0, 1] images: flip + random-resized-crop + color
    jitter, same random draw applied to the whole batch (fresh draw per call -- callers make
    repeated calls to build up independent-ish padding copies for a class)."""
    n, c, h, w = x_01.shape
    out = _FLIP(x_01)

    scale = rng.uniform(0.7, 1.0)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    top = int(rng.integers(0, h - new_h + 1))
    left = int(rng.integers(0, w - new_w + 1))
    out = out[:, :, top:top + new_h, left:left + new_w]
    out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

    out = _COLOR_JITTER(out)
    return out.clamp(0, 1)


def compute_task_stats(task_id, net, augment_to=None, seed=1):
    """Returns ((pooled_mean, pooled_cov), [(mean, cov) for each of the 20 classes]).

    pooled stats always come from the real (non-augmented) 10,000-image trainset -- never
    rank-deficient regardless of score1/score2, since 10,000 >> any embedding dim used here.
    per-class stats come from the real ~500 per-class images alone when augment_to is None
    (score1's lower-dimension embeddings); when augment_to is set (score2), each class's real
    images are padded with random-augmented copies up to augment_to samples before embedding.
    """
    rng = np.random.default_rng(seed + task_id)
    torch.manual_seed(seed + task_id)

    x, y = torch.load(f"saved_split_task/data/trainset_{task_id}.pt")  # ImageNet-normalized

    pooled_mean, pooled_cov = util.calculate_mean_covariance(embed_norm_images(x, net))

    per_class = []
    for c in range(CLASSES_PER_TASK):
        # Only this class's ~500 images (or up to augment_to) get denormalized at once -- far
        # smaller than the full task, so no repeat of the pooled-stats OOM above.
        x_c = denormalize_imagenet(x[y == c])
        if augment_to is not None and len(x_c) < augment_to:
            copies = [x_c]
            total = len(x_c)
            while total < augment_to:
                copy = augmented_copy(x_c, rng)
                copies.append(copy)
                total += len(copy)
            x_c = torch.cat(copies)[:augment_to]
        feats_c = embed_01_images(x_c, net)
        per_class.append(util.calculate_mean_covariance(feats_c))
        del x_c

    del x
    return (pooled_mean, pooled_cov), per_class


def load_or_compute_task_stats(task_id, net, cache_dir, augment_to=None, seed=1):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/task_{task_id}.pt"
    if os.path.exists(cache_path):
        # weights_only=False: the cache holds numpy mean/cov arrays (from
        # util.calculate_mean_covariance), not just tensors -- these are our own files, so the
        # usual untrusted-pickle concern with weights_only=False doesn't apply.
        cached = torch.load(cache_path, weights_only=False)
        return cached['pooled'], cached['per_class']
    pooled, per_class = compute_task_stats(task_id, net, augment_to=augment_to, seed=seed)
    torch.save({'pooled': pooled, 'per_class': per_class}, cache_path)
    return pooled, per_class


def fcc_for_pairs(task_stats):
    """task_stats: dict[task_id] -> (pooled, per_class). Returns [(source, target, mean, ratio)]
    for all 90 source != target pairs -- source's pooled stats vs. target's per-class stats,
    aggregated exactly like augment.py's FCC block (mean and mean/max across target classes)."""
    rows = []
    tasks = sorted(task_stats.keys())
    for source in tasks:
        src_mean, src_cov = task_stats[source][0]
        for target in tasks:
            if source == target:
                continue
            tgt_per_class = task_stats[target][1]
            per_class_fcc = np.array([
                util.calculate_fid(src_mean, src_cov, mu_c, sigma_c, ridge=FCC_RIDGE, max_imag=FCC_MAX_IMAG)
                for mu_c, sigma_c in tgt_per_class
            ])
            rows.append((source, target, float(per_class_fcc.mean()),
                        float(per_class_fcc.mean() / per_class_fcc.max())))
    return rows
