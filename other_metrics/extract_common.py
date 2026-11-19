"""
Shared helpers for the LogME / LEEP extraction scripts (other_metrics/aug_scores.py,
other_metrics/imagenet1k_scores.py).

LogME needs per-image features from the SOURCE model on the TARGET training images plus target
labels; LEEP needs the SOURCE model's softmax over its own source classes on the same images plus
target labels. Both come out of one forward pass, so extract() returns (features, logits, labels)
together and score_logme_leep() computes both scores from them.

Conventions (decided with the user, 2026-11-14):
  * source model in plain eval mode with the saved/pretrained source batch-norm statistics
  * fp32 forward pass (training used fp16 autocast on GPU; extraction deliberately does not)
  * scored on the target's full native TRAIN set (the gain being correlated against is measured
    on the test set), no subsampling
  * only scores are cached (CSV), never the features
The LogME/LEEP implementations are the unmodified files in LogME/ (LogME.py needs numba).
"""
import csv
import os
import time

import numpy as np
import torch


def set_strict_fp32():
    """Disable TF32 so the fp32 forward pass really is fp32 on Ampere+ GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def softmax64(logits):
    """Row-wise softmax in float64. LEEP needs probabilities (not logits), and float64 avoids the
    exact-zero columns that make LEEP's P(y|z) division return NaN."""
    z = np.asarray(logits, dtype=np.float64).copy()
    z -= z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def limit_dataset(dataset, limit):
    """Evenly spaced subset of `limit` items (smoke tests only; keeps all classes in play)."""
    if limit is None or limit >= len(dataset):
        return dataset
    idx = torch.linspace(0, len(dataset) - 1, limit).long().tolist()
    return torch.utils.data.Subset(dataset, idx)


@torch.no_grad()
def extract(model_fn, loader, device):
    """Run model_fn(x) -> (features [B, D], logits [B, C_s]) over the loader (which must not
    shuffle). Returns float32 numpy features, float32 numpy logits, int64 numpy labels."""
    feats, logits, labels = [], [], []
    for x, y in loader:
        f, lg = model_fn(x.to(device, non_blocking=True))
        feats.append(f.float().cpu())
        logits.append(lg.float().cpu())
        labels.append(y.cpu())
    return (torch.cat(feats).numpy(), torch.cat(logits).numpy(),
            torch.cat(labels).numpy().astype(np.int64))


def score_logme_leep(features, logits, labels):
    """Return {'logme': float, 'leep': float}. Raises if either score is not finite."""
    from LogME.LogME import LogME  # needs numba
    from LogME.LEEP import LEEP

    labels = np.asarray(labels).astype(np.int64)
    present = np.unique(labels)
    if labels.min() != 0 or len(present) != labels.max() + 1:
        print(f'  WARNING: target labels are not exactly 0..C-1 (present: {present.tolist()[:20]}...); '
              f'LogME/LEEP treat max(label)+1 as the class count, so a missing class distorts scores')

    logme = float(LogME(regression=False).fit(features, labels))
    leep = float(LEEP(softmax64(logits), labels))
    if not (np.isfinite(logme) and np.isfinite(leep)):
        raise ValueError(f'non-finite score: logme={logme}, leep={leep}')
    return {'logme': logme, 'leep': leep}


def timed(fn, *args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - start


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def upsert_row(path, fieldnames, key, row):
    """Insert `row` into the CSV at `path`, replacing any existing row with the same `key` value.
    Rewrites the file atomically so a killed job never leaves a half-written CSV."""
    rows = [r for r in read_rows(path) if r[key] != str(row[key])]
    rows.append({k: row[k] for k in fieldnames})
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path + '.tmp', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path + '.tmp', path)
