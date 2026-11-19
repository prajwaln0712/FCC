"""
Target dataset loaders for the ImageNet-1K FCC experiment (7 targets, final list per
Design_choices_imagenet1k.txt item 1): CIFAR-10, Places365-subset, EuroSAT, SVHN, MNIST,
Fashion-MNIST, KMNIST.

Every loader returns a uniform dict: {'train_full': Dataset, 'test': Dataset, 'num_classes':
int, 'class_names': [...], 'train_targets': np.ndarray of int labels for train_full, same order}.
`train_targets` is returned alongside `train_full` (rather than requiring callers to reach into
the dataset's own `.targets`/`.labels` attribute, whose name differs per dataset) so every
downstream stage can do per-class indexing uniformly.

Two transform variants, chosen via `variant`:
  'norm' -- Resize(224) [+ grayscale replication for MNIST-family] -> ToTensor -> Normalize
            (ImageNet mean/std). Used by both training arms (imagenet1k/imagenet1k_test.py).
  '01'   -- same resize, no Normalize ([0, 1] range). Used by the FCC stage
            (imagenet1k/target_fcc_stats.py) and by tinyimagenet/fcc_common.py's augmented_copy,
            which expects [0, 1] tensors.
"""
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset

import data as data_module

DATA_ROOT = './data'
SPLITS_DIR = 'imagenet1k/splits'

TARGETS = ['cifar10', 'places365', 'eurosat', 'svhn', 'mnist', 'fashionmnist', 'kmnist']

EUROSAT_SPLIT_SEED = 20_260_001

IMAGENET_MEAN, IMAGENET_STD = data_module.DATASET_NORMALIZATION['ImageNet']

# Keywords used to pick a 15-class Places365 subset out of its 365 categories, matched against
# Places365's raw category strings (e.g. '/a/airport_terminal', or '/s/stadium/football' for
# categories with a sport/type variant) via select_places365_subset() below -- see
# Design_choices_imagenet1k.txt item 1 / the implementation plan's Places365 note.
#
# CONFIRMED (2026-11-07, real verify_targets run on RC): plain substring matching was too loose
# -- 'beach' matched both '/b/beach' and '/b/beach_house'. 'stadium/football' is a confident fix
# (Places365 nests sport variants under stadium/baseball, stadium/football, stadium/soccer --
# a bare 'stadium' keyword would be ambiguous across all three, so it's written with the variant
# included directly). 'desert_sand' is changed to 'desert/sand' on the same reasoning but is a
# less certain guess at the real nesting -- if it still fails to match, that's the one to report
# back with the real category string.
PLACES365_KEYWORDS = [
    'airport_terminal', 'beach', 'bedroom', 'bridge', 'campus',
    'desert/sand', 'forest_path', 'harbor', 'highway', 'kitchen',
    'mountain', 'office', 'restaurant', 'stadium/football', 'waterfall',
]


def stratified_split(targets, fractions, seed):
    """Per-class index split preserving class balance. `targets`: array-like of int labels.
    `fractions`: list of floats summing to 1 (e.g. [0.8, 0.2]). Returns a list of np.ndarray of
    indices into `targets`, one array per fraction, in the same order as `fractions`. Used both
    for EuroSAT's 80/20 train/test carve (no native split exists) and for every target's 80/20
    train/validation split in the adaptive-stopping training loop."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    cum_fractions = np.cumsum(fractions)
    splits = [[] for _ in fractions]
    for c in np.unique(targets):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        boundaries = [0] + [int(round(f * n)) for f in cum_fractions]
        boundaries[-1] = n  # absorb any rounding slack into the last split
        for i in range(len(fractions)):
            splits[i].append(idx_c[boundaries[i]:boundaries[i + 1]])
    return [np.concatenate(s) if s else np.array([], dtype=int) for s in splits]


class RelabeledSubset(Dataset):
    """Wraps base[indices], remapping each sample's original label via label_map. Used for the
    Places365 subset (filter to 15 classes, remap their original indices to 0..14)."""

    def __init__(self, base, indices, label_map):
        self.base = base
        self.indices = np.asarray(indices)
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[int(self.indices[i])]
        return x, self.label_map[y]


def select_places365_subset(classes, keywords):
    """Match each keyword to exactly one class name in `classes`.

    Tries an exact match on the category's final '/'-separated segment first (e.g. keyword
    'beach' against '/b/beach' -> segment 'beach', a match; against '/b/beach_house' -> segment
    'beach_house', not a match) -- this is what actually distinguishes a category from its
    same-prefix variants (beach vs beach_house, office vs office_building, restaurant vs
    restaurant_patio, etc.), confirmed necessary against a real verify_targets run on RC
    (2026-11-07), where plain substring matching let 'beach' match both '/b/beach' and
    '/b/beach_house'. Falls back to substring containment only if no exact segment match exists,
    for keywords written with their own '/' already included (e.g. 'stadium/football' against
    '/s/stadium/football'). Either way, raises a clear, actionable error if a keyword doesn't
    resolve to exactly one class, so a bad guess fails loudly instead of silently selecting the
    wrong category."""
    selected = []
    for kw in keywords:
        exact = [i for i, name in enumerate(classes) if name.rsplit('/', 1)[-1] == kw]
        matches = exact if exact else [i for i, name in enumerate(classes) if kw in name]
        if len(matches) != 1:
            raise ValueError(
                f"Places365 keyword '{kw}' matched {len(matches)} classes (expected exactly 1): "
                f"{[classes[i] for i in matches]}. Inspect Places365's .classes list on disk and "
                f"fix PLACES365_KEYWORDS in imagenet1k/data_targets.py -- this was a known pending "
                f"item when this file was written, not a bug.")
        selected.append(matches[0])
    return selected


def _build_transform(variant, grayscale_to_rgb=False):
    assert variant in ('norm', '01'), f'unknown variant {variant!r}'
    ops = []
    if grayscale_to_rgb:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.Resize((224, 224)))
    ops.append(T.ToTensor())
    if variant == 'norm':
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(ops)


def _cifar10(variant, root):
    tfm = _build_transform(variant)
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    return dict(train_full=train, test=test, num_classes=10, class_names=train.classes,
                train_targets=np.asarray(train.targets))


def _svhn(variant, root):
    tfm = _build_transform(variant)
    train = torchvision.datasets.SVHN(root, split='train', download=True, transform=tfm)
    test = torchvision.datasets.SVHN(root, split='test', download=True, transform=tfm)
    return dict(train_full=train, test=test, num_classes=10,
                class_names=[str(i) for i in range(10)], train_targets=np.asarray(train.labels))


def _mnist_family(cls, variant, root):
    tfm = _build_transform(variant, grayscale_to_rgb=True)
    train = cls(root, train=True, download=True, transform=tfm)
    test = cls(root, train=False, download=True, transform=tfm)
    return dict(train_full=train, test=test, num_classes=10,
                class_names=[str(i) for i in range(10)],
                train_targets=np.asarray(train.targets))


def _mnist(variant, root):
    return _mnist_family(torchvision.datasets.MNIST, variant, root)


def _fashionmnist(variant, root):
    return _mnist_family(torchvision.datasets.FashionMNIST, variant, root)


def _kmnist(variant, root):
    return _mnist_family(torchvision.datasets.KMNIST, variant, root)


def _eurosat(variant, root):
    """EuroSAT has no native train/test split (a single ~27,000-image folder of 10 classes) --
    carve a seeded, deterministic, per-class 80/20 split and persist the chosen indices so
    re-runs are exactly reproducible without recomputation."""
    tfm = _build_transform(variant)
    full = torchvision.datasets.EuroSAT(root, download=True, transform=tfm)
    targets = np.asarray(full.targets)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    split_path = f'{SPLITS_DIR}/eurosat_splits.json'
    if os.path.exists(split_path):
        with open(split_path) as f:
            saved = json.load(f)
        train_idx, test_idx = np.array(saved['train_idx']), np.array(saved['test_idx'])
    else:
        train_idx, test_idx = stratified_split(targets, [0.8, 0.2], seed=EUROSAT_SPLIT_SEED)
        with open(split_path, 'w') as f:
            json.dump({'train_idx': train_idx.tolist(), 'test_idx': test_idx.tolist()}, f)

    return dict(train_full=Subset(full, train_idx), test=Subset(full, test_idx), num_classes=10,
                class_names=full.classes, train_targets=targets[train_idx])


def _places365(variant, root):
    """10-20 class subset (15 chosen, see PLACES365_KEYWORDS) of Places365. 'train-standard' for
    training; 'val' for the held-out test set, since Places365's true test-split labels are never
    publicly released (this is the standard substitute, 100 images/class)."""
    tfm = _build_transform(variant)
    train_full_ds = torchvision.datasets.Places365(root, split='train-standard', small=True,
                                                     download=True, transform=tfm)
    test_full_ds = torchvision.datasets.Places365(root, split='val', small=True,
                                                    download=True, transform=tfm)
    all_classes = train_full_ds.classes
    subset_orig_idx = select_places365_subset(all_classes, PLACES365_KEYWORDS)
    remap = {orig: new for new, orig in enumerate(subset_orig_idx)}
    class_names = [all_classes[i] for i in subset_orig_idx]

    def _filtered_indices(ds):
        targets = np.asarray(ds.targets)
        mask = np.isin(targets, subset_orig_idx)
        idx = np.where(mask)[0]
        return idx, np.array([remap[t] for t in targets[idx]])

    train_idx, train_targets = _filtered_indices(train_full_ds)
    test_idx, _ = _filtered_indices(test_full_ds)

    return dict(train_full=RelabeledSubset(train_full_ds, train_idx, remap),
                test=RelabeledSubset(test_full_ds, test_idx, remap),
                num_classes=len(subset_orig_idx), class_names=class_names,
                train_targets=train_targets)


_LOADERS = {
    'cifar10': _cifar10,
    'places365': _places365,
    'eurosat': _eurosat,
    'svhn': _svhn,
    'mnist': _mnist,
    'fashionmnist': _fashionmnist,
    'kmnist': _kmnist,
}


def load_target(name, variant='norm', root=DATA_ROOT):
    assert name in TARGETS, f'unknown target {name!r}, expected one of {TARGETS}'
    return _LOADERS[name](variant, root)
