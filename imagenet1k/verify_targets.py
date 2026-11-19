"""
Per-class image-count audit for all 7 imagenet1k target datasets (resolves the pending
EuroSAT/Places365-subset per-class-count verification item from the design-choices log --
N=2000 was assumed from general recollection, not verified against real file counts; this
checks all 7 targets, not just those two, since the same pattern covers all of them for free).

Modeled on nist/verify_mnist_datasets.py's shape/class-count check pattern.
"""
import argparse
import sys

import numpy as np

from imagenet1k.data_targets import TARGETS, load_target


def audit(name, root):
    info = load_target(name, variant='01', root=root)
    y_train = info['train_targets']
    counts = np.bincount(y_train, minlength=info['num_classes'])
    n_test = len(info['test'])

    problems = []
    if len(counts) != info['num_classes']:
        problems.append(f'bincount length {len(counts)} != num_classes {info["num_classes"]}')
    if counts.min() == 0:
        problems.append(f'at least one class has zero training images (counts={counts.tolist()})')

    status = 'OK' if not problems else 'FAIL: ' + '; '.join(problems)
    print(f'{name:14s} classes={info["num_classes"]:3d}  n_train={len(y_train):7d}  '
          f'n_test={n_test:6d}  min/class={counts.min():5d}  max/class={counts.max():5d}  '
          f'median/class={int(np.median(counts)):5d}  {status}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS)
    ap.add_argument('--data-root', default='./data')
    args = ap.parse_args()

    failed = []
    for name in args.targets:
        try:
            problems = audit(name, args.data_root)
            if problems:
                failed.append(name)
        except Exception as e:
            print(f'{name:14s} FAIL: {e}')
            failed.append(name)

    if failed:
        print(f'\n{len(failed)} target(s) failed verification: {failed}')
        sys.exit(1)
    print(f'\nAll {len(args.targets)} targets downloaded and verified successfully.')


if __name__ == '__main__':
    main()
