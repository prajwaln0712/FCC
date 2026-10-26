import sys
import torch
import data

EXPECTED_DIM = 784
EXPECTED_CLASSES = {
    'MNIST': 10,
    'KMNIST': 10,
    'FashionMNIST': 10,
    'USPS': 10,
    'EMNIST-letters': 26,
}


def verify(name, ds):
    x, y = ds[0]
    assert x.shape == (EXPECTED_DIM,), f'{name}: expected shape ({EXPECTED_DIM},), got {tuple(x.shape)}'
    assert x.dtype == torch.float32, f'{name}: expected float32, got {x.dtype}'

    X, Y = data.dataset_to_tensors(ds)
    assert X.shape == (len(ds), EXPECTED_DIM), f'{name}: bad stacked shape {tuple(X.shape)}'
    assert torch.isfinite(X).all(), f'{name}: non-finite values found'

    labels = torch.unique(Y)
    n_classes = len(labels)
    expected = EXPECTED_CLASSES[name]
    assert n_classes == expected, f'{name}: expected {expected} classes, got {n_classes} ({labels.tolist()})'

    print(f'[OK] {name:15s} n={len(ds):6d} dim={EXPECTED_DIM} classes={n_classes} '
          f'labels=[{labels.min().item()}..{labels.max().item()}] '
          f'mean={X.mean():.4f} std={X.std():.4f}')


def main():
    datasets = data.load_mnist_family_datasets(train=True, download=True)
    failed = []
    for name, ds in datasets.items():
        try:
            verify(name, ds)
        except Exception as e:
            print(f'[FAIL] {name}: {e}')
            failed.append(name)

    if failed:
        print(f'\n{len(failed)} dataset(s) failed verification: {failed}')
        sys.exit(1)
    print(f'\nAll {len(datasets)} datasets downloaded and verified successfully.')


if __name__ == '__main__':
    main()
