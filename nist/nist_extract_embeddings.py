"""
Extract Inception-v3 and DINOv3 embeddings for the five *NIST datasets (train split).

Independent of the raw-pixel pipeline (data.py's MNIST normalisation and flattening are NOT used):
  raw 28x28 uint8 -> [0,1] -> (USPS handling) -> bicubic resize (299 Inception / 224 DINO)
  -> 3-channel replication -> network-specific normalisation -> eval-mode forward pass.

Layers: Inception = 2048-d pool3 output (before the classifier); DINOv3 = class token after the final
layer norm (the backbone has no classification head).

Output: <out-dir>/<dataset>_<network>.pt = {'features': float32 [N, D], 'labels': int64 [N] (0-based),
'meta': {...}}. Samples stay in the dataset's native order (no shuffling). EMNIST images are transposed
to fix the orientation torchvision stores them in.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

DATASETS = ['MNIST', 'FashionMNIST', 'KMNIST', 'USPS', 'EMNIST-letters']
EXPECTED = {  # (train size, number of classes)
    'MNIST': (60000, 10), 'FashionMNIST': (60000, 10), 'KMNIST': (60000, 10),
    'USPS': (7291, 10), 'EMNIST-letters': (124800, 26),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_raw(name, root):
    """Raw uint8 images [N, H, W] and 0-based int64 labels, in the dataset's native order."""
    if name == 'EMNIST-letters':
        ds = torchvision.datasets.EMNIST(root, split='letters', train=True, download=True)
    else:
        cls = {'MNIST': torchvision.datasets.MNIST, 'FashionMNIST': torchvision.datasets.FashionMNIST,
               'KMNIST': torchvision.datasets.KMNIST, 'USPS': torchvision.datasets.USPS}[name]
        ds = cls(root, train=True, download=True)
    x = torch.as_tensor(np.asarray(ds.data), dtype=torch.uint8)
    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.int64)
    if name == 'EMNIST-letters':
        x = x.transpose(1, 2)  # torchvision stores EMNIST transposed relative to MNIST
    return x.contiguous(), y - y.min()  # EMNIST-letters labels run 1..26


def save_orientation_grid(raw, out_dir, n=12):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping orientation_check.png')
        return
    fig, axes = plt.subplots(len(raw), n, squeeze=False, figsize=(n * 0.9, len(raw) * 1.1 + 0.5))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.01)
    for r, (name, (x, y)) in enumerate(raw.items()):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(x[c].numpy(), cmap='gray', vmin=0, vmax=255)
            ax.set_title(str(int(y[c])), fontsize=7)
            ax.axis('off')
        pos = axes[r, 0].get_position()
        fig.text(0.005, (pos.y0 + pos.y1) / 2, name, ha='left', va='center', fontsize=9)
    fig.suptitle('Raw inputs after orientation fix (titles = 0-based labels)', fontsize=10, y=0.985)
    path = os.path.join(out_dir, 'orientation_check.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def build_network(network, args, device):
    """Return dict(forward, size, mean, std, meta) for an eval-mode feature extractor."""
    if network == 'inception' and args.inception_weights == 'fid':
        from pytorch_fid.inception import InceptionV3
        # Earlier blocks give lower-dimensional features (64/192/768/2048-d) -- needed when the
        # per-class sample count is too small relative to 2048-d for a full-rank covariance.
        inception_dim = getattr(args, 'inception_dim', 2048)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dim]
        model = InceptionV3([block_idx], resize_input=False, normalize_input=False)
        model.eval().to(device)
        # pytorch_fid's InceptionV3 only global-average-pools internally at the final (2048-d)
        # block -- an earlier block's raw output still has spatial extent (e.g. 192-d is
        # [B, 192, 35, 35]), so .flatten(1) alone would wrongly fold the spatial grid into the
        # feature dimension instead of averaging over it. mean(dim=(2,3)) pools correctly for
        # every block, and is a no-op (so unchanged) for the already-1x1 final block.
        return dict(
            forward=lambda x: model(x)[0].mean(dim=(2, 3)), size=299, mean=(0.5,) * 3, std=(0.5,) * 3,
            meta=dict(weights='pytorch-fid FID Inception (pt_inception-2015-12-05)',
                      layer=f'block {block_idx}, {inception_dim}-d (before classifier)', normalization='[-1, 1]'))
    if network == 'inception':
        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        return dict(
            forward=lambda x: model(x), size=299, mean=IMAGENET_MEAN, std=IMAGENET_STD,
            meta=dict(weights='torchvision Inception_V3_Weights.IMAGENET1K_V1',
                      layer='avgpool 2048-d (fc replaced by Identity)', normalization='ImageNet mean/std'))

    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.dino_model)
    model.eval().to(device)
    checked = []

    def forward(x):
        out = model(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        if not checked and getattr(out, 'pooler_output', None) is not None:
            assert torch.allclose(out.pooler_output, cls, atol=1e-4), 'pooler_output differs from the CLS token'
            checked.append(True)
        return cls

    return dict(
        forward=forward, size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        meta=dict(weights=args.dino_model, layer=f'CLS token after final norm, {model.config.hidden_size}-d',
                  normalization='ImageNet mean/std'))


def resize(x, size):
    return F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).clamp_(0, 1)


def preprocess(xb, dataset, net, usps_mode, device):
    x = xb.to(device, non_blocking=True).float().div_(255).unsqueeze(1)  # [B, 1, H, W] in [0, 1]
    if dataset == 'USPS':
        if usps_mode == 'two_stage':
            x = resize(x, 28)
        elif usps_mode == 'pad20':
            x = F.pad(resize(x, 20), (4, 4, 4, 4))
    x = resize(x, net['size']).repeat(1, 3, 1, 1)
    mean = torch.tensor(net['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(net['std'], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def extract(x_u8, dataset, net, args, device):
    feats, n, start = [], len(x_u8), time.time()
    for i, s in enumerate(range(0, n, args.batch_size)):
        batch = preprocess(x_u8[s:s + args.batch_size], dataset, net, args.usps_mode, device)
        feats.append(net['forward'](batch).float().cpu())
        if i % 50 == 0:
            print(f'    {min(s + args.batch_size, n):>7d}/{n}  ({(s + args.batch_size) / (time.time() - start):.0f} img/s)')
    return torch.cat(feats)


def verify_payload(p, dataset, limit):
    X, y = p['features'], p['labels']
    n_exp, k = EXPECTED[dataset]
    problems = []
    if X.ndim != 2 or len(X) != len(y):
        problems.append(f'bad shapes {tuple(X.shape)} / {tuple(y.shape)}')
    if X.dtype != torch.float32 or y.dtype != torch.int64:
        problems.append(f'bad dtypes {X.dtype} / {y.dtype}')
    if not torch.isfinite(X).all():
        problems.append('non-finite features')
    if X.std(dim=0).max() == 0:
        problems.append('all feature dimensions are constant')
    if y.min() != 0 or y.max() >= k:
        problems.append(f'labels outside 0..{k - 1}')
    if limit is None and (len(X) != n_exp or len(torch.unique(y)) != k):
        problems.append(f'expected N={n_exp}, {k} classes; got N={len(X)}, {len(torch.unique(y))} classes')
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--network', choices=['inception', 'dino', 'both'], default='both')
    ap.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--out-dir', default='nist_embeddings')
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--dino-model', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--inception-weights', choices=['fid', 'torchvision'], default='fid')
    ap.add_argument('--usps-mode', choices=['two_stage', 'direct', 'pad20'], default='two_stage')
    ap.add_argument('--limit', type=int, default=None, help='smoke test: strided subset of N images per dataset')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(argv)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == 'cpu':
        print('WARNING: running on CPU; this will be very slow for the full datasets')

    out_dir = os.path.join(args.out_dir, f'smoke_{args.limit}') if args.limit else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    networks = ['inception', 'dino'] if args.network == 'both' else [args.network]

    print('Loading raw datasets...')
    raw = {name: load_raw(name, args.data_root) for name in args.datasets}
    save_orientation_grid(raw, out_dir)
    if args.limit:
        for name, (x, y) in raw.items():
            idx = torch.linspace(0, len(x) - 1, min(args.limit, len(x))).long()
            raw[name] = (x[idx], y[idx])

    rows, failed = [], []
    for network in networks:
        print(f'\n=== {network} ===')
        net = build_network(network, args, device)
        for name in args.datasets:
            path = os.path.join(out_dir, f'{name}_{network}.pt')
            if os.path.exists(path) and not args.overwrite:
                print(f'  {name}: {path} exists, verifying (use --overwrite to redo)')
                payload = torch.load(path)
            else:
                x, y = raw[name]
                print(f'  {name}: extracting {len(x)} images in batches of {args.batch_size}')
                start = time.time()
                feats = extract(x, name, net, args, device)
                meta = dict(
                    dataset=name, network=network, split='train', n=len(feats), dim=feats.shape[1],
                    input_size=net['size'], resize='bicubic (torch.nn.functional.interpolate)',
                    channels='grayscale replicated to 3', usps_mode=args.usps_mode if name == 'USPS' else None,
                    emnist_transposed=name == 'EMNIST-letters', batch_size=args.batch_size, limit=args.limit,
                    torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                    extracted_at=datetime.datetime.now().isoformat(timespec='seconds'), **net['meta'])
                payload = {'features': feats, 'labels': y, 'meta': meta}
                torch.save(payload, path + '.tmp')
                os.replace(path + '.tmp', path)
                print(f'  {name}: saved {path} in {time.time() - start:.1f}s')
            problems = verify_payload(payload, name, args.limit)
            X = payload['features']
            rows.append((name, network, len(X), X.shape[1], len(torch.unique(payload['labels'])),
                         X.mean().item(), X.std().item(), 'OK' if not problems else 'FAIL: ' + '; '.join(problems)))
            failed += [name] if problems else []
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f'\n{"dataset":<16}{"network":<11}{"N":>8}{"D":>6}{"classes":>9}{"mean":>10}{"std":>10}  status')
    for name, network, n, d, k, m, s, status in rows:
        print(f'{name:<16}{network:<11}{n:>8}{d:>6}{k:>9}{m:>10.4f}{s:>10.4f}  {status}')
    if failed:
        print(f'\nVerification FAILED for: {sorted(set(failed))}')
        sys.exit(1)
    print(f'\nAll {len(rows)} embedding files verified in {out_dir}')


if __name__ == '__main__':
    main()
