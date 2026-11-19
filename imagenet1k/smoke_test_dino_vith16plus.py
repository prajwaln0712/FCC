"""
End-to-end smoke test for DINOv3-ViT-H+ (facebook/dinov3-vith16plus-pretrain-lvd1689m, 1280-d).
Only its config has been verified so far (via transformers.AutoConfig, confirmed hidden_size=
1280) -- this actually loads the pretrained weights and runs a forward pass on a handful of real
images, to confirm the 1280-d output shape before relying on it for the full
imagenet_source_stats.py / target_fcc_stats.py runs. Resolves the pending DINOv3-ViT-H+
verification item in Design_choices_imagenet1k.txt item 3.
"""
import argparse

import torch

from imagenet1k.data_targets import load_target
from nist.nist_extract_embeddings import build_network
from tinyimagenet.fcc_common import embed_01_images

EXPECTED_DIM = 1280


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dino-model', default='facebook/dinov3-vith16plus-pretrain-lvd1689m')
    ap.add_argument('--n-images', type=int, default=8)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'Loading {args.dino_model} on {device}...')
    net = build_network('dino', argparse.Namespace(dino_model=args.dino_model), device)
    print(f"Loaded. meta={net['meta']}")

    print(f'Loading {args.n_images} real images (CIFAR-10 train) to embed...')
    info = load_target('cifar10', variant='01', root=args.data_root)
    xs = torch.stack([info['train_full'][i][0] for i in range(args.n_images)])

    feats = embed_01_images(xs, net, batch_size=args.n_images, device=device)
    print(f'Output shape: {tuple(feats.shape)}')
    assert feats.shape == (args.n_images, EXPECTED_DIM), (
        f'expected ({args.n_images}, {EXPECTED_DIM}), got {tuple(feats.shape)}')
    assert torch.isfinite(feats).all(), 'non-finite values in embeddings'
    print(f'[OK] DINOv3-ViT-H+ smoke test passed: {args.n_images}x{EXPECTED_DIM}, finite, '
          f'mean={feats.mean().item():.4f} std={feats.std().item():.4f}')


if __name__ == '__main__':
    main()
