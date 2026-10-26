"""Ad-hoc smoke test for models/trainer.py -- not part of the repo, do not commit.

Run from the repo root (matches every other RC job script's `cd ~/fcc` convention):
    python smoke_test_trainer.py
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.resnet import ResNet18
from models.trainer import train, test_func, frozen_module


def run(use_amp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} use_amp={use_amp}")

    X = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    backbone = ResNet18().to(device)
    head = nn.Linear(backbone.latent_dims, 10).to(device)
    backbone_opt = torch.optim.SGD(backbone.parameters(), lr=0.01)
    head_opt = torch.optim.SGD(head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    before_backbone = {k: v.clone() for k, v in backbone.state_dict().items()}
    before_head = {k: v.clone() for k, v in head.state_dict().items()}

    train(feature_extractor=backbone, classifier=head, device=device, train_loader=loader,
          criterion=criterion, ft_extractor_optimizer=backbone_opt, classifier_optimizer=head_opt,
          use_amp=use_amp)

    backbone_changed = any(not torch.equal(v, backbone.state_dict()[k]) for k, v in before_backbone.items())
    head_changed = any(not torch.equal(v, head.state_dict()[k]) for k, v in before_head.items())
    print(f"[1] backbone changed={backbone_changed} head changed={head_changed} (expect True, True)")

    frozen_module(backbone)
    before_backbone_params = {k: v.clone() for k, v in backbone.named_parameters()}
    before_bn_buffers = {k: v.clone() for k, v in backbone.state_dict().items() if "running_mean" in k}
    bn_key = next(iter(before_bn_buffers))

    train(feature_extractor=backbone, classifier=head, device=device, train_loader=loader,
          criterion=criterion, ft_extractor_optimizer=None, classifier_optimizer=head_opt,
          use_amp=use_amp)

    backbone_identical = all(
        torch.equal(v, dict(backbone.named_parameters())[k]) for k, v in before_backbone_params.items())
    bn_changed = not torch.equal(before_bn_buffers[bn_key], backbone.state_dict()[bn_key])
    print(f"[2] backbone bit-identical={backbone_identical} BN running_mean changed={bn_changed} (expect True, True)")

    loss_no_epoch = train(feature_extractor=backbone, classifier=head, device=device, train_loader=loader,
                           criterion=criterion, ft_extractor_optimizer=None, classifier_optimizer=head_opt,
                           use_amp=use_amp)
    print(f"[3] train() without epoch kwarg ok, loss={loss_no_epoch}")

    acc = test_func(backbone, head, device, loader, use_amp)
    print(f"[4] test_func positional args ok, accuracy={acc} in [0,100]={0 <= acc <= 100}")


if __name__ == "__main__":
    run(use_amp=True)
    run(use_amp=False)
