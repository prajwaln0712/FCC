"""
Reconstruction of the original trainer.py, which was never committed/published.
Reconstructed from its call sites in dataaugmentation/augment.py (transfer_learning()).

Design decisions baked into this reconstruction:
  - train() puts both feature_extractor and classifier into .train() mode at the start of every
    call, because test_func() leaves both modules in .eval() mode and train() is always called
    again afterward (e.g. CIFAR-10 head-only adaptation runs right after the ImageNet no-adapt
    test_func() call). One consequence: during head-only adaptation the frozen backbone's
    BatchNorm running statistics keep updating on the target data even though its weights are
    frozen. This is intentional.
  - When ft_extractor_optimizer is None, the backbone forward pass runs inside torch.no_grad()
    (no backbone gradients are computed). BatchNorm running stats still update because buffer
    updates happen during the forward pass itself, independent of autograd -- only gradient
    tracking is disabled.
  - Mixed precision uses a single module-level GradScaler, created lazily and reused across every
    train() call (a fresh scaler per call defeats its own scale-factor estimation).
  - test_func() returns accuracy as a percentage (0-100), not a 0-1 fraction.
"""
import torch
import torch.nn as nn

_SCALER = None


def _get_scaler():
    global _SCALER
    if _SCALER is None:
        try:
            _SCALER = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            _SCALER = torch.cuda.amp.GradScaler()
    return _SCALER


def _autocast(device, use_amp):
    if use_amp and device.type == 'cuda':
        try:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        except AttributeError:
            return torch.cuda.amp.autocast(dtype=torch.float16)
    return torch.autocast(device_type='cpu', enabled=False)


def train(feature_extractor, classifier, device, train_loader, epoch=None,
          criterion=None, ft_extractor_optimizer=None, classifier_optimizer=None,
          use_amp=True):
    feature_extractor.train()
    classifier.train()

    amp_enabled = use_amp and device.type == 'cuda'
    scaler = _get_scaler() if amp_enabled else None

    total_loss = 0.0
    total_samples = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        for opt in (ft_extractor_optimizer, classifier_optimizer):
            if opt is not None:
                opt.zero_grad(set_to_none=True)

        with _autocast(device, use_amp):
            if ft_extractor_optimizer is not None:
                features = feature_extractor(x)
            else:
                with torch.no_grad():
                    features = feature_extractor(x)
            logits = classifier(features)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            for opt in (ft_extractor_optimizer, classifier_optimizer):
                if opt is not None:
                    scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            for opt in (ft_extractor_optimizer, classifier_optimizer):
                if opt is not None:
                    opt.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def test_func(feature_extractor, classifier, device, test_loader, use_amp=True):
    feature_extractor.eval()
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            with _autocast(device, use_amp):
                features = feature_extractor(x)
                logits = classifier(features)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def frozen_module(module):
    for p in module.parameters():
        p.requires_grad = False
    return module
