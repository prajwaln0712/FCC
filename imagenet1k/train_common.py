"""
Shared training helpers for imagenet1k/imagenet1k_test.py: the seeding convention and the adaptive
early-stopping training loop used by both arms (frozen-backbone transfer, full-from-scratch).

Stopping-rule parameters, confirmed in Design_choices_imagenet1k.txt item 7: seeded 80/20
train/validation split (carved by the caller via data_targets.stratified_split -- test sets are
never touched by any stopping decision), stop once 5 consecutive epochs fail to improve
validation accuracy by more than 0.4 percentage points, capped at 30 epochs total, reload the
best-validation-epoch checkpoint before the single final test-set evaluation.
"""
import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.trainer import train, test_func

SPLIT_BASE_SEED = 30_000_000
RUN_BASE_SEED = 31_000_000

VAL_FRACTION = 0.2
IMPROVE_THRESHOLD = 0.4  # percentage points
PATIENCE = 5
MAX_EPOCHS = 30
BATCH_SIZE = 64


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_seed_for(target_idx):
    """Fixed per target, shared by both arms -- so frozen-transfer and full-from-scratch see the
    identical 80/20 validation split for a fair Gain comparison."""
    return SPLIT_BASE_SEED + target_idx


def run_seed_for(target_idx, arm_idx):
    """Varies per arm -- model init / loader shuffling differ between the two arms."""
    return RUN_BASE_SEED + target_idx * 10 + arm_idx


def run_arm(feature_extractor, classifier, device, train_full, train_idx, val_idx, test_dataset,
            ft_extractor_optimizer, classifier_optimizer, out_prefix,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            improve_threshold=IMPROVE_THRESHOLD):
    """Adaptive early-stopping training loop, shared by both arms.

    Returns (results: dict, best_state: dict) -- results is ready to be extended by the caller
    with target/arm/seed fields and written to JSON; best_state is {'feature_extractor':
    state_dict, 'classifier': state_dict} for the best-validation-epoch, ready to torch.save.
    """
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_full, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    epochs_run = 0
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train(feature_extractor=feature_extractor, classifier=classifier, device=device,
              train_loader=train_loader, epoch=epoch, criterion=criterion,
              ft_extractor_optimizer=ft_extractor_optimizer, classifier_optimizer=classifier_optimizer)
        val_acc = test_func(feature_extractor=feature_extractor, classifier=classifier,
                             device=device, test_loader=val_loader)
        epochs_run = epoch
        print(f'[{out_prefix}] epoch {epoch}  val_acc={val_acc:.2f}  '
              f'best_so_far={best_val_acc:.2f} (epoch {best_epoch})')

        if val_acc > best_val_acc + improve_threshold:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                'feature_extractor': copy.deepcopy(feature_extractor.state_dict()),
                'classifier': copy.deepcopy(classifier.state_dict()),
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{out_prefix}] early stop at epoch {epoch} '
                      f'(no >{improve_threshold}pp improvement for {patience} epochs)')
                break

    elapsed = time.time() - t_start

    # Reload the best-validation checkpoint; evaluate on the native (untouched) test set exactly
    # once. Matters even for the frozen-transfer arm: its backbone weights never change, but
    # BatchNorm running statistics keep updating during training (models/trainer.py's train()
    # docstring), so the best-epoch state is not identical to the final-epoch state for that arm
    # either.
    feature_extractor.load_state_dict(best_state['feature_extractor'])
    classifier.load_state_dict(best_state['classifier'])
    final_acc = test_func(feature_extractor=feature_extractor, classifier=classifier,
                           device=device, test_loader=test_loader)

    results = {
        'epochs_run': epochs_run,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_acc': final_acc,
        'train_seconds': elapsed,
    }
    return results, best_state
