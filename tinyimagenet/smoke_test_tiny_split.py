"""Ad-hoc smoke test for tiny_split.py + models/resnet18.py together -- NOT part of the repo, do not commit.

Run after tiny_split.py's main() has produced saved_split_task/data/trainset_{0..9}.pt /
test_{0..9}.pt on RC. Checks:
  1. Each task has exactly 20 unique labels, remapped to 0..19.
  2. The 10 tasks' counts sum to TinyImageNet's real totals (100,000 train / 10,000 test).
  3. A real batch from task 0 produces a (B, 512) feature vector through the new ResNet18.
"""
import torch

from models.resnet18 import ResNet18

NUM_TASKS = 10
LABELS_PER_TASK = 20
EXPECTED_TRAIN_TOTAL = 100_000
EXPECTED_TEST_TOTAL = 10_000


def check_task(task_id, split, expected_count_hint):
    x, y = torch.load(f"saved_split_task/data/{split}_{task_id}.pt")
    unique_labels = sorted(y.unique().tolist())
    ok_labels = unique_labels == list(range(LABELS_PER_TASK))
    print(f"[{split}_{task_id}] count={x.shape[0]} labels_ok={ok_labels} "
          f"(expect 0..{LABELS_PER_TASK - 1}, got {unique_labels[:3]}...{unique_labels[-3:]})")
    assert ok_labels, f"{split}_{task_id}: labels not remapped to 0..{LABELS_PER_TASK - 1}"
    return x.shape[0]


def main():
    train_total = sum(check_task(i, "trainset", EXPECTED_TRAIN_TOTAL) for i in range(NUM_TASKS))
    test_total = sum(check_task(i, "test", EXPECTED_TEST_TOTAL) for i in range(NUM_TASKS))

    print(f"[1] train total={train_total} (expect {EXPECTED_TRAIN_TOTAL}) "
          f"ok={train_total == EXPECTED_TRAIN_TOTAL}")
    print(f"[2] test total={test_total} (expect {EXPECTED_TEST_TOTAL}) "
          f"ok={test_total == EXPECTED_TEST_TOTAL}")
    assert train_total == EXPECTED_TRAIN_TOTAL
    assert test_total == EXPECTED_TEST_TOTAL

    x0, y0 = torch.load("saved_split_task/data/trainset_0.pt")
    batch = x0[:8]
    print(f"[3] task 0 batch shape in: {tuple(batch.shape)}")

    net = ResNet18()
    net.eval()
    with torch.no_grad():
        out = net(batch)
    print(f"[3] ResNet18 output shape: {tuple(out.shape)} (expect (8, 512))")
    assert tuple(out.shape) == (8, 512)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
