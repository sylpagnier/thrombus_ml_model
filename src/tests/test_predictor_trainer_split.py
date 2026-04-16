"""Tests for shared dataset split logic in ``DEQPredictorTrainer``."""

import torch

from src.training.trainer import DEQPredictorTrainer


class _GraphStub:
    def __init__(self, is_anchor: bool, idx: int):
        self.is_anchor = torch.tensor([is_anchor], dtype=torch.bool)
        self.idx = idx


def _make_dataset(n_anchor: int, n_phys: int):
    anchors = [_GraphStub(True, i) for i in range(n_anchor)]
    phys = [_GraphStub(False, 10_000 + i) for i in range(n_phys)]
    return anchors + phys


def test_split_anchor_physics_counts_and_ratio():
    dataset = _make_dataset(n_anchor=10, n_phys=20)
    split = DEQPredictorTrainer(seed=42, train_ratio=0.9).split_anchor_physics(dataset)
    assert split.train_anchors == 9
    assert split.train_physics == 18
    assert len(split.train) == 27
    assert len(split.val) == 3
    assert sum(int(d.is_anchor.any().item()) for d in split.train) == 9
    assert sum(int(d.is_anchor.any().item()) for d in split.val) == 1


def test_split_deterministic_for_same_seed():
    dataset = _make_dataset(n_anchor=6, n_phys=6)
    t1 = DEQPredictorTrainer(seed=7, train_ratio=0.5)
    t2 = DEQPredictorTrainer(seed=7, train_ratio=0.5)
    s1 = t1.split_anchor_physics(dataset)
    s2 = t2.split_anchor_physics(dataset)
    assert [d.idx for d in s1.train] == [d.idx for d in s2.train]
    assert [d.idx for d in s1.val] == [d.idx for d in s2.val]


def test_split_changes_with_seed():
    dataset = _make_dataset(n_anchor=8, n_phys=8)
    s1 = DEQPredictorTrainer(seed=1, train_ratio=0.75).split_anchor_physics(dataset)
    s2 = DEQPredictorTrainer(seed=99, train_ratio=0.75).split_anchor_physics(dataset)
    # Different RNG seed should generally produce different ordering in split membership.
    assert [d.idx for d in s1.train] != [d.idx for d in s2.train]
