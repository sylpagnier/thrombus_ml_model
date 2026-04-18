"""Tests for Tier 1 anchor/physics train/val split (``split_anchor_physics``)."""

import torch

from src.training.train_t1_predictor import split_anchor_physics


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
    split = split_anchor_physics(dataset, seed=42, train_ratio=0.9)
    assert split.train_anchors == 9
    assert split.train_physics == 18
    assert len(split.train) == 27
    assert len(split.val) == 3
    assert sum(int(d.is_anchor.any().item()) for d in split.train) == 9
    assert sum(int(d.is_anchor.any().item()) for d in split.val) == 1


def test_split_deterministic_for_same_seed():
    dataset = _make_dataset(n_anchor=6, n_phys=6)
    s1 = split_anchor_physics(dataset, seed=7, train_ratio=0.5)
    s2 = split_anchor_physics(dataset, seed=7, train_ratio=0.5)
    assert [d.idx for d in s1.train] == [d.idx for d in s2.train]
    assert [d.idx for d in s1.val] == [d.idx for d in s2.val]


def test_split_changes_with_seed():
    dataset = _make_dataset(n_anchor=8, n_phys=8)
    s1 = split_anchor_physics(dataset, seed=1, train_ratio=0.75)
    s2 = split_anchor_physics(dataset, seed=99, train_ratio=0.75)
    # Different RNG seed should generally produce different ordering in split membership.
    assert [d.idx for d in s1.train] != [d.idx for d in s2.train]
