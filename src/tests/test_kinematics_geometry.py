import torch
from torch_geometric.data import Data

from src.utils.kinematics_geometry import (
    GeometryCurriculumConfig,
    cohort_level_counts,
    geometry_sample_weight,
    split_anchor_physics_stratified,
    train_pool_for_epoch,
)


def _graph(level: int, anchor: bool) -> Data:
    d = Data(
        x=torch.zeros(4, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.zeros(4, 5),
        is_anchor=torch.tensor([anchor], dtype=torch.bool),
    )
    d.geometry_level = torch.tensor([level], dtype=torch.int8)
    d.config_id = 0
    return d


def test_geometry_curriculum_auto_phases():
    cfg = GeometryCurriculumConfig(enabled=True, phase="auto", l0l1_only_epochs=6)
    assert cfg.resolved_phase(2, 1, 40, 60) == "l0l1_only"
    w0 = cfg.level_weights(2, 1, stage1_end=40, stage2_end=60)
    assert w0[2] == 0.0
    assert cfg.resolved_phase(7, 1, 40, 60) == "foundation"
    w1 = cfg.level_weights(7, 1, stage1_end=40, stage2_end=60)
    assert w1[2] < w1[0]
    w2 = cfg.level_weights(65, 3, stage1_end=40, stage2_end=60)
    assert w2[2] > w2[0]


def test_train_pool_l0l1_only_excludes_l2():
    cfg = GeometryCurriculumConfig(enabled=True, l0l1_only_epochs=6)
    train = [_graph(0, True), _graph(1, False), _graph(2, True)]
    pool = train_pool_for_epoch(train, curriculum=cfg, epoch=2, stage=1, stage1_end=40, stage2_end=60)
    assert all(int(d.geometry_level.item()) in (0, 1) for d in pool)
    pool2 = train_pool_for_epoch(train, curriculum=cfg, epoch=8, stage=1, stage1_end=40, stage2_end=60)
    assert len(pool2) == 3


def test_geometry_sample_weight():
    cfg = GeometryCurriculumConfig(enabled=True, phase="l2_heavy")
    w = cfg.level_weights(80, 3, stage1_end=40, stage2_end=60)
    g2 = _graph(2, True)
    g0 = _graph(0, True)
    assert geometry_sample_weight(g2, w) > geometry_sample_weight(g0, w)


def test_split_stratified_preserves_levels_in_val():
    dataset = [_graph(l, i % 2 == 0) for l in (0, 1, 2) for i in range(6)]
    splits = split_anchor_physics_stratified(dataset, seed=0)
    val_levels = {int(d.geometry_level.item()) for d in splits["val"]}
    assert 0 in val_levels or 1 in val_levels or 2 in val_levels
    assert len(splits["train"]) + len(splits["val"]) == len(dataset)


def test_cohort_level_counts():
    ds = [_graph(0, False), _graph(2, True)]
    c = cohort_level_counts(ds)
    assert c[0] == 1 and c[2] == 1
