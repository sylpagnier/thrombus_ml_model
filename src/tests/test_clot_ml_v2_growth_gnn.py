"""V3 growth GNN unit / smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.training.clot_ml_v2_growth_gnn import (
    ClotGrowthRateGNN,
    apply_step3_v3_env,
    eligibility_union_mask,
    growth_gnn_feature_dim,
    rate_scale_from_env,
)


def test_growth_gnn_feature_dim():
    assert growth_gnn_feature_dim() == 8


def test_rate_scale_env(monkeypatch):
    monkeypatch.setenv("CLOT_V3_RATE_SCALE", "3.5")
    assert rate_scale_from_env() == 3.5


def test_eligibility_union_nonempty():
    class _G:
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        y = torch.zeros(3, 4, 16)
        mask_wall = torch.tensor([1.0, 0.0, 0.0, 0.0])

    g = _G()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    phi_by_t = {
        0: torch.zeros(4),
        1: torch.tensor([0.6, 0.0, 0.0, 0.0]),
    }
    union = eligibility_union_mask(g, phi_by_t, device=device, phys_cfg=phys, bio_cfg=bio)
    assert bool(union.any().item())


@pytest.mark.skipif(
    not Path("data/processed/graphs_biochem_anchors/patient007.pt").exists(),
    reason="patient007 anchor missing",
)
def test_v3_rollout_forward_smoke():
    from src.training.clot_ml_device import resolve_clot_ml_eval_device
    from src.training.clot_ml_v2_growth_gnn import resolve_v3_rule_cfg, rollout_v3_growth_gnn
    from src.utils.paths import get_project_root

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    apply_step3_v3_env()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors/patient007.pt",
        map_location=device,
        weights_only=False,
    )
    rule_cfg = resolve_v3_rule_cfg(root / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    model = ClotGrowthRateGNN(in_dim=growth_gnn_feature_dim(), hidden=16).to(device)
    model.eval()
    phi = rollout_v3_growth_gnn(
        model, data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio
    )
    assert len(phi) == int(data.y.shape[0])
    assert float(phi[max(phi.keys())].max().item()) >= 0.0
