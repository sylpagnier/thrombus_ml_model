"""Smoke tests for V3.2 ranker rollout."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.training.clot_ml_step0_coef import discover_anchor_paths
from src.training.clot_ml_v2_growth_gnn import growth_gnn_feature_dim
from src.training.clot_ml_v32_growth_ranker import (
    ClotGrowthRankGNN,
    apply_v32_env,
    resolve_rule_cfg,
    rollout_v32_ranker,
)
from src.utils.paths import get_project_root


@pytest.fixture
def anchor_graph():
    root = get_project_root()
    paths = discover_anchor_paths(root / "data/processed/graphs_biochem_anchors")
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(paths[0], map_location="cpu", weights_only=False)


def test_ranker_rollout_smoke(anchor_graph) -> None:
    apply_v32_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_rule_cfg(
        get_project_root() / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    )
    if not Path(
        get_project_root() / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    ).exists():
        pytest.skip("step0 coef json missing")

    model = ClotGrowthRankGNN(in_dim=growth_gnn_feature_dim(), hidden=16)
    model.eval()
    phi_by_t = rollout_v32_ranker(
        model,
        anchor_graph,
        rule_cfg,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
    )
    assert len(phi_by_t) >= 2
    for t, phi in phi_by_t.items():
        assert phi.shape[0] == anchor_graph.num_nodes
        assert float(phi.min()) >= -1e-5
        assert float(phi.max()) <= 1.0 + 1e-5
