"""Smoke tests for shear-gradient localized risk blends."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_localized_spatial import LocalizedSpatialConfig
from src.core_physics.clot_temporal_growth_rules import (
    compute_localized_risk_score,
    eval_temporal_rule_on_anchor,
    shear_risk_rule_grid,
)
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 anchor missing")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_shear_grid_nonempty():
    rules = shear_risk_rule_grid()
    assert len(rules) >= 12
    assert all(r.name.startswith("sh_") and r.name.endswith("_inc40") for r in rules)


def test_shear_blend_changes_risk(patient007):
    device = torch.device("cpu")
    bio = BiochemConfig(phase="biochem")
    ceiling = resolve_ceiling_mask(patient007, device, bio)
    pool = ceiling.reshape(-1).bool()
    base_loc = LocalizedSpatialConfig(
        mode="wall_half",
        segment_top_frac=0.20,
        skip_wall_arc_frac=0.0,
        neg_dx_risk_weight=0.25,
        normalize_risk_per_half=True,
    )
    neg_loc = LocalizedSpatialConfig(
        mode="wall_half",
        segment_top_frac=0.20,
        skip_wall_arc_frac=0.0,
        neg_dx_risk_weight=0.70,
        sep_stream_risk_weight=0.10,
        stasis_risk_weight=0.10,
        low_grad_risk_weight=0.10,
        normalize_risk_per_half=True,
    )
    r0 = compute_localized_risk_score(
        patient007, device=device, bio_cfg=bio, t_in=0, ceiling=ceiling, pool=pool,
        spatial_rule=None, loc=base_loc,
    )
    r1 = compute_localized_risk_score(
        patient007, device=device, bio_cfg=bio, t_in=0, ceiling=ceiling, pool=pool,
        spatial_rule=None, loc=neg_loc,
    )
    assert not torch.allclose(r0, r1)


def test_shear_rule_eval_smoke(patient007):
    rule = shear_risk_rule_grid()[0]
    row = eval_temporal_rule_on_anchor(
        patient007,
        rule,
        stem="patient007",
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert row["n_pairs"] > 0
    assert row["tfinal_clot_shape"] == row["tfinal_clot_shape"]
