"""Pocket-contrast exclusive wrong-pocket loss + WG_prec_pocket leg wiring."""

from __future__ import annotations

import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.architecture.runtime_config import BiochemRuntimeConfig
from src.biochem_gnn.mat_growth_simple import (
    get_mat_growth_config_kwargs,
    get_mat_growth_runtime_kwargs,
    mat_growth_leg_spec,
)
from src.core_physics.species_pushforward_continuous import (
    continuous_frontier_hops,
    continuous_nucleation_topk,
    continuous_pocket_contrast_hops,
    continuous_pocket_contrast_weight,
    gt_first_seed_mat_mask,
    pocket_allowed_from_gt_seed,
    pocket_contrast_aux_loss,
    soft_mat_commit_prob,
)


def test_wg_prec_pocket_leg_no_hard_mask():
    leg = mat_growth_leg_spec("WG_prec_pocket")
    assert int(leg.config_kwargs.get("frontier_hops", -1)) == 0
    assert float(leg.config_kwargs.get("nucleation_topk", -1.0)) == 0.0
    assert float(leg.config_kwargs.get("seed_aux_weight") or 0.0) == 0.0
    assert leg.config_kwargs.get("physical_fp_gating") in (False, None)
    assert float(leg.config_kwargs.get("pocket_contrast_weight") or 0.0) > 0.0
    assert int(leg.config_kwargs.get("pocket_contrast_hops") or 0) >= 2
    assert float(leg.runtime_kwargs.get("select_clot_f1_weight") or 0.0) >= 0.7
    assert float(leg.runtime_kwargs.get("select_mass_hard_min") or 0.0) >= 0.5
    assert "WG_prec_iter" in str(leg.init_ckpt)


def test_wg_prec_pocket_binds_typed_config():
    cfg = PushforwardConfig.from_meta({"config_kwargs": get_mat_growth_config_kwargs("WG_prec_pocket")})
    rt = BiochemRuntimeConfig.from_meta(
        {"runtime_kwargs": get_mat_growth_runtime_kwargs("WG_prec_pocket")}
    )
    assert cfg.pocket_contrast_weight > 0.0
    assert cfg.frontier_hops == 0
    assert cfg.physical_fp_gating is False
    assert rt.scoring.select_clot_f1_weight >= 0.7
    with use_pushforward_config(cfg):
        assert continuous_pocket_contrast_weight() > 0.0
        assert continuous_pocket_contrast_hops() >= 2
        assert continuous_frontier_hops() == 0
        assert continuous_nucleation_topk() == 0.0


def test_gt_first_seed_is_earliest_only():
    # t0 inactive; t1 seeds node 0; t2 adds node 2 -- first-seed must be only node 0.
    # log_state shape [N,1] Mat-only; active when > mat_commit_thresh (default snapshot ~1e-4).
    s0 = torch.zeros(4, 1)
    s1 = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    s2 = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    band = torch.ones(4, dtype=torch.bool)
    with use_pushforward_config(PushforwardConfig(species_scope="mat", channels=(11,), mat_commit_thresh=1e-4)):
        seed = gt_first_seed_mat_mask([s0, s1, s2], early_steps=3, band_mask=band)
    assert seed.tolist() == [True, False, False, False]


def test_pocket_contrast_penalizes_outside_not_inside():
    # Line 0-1-2-3; allowed = {0,1}; soft mass on 3 should cost more than on 0.
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    seed = torch.tensor([True, False, False, False])
    allowed = pocket_allowed_from_gt_seed(seed, edge, hops=1)
    assert allowed.tolist() == [True, True, False, False]
    band = torch.ones(4, dtype=torch.bool)
    soft_bad = torch.tensor([0.1, 0.1, 0.1, 0.9])
    soft_good = torch.tensor([0.9, 0.5, 0.0, 0.0])
    bad = pocket_contrast_aux_loss(soft_bad, allowed, band, weight=1.0, inside_weight=0.0)
    good = pocket_contrast_aux_loss(soft_good, allowed, band, weight=1.0, inside_weight=0.0)
    assert float(bad.item()) > float(good.item())


def test_pocket_contrast_no_hard_mask_on_allowed():
    """Inside allowed pocket, outside-weight alone does not punish high soft mass."""
    allowed = torch.tensor([True, True, False, False])
    band = torch.ones(4, dtype=torch.bool)
    soft = torch.tensor([0.95, 0.9, 0.0, 0.0])
    loss = pocket_contrast_aux_loss(soft, allowed, band, weight=1.0, inside_weight=0.0)
    assert float(loss.item()) == 0.0


def test_soft_mat_commit_prob_shape():
    with use_pushforward_config(PushforwardConfig(species_scope="mat", channels=(11,), mat_commit_thresh=1e-4)):
        log_state = torch.tensor([[0.0], [1.0], [0.5]])
        p = soft_mat_commit_prob(log_state)
    assert p.shape == (3,)
    assert float(p[0].item()) < float(p[1].item())
