"""Early seed-location aux: wiring + physical location (not mass) checks."""

from __future__ import annotations

import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.architecture.runtime_config import BiochemRuntimeConfig
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env, mat_growth_leg_spec
from src.core_physics.species_pushforward_continuous import (
    continuous_frontier_hops,
    continuous_nucleation_topk,
    continuous_seed_aux_weight,
    gt_early_mat_pocket,
    soft_seed_location_aux_loss,
)
from src.training.train_species_pushforward_continuous import select_checkpoint_score


def test_wg_prec_seed_aux_leg_no_hard_frontier():
    leg = mat_growth_leg_spec("WG_prec_seed_aux")
    assert int(leg.config_kwargs.get("frontier_hops", -1)) == 0
    assert float(leg.config_kwargs.get("nucleation_topk", -1.0)) == 0.0
    assert float(leg.config_kwargs.get("seed_aux_weight", 0.0)) > 0.0
    assert float(leg.config_kwargs.get("seed_aux_weight", 0.0)) < 1.0  # must stay small
    assert int(leg.config_kwargs.get("seed_aux_early_steps", 0)) >= 1
    assert float(leg.runtime_kwargs.get("select_seed_prec_lambda", 0.0)) > 0.0
    # Prec mass/FP retained.
    assert float(leg.config_kwargs.get("step_mass_penalty", 0.0)) > 0.0
    assert float(leg.config_kwargs.get("final_mass_penalty", 0.0)) > 0.0
    assert "WG_prec_iter" in str(leg.init_ckpt)


def test_wg_prec_seed_aux_binds_typed_config():
    from src.biochem_gnn.config import _bind_typed_configs
    from src.architecture.pushforward_config import PushforwardConfig as PF
    from src.architecture.runtime_config import BiochemRuntimeConfig as RT

    try:
        apply_mat_growth_leg_env("WG_prec_seed_aux", force=True)
        assert continuous_frontier_hops() == 0
        assert continuous_nucleation_topk() == 0.0
        assert continuous_seed_aux_weight() == 0.15
    finally:
        _bind_typed_configs(PF(), RT())


def test_seed_aux_is_location_not_mass():
    """Equal soft mass, wrong place -> higher loss than correct place."""
    n = 6
    band = torch.ones(n, dtype=torch.bool)
    gt = torch.zeros(n, dtype=torch.bool)
    gt[1] = True
    gt[2] = True
    # Same total soft mass (~2.0), correct vs wrong support.
    good = torch.tensor([0.05, 0.95, 0.95, 0.05, 0.05, 0.05])
    bad = torch.tensor([0.95, 0.05, 0.05, 0.95, 0.05, 0.05])
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    l_good = soft_seed_location_aux_loss(
        good, gt, band, edge_index=ei, weight=1.0, compact_weight=0.0, pos_weight=4.0
    )
    l_bad = soft_seed_location_aux_loss(
        bad, gt, band, edge_index=ei, weight=1.0, compact_weight=0.0, pos_weight=4.0
    )
    assert float(l_bad) > float(l_good) + 0.05


def test_seed_aux_compactness_prefers_connected():
    n = 5
    band = torch.ones(n, dtype=torch.bool)
    gt = torch.zeros(n, dtype=torch.bool)
    gt[1] = True
    gt[2] = True
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    connected = torch.tensor([0.1, 0.9, 0.9, 0.1, 0.1])
    split = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.9])
    # Location weight 0 so only compactness differs.
    c_conn = soft_seed_location_aux_loss(
        connected, gt, band, edge_index=ei, weight=0.0, compact_weight=1.0
    )
    c_split = soft_seed_location_aux_loss(
        split, gt, band, edge_index=ei, weight=0.0, compact_weight=1.0
    )
    assert float(c_split) > float(c_conn)


def test_gt_early_mat_pocket_first_new_only():
    # Mat-only state dim 1.
    with use_pushforward_config(PushforwardConfig(dual_head=True, species_scope="mat")):
        s0 = torch.zeros(4, 1)
        s1 = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
        s2 = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        band = torch.ones(4, dtype=torch.bool)
        pocket = gt_early_mat_pocket([s0, s1, s2], early_steps=2, band_mask=band)
        assert bool(pocket[1]) and bool(pocket[2])
        assert not bool(pocket[0]) and not bool(pocket[3])


def test_select_score_uses_seed_panel():
    row = {
        "deploy_eval_t": 1,
        "loss": 1.0,
        "mat_seed_prec": 0.8,
        "mat_front_speed_ratio": 1.0,
        "deploy_clot_fn": 10.0,
        "deploy_clot_fp": 10.0,
    }
    base_kw = dict(
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.3,
        deploy_mat_f1=0.2,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.9,
        select_clot_score_weight=0.9,
        select_mat_f1_weight=0.1,
    )
    s0, _ = select_checkpoint_score(row, **base_kw)
    s1, mode = select_checkpoint_score(
        row, **base_kw, select_seed_prec_lambda=0.2, select_front_speed_lambda=0.1
    )
    assert mode == "deploy_only"
    assert s1 > s0
