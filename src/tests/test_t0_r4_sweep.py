"""T0 Rung4 sweep recipe wiring."""

from __future__ import annotations

import os

import torch

from src.core_physics.t0_rung4_ladder import (
    S0_RULE_ENV_KEYS,
    rung4_step_uses_coupled_species_rollout,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_r4_sweep import (
    RECIPES,
    T0R4SweepGNN,
    _build_models,
    load_sweep_bundle,
    recipe_from_id,
    save_sweep_checkpoint,
)


def test_recipe_catalog():
    assert "s_star_full" in RECIPES
    assert "s_star_g0_rules" in RECIPES
    assert "s4_delta_gnn" in RECIPES
    assert "s5_gnode_fimat" in RECIPES
    assert RECIPES["s5_gnode_fimat"].species == "gnn_delta"
    assert RECIPES["s5_gnode_fimat"].w_species > 0.0
    assert RECIPES["s_star_gate"].gate == "gnn"
    assert RECIPES["s_star_species"].species == "gnn_delta"
    assert RECIPES["s_star_dyn"].dyn == "gru_smooth"


def test_sweep_gnn_zero_init():
    m = T0R4SweepGNN(in_dim=11, out_dim=1, hidden=16)
    n = 24
    x = torch.randn(n, 11)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = m(x, ei)
    assert y.shape == (n, 1)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_s_star_gate_uses_coupled_rollout():
    assert rung4_step_uses_coupled_species_rollout("s_star_gate") is True
    assert rung4_step_uses_coupled_species_rollout("s0") is False
    assert rung4_step_uses_coupled_species_rollout("s_star_g0_rules") is False


def test_s0_rule_env_keys_cover_sweep_knobs():
    for key in (
        "T0_R4_S0_SPATIAL_TOP_FRAC",
        "T0_R4_S0_ONSET_TAU_START",
        "T0_R4_S0_ONSET_TAU_END",
        "T0_R4_S0_FI_MAT_GAIN",
        "T0_R4_S0_SPREAD_HOPS",
        "T0_R4_S0_SPREAD_DECAY",
    ):
        assert key in S0_RULE_ENV_KEYS


def test_rung4_deploy_band_default():
    os.environ.pop("T0_RUNG4_USE_DGAMMA_WALL_SEED", None)
    assert rung4_use_dgamma_wall_seed() is False
    os.environ["T0_RUNG4_USE_DGAMMA_WALL_SEED"] = "1"
    try:
        assert rung4_use_dgamma_wall_seed() is True
    finally:
        os.environ.pop("T0_RUNG4_USE_DGAMMA_WALL_SEED", None)


def test_s5_delta_is_tanh_bounded():
    recipe = recipe_from_id("s5_gnode_fimat")
    bundle = _build_models(recipe, torch.device("cpu"))
    n = 12
    x = torch.randn(n, bundle.in_dim) * 5.0
    ei = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).T
    raw = bundle.species_model(x, ei)
    delta = torch.tanh(raw)
    assert delta.abs().max().item() <= 1.0 + 1e-6


def test_sweep_ckpt_roundtrip(tmp_path):
    recipe = recipe_from_id("s_star_small_ml")
    bundle = _build_models(recipe, torch.device("cpu"))
    ckpt = tmp_path / "leg.pth"
    save_sweep_checkpoint(ckpt, bundle, meta={"epoch": 1})
    loaded = load_sweep_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert loaded is not None
    assert loaded.recipe.id == "s_star_small_ml"
