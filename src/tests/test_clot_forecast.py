"""Tests for clot forecast ladder helpers."""

from __future__ import annotations

import os

import pytest
import torch

from src.core_physics.clot_forecast import (
    append_forecast_input_features,
    clot_forecast_extra_feature_dim,
    clot_forecast_mask_mode,
    clot_forecast_one_step_enabled,
    clot_forecast_pair_schedule,
    iter_forecast_pairs,
    snapshot_clot_forecast_config,
)
from src.core_physics.clot_phi_simple import ClotPhiMPNNHybrid, build_clot_phi_model, clot_phi_feature_dim


def test_forecast_env_dims(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "0")
    assert clot_forecast_one_step_enabled()
    assert clot_forecast_extra_feature_dim() == 0
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "1")
    assert clot_forecast_extra_feature_dim() == 1


def test_append_forecast_input_features(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "1")
    feats = torch.zeros(4, 3)
    log_mu = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = append_forecast_input_features(
        feats, log_mu, n_nodes=4, device=torch.device("cpu"), dtype=torch.float32
    )
    assert out.shape == (4, 4)
    assert out[0, 3].item() == pytest.approx(0.1)


def test_mpnn_hybrid_forward():
    model = ClotPhiMPNNHybrid(in_dim=3, hidden=16)
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    logits = model.forward_logits(x, edge_index)
    dlog = model.forward_delta_log_mu(x, edge_index)
    assert logits.shape == (5,)
    assert dlog.shape == (5,)


def test_build_mpnn_via_factory(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_HYBRID", "1")
    monkeypatch.setenv("CLOT_PHI_MODEL", "mpnn")
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    model = build_clot_phi_model(in_dim=clot_phi_feature_dim(), hidden=16)
    assert isinstance(model, ClotPhiMPNNHybrid)


def test_deploy_pred_mask_mlp_forward(monkeypatch):
    """MLP hybrid forward_logits must not receive edge_index."""
    from src.core_physics.clot_forecast import resolve_forecast_deploy_mask_from_model
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.clot_phi_simple import ClotPhiStepBatch

    monkeypatch.setenv("CLOT_FORECAST_MASK", "deploy_pred")
    monkeypatch.setenv("CLOT_PHI_HYBRID", "1")
    model = build_clot_phi_model(in_dim=4, hidden=8)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")

    class _MiniGraph:
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        mask_wall = torch.tensor([True, True, False, False])

    data = _MiniGraph()
    feats = torch.randn(4, 4)
    step = ClotPhiStepBatch(
        features=feats,
        phi_gt=torch.zeros(4),
        mu_c_si=torch.full((4, 1), 0.04),
        mu_gt_cap=torch.tensor([0.04, 0.08, 0.04, 0.04]),
        region=torch.zeros(4),
        loss_mask=torch.zeros(4, dtype=torch.bool),
        species_log_gt=torch.zeros(4, 12),
        u_flow_nd=torch.zeros(4),
        v_flow_nd=torch.zeros(4),
        mu_in_cap=torch.tensor([0.04, 0.08, 0.04, 0.04]),
        phi_in_gt=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    mask = resolve_forecast_deploy_mask_from_model(
        data,
        step=step,
        model=model,
        edge_index=None,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        hybrid=True,
    )
    assert mask.shape == (4,)
    assert bool(mask.any())


def test_resolve_rollout_prev_mu_si():
    from src.core_physics.clot_forecast import resolve_rollout_prev_mu_si
    from src.core_physics.clot_phi_rollout import ClotPhiRolloutState
    from src.core_physics.clot_phi_simple import ClotPhiStepBatch

    device = torch.device("cpu")
    step = ClotPhiStepBatch(
        features=torch.zeros(3, 4),
        phi_gt=torch.zeros(3),
        mu_c_si=torch.full((3, 1), 0.04),
        mu_gt_cap=torch.tensor([0.04, 0.06, 0.04]),
        region=torch.ones(3),
        loss_mask=torch.ones(3, dtype=torch.bool),
        species_log_gt=torch.zeros(3, 12),
        u_flow_nd=torch.zeros(3),
        v_flow_nd=torch.zeros(3),
    )
    rs = ClotPhiRolloutState(log_mu_prev=torch.log(torch.tensor([0.04, 0.08, 0.04])))
    mu = resolve_rollout_prev_mu_si(rs, step, device)
    assert mu[1].item() == pytest.approx(0.08, rel=1e-4)
    mu0 = resolve_rollout_prev_mu_si(None, step, device)
    assert mu0[1].item() == pytest.approx(0.06, rel=1e-4)


def test_forecast_mask_mode(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MASK", "deploy_pred")
    assert clot_forecast_mask_mode() == "deploy_pred"
    monkeypatch.setenv("CLOT_FORECAST_MASK", "deploy_input")
    assert clot_forecast_mask_mode() == "deploy_input"
    monkeypatch.setenv("CLOT_FORECAST_MASK", "input")
    assert clot_forecast_mask_mode() == "input"
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "1")
    cfg = snapshot_clot_forecast_config()
    assert cfg["forecast_one_step"] is True
    assert cfg["forecast_input_mu"] is True


def test_forecast_pair_schedules(monkeypatch):
    t_steps = 54
    monkeypatch.setenv("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
    assert clot_forecast_pair_schedule() == "static_final"
    assert iter_forecast_pairs(t_steps, time_stride=1, pair_stride=1) == [(0, 53)]

    monkeypatch.setenv("CLOT_FORECAST_PAIR_SCHEDULE", "from_t0")
    pairs = iter_forecast_pairs(t_steps, time_stride=5, pair_stride=1)
    assert pairs[0] == (0, 1)
    assert all(t_in == 0 for t_in, _ in pairs)
    assert pairs[-1][1] == 51

    monkeypatch.delenv("CLOT_FORECAST_PAIR_SCHEDULE", raising=False)
    rolling = iter_forecast_pairs(10, time_stride=2, pair_stride=3)
    assert rolling == [(0, 3), (2, 5), (4, 7), (6, 9)]
