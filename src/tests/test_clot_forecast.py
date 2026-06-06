"""Tests for clot forecast ladder helpers."""

from __future__ import annotations

import os

import pytest
import torch

from src.core_physics.clot_forecast import (
    append_forecast_input_features,
    clot_forecast_extra_feature_dim,
    clot_forecast_one_step_enabled,
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


def test_snapshot_config(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "1")
    cfg = snapshot_clot_forecast_config()
    assert cfg["forecast_one_step"] is True
    assert cfg["forecast_input_mu"] is True
