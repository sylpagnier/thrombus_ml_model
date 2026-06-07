"""Tests for deploy-faithful forecast mu carry."""

from __future__ import annotations

import os

import pytest
import torch

from src.core_physics.clot_forecast import (
    clot_forecast_mu_carry_enabled,
    resolve_forecast_log_mu_in,
    resolve_forecast_mu_in_si,
    snapshot_clot_forecast_config,
)
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState


def test_mu_carry_disabled_uses_gt(monkeypatch):
    monkeypatch.delenv("CLOT_FORECAST_MU_CARRY", raising=False)
    gt = torch.tensor([0.04, 0.08, 0.04])
    mu_c = torch.full((3, 1), 0.035)
    log_mu = resolve_forecast_log_mu_in(
        gt_mu_cap_si=gt,
        mu_c_si=mu_c,
        forecast_state=None,
        time_index=1,
        train_epoch=None,
        device=torch.device("cpu"),
    )
    assert log_mu[1].item() == pytest.approx(float(torch.log(gt[1]).item()))


def test_mu_carry_uses_state_after_warmup(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MU_CARRY", "1")
    monkeypatch.setenv("CLOT_FORECAST_MU_INIT", "carreau")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "0")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_STEPS", "0")
    assert clot_forecast_mu_carry_enabled()
    gt = torch.tensor([0.04, 0.08, 0.04])
    mu_c = torch.full((3, 1), 0.035)
    state = ClotPhiRolloutState(log_mu_prev=torch.log(torch.tensor([0.04, 0.05, 0.04])))
    log_mu = resolve_forecast_log_mu_in(
        gt_mu_cap_si=gt,
        mu_c_si=mu_c,
        forecast_state=state,
        time_index=2,
        train_epoch=10,
        device=torch.device("cpu"),
    )
    assert log_mu[1].item() == pytest.approx(float(torch.log(torch.tensor(0.05)).item()))


def test_mu_carry_cold_start_carreau(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MU_CARRY", "1")
    monkeypatch.setenv("CLOT_FORECAST_MU_INIT", "carreau")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "0")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_STEPS", "0")
    gt = torch.tensor([0.04, 0.08, 0.04])
    mu_c = torch.full((3, 1), 0.035)
    log_mu = resolve_forecast_log_mu_in(
        gt_mu_cap_si=gt,
        mu_c_si=mu_c,
        forecast_state=None,
        time_index=0,
        train_epoch=10,
        device=torch.device("cpu"),
    )
    assert log_mu[0].item() == pytest.approx(float(torch.log(torch.tensor(0.035)).item()))


def test_snapshot_includes_mu_carry(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    monkeypatch.setenv("CLOT_FORECAST_INPUT_MU", "1")
    monkeypatch.setenv("CLOT_FORECAST_MU_CARRY", "1")
    cfg = snapshot_clot_forecast_config()
    assert cfg["forecast_mu_carry"] is True
    assert cfg["forecast_input_mu"] is True


def test_resolve_forecast_mu_in_si_roundtrip(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MU_CARRY", "0")
    gt = torch.tensor([0.04, 0.06])
    mu_c = torch.full((2, 1), 0.035)
    mu_in = resolve_forecast_mu_in_si(
        gt_mu_cap_si=gt,
        mu_c_si=mu_c,
        forecast_state=None,
        time_index=0,
        train_epoch=None,
        device=torch.device("cpu"),
    )
    assert mu_in[1].item() == pytest.approx(0.06, rel=1e-4)
