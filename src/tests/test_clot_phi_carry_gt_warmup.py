"""Bridge A: GT log-mu carry warm-up for clot-phi rollout."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.clot_forecast import resolve_rollout_prev_mu_si
from src.core_physics.clot_phi_rollout import (
    ClotPhiRolloutState,
    carry_gt_fade_alpha,
    carry_gt_warmup_active,
    resolve_carry_log_mu_feature,
)
from src.core_physics.clot_phi_simple import ClotPhiStepBatch


def _step(mu_gt: list[float]) -> ClotPhiStepBatch:
    t = torch.tensor(mu_gt, dtype=torch.float32)
    return ClotPhiStepBatch(
        features=torch.zeros(len(mu_gt), 4),
        phi_gt=torch.zeros(len(mu_gt)),
        mu_c_si=torch.full((len(mu_gt), 1), 0.04),
        mu_gt_cap=t,
        region=torch.ones(len(mu_gt)),
        loss_mask=torch.ones(len(mu_gt), dtype=torch.bool),
        species_log_gt=torch.zeros(len(mu_gt), 12),
        u_flow_nd=torch.zeros(len(mu_gt)),
        v_flow_nd=torch.zeros(len(mu_gt)),
    )


def test_carry_gt_warmup_active_train_only(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "10")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_STEPS", "0")
    assert carry_gt_warmup_active(0, 5) is True
    assert carry_gt_warmup_active(99, 5) is True
    assert carry_gt_warmup_active(0, 10) is False
    assert carry_gt_warmup_active(0, None) is False


def test_carry_gt_warmup_steps(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "0")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_STEPS", "5")
    assert carry_gt_warmup_active(4, 20) is True
    assert carry_gt_warmup_active(5, 20) is False


def test_carry_gt_fade_alpha(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "10")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_FADE_EPOCHS", "4")
    assert carry_gt_fade_alpha(9) is None
    assert carry_gt_fade_alpha(10) == pytest.approx(0.0)
    assert carry_gt_fade_alpha(12) == pytest.approx(0.5)
    assert carry_gt_fade_alpha(14) is None


def test_resolve_carry_log_mu_gt_then_pred(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_ROLLOUT", "1")
    monkeypatch.setenv("CLOT_PHI_CARRY_LOG_MU", "1")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "5")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_FADE_EPOCHS", "0")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_STEPS", "0")
    device = torch.device("cpu")
    mu_gt = torch.tensor([0.04, 0.08, 0.04])
    rs = ClotPhiRolloutState(log_mu_prev=torch.log(torch.tensor([0.05, 0.10, 0.05])))
    log_gt = resolve_carry_log_mu_feature(
        time_index=0,
        train_epoch=2,
        gt_mu_cap_si=mu_gt,
        rollout_state=rs,
        device=device,
    )
    assert log_gt is not None
    assert log_gt[1].item() == pytest.approx(torch.log(torch.tensor(0.08)).item(), rel=1e-4)
    log_pred = resolve_carry_log_mu_feature(
        time_index=0,
        train_epoch=10,
        gt_mu_cap_si=mu_gt,
        rollout_state=rs,
        device=device,
    )
    assert log_pred is not None
    assert log_pred[1].item() == pytest.approx(torch.log(torch.tensor(0.10)).item(), rel=1e-4)


def test_resolve_rollout_prev_mu_si_bridge(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_WARMUP_EPOCHS", "5")
    monkeypatch.setenv("CLOT_PHI_CARRY_GT_FADE_EPOCHS", "0")
    device = torch.device("cpu")
    step = _step([0.04, 0.06, 0.04])
    rs = ClotPhiRolloutState(log_mu_prev=torch.log(torch.tensor([0.04, 0.08, 0.04])))
    mu_warm = resolve_rollout_prev_mu_si(rs, step, device, time_index=0, train_epoch=2)
    assert mu_warm[1].item() == pytest.approx(0.06, rel=1e-4)
    mu_pred = resolve_rollout_prev_mu_si(rs, step, device, time_index=0, train_epoch=None)
    assert mu_pred[1].item() == pytest.approx(0.08, rel=1e-4)
