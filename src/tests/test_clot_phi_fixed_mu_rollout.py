"""Phi-only rollout: mu from log_blend(mu_c, phi_prev); no mu carry."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.clot_forecast import resolve_rollout_prev_mu_si
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState
from src.core_physics.clot_phi_simple import (
    ClotPhiStepBatch,
    clot_phi_fixed_mu_from_phi_enabled,
    log_blend_mu_eff_si,
    mu_eff_from_carried_phi,
)


def _step(mu_c: list[float], mu_gt: list[float] | None = None) -> ClotPhiStepBatch:
    mc = torch.tensor(mu_c, dtype=torch.float32)
    mg = torch.tensor(mu_gt if mu_gt is not None else mu_c, dtype=torch.float32)
    return ClotPhiStepBatch(
        features=torch.zeros(len(mu_c), 4),
        phi_gt=torch.zeros(len(mu_c)),
        mu_c_si=mc,
        mu_gt_cap=mg,
        region=torch.ones(len(mu_c)),
        loss_mask=torch.ones(len(mu_c), dtype=torch.bool),
        species_log_gt=torch.zeros(len(mu_c), 12),
        u_flow_nd=torch.zeros(len(mu_c)),
        v_flow_nd=torch.zeros(len(mu_c)),
    )


def test_mu_eff_from_carried_phi_bulk_when_no_phi():
    device = torch.device("cpu")
    mu = mu_eff_from_carried_phi(torch.tensor([0.04, 0.05]), None, device=device)
    assert mu[0].item() == pytest.approx(0.04, rel=1e-4)
    assert mu[1].item() == pytest.approx(0.05, rel=1e-4)


def test_mu_eff_from_carried_phi_clot_level(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MU_SOLID_SI", "0.10")
    device = torch.device("cpu")
    mu_c = torch.tensor([0.04, 0.04])
    phi = torch.tensor([1.0, 0.0])
    mu = mu_eff_from_carried_phi(mu_c, phi, device=device)
    assert mu[0].item() == pytest.approx(0.10, rel=1e-4)
    assert mu[1].item() == pytest.approx(0.04, rel=1e-4)


def test_resolve_rollout_prev_mu_si_fixed_mu(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    monkeypatch.setenv("CLOT_PHI_MU_SOLID_SI", "0.10")
    assert clot_phi_fixed_mu_from_phi_enabled()
    device = torch.device("cpu")
    step = _step([0.04, 0.04], [0.04, 0.08])
    rs = ClotPhiRolloutState(phi_prev=torch.tensor([1.0, 0.0]))
    mu = resolve_rollout_prev_mu_si(rs, step, device)
    assert mu[0].item() == pytest.approx(0.10, rel=1e-4)
    assert mu[1].item() == pytest.approx(0.04, rel=1e-4)
    mu0 = resolve_rollout_prev_mu_si(None, step, device)
    assert mu0[0].item() == pytest.approx(0.04, rel=1e-4)


def test_log_blend_matches_carried_phi_at_full_phi(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MU_SOLID_SI", "0.10")
    mu_c = torch.tensor([0.04])
    phi = torch.tensor([0.8])
    mu_a = log_blend_mu_eff_si(mu_c, phi)
    mu_b = mu_eff_from_carried_phi(mu_c, phi, device=torch.device("cpu"))
    assert mu_a.item() == pytest.approx(mu_b.item(), rel=1e-5)
