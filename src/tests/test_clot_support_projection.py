"""Hard support projection: mu = Carreau off-band, model mu inside B_t."""

from __future__ import annotations

import os

import torch

from src.core_physics.clot_phi_simple import (
    apply_clot_support_projection,
    clot_phi_hard_support_projection_enabled,
    project_mu_from_phi_with_support,
)


def test_apply_clot_support_projection_off_band_is_carreau():
    mu_c = torch.tensor([0.04, 0.04, 0.05, 0.06], dtype=torch.float32)
    mu_model = torch.tensor([0.10, 0.10, 0.10, 0.10], dtype=torch.float32)
    band = torch.tensor([False, True, True, False])
    out = apply_clot_support_projection(mu_c, mu_model, band)
    assert out[0].item() == mu_c[0].item()
    assert out[1].item() == mu_model[1].item()
    assert out[3].item() == mu_c[3].item()


def test_project_mu_from_phi_with_support():
    mu_c = torch.tensor([0.04, 0.04], dtype=torch.float32)
    phi = torch.tensor([1.0, 0.0], dtype=torch.float32)
    band = torch.tensor([True, False])
    out = project_mu_from_phi_with_support(mu_c, phi, band, mu_solid_si=0.10)
    assert out[0].item() > 0.09
    assert abs(out[1].item() - 0.04) < 1e-5


def test_hard_support_defaults_on_for_forecast(monkeypatch):
    monkeypatch.delenv("CLOT_PHI_HARD_SUPPORT_PROJECTION", raising=False)
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    assert clot_phi_hard_support_projection_enabled()


def test_hard_support_off_when_disabled(monkeypatch):
    monkeypatch.setenv("CLOT_FORECAST_MODE", "one_step")
    monkeypatch.setenv("CLOT_PHI_HARD_SUPPORT_PROJECTION", "0")
    assert not clot_phi_hard_support_projection_enabled()
