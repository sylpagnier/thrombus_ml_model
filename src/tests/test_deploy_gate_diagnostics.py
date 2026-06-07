"""Deploy gate frame diagnostics (allowed band phi / mu_mlp / commit)."""

from __future__ import annotations

import pytest
import torch

from src.config import PhysicsConfig
from src.core_physics.clot_phi_mu_inject import (
    mlp_deploy_no_commit_at_t0,
    resolve_deploy_mlp_band_commit_mask,
)


def test_bottleneck_phi_low_when_phi_below_thr(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "mlp_band")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_THRESH", "0.5")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    allowed = torch.tensor([True, True, False, False])
    phi = torch.tensor([0.1, 0.1, 0.9, 0.9])
    mu_mlp = torch.full((4,), 0.10)
    mu_c = torch.full((4,), 0.04)
    gate = resolve_deploy_mlp_band_commit_mask(
        allowed, phi, mu_mlp, mu_c, phys_cfg=PhysicsConfig(phase="biochem")
    )
    assert not bool(gate.any())


def test_no_commit_t0_flag(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "mlp_band")
    monkeypatch.setenv("BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0", "1")
    assert mlp_deploy_no_commit_at_t0()
