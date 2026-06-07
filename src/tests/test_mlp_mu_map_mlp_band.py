"""Wired deploy mlp_band commit (MLP mu inside allowed vision)."""

from __future__ import annotations

import pytest
import torch

from src.config import PhysicsConfig
from src.core_physics.clot_phi_mu_inject import (
    mlp_mu_map_mask_mode,
    normalize_mlp_mu_map_mask_mode,
    resolve_clot_trigger_gate,
    resolve_deploy_mlp_band_commit_mask,
)


def test_mlp_band_mode_normalized():
    assert normalize_mlp_mu_map_mask_mode("mlp_band") == "mlp_band"
    assert normalize_mlp_mu_map_mask_mode("wired") == "mlp_band"


def test_mlp_band_commit_inside_allowed_only(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "mlp_band")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_THRESH", "0.5")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    allowed = torch.tensor([True, True, False, False])
    phi = torch.tensor([0.9, 0.1, 0.9, 0.1])
    mu_mlp = torch.tensor([0.10, 0.10, 0.10, 0.10])
    mu_c = torch.tensor([0.04, 0.04, 0.04, 0.04])
    gate = resolve_deploy_mlp_band_commit_mask(allowed, phi, mu_mlp, mu_c, phys_cfg=phys)
    assert gate.tolist() == [True, False, False, False]


def test_resolve_clot_trigger_gate_mlp_band(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "mlp_band")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_THRESH", "0.5")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    allowed = torch.tensor([True, False, True])
    phi = torch.tensor([0.8, 0.8, 0.2])
    mu_mlp = torch.full((3, 1), 0.10)
    mu_c = torch.full((3, 1), 0.04)
    gate = resolve_clot_trigger_gate(
        phi,
        mu_c,
        mu_mlp,
        phys_cfg=phys,
        allowed_commit_mask=allowed,
    )
    assert int(gate.sum().item()) == 1
    assert mlp_mu_map_mask_mode() == "mlp_band"
