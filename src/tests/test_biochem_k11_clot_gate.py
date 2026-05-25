"""K11 binary clot gate helpers and forward policy."""

from __future__ import annotations

import torch

from src.architecture import gnode_biochem as gb


def test_k11_clot_region_excludes_wall(monkeypatch):
    monkeypatch.setenv("BIOCHEM_K11_D_PEAK_ND", "0.008")
    monkeypatch.setenv("BIOCHEM_K11_SIGMA_ND", "0.008")
    sdf = torch.tensor([0.0, 0.008, 0.02, 0.05], dtype=torch.float32)
    wall = torch.tensor([True, False, False, False])
    m = gb.k11_clot_region_mask(sdf, wall).reshape(-1)
    assert float(m[0].item()) == 0.0
    assert float(m[1].item()) > 0.4


def test_k11_mu_clot_si_default():
    from src.config import PhysicsConfig

    cfg = PhysicsConfig()
    assert gb.k11_mu_clot_si(cfg) >= 0.09


def test_k11_policy_snapshot(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_K11_CLOT_GATE", "1")
    monkeypatch.delenv("BIOCHEM_MU_K10E_SIMPLE", raising=False)
    fp = gb.snapshot_biochem_forward_policy()
    assert fp["mu_k11_clot_gate"] is True
    assert not gb._biochem_mu_k10e_simple_enabled()
