"""K10e wall-adjacent μ mask and forward policy."""

from __future__ import annotations

import torch

from src.architecture import gnode_biochem as gb


def test_k10e_wall_adjacent_mask_zero_on_wall(monkeypatch):
    monkeypatch.setenv("BIOCHEM_K10E_D_PEAK_ND", "0.004")
    monkeypatch.setenv("BIOCHEM_K10E_SIGMA_ND", "0.003")
    sdf = torch.tensor([0.0, 0.004, 0.01, 0.05], dtype=torch.float32)
    wall = torch.tensor([True, False, False, False])
    m = gb.k10e_wall_adjacent_mask(sdf, wall).reshape(-1)
    assert float(m[0].item()) == 0.0
    assert float(m[1].item()) > 0.5
    assert float(m[3].item()) == 0.0


def test_k10e_simple_env_and_policy(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_K10E_SIMPLE", "1")
    monkeypatch.delenv("BIOCHEM_MU_K10D_SIMPLE", raising=False)
    assert gb._biochem_mu_k10e_simple_enabled()
    assert not gb._biochem_mu_k10d_simple_enabled()
    fp = gb.snapshot_biochem_forward_policy()
    assert fp["mu_k10e_simple"] is True
