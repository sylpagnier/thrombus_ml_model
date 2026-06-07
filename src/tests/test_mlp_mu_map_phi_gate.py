"""Leg B v2: clot trigger mask keeps Carreau bulk; MLP mu only on clot sites."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_mu_inject import (
    ClotPhiMuInjector,
    assemble_committed_mu_map,
    biochem_mlp_mu_map_enabled,
)
from src.core_physics.clot_phi_simple import cap_mu_eff_si, phi_gt_binary


class _FakeHybrid(torch.nn.Module):
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = torch.zeros(n, device=x.device, dtype=x.dtype)
        logits[0] = 10.0
        return logits

    def forward_delta_log_mu(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        d = torch.zeros(n, device=x.device, dtype=x.dtype)
        d[0] = 1.0
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class _FakeData:
    num_nodes = 3
    y = torch.zeros(1, 3, 16)
    mask_wall = torch.tensor([True, False, False])


def test_apply_mu_map_phi_gate_carreau_bulk(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_GATE", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "phi")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_THRESH", "0.5")
    monkeypatch.setenv("BIOCHEM_MLP_CLOT_BLEND", "1.0")
    assert biochem_mlp_mu_map_enabled()

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    inj = ClotPhiMuInjector.__new__(ClotPhiMuInjector)
    inj.ckpt_path = None
    inj.device = device
    inj.phys_cfg = phys
    inj.bio_cfg = bio
    inj._model = _FakeHybrid()
    inj._cfg = {"hybrid": True}
    inj.last_diag = None

    mu_c = torch.tensor([[0.04], [0.04], [0.04]], dtype=torch.float32)
    mu_mlp = torch.tensor([[0.10], [0.10], [0.04]], dtype=torch.float32)
    region = torch.tensor([True, True, True])
    phi = torch.tensor([1.0, 0.0, 0.0])

    monkeypatch.setattr(
        inj,
        "predict_mu_mlp_full",
        lambda *a, **k: (mu_mlp, mu_c, region, phi),
    )

    out = inj.apply_mu_map(
        _FakeData(),
        0,
        u_nd=torch.zeros(3),
        v_nd=torch.zeros(3),
        species_log=torch.zeros(3, 12),
    )
    assert float(out[0].item()) == pytest.approx(0.10, rel=1e-4)
    assert float(out[1].item()) == pytest.approx(0.04, rel=1e-4)
    assert float(out[2].item()) == pytest.approx(0.04, rel=1e-4)


def test_assemble_gt_clot_mask_carreau_bulk(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_PHI_GATE", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "gt_clot")
    phys = PhysicsConfig(phase="biochem")

    mu_c = torch.full((4, 1), 0.04)
    mu_mlp = torch.full((4, 1), 0.10)
    phi = torch.full((4,), 0.9)
    region = torch.tensor([True, True, True, False])
    mu_gt = torch.tensor([0.10, 0.04, 0.04, 0.04])
    mu_cap = cap_mu_eff_si(mu_gt)
    gate = phi_gt_binary(mu_cap, region, phys)
    assert int(gate.sum().item()) == 1

    out = assemble_committed_mu_map(
        mu_c,
        mu_mlp,
        phi,
        region=region,
        mu_gt_cap_si=mu_cap,
        phys_cfg=phys,
    )
    assert float(out[0].item()) == pytest.approx(0.10, rel=1e-4)
    assert float(out[1].item()) == pytest.approx(0.04, rel=1e-4)
    assert float(out[2].item()) == pytest.approx(0.04, rel=1e-4)
    assert float(out[3].item()) == pytest.approx(0.04, rel=1e-4)


def test_mu_map_cap_low_shear_bulk(monkeypatch):
    from src.core_physics import clot_phi_mu_inject as inj_mod

    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_BULK", "cap_low_shear")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND", "0.01")
    phys = PhysicsConfig(phase="biochem")
    monkeypatch.setattr(
        inj_mod,
        "carreau_mu_si_from_uv",
        lambda *a, **k: torch.full((2,), float(phys.mu_0)),
    )
    monkeypatch.setattr(
        inj_mod,
        "gamma_dot_nd_from_uv",
        lambda *a, **k: torch.tensor([0.001, 0.5]),
    )
    out = inj_mod.mu_map_carreau_baseline_si(object(), torch.zeros(2), torch.zeros(2), phys)
    assert float(out[0].item()) == pytest.approx(float(phys.mu_inf), rel=1e-4)
    assert float(out[1].item()) == pytest.approx(float(phys.mu_0), rel=1e-4)
