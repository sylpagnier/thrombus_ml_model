"""Rollout health catches frozen wall-ring predictions."""

from __future__ import annotations

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.evaluation import rung4_rollout_health as rh


def test_frozen_wall_ring_detected(monkeypatch):
    device = torch.device("cpu")
    n = 100
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:20] = True
    phi_traj: dict[int, torch.Tensor] = {}
    phi_ring = torch.zeros(n)
    phi_ring[wall] = 0.9
    for t in [0, 10, 20]:
        phi_traj[t] = phi_ring.clone()

    class _Data:
        num_nodes = n
        y = torch.zeros(21, n, 16)

    data = _Data()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    monkeypatch.setattr(rh, "gt_clot_phi_at_time", lambda *a, **k: torch.zeros(n))
    monkeypatch.setattr(rh, "_wall_mask_from_data", lambda *a, **k: wall)
    monkeypatch.setattr(rh, "macro_tau_at_index", lambda data, t, bio_cfg=None: float(t) / 20.0)

    health = rh.compute_rung4_rollout_health(phi_traj, data, phys, bio, device, times=[0, 20])
    assert health["frozen_wall_ring"] is True
    assert health["health_pass"] is False
    assert health["early_phi_wall_max"] > 0.15


def test_wall_carpet_detected(monkeypatch):
    device = torch.device("cpu")
    n = 200
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:40] = True
    gt = torch.zeros(n)
    gt[:10] = 1.0  # GT clot on small wall-adjacent subset

    phi_pred = torch.zeros(n)
    phi_pred[wall] = 1.0  # predict entire wall

    phi_traj = {0: phi_pred.clone(), 50: phi_pred.clone()}

    class _Data:
        num_nodes = n
        y = torch.zeros(51, n, 16)

    monkeypatch.setattr(rh, "gt_clot_phi_at_time", lambda data, t, phys, device: gt)
    monkeypatch.setattr(rh, "_wall_mask_from_data", lambda *a, **k: wall)
    monkeypatch.setattr(rh, "macro_tau_at_index", lambda data, t, bio_cfg=None: 1.0)

    health = rh.compute_rung4_rollout_health(
        phi_traj, _Data(), PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem"), device, times=[50],
    )
    assert health["wall_carpet"] is True
    assert health["health_pass"] is False
