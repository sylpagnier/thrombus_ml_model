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


def test_k11_wall_prox_apply_includes_wall_node(monkeypatch):
    monkeypatch.setenv("BIOCHEM_K11_APPLY_MODE", "wall_prox")
    sdf = torch.tensor([0.0, 0.008, 0.05], dtype=torch.float32)
    wall = torch.tensor([True, False, False])
    m = gb.k11_clot_apply_mask(sdf, wall).reshape(-1)
    assert float(m[0].item()) > 0.9
    assert float(m[1].item()) > 0.2


def test_k11_policy_snapshot(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_K11_CLOT_GATE", "1")
    monkeypatch.delenv("BIOCHEM_MU_K10E_SIMPLE", raising=False)
    fp = gb.snapshot_biochem_forward_policy()
    assert fp["mu_k11_clot_gate"] is True
    assert not gb._biochem_mu_k10e_simple_enabled()


def test_k11_growth_off_by_default(monkeypatch):
    monkeypatch.delenv("BIOCHEM_K11_CLOT_GROWTH", raising=False)
    assert not gb._biochem_env_truthy("BIOCHEM_K11_CLOT_GROWTH", default=False)


def test_k11_bio_trigger_high_fi_mat(monkeypatch):
    class _Stub:
        fi_crit = 0.6
        mat_crit = 2e7

        def species_log_nd_to_si(self, species_log):
            return torch.expm1(species_log.clamp(-10, 8))

    n = 2
    species_log = torch.zeros(n, 12)
    species_log[:, 8] = 8.0
    species_log[:, 11] = 10.0
    bio = gb.k11_bio_trigger_score(_Stub(), species_log)
    assert float(bio.max().item()) > 0.4
    assert float(bio.min().item()) >= 0.0


def test_k11_adjacent_apply_excludes_wall_node(monkeypatch):
    monkeypatch.setenv("BIOCHEM_K11_APPLY_MODE", "adjacent")
    sdf = torch.tensor([0.0, 0.008, 0.05], dtype=torch.float32)
    wall = torch.tensor([True, False, False])
    m = gb.k11_clot_apply_mask(sdf, wall).reshape(-1)
    assert float(m[0].item()) == 0.0
    assert float(m[1].item()) > 0.2


def test_k11_gt_label_not_degenerate(monkeypatch):
    from src.config import PhysicsConfig
    from src.training.train_biochem_corrector import _k11_clot_gt_label

    cfg = PhysicsConfig()
    mu = torch.tensor([0.004, 0.040, 0.060, 0.100], dtype=torch.float32)
    y = _k11_clot_gt_label(mu, cfg)
    assert int(y.sum().item()) == 2


def test_k11_trigger_apply_env_default_off_in_k11e(monkeypatch):
    monkeypatch.setenv("BIOCHEM_K11_TRIGGER_APPLY", "0")
    assert not gb._k11_trigger_apply_enabled()


def test_k11_bce_support_adjacent_trigger(monkeypatch):
    from src.training.train_biochem_corrector import _k11_bce_node_mask

    monkeypatch.setenv("BIOCHEM_K11_APPLY_MODE", "adjacent")
    monkeypatch.setenv("BIOCHEM_K11_BCE_SUPPORT", "adjacent_trigger")
    monkeypatch.setenv("BIOCHEM_K11_BCE_TRIG_THRESH", "0.5")
    n = 4
    data = type("G", (), {})()
    data.x = torch.tensor([[0, 0, 0.0], [0, 0, 0.008], [0, 0, 0.02], [0, 0, 0.05]], dtype=torch.float32)
    data.mask_wall = torch.tensor([True, False, False, False])
    truth = torch.ones(n, dtype=torch.bool)
    y_clot = torch.tensor([0.0, 1.0, 0.0, 0.0])
    trig = torch.tensor([0.0, 0.1, 0.6, 0.0])
    m = _k11_bce_node_mask(data, truth, y_clot, trig, torch.device("cpu"))
    assert m[0].item() is False
    assert m[1].item() is True
    assert m[2].item() is True
    assert m[3].item() is False


def test_k11_forward_policy_snapshot_trigger_apply(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_K11_CLOT_GATE", "1")
    monkeypatch.setenv("BIOCHEM_K11_TRIGGER_APPLY", "0")
    monkeypatch.setenv("BIOCHEM_K11_APPLY_MODE", "adjacent")
    fp = gb.snapshot_biochem_forward_policy()
    assert fp["k11_trigger_apply"] is False
    assert fp["k11_apply_mode"] == "adjacent"
