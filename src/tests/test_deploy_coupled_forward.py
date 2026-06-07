"""Unit tests for deploy coupled forward (soft mlp_band + promote score)."""

from __future__ import annotations

import torch

from src.config import PhysicsConfig
from src.training.deploy_coupled_forward import (
    compute_deploy_coupled_step_losses,
    deploy_coupled_promote_score,
    resolve_supervise_clot_mask,
    soft_committed_mu_si,
    soft_mlp_band_gate,
)


def test_soft_mlp_band_gate_high_phi_mu_near_one() -> None:
    allowed = torch.ones(4, dtype=torch.bool)
    phi = torch.tensor([0.9, 0.9, 0.1, 0.9])
    mu = torch.tensor([0.06, 0.04, 0.06, 0.06])
    g = soft_mlp_band_gate(allowed, phi, mu, phi_thresh=0.5, mu_thresh_si=0.055)
    assert float(g[0]) > float(g[1])
    assert float(g[0]) > 0.5
    assert float(g[2]) < 0.5


def test_soft_committed_mu_blends_toward_mlp() -> None:
    bulk = torch.tensor([0.01, 0.01])
    mlp = torch.tensor([0.06, 0.06])
    gate = torch.tensor([1.0, 0.0])
    out = soft_committed_mu_si(bulk, mlp, gate, blend=1.0)
    assert abs(float(out[0]) - 0.06) < 1e-5
    assert abs(float(out[1]) - 0.01) < 1e-5


def test_deploy_coupled_losses_backward() -> None:
    phys = PhysicsConfig(phase="biochem")
    n = 8
    allowed = torch.ones(n, dtype=torch.bool)
    phi = torch.sigmoid(torch.randn(n, requires_grad=True))
    mu_raw = torch.randn(n, requires_grad=True)
    mu_mlp = (0.02 + 0.05 * torch.sigmoid(mu_raw)).clamp(min=1e-4)
    mu_bulk = torch.full((n,), 0.01)
    mu_gt = torch.full((n,), 0.06)
    phi_gt = torch.zeros(n)
    gt_clot = allowed.clone()
    loss, terms = compute_deploy_coupled_step_losses(
        phi_pred=phi,
        mu_mlp=mu_mlp,
        mu_bulk=mu_bulk,
        mu_gt_cap=mu_gt,
        phi_gt=phi_gt,
        allowed=allowed,
        gt_clot=gt_clot,
        phys_cfg=phys,
        phi_thresh=0.5,
        mu_log_lambda=1.0,
        hinge_lambda=1.0,
        allowed_hinge_lambda=1.0,
        soft_commit_lambda=1.0,
        phi_lambda=0.0,
        pos_weight=1.0,
        logits=None,
    )
    loss.backward()
    assert loss.item() > 0.0
    assert mu_raw.grad is not None
    assert float(terms["allowed_hinge"].item()) >= 0.0


def test_supervise_mask_empty_when_no_clot_signal() -> None:
    allowed = torch.ones(6, dtype=torch.bool)
    gt_clot = torch.zeros(6, dtype=torch.bool)
    phi_gt = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
    sup = resolve_supervise_clot_mask(
        gt_clot=gt_clot, phi_gt=phi_gt, allowed=allowed, phi_thresh=0.5
    )
    assert not bool(sup.any().item())


def test_allowed_hinge_grad_without_gt_clot() -> None:
    phys = PhysicsConfig(phase="biochem")
    n = 8
    allowed = torch.ones(n, dtype=torch.bool)
    phi = torch.full((n,), 0.9, requires_grad=False)
    mu_raw = torch.randn(n, requires_grad=True)
    mu_mlp = (0.02 + 0.02 * torch.sigmoid(mu_raw)).clamp(min=1e-4)
    mu_bulk = torch.full((n,), 0.01)
    mu_gt = torch.full((n,), 0.01)
    phi_gt = torch.zeros(n)
    gt_clot = torch.zeros(n, dtype=torch.bool)
    loss, _ = compute_deploy_coupled_step_losses(
        phi_pred=phi,
        mu_mlp=mu_mlp,
        mu_bulk=mu_bulk,
        mu_gt_cap=mu_gt,
        phi_gt=phi_gt,
        allowed=allowed,
        gt_clot=gt_clot,
        phys_cfg=phys,
        phi_thresh=0.5,
        mu_log_lambda=0.0,
        hinge_lambda=0.0,
        allowed_hinge_lambda=5.0,
        soft_commit_lambda=0.0,
        phi_lambda=0.0,
        pos_weight=1.0,
        logits=None,
    )
    loss.backward()
    assert mu_raw.grad is not None


def test_promote_score_prefers_deploy_metrics() -> None:
    low = deploy_coupled_promote_score(
        {
            "gate_frac_mu": 0.0,
            "frac_both_allowed": 0.01,
            "frac_mu_ok_allowed": 0.01,
            "frac_rollout_mu_ok_allowed": 0.0,
            "mu_mlp_p90_allowed": 0.03,
            "allowed_hinge": 0.1,
            "mu_log_mae": 0.5,
        }
    )
    high = deploy_coupled_promote_score(
        {
            "gate_frac_mu": 0.02,
            "frac_both_allowed": 0.15,
            "frac_mu_ok_allowed": 0.20,
            "frac_rollout_mu_ok_allowed": 0.10,
            "gate_mu_p90": 0.06,
            "allowed_hinge": 0.02,
            "mu_log_mae": 0.3,
        }
    )
    assert high > low
