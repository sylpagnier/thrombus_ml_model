"""Smoke tests for clot trigger stack forward."""

from __future__ import annotations

import os

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import build_clot_phi_model, build_clot_phi_step, clot_phi_feature_dim
from src.training.clot_ml_step0_coef import discover_anchor_paths
from src.core_physics.clot_growth_masks import (
    gt_growth_commit_mask_at_time,
    growth_seed_mode,
)
from src.core_physics.clot_phi_simple import clot_phi_loss_scope
from src.training.clot_trigger_stack import (
    apply_clot_trigger_deploy_env,
    apply_clot_trigger_honest_env,
    apply_clot_trigger_nucleation_env,
    apply_clot_trigger_oracle_debug_env,
    apply_clot_trigger_oracle_forward_env,
    apply_deploy_nucleation_mask_env,
    apply_oracle_neighbor_mask_env,
    apply_star1_train_env,
    apply_star2_eval_env,
    apply_star6_coupled_env,
    clot_phi_trigger_rollout_enabled,
    deploy_env_is_faithful,
    forward_clot_trigger_hybrid,
    forward_physics_trigger_phi,
    reset_star2_kinematics_cache,
    reset_star6_caches,
)
from src.core_physics.clot_trigger_rollout import (
    clot_trigger_nucleation_enabled,
    forward_path_uses_gt_commits,
    lumen_false_positive_frac,
    rollout_clot_trigger_hybrid_trainable,
    rollout_clot_trigger_physics,
)
from src.utils.paths import get_project_root


@pytest.fixture
def anchor_graph():
    paths = discover_anchor_paths(get_project_root() / "data/processed/graphs_biochem_anchors")
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(paths[0], map_location="cpu", weights_only=False)


def test_physics_trigger_forward(anchor_graph) -> None:
    apply_star1_train_env(fast=True)
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    phi, mu = forward_physics_trigger_phi(
        step, anchor_graph, phys_cfg=phys, bio_cfg=bio, device=device
    )
    assert phi.shape[0] == anchor_graph.num_nodes
    assert float(phi.min()) >= 0.0
    assert float(mu.min()) > 0.0


def test_hybrid_trigger_forward(anchor_graph) -> None:
    apply_star1_train_env(fast=True)
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    in_dim = clot_phi_feature_dim()
    model = build_clot_phi_model(in_dim=in_dim, hidden=16)
    model.eval()
    step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    out = forward_clot_trigger_hybrid(
        model, step, anchor_graph, phys_cfg=phys, bio_cfg=bio, device=device
    )
    assert "phi_hybrid" in out
    assert out["phi_hybrid"].shape[0] == anchor_graph.num_nodes


def test_star2_pred_kine_step(anchor_graph, monkeypatch) -> None:
    apply_star2_eval_env(kine_ckpt="outputs/kinematics/kinematics_best.pth")
    reset_star2_kinematics_cache()
    root = get_project_root()
    kine = root / "outputs/kinematics/kinematics_best.pth"
    if not kine.is_file():
        pytest.skip("missing kinematics_best.pth")
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step_gt = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    step_pred = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    u_gt = anchor_graph.y[0, :, 0]
    u_diff = (step_pred.u_flow_nd.cpu() - u_gt).abs().max().item()
    assert u_diff > 1e-6, "pred kine should differ from GT u on anchor"
    assert step_pred.features.shape == step_gt.features.shape


def test_star6_coupled_flow_feedback(anchor_graph, monkeypatch) -> None:
    """Coupled rollout should change [u,v] after phi/mu commit (vs frozen steady kine)."""
    root = get_project_root()
    kine = root / "outputs/kinematics/kinematics_best.pth"
    if not kine.is_file():
        pytest.skip("missing kinematics_best.pth")
    apply_star6_coupled_env(kine_ckpt=str(kine.relative_to(root)), species_live=False)
    reset_star6_caches()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    in_dim = clot_phi_feature_dim()
    model = build_clot_phi_model(in_dim=in_dim, hidden=16)
    model.eval()
    from src.training.clot_trigger_stack import (
        advance_coupled_trigger_state,
        build_clot_trigger_coupled_step,
        init_coupled_trigger_rollout,
    )

    rollout_state, kine_provider = init_coupled_trigger_rollout(anchor_graph, device=device)
    step0 = build_clot_trigger_coupled_step(
        anchor_graph,
        0,
        pred_species_series=None,
        rollout_state=rollout_state,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
    )
    u0 = step0.u_flow_nd.clone()
    out0 = forward_clot_trigger_hybrid(
        model, step0, anchor_graph, phys_cfg=phys, bio_cfg=bio, device=device
    )
    advance_coupled_trigger_state(
        anchor_graph,
        out0["phi_hybrid"],
        out0["mu_hybrid"],
        rollout_state=rollout_state,
        kine_provider=kine_provider,
        detach=True,
    )
    step1 = build_clot_trigger_coupled_step(
        anchor_graph,
        1,
        pred_species_series=None,
        rollout_state=rollout_state,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
    )
    u_diff = (step1.u_flow_nd - u0).abs().max().item()
    assert u_diff > 1e-8, "coupled kine should update flow after phi/mu commit"


def test_deploy_ceiling_loss_mask(anchor_graph) -> None:
    apply_clot_trigger_deploy_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    assert clot_phi_loss_scope() == "ceiling"
    assert bool(step.loss_mask.any().item())
    assert not bool(step.loss_mask.all().item())
    assert torch.equal(step.loss_mask, step.region)


def test_deploy_nucleation_mask_differs_from_oracle(anchor_graph) -> None:
    """Nucleation-band loss mask should differ from legacy oracle band."""
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    apply_oracle_neighbor_mask_env()
    step_oracle = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    apply_deploy_nucleation_mask_env()
    step_deploy = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    assert step_oracle.loss_mask.shape == step_deploy.loss_mask.shape
    assert not torch.equal(step_oracle.loss_mask, step_deploy.loss_mask)


def test_star3_species_override_changes_features(anchor_graph) -> None:
    apply_star1_train_env(fast=True)
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step_gt = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    fake_sp = torch.zeros_like(anchor_graph.y[0, :, 4:16]) + 0.5
    step_pred = build_clot_phi_step(
        anchor_graph, 0, phys, bio, device, species_log_override=fake_sp
    )
    assert not torch.allclose(step_gt.features, step_pred.features)
    assert step_gt.phi_gt.shape == step_pred.phi_gt.shape


def test_honest_env_enables_nucleation_projection() -> None:
    apply_clot_trigger_honest_env()
    assert clot_trigger_nucleation_enabled()
    assert not forward_path_uses_gt_commits()
    assert growth_seed_mode() == "pred"
    assert deploy_env_is_faithful()


def test_oracle_debug_env_uses_gt_growth_seed() -> None:
    apply_clot_trigger_oracle_debug_env()
    assert growth_seed_mode() == "gt"
    assert forward_path_uses_gt_commits()
    assert not deploy_env_is_faithful()


def test_oracle_forward_uses_gt_commits() -> None:
    apply_clot_trigger_oracle_forward_env()
    assert forward_path_uses_gt_commits()


def test_nucleation_rollout_reduces_lumen_fp_vs_raw(anchor_graph) -> None:
    apply_clot_trigger_honest_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    traj = rollout_clot_trigger_physics(
        anchor_graph, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1
    )
    assert traj, "rollout should produce at least one step"
    t0 = traj[0]
    assert t0["phi_raw"].shape == t0["phi"].shape
    step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    fp_raw = lumen_false_positive_frac(
        t0["phi_raw"], step.phi_gt, data=anchor_graph, device=device
    )
    fp_proj = lumen_false_positive_frac(
        t0["phi"], step.phi_gt, data=anchor_graph, device=device
    )
    assert fp_proj <= fp_raw + 1e-9


def test_gt_growth_labels_zero_at_t0(anchor_graph) -> None:
    apply_clot_trigger_honest_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
    assert float(step.phi_gt.max().item()) <= 0.0 + 1e-6


def test_rollout_ic_phi_zero_at_t0(anchor_graph) -> None:
    apply_clot_trigger_honest_env()
    from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env

    apply_physics_trigger_baseline_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    traj = rollout_clot_trigger_physics(
        anchor_graph, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1
    )
    phi0 = traj[0]["phi"]
    assert float(phi0.max().item()) <= 0.0 + 1e-6
    assert float(phi0.sum().item()) <= 0.0 + 1e-6


def test_gt_clot_phi_excludes_t0_baseline_without_growth(anchor_graph) -> None:
    """Absolute mu threshold can count t=0 baseline nodes; growth mask must not."""
    from src.core_physics.clot_phi_simple import cap_mu_eff_si, clot_phi_thresh_si

    apply_clot_trigger_deploy_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    tf = int(anchor_graph.y.shape[0]) - 1
    y = anchor_graph.y[tf]
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]))
    abs_clot = int((mu_cap >= clot_phi_thresh_si(phys)).sum().item())
    growth_clot = int(
        gt_growth_commit_mask_at_time(anchor_graph, tf, phys, device).sum().item()
    )
    assert growth_clot <= abs_clot
    assert int(gt_growth_commit_mask_at_time(anchor_graph, 0, phys, device).sum().item()) == 0


def test_gt_growth_commits_zero_at_t0_and_early(anchor_graph) -> None:
    apply_clot_trigger_deploy_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    for t in (0, 17):
        mask = gt_growth_commit_mask_at_time(anchor_graph, t, phys, device)
        assert int(mask.sum().item()) == 0


def test_gt_clot_phi_at_time_subtracts_t0_baseline(anchor_graph) -> None:
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    apply_clot_trigger_deploy_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    for t in (0, 17):
        phi = gt_clot_phi_at_time(anchor_graph, t, phys, device=device)
        assert float(phi.sum().item()) <= 0.0 + 1e-6


def test_star1_enables_trigger_rollout_training() -> None:
    apply_star1_train_env(fast=True)
    assert clot_phi_trigger_rollout_enabled()


def test_trigger_rollout_trainable_smoke(anchor_graph) -> None:
    apply_star1_train_env(fast=True)
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    in_dim = clot_phi_feature_dim()
    model = build_clot_phi_model(in_dim=in_dim, hidden=16)
    model.train()
    traj = rollout_clot_trigger_hybrid_trainable(
        model,
        anchor_graph,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        time_stride=4,
    )
    assert traj
    assert any(float(v["phi"].max().item()) >= 0.0 for v in traj.values())


def test_nucleation_projection_zeros_ineligible_at_t0(anchor_graph) -> None:
    apply_clot_trigger_nucleation_env()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n = int(anchor_graph.num_nodes)
    from src.core_physics.clot_nucleation_mask import (
        project_phi_with_nucleation,
        resolve_nucleation_eligibility,
    )

    elig = resolve_nucleation_eligibility(
        anchor_graph, 0, device, phys, bio, growth_seed="pred", phi_pred_by_time={}
    )
    raw = torch.ones(n, device=device)
    out = project_phi_with_nucleation(raw, None, elig, commit_thresh=0.5)
    assert float(out[~elig].max().item()) <= 0.0 + 1e-6
