"""Stage A: ADR mass conservation checks (reaction sources, fibrin pair, soft-step gradients)."""

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels, SoftStepSTE
from src.core_physics.physics_kernels import PhysicsKernels


def test_fibrin_pair_conserves_mass_in_rate_equations():
    """``R_FG + R_FI`` vanishes for the fibrinogen→fibrin branch (no artificial creation)."""
    cfg = BiochemConfig(tier="tier3")
    core = PhysicsKernels(PhysicsConfig(tier="tier3"))
    bio = BiochemPhysicsKernels(cfg, core)
    kin = bio.kinetics

    T = torch.tensor([1e-7, 2e-6], dtype=torch.float32)
    FG = torch.tensor([5e-3, 1e-3], dtype=torch.float32)
    FI = torch.tensor([1e-5, 2e-4], dtype=torch.float32)
    R_FG, R_FI = kin.compute_fibrin_kinetics(T, FG, FI)
    pair_sum = R_FG + R_FI
    assert torch.allclose(pair_sum, torch.zeros_like(pair_sum), atol=1e-9, rtol=0.0)


def test_bulk_reaction_sources_balance_platelet_transfer():
    """RP drain equals AP source: ``R_RP + R_AP == 0`` for the activation channel."""
    cfg = BiochemConfig(tier="tier3")
    core = PhysicsKernels(PhysicsConfig(tier="tier3"))
    bio = BiochemPhysicsKernels(cfg, core)
    kin = bio.kinetics

    species_dict = {
        "RP": torch.tensor(1e-6),
        "AP": torch.tensor(0.5e-6),
        "APR": torch.tensor(1e-9),
        "APS": torch.tensor(1e-9),
        "PT": torch.tensor(1e-6),
        "T": torch.tensor(1e-8),
        "AT": torch.tensor(1e-6),
        "FG": torch.tensor(5e-3),
        "FI": torch.tensor(1e-6),
    }
    shear = torch.tensor(5000.0)
    R = kin.compute_species_reactions(species_dict, shear)
    rp_ap = R["RP"] + R["AP"]
    assert torch.isfinite(rp_ap)
    assert abs(float(rp_ap.item())) < 1e-18


def test_adr_residual_finite_under_static_flow_stub():
    """With zero velocity and uniform species, ADR residuals stay finite (sanity for Stage A)."""
    cfg = BiochemConfig(tier="tier3")
    core = PhysicsKernels(PhysicsConfig(tier="tier3"))
    bio = BiochemPhysicsKernels(cfg, core)

    n = 8
    species_preds = torch.full((n, 9), -2.0, dtype=torch.float32)
    velocity_field = torch.zeros(n, 2, dtype=torch.float32)
    spatial_props = {"u_ref": torch.ones(n) * 0.1, "d_bar": torch.ones(n) * 0.01}

    idx = torch.stack([torch.arange(n), torch.arange(n)])
    val = torch.ones(n, dtype=torch.float32)
    eye = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    class _Stub:
        G_x = eye
        G_y = eye
        Laplacian = eye

    data = _Stub()
    lf, ls = bio.biochem_adr_residual(species_preds, velocity_field, spatial_props, data, d_pred_dt=None)
    assert torch.isfinite(lf) and torch.isfinite(ls)


def test_soft_step_ste_backward_is_smooth_near_threshold():
    """STE backward uses sigmoid; gradients stay finite around the threshold (DEQ root-finding)."""
    x = torch.linspace(-0.5, 0.5, steps=32, requires_grad=True)
    y = SoftStepSTE.apply(x, 0.0, 0.1, 1.0)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_kinetics_soft_steps_finite_gradients():
    """Temperature-scaled soft gates yield finite grads w.r.t. inputs."""
    cfg = BiochemConfig(tier="tier3")
    core = PhysicsKernels(PhysicsConfig(tier="tier3"))
    bio = BiochemPhysicsKernels(cfg, core)
    kin = bio.kinetics

    omega = torch.tensor([0.3, 1.2, 600.0], requires_grad=True)
    shear = torch.tensor([100.0, 8000.0, 12000.0], requires_grad=True)
    kpa = kin.compute_k_pa(omega, shear)
    kpa.sum().backward()
    assert torch.isfinite(omega.grad).all() and torch.isfinite(shear.grad).all()
