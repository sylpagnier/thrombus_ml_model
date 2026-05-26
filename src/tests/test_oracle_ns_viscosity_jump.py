"""
Step 1 — Oracle NS baseline (verify discrete physics before ML).

Zero-ML sanity: ground-truth ``(u, v, p)`` and oracle ``mu_eff`` into
:meth:`PhysicsKernels.navier_stokes_residual`. Two viscosity recipes:

1. **Legacy linear blend** (binary mask ``phi``):
       mu = mu_bulk * (1 - phi) + mu_clot * phi

2. **Multiplicative smooth-step** (target biochem formulation):
       mu_eff = mu_carreau * (1 + (mu_max_ratio - 1) * phi_clot)
   with ``phi_clot`` from GT Mat / FI sigmoids (``viscosity_mat_crit``,
   ``viscosity_fi_crit``, GNODE temperatures).

Goals:
- Momentum loss and interior residuals stay **finite** through an ~80x jump
  (default ``mu_max_ratio=80``).
- On COMSOL anchors, GT ``mu_eff`` channel should yield **low** NS loss (WLS
  discretization of the exported solution); multiplicative oracle uses the same
  GT flow fields and species triggers.

Environment:
    ORACLE_NS_VISCOSITY_RATIO — linear-blend clot/bulk ratio (default ``1e9``)
    ORACLE_NS_MAX_LOSS — optional upper cap on Huber loss (synthetic smoke)
    ORACLE_NS_MOM_NEAR_ZERO_MAX — max Huber loss for COMSOL GT-mu path (default ``120``)
    ORACLE_NS_PHI_COMBINE — ``max`` | ``product`` for Mat/FI triggers (default ``max``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig, PredChannels, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.physics_kernels import PhysicsKernels
from src.tests.test_kinematics_physics_kernels import create_physical_test_graph
from src.utils.paths import get_project_root
from src.utils.rheology import (
    multiplicative_clot_mu_eff_nd,
    phi_clot_from_mat_fi,
)


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def oracle_viscosity_ratio() -> float:
    return float(_env_float("ORACLE_NS_VISCOSITY_RATIO", 1e9) or 1e9)


def oracle_mu_max_ratio() -> float:
    return float(_env_float("ORACLE_NS_MU_MAX_RATIO", 80.0) or 80.0)


def oracle_phi_combine() -> str:
    return (os.environ.get("ORACLE_NS_PHI_COMBINE") or "max").strip().lower()


def build_oracle_mu_nd(
    phi_true: torch.Tensor,
    *,
    mu_bulk_nd: float,
    viscosity_ratio: float,
) -> torch.Tensor:
    """Oracle ND viscosity from a binary/soft clot mask (linear blend, no learned head)."""
    phi = phi_true.reshape(-1, 1).to(dtype=torch.float32).clamp(0.0, 1.0)
    mu_bulk = torch.as_tensor(mu_bulk_nd, dtype=torch.float32, device=phi.device)
    mu_clot = mu_bulk * float(viscosity_ratio)
    return mu_bulk * (1.0 - phi) + mu_clot * phi


def species_si_from_state_row(
    y_row: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SI Mat and FI from a Phase-3 state row ``[N, >=16]`` (log1p species at 4:16)."""
    if y_row.shape[1] < 16:
        raise ValueError(f"expected state channels >= 16, got {y_row.shape[1]}")
    scales = bio_cfg.get_species_scales(device=y_row.device)
    species_log = y_row[:, 4:16].to(torch.float32)
    species_nd = torch.expm1(torch.clamp(species_log, min=-10.0, max=8.0))
    species_si = species_nd * scales.to(dtype=species_nd.dtype)
    fi_si = species_si[:, 8:9]
    mat_si = species_si[:, 11:12]
    return mat_si, fi_si


def carreau_mu_nd_from_gt_velocity(
    core: PhysicsKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    data,
    props: dict,
) -> torch.Tensor:
    """Non-dimensional Carreau viscosity from GT ``u, v`` and graph WLS operators."""
    c_u = core._compute_derivatives(u.reshape(-1, 1), props)
    c_v = core._compute_derivatives(v.reshape(-1, 1), props)
    du_ij = torch.stack(
        [c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]],
        dim=1,
    )
    return core._compute_carreau_viscosity(du_ij, data)


def build_multiplicative_oracle_mu_nd(
    core: PhysicsKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    mat_si: torch.Tensor,
    fi_si: torch.Tensor,
    data,
    props: dict,
    bio_cfg: BiochemConfig,
    *,
    mu_max_ratio: Optional[float] = None,
    phi_combine: Optional[str] = None,
) -> torch.Tensor:
    """mu_eff = mu_carreau(u,v) * (1 + (mu_max_ratio - 1) * phi_clot(Mat, FI))."""
    mu_carreau = carreau_mu_nd_from_gt_velocity(core, u, v, data, props)
    phi = phi_clot_from_mat_fi(
        mat_si,
        fi_si,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
        combine=phi_combine or oracle_phi_combine(),
    )
    ratio = float(mu_max_ratio) if mu_max_ratio is not None else oracle_mu_max_ratio()
    return multiplicative_clot_mu_eff_nd(mu_carreau, phi, ratio)


def phi_true_from_gt_mu(
    mu_eff_nd: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    mu_floor_si: Optional[float] = None,
) -> torch.Tensor:
    """Binary clot mask from COMSOL ``mu_eff`` channel (GT oracle for phi)."""
    mu_si = phys_cfg.viscosity_nd_to_si(mu_eff_nd.reshape(-1))
    floor = float(mu_floor_si) if mu_floor_si is not None else max(
        float(os.environ.get("BIOCHEM_K11_CLOT_MU_SI_MIN", "0.055") or "0.055"),
        float(phys_cfg.mu_inf),
    )
    return (mu_si >= floor).to(dtype=torch.float32)


def _graph_props(core: PhysicsKernels, data) -> dict:
    props = core._get_geometric_props(data)
    num_nodes = int(data.num_nodes)
    u_ref = data.u_ref if torch.is_tensor(data.u_ref) else torch.tensor([float(data.u_ref)])
    d_bar = data.d_bar if torch.is_tensor(data.d_bar) else torch.tensor([float(data.d_bar)])
    if u_ref.numel() == 1:
        u_ref = u_ref.view(1).expand(num_nodes)
    if d_bar.numel() == 1:
        d_bar = d_bar.view(1).expand(num_nodes)
    props["u_ref"] = u_ref
    props["d_bar"] = d_bar
    return props


def interior_momentum_components(
    core: PhysicsKernels,
    pred_uvp_mu: torch.Tensor,
    data,
    props: dict,
    *,
    re_ref: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node strong-form momentum residual (same recipe as ``navier_stokes_residual``)."""
    u = pred_uvp_mu[:, PredChannels.U]
    v = pred_uvp_mu[:, PredChannels.V]
    p = pred_uvp_mu[:, PredChannels.P]

    c_u = core._compute_derivatives(u.unsqueeze(1), props)
    c_v = core._compute_derivatives(v.unsqueeze(1), props)
    c_p = core._compute_derivatives(p.unsqueeze(1), props)

    u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
    u_xx, u_yy = c_u[:, 2, 0], c_u[:, 4, 0]
    u_xy = c_u[:, 3, 0]
    v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]
    v_xx, v_yy = c_v[:, 2, 0], c_v[:, 4, 0]
    v_xy = c_v[:, 3, 0]
    p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

    Re = core.cfg.get_re(props["u_ref"], props["d_bar"])
    if re_ref is not None:
        Re = torch.as_tensor(re_ref, device=Re.device, dtype=Re.dtype).expand_as(Re)

    mu_eff = pred_uvp_mu[:, PredChannels.MU_EFF_ND]
    mu_for_grad = mu_eff.detach() if core.cfg.detach_mu_for_ns_gradient else mu_eff
    mu_for_grad = torch.clamp(mu_for_grad, min=1e-6)
    log_mu = torch.log(mu_for_grad)
    c_log_mu = core._compute_derivatives(log_mu.unsqueeze(1), props)
    log_mu_x, log_mu_y = c_log_mu[:, 0, 0], c_log_mu[:, 1, 0]
    mu_x = mu_for_grad * log_mu_x
    mu_y = mu_for_grad * log_mu_y
    max_grad = 5.0 * core.cfg.mu_viscosity_nd_scale
    mu_x = torch.clamp(mu_x, min=-max_grad, max=max_grad)
    mu_y = torch.clamp(mu_y, min=-max_grad, max=max_grad)

    visc_x = (1.0 / Re) * (2 * mu_x * u_x + mu_y * (u_y + v_x) + mu_eff * (2 * u_xx + u_yy + v_xy))
    visc_y = (1.0 / Re) * (2 * mu_y * v_y + mu_x * (u_y + v_x) + mu_eff * (2 * v_yy + v_xx + u_xy))

    if core.advect_detach:
        mom_x = (u.detach() * u_x + v.detach() * u_y) + p_x - visc_x
        mom_y = (u.detach() * v_x + v.detach() * v_y) + p_y - visc_y
    else:
        mom_x = (u * u_x + v * u_y) + p_x - visc_x
        mom_y = (u * v_x + v * v_y) + p_y - visc_y
    return mom_x, mom_y


def navier_stokes_loss_from_fields(
    core: PhysicsKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    mu_nd: torch.Tensor,
    data,
    *,
    re_ref: Optional[float] = None,
) -> torch.Tensor:
    pred = torch.cat(
        [
            u.reshape(-1, 1),
            v.reshape(-1, 1),
            p.reshape(-1, 1),
            mu_nd.reshape(-1, 1),
        ],
        dim=1,
    )
    props = _graph_props(core, data)
    return core.navier_stokes_residual(pred, data, props=props, re_ref=re_ref)


def oracle_navier_stokes_loss(
    core: PhysicsKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    phi_true: torch.Tensor,
    data,
    *,
    mu_bulk_nd: Optional[float] = None,
    viscosity_ratio: Optional[float] = None,
    re_ref: Optional[float] = None,
) -> torch.Tensor:
    """GT velocity/pressure + mask-defined linear-blend oracle viscosity -> NS loss."""
    bulk = float(mu_bulk_nd) if mu_bulk_nd is not None else float(core.mu_inf_nd)
    ratio = float(viscosity_ratio) if viscosity_ratio is not None else oracle_viscosity_ratio()
    mu = build_oracle_mu_nd(phi_true, mu_bulk_nd=bulk, viscosity_ratio=ratio)
    return navier_stokes_loss_from_fields(core, u, v, p, mu, data, re_ref=re_ref)


def multiplicative_oracle_navier_stokes_loss(
    core: PhysicsKernels,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    mat_si: torch.Tensor,
    fi_si: torch.Tensor,
    data,
    bio_cfg: BiochemConfig,
    *,
    mu_max_ratio: Optional[float] = None,
    re_ref: Optional[float] = None,
) -> torch.Tensor:
    """GT flow + Carreau baseline x multiplicative clot gate from GT Mat/FI."""
    props = _graph_props(core, data)
    mu = build_multiplicative_oracle_mu_nd(
        core, u, v, mat_si, fi_si, data, props, bio_cfg, mu_max_ratio=mu_max_ratio
    )
    return navier_stokes_loss_from_fields(core, u, v, p, mu, data, re_ref=re_ref)


def _find_biochem_graph() -> Optional[Path]:
    root = get_project_root()
    for directory in (
        root / "data/processed/graphs_biochem_anchors",
        root / "data/processed/graphs_biochem",
    ):
        if not directory.is_dir():
            continue
        candidates = sorted(p for p in directory.glob("*.pt") if p.is_file())
        if candidates:
            return candidates[0]
    return None


def _re_ref_from_data(data) -> Optional[float]:
    if hasattr(data, "re_actual") and data.re_actual is not None:
        if torch.is_tensor(data.re_actual):
            return float(data.re_actual.mean().item())
        return float(data.re_actual)
    return None


@pytest.fixture
def biochem_phys_cfg() -> PhysicsConfig:
    return PhysicsConfig(phase="biochem")


@pytest.fixture
def biochem_bio_cfg() -> BiochemConfig:
    return BiochemConfig(phase="biochem")


@pytest.fixture
def biochem_kernels(biochem_phys_cfg: PhysicsConfig) -> PhysicsKernels:
    return PhysicsKernels(biochem_phys_cfg)


def test_oracle_ns_synthetic_1e9_jump_finite(biochem_kernels: PhysicsKernels):
    """WLS channel flow + half-domain clot mask; production Huber NS must stay finite."""
    data, nodes, _ = create_physical_test_graph()
    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm**2)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    phi = (nodes[:, 0] > 0.005).float()

    loss = oracle_navier_stokes_loss(
        biochem_kernels, u, v, p, phi, data, viscosity_ratio=oracle_viscosity_ratio()
    )
    assert torch.isfinite(loss), f"Oracle NS loss not finite: {loss.item()}"
    assert float(loss.item()) >= 0.0

    props = _graph_props(biochem_kernels, data)
    mu = build_oracle_mu_nd(phi, mu_bulk_nd=float(biochem_kernels.mu_inf_nd), viscosity_ratio=oracle_viscosity_ratio())
    pred = torch.cat([u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), mu], dim=1)
    mom_x, mom_y = interior_momentum_components(biochem_kernels, pred, data, props)
    interior = biochem_kernels.fluid_interior_mask(data)
    mx = mom_x[interior]
    my = mom_y[interior]
    assert torch.isfinite(mx).all(), "Non-finite mom_x on interior (oracle synthetic)"
    assert torch.isfinite(my).all(), "Non-finite mom_y on interior (oracle synthetic)"

    cap = _env_float("ORACLE_NS_MAX_LOSS", None)
    if cap is not None:
        assert float(loss.item()) <= cap, f"Oracle NS loss {loss.item():.4e} > cap {cap}"

    log_mu = torch.log(mu.clamp(min=1e-6))
    c_log = biochem_kernels._compute_derivatives(log_mu, props)
    assert torch.isfinite(c_log).all(), "Non-finite WLS derivatives of log(mu_oracle)"


def test_oracle_ns_mse_mode_also_finite_at_1e9(biochem_phys_cfg: PhysicsConfig):
    """Document that raw MSE can be large but should not NaN at the same jump."""
    data, nodes, _ = create_physical_test_graph()
    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm**2)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    phi = (nodes[:, 0] > 0.005).float()

    mse_kernels = PhysicsKernels(biochem_phys_cfg)
    mse_kernels.momentum_loss_mode = "mse"
    loss = oracle_navier_stokes_loss(
        mse_kernels, u, v, p, phi, data, viscosity_ratio=oracle_viscosity_ratio()
    )
    assert torch.isfinite(loss), f"MSE oracle NS loss not finite: {loss.item()}"


def test_oracle_ns_multiplicative_synthetic_80x_finite(
    biochem_kernels: PhysicsKernels,
    biochem_bio_cfg: BiochemConfig,
):
    """Multiplicative smooth-step oracle: ~80x mu jump must not NaN the WLS NS residual."""
    data, nodes, _ = create_physical_test_graph()
    num_nodes = int(data.num_nodes)
    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm**2)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)

    half = num_nodes // 2
    mat_si = torch.full((num_nodes, 1), 1.0e6, dtype=torch.float32)
    fi_si = torch.full((num_nodes, 1), 0.01, dtype=torch.float32)
    mat_si[half:] = 5.0e7
    fi_si[half:] = 2.0

    props = _graph_props(biochem_kernels, data)
    mu = build_multiplicative_oracle_mu_nd(
        biochem_kernels, u, v, mat_si, fi_si, data, props, biochem_bio_cfg, mu_max_ratio=80.0
    )
    ratio = mu[half:].median() / mu[:half].median().clamp(min=1e-6)
    assert float(ratio.item()) > 10.0, f"Expected large mu jump across clot band, got ratio ~{ratio.item():.2f}"

    loss = navier_stokes_loss_from_fields(biochem_kernels, u, v, p, mu, data)
    assert torch.isfinite(loss), f"Multiplicative oracle NS loss not finite: {loss.item()}"

    pred = torch.cat([u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), mu], dim=1)
    mom_x, mom_y = interior_momentum_components(biochem_kernels, pred, data, props)
    interior = biochem_kernels.fluid_interior_mask(data)
    assert torch.isfinite(mom_x[interior]).all()
    assert torch.isfinite(mom_y[interior]).all()

    log_mu = torch.log(mu.clamp(min=1e-6))
    c_log = biochem_kernels._compute_derivatives(log_mu, props)
    assert torch.isfinite(c_log).all(), "Non-finite WLS derivatives of log(mu_multiplicative_oracle)"


def test_oracle_ns_comsol_anchor_gt_fields(biochem_kernels: PhysicsKernels, biochem_phys_cfg: PhysicsConfig):
    """Linear-blend oracle on an extracted Phase-3 graph when available."""
    graph_path = _find_biochem_graph()
    if graph_path is None:
        pytest.skip("No extracted biochem graph under data/processed/graphs_biochem*.")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not hasattr(data, "y") or data.y.dim() != 3 or data.y.shape[-1] <= STATE_CHANNEL_MU_EFF_ND:
        pytest.skip(f"{graph_path.name}: missing trajectory y[T,N,C] with mu channel.")

    t_idx = int(data.y.shape[0] // 2)
    y1 = data.y[t_idx].detach()
    u, v, p = y1[:, 0], y1[:, 1], y1[:, 2]
    phi = phi_true_from_gt_mu(y1[:, STATE_CHANNEL_MU_EFF_ND], biochem_phys_cfg)
    if float(phi.sum()) < 1.0:
        pytest.skip(f"{graph_path.name} t={t_idx}: no GT clot nodes above mu floor for phi_true.")

    re_ref = _re_ref_from_data(data)
    loss = oracle_navier_stokes_loss(
        biochem_kernels,
        u,
        v,
        p,
        phi,
        data,
        viscosity_ratio=oracle_viscosity_ratio(),
        re_ref=re_ref,
    )
    assert torch.isfinite(loss), (
        f"Oracle NS loss not finite on {graph_path.name} @ t={t_idx}: {loss.item()}"
    )

    props = _graph_props(biochem_kernels, data)
    mu = build_oracle_mu_nd(
        phi, mu_bulk_nd=float(biochem_kernels.mu_inf_nd), viscosity_ratio=oracle_viscosity_ratio()
    )
    pred = torch.cat([u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), mu], dim=1)
    mom_x, mom_y = interior_momentum_components(
        biochem_kernels, pred, data, props, re_ref=re_ref
    )
    interior = biochem_kernels.fluid_interior_mask(data)
    assert torch.isfinite(mom_x[interior]).all()
    assert torch.isfinite(mom_y[interior]).all()


def test_oracle_ns_comsol_multiplicative_smooth_step_gt(
    biochem_kernels: PhysicsKernels,
    biochem_phys_cfg: PhysicsConfig,
    biochem_bio_cfg: BiochemConfig,
):
    """GT u,v,p + mu_carreau x (1 + 79*phi_clot(Mat,FI)); NS must stay finite (near-zero with GT mu)."""
    graph_path = _find_biochem_graph()
    if graph_path is None:
        pytest.skip("No extracted biochem graph under data/processed/graphs_biochem*.")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not hasattr(data, "y") or data.y.dim() != 3 or data.y.shape[-1] < 16:
        pytest.skip(f"{graph_path.name}: missing trajectory y[T,N,>=16].")

    t_idx = int(data.y.shape[0] // 2)
    y1 = data.y[t_idx].detach()
    u, v, p = y1[:, 0], y1[:, 1], y1[:, 2]
    mat_si, fi_si = species_si_from_state_row(y1, biochem_bio_cfg)
    phi = phi_clot_from_mat_fi(
        mat_si,
        fi_si,
        mat_crit=biochem_bio_cfg.viscosity_mat_crit,
        fi_crit=biochem_bio_cfg.viscosity_fi_crit,
        temp_mat=biochem_bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=biochem_bio_cfg.viscosity_gnode_temp_fi,
        combine=oracle_phi_combine(),
    )
    if float(phi.max()) < 0.05:
        pytest.skip(f"{graph_path.name} t={t_idx}: phi_clot never activates on GT Mat/FI.")

    re_ref = _re_ref_from_data(data)

    loss_mult = multiplicative_oracle_navier_stokes_loss(
        biochem_kernels, u, v, p, mat_si, fi_si, data, biochem_bio_cfg, re_ref=re_ref
    )
    assert torch.isfinite(loss_mult), (
        f"Multiplicative oracle NS not finite on {graph_path.name} @ t={t_idx}: {loss_mult.item()}"
    )

    mu_gt = y1[:, STATE_CHANNEL_MU_EFF_ND : STATE_CHANNEL_MU_EFF_ND + 1]
    loss_gt_mu = navier_stokes_loss_from_fields(
        biochem_kernels, u, v, p, mu_gt, data, re_ref=re_ref
    )
    assert torch.isfinite(loss_gt_mu), f"GT-mu NS loss not finite: {loss_gt_mu.item()}"

    near_zero_cap = float(_env_float("ORACLE_NS_MOM_NEAR_ZERO_MAX", 120.0) or 120.0)
    assert float(loss_gt_mu.item()) <= near_zero_cap, (
        f"GT mu_eff + GT (u,v,p) should yield low NS Huber on WLS graph "
        f"(got {loss_gt_mu.item():.4e}, cap {near_zero_cap}); "
        "increase momentum_huber_delta or smooth GT fields if this fails."
    )

    props = _graph_props(biochem_kernels, data)
    mu_mult = build_multiplicative_oracle_mu_nd(
        biochem_kernels, u, v, mat_si, fi_si, data, props, biochem_bio_cfg
    )
    pred = torch.cat([u.unsqueeze(1), v.unsqueeze(1), p.unsqueeze(1), mu_mult], dim=1)
    mom_x, mom_y = interior_momentum_components(
        biochem_kernels, pred, data, props, re_ref=re_ref
    )
    interior = biochem_kernels.fluid_interior_mask(data)
    assert torch.isfinite(mom_x[interior]).all()
    assert torch.isfinite(mom_y[interior]).all()
