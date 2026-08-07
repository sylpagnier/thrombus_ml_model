"""Deploy-legal prior fields for the RGP-DEQ input block (WALL_MODEL_PLAN.md s16.1, s17 Z1/Z2).

**Why this module exists.** `data.x[:, UV_PRIOR|MU_PRIOR]` as stored in the anchor packs are
bit-identical to the converged clot-free CFD solution `y[0]` -- they contain backflow, which the
clamped parabolic magnitude in `build_poiseuille_priors` cannot produce, and `wss_prior_nd` is
identically zero because it has no `y[0]` counterpart to be overwritten with (s16.1).

The RGP-DEQ consumes those columns as *inputs* (`ginodeq.py:438-440`), so the flow surrogate is
handed the field it exists to predict.

**The deployment contract (s17 Z2, decided):** at deploy we are given **geometry + initial and
boundary conditions only**. No clot-free CFD solve is available. The stored priors are therefore
not legal inputs, and anything trained on them is trained on information that will not exist.

This module supplies legal replacements computable from `(sdf_nd, width_nd, mask_inlet,
mask_outlet, edge_index)` alone.
"""
from __future__ import annotations

import torch

from src.config import PhysicsConfig
from src.data_gen.lib.node_feature_assembly import (
    mass_conserving_umax_nd,
    width_nd_to_radius_nd,
)

# data.x column layout (kine_x_v1_18ch)
COL_XY = slice(0, 2)
COL_SDF = 2
COL_WALL_NORMAL = slice(4, 6)
COL_U_PRIOR = 11
COL_V_PRIOR = 12
COL_MU_PRIOR = 13
COL_WSS_PRIOR = 14
COL_WIDTH = 15

PRIOR_SOURCES = ("stored", "analytic", "zero")


def potential_flow_direction(
    data, *, iters: int = 3000, tol: float = 1e-8, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit flow direction from a graph Laplace solve with inlet/outlet Dirichlet BCs.

    Potential-flow approximation: solve ``div(grad phi) = 0`` with ``phi=1`` on the inlet and
    ``phi=0`` on the outlet, then take the normalised ``-grad phi`` as the streamwise direction.
    Uses only geometry and the boundary masks, so it is legal under the s17 Z2 contract.

    Solved by conjugate gradient on the free nodes. **Jacobi is not viable here**: these meshes
    run ~274 hops inlet-to-outlet, so Jacobi needs O(diameter^2) ~ 75k sweeps to converge, and
    an under-converged potential yields a direction field uncorrelated with the flow.

    Returns ``(dir_x, dir_y)``, each ``[N]``, unit-norm where the gradient is resolvable.
    """
    dev = device or data.x.device
    n = int(data.num_nodes)
    row, col = data.edge_index.to(dev)
    pos = data.x[:, COL_XY].to(dev, torch.float32)

    inlet = _mask(data, "mask_inlet", n, dev)
    outlet = _mask(data, "mask_outlet", n, dev)
    fixed = inlet | outlet
    free = ~fixed

    deg = torch.zeros(n, device=dev)
    deg.index_add_(0, row, torch.ones(row.shape[0], device=dev))

    def lap(v: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(n, device=dev)
        acc.index_add_(0, row, v[col])
        return deg * v - acc

    phi_fixed = torch.zeros(n, device=dev)
    phi_fixed[inlet] = 1.0
    # Solve L[free,free] x = -L[free,fixed] phi_fixed  by CG on the free block.
    b = (-lap(phi_fixed))[free]
    x = torch.zeros(int(free.sum()), device=dev)

    def A(v: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(n, device=dev)
        full[free] = v
        return lap(full)[free]

    r = b - A(x)
    p = r.clone()
    rs = torch.dot(r, r)
    for _ in range(int(iters)):
        if rs.sqrt() < tol:
            break
        Ap = A(p)
        alpha = rs / torch.dot(p, Ap).clamp(min=1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        p = r + (rs_new / rs.clamp(min=1e-30)) * p
        rs = rs_new

    phi = phi_fixed.clone()
    phi[free] = x

    gx, gy = _lsq_gradient(phi, pos, row, col, n, dev)
    # Flow runs down the potential gradient.
    dx, dy = -gx, -gy
    mag = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-9)
    return dx / mag, dy / mag


def _mask(data, name: str, n: int, dev) -> torch.Tensor:
    m = getattr(data, name, None)
    if m is None:
        return torch.zeros(n, dtype=torch.bool, device=dev)
    return m.reshape(-1).to(dev).bool()


def _lsq_gradient(f, pos, row, col, n, dev):
    """Per-node least-squares gradient of a scalar field over graph edges."""
    dv = pos[col] - pos[row]
    df = f[col] - f[row]
    w = 1.0 / dv.norm(dim=1).clamp(min=1e-9) ** 2
    A = torch.zeros(n, 2, 2, device=dev)
    b = torch.zeros(n, 2, device=dev)
    for k in range(2):
        for j in range(2):
            A[:, k, j].index_add_(0, row, w * dv[:, k] * dv[:, j])
        b[:, k].index_add_(0, row, w * dv[:, k] * df)
    # Scale-aware ridge: nodes whose edge vectors are collinear (common on boundary rows)
    # give a singular 2x2, so regularise relative to the local matrix magnitude rather than
    # with a fixed epsilon, and fall back to the pseudo-inverse if anything is still singular.
    scale = A.abs().amax(dim=(1, 2)).clamp(min=1e-30).reshape(-1, 1, 1)
    A = A + torch.eye(2, device=dev).unsqueeze(0) * scale * 1e-6
    try:
        g = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    except Exception:
        g = (torch.linalg.pinv(A) @ b.unsqueeze(-1)).squeeze(-1)
    return g[:, 0], g[:, 1]


def build_analytic_priors(
    data, *, phys_cfg: PhysicsConfig | None = None, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical Poiseuille priors from geometry + BCs only. Returns (u, v, mu, wss).

    The magnitude, shear rate, Carreau viscosity and wall shear stress are pure functions of
    ``(sdf_nd, width_nd)`` -- no flow direction needed. Direction is supplied by
    :func:`potential_flow_direction` and only sets the sign/orientation of ``u``/``v``.
    """
    dev = device or data.x.device
    ph = phys_cfg or PhysicsConfig()
    x = data.x.to(dev)
    sdf = x[:, COL_SDF].reshape(-1).clamp_min(0.0)
    width = x[:, COL_WIDTH].reshape(-1)

    r_nd = width_nd_to_radius_nd(width).reshape(-1)
    u_max = mass_conserving_umax_nd(r_nd).reshape(-1)
    r_lane = (r_nd - torch.minimum(sdf, r_nd)).clamp_min(0.0)

    mag = torch.clamp(u_max * (1.0 - (r_lane**2 / (r_nd**2 + 1e-12))), min=0.0)
    gamma = torch.abs(-2.0 * u_max * r_lane / (r_nd**2 + 1e-12))

    if getattr(ph, "viscosity_model", "carreau") == "newtonian":
        mu = torch.ones_like(mag)
    else:
        ref = float(ph.mu_viscosity_nd_scale)
        u_ref = float(getattr(data, "u_ref", torch.tensor(1.0)).reshape(-1)[0])
        d_bar = float(getattr(data, "d_bar", torch.tensor(1.0)).reshape(-1)[0])
        lam_nd = ph.lam * (u_ref / max(d_bar, 1e-12))
        mu = (ph.mu_inf / ref) + ((ph.mu_0 / ref) - (ph.mu_inf / ref)) * (
            1.0 + (lam_nd * gamma) ** ph.a
        ) ** ((ph.n - 1.0) / ph.a)

    wall = _mask(data, "mask_wall", int(data.num_nodes), dev)
    wss = mu * gamma * wall.to(mu.dtype)

    dx, dy = potential_flow_direction(data, device=dev)
    return mag * dx, mag * dy, mu, wss


def resolve_prior_source(default: str = "stored") -> str:
    """Active prior source from the runtime config, else ``SPECIES_PRIOR_SOURCE``, else default."""
    import os

    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return str(rt.rollout.prior_source or default).strip().lower()
    except Exception:
        pass
    return (os.environ.get("SPECIES_PRIOR_SOURCE") or default).strip().lower()


def assert_train_deploy_prior_parity(train_source: str, deploy_source: str) -> None:
    """Fail loudly when training uses a prior block deploy will not have (s17 Z3).

    v1-v10 trained with the leaked CFD priors and deployed against a predicted field. That is a
    distribution shift sitting under every result in sections 9-13, and it was never checked.
    """
    t, d = (train_source or "").strip().lower(), (deploy_source or "").strip().lower()
    if t == d:
        return
    raise ValueError(
        f"prior_source mismatch: training uses {t!r} but deploy uses {d!r}. "
        "Under the s17 Z2 contract the model must never train on a prior block it will not "
        "have at deploy. Set both to 'analytic', or pass them equal deliberately."
    )


def apply_prior_source(data, source: str = "analytic", *, phys_cfg: PhysicsConfig | None = None):
    """Return ``data`` with the four prior columns rewritten according to ``source``.

    * ``stored``   -- leave as-is. **Illegal under the s17 Z2 contract** (these are GT CFD).
    * ``analytic`` -- Poiseuille magnitude + potential-flow direction. Legal.
    * ``zero``     -- all four columns zeroed. The ablation control for Z1.

    Mutates a shallow clone, never the caller's object, so cached packs stay clean.
    """
    src = (source or "stored").strip().lower()
    if src not in PRIOR_SOURCES:
        raise ValueError(f"prior source must be one of {PRIOR_SOURCES}, got {source!r}")
    if src == "stored":
        return data

    out = data.clone()
    x = out.x.clone()
    if src == "zero":
        x[:, COL_U_PRIOR] = 0.0
        x[:, COL_V_PRIOR] = 0.0
        x[:, COL_MU_PRIOR] = 0.0
        x[:, COL_WSS_PRIOR] = 0.0
    else:
        u, v, mu, wss = build_analytic_priors(out, phys_cfg=phys_cfg)
        x[:, COL_U_PRIOR] = u.to(x.dtype)
        x[:, COL_V_PRIOR] = v.to(x.dtype)
        x[:, COL_MU_PRIOR] = mu.to(x.dtype)
        x[:, COL_WSS_PRIOR] = wss.to(x.dtype)
    out.x = x
    return out
