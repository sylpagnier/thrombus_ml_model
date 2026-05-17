"""Kinematics-derived clot-deposition-rate prior aligned with COMSOL adhesion flux.

The score is a physics-anchored estimate of local clot risk in ``[0, 1]``,
computed from kinematics only (``u``, ``v``, ``G_x``, ``G_y``, ``sdf_nd``). Two
components mirror the ``BiochemPhysicsKernels.biochem_wall_residual`` adhesion
flux structure:

1. **Pathological adhesion** (streamwise shear-gradient driven): proxy for the
   COMSOL ``is_separation * (L_char/gamma_m) * |dshear_ds| * k_rs * RP_wall``
   contribution. Dimensional factors are absorbed into per-graph max
   normalisation; the dimensionless gate is
   ``is_separation * |dshear_ds_phys| / |sgt|``.
2. **Stagnation / recirculation adhesion** (low-shear driven): proxy for the
   COMSOL ``is_low_shear * k_rs * RP_wall`` contribution, dose-weighted by a
   local residence factor ``exp(-|u|/u_ref)`` so slow-flow vortex zones get
   lifted (longer contact time → more deposition per unit RP).

Both components are wall-localised via a *smooth* SDF gate
``exp(-sdf_nd / lambda_w)`` (replaces the v1 hard ``mask_wall + bulk_floor``
that produced ``Pearson(p_clot, prior) ≈ 0.04`` against GT clot location even
with ``bulk_floor=0.10``).

Two physics-anchored output channels (further columns are filled with a
wall-proximity-only floor so the encoder still sees a boundary-layer signal):

- **Column 0** — combined wall-localised clot-risk score (loss target / encoder
  cue). Probabilistic-OR union of the two component fields, so the channel is
  guaranteed to dominate any single component.
- **Column 1** — pathological subset (weighted separation flux × wall
  proximity). Lets the bio_encoder learn streamwise-shear-gradient adhesion
  separately from low-shear/recirculation adhesion.

Env knobs (all optional, sensible physical defaults):

- ``BIOCHEM_PRIOR_W_PATHOLOGICAL`` (default ``1.0``) — scalar multiplier on the
  normalised pathological component (set to 0 to disable).
- ``BIOCHEM_PRIOR_W_STAGNATION`` (default ``0.25``) — scalar multiplier on the
  normalised stagnation component (set to 0 to disable).
- ``BIOCHEM_PRIOR_RESIDENCE_BOOST`` (default ``0.5``) — ``beta`` in
  ``flux_stag = is_low_shear * (1 + beta * exp(-|u|/u_ref))``.
- ``BIOCHEM_PRIOR_STAGNATION_POWER`` (default ``1.5``) — sharpens broad
  low-shear regions after per-graph normalisation.
- ``BIOCHEM_PRIOR_TOTAL_POWER`` (default ``1.5``) — sharpens the combined
  risk map before wall localisation. Calibrated against available COMSOL
  anchors so raw prior prevalence starts close to viscosity-threshold clot
  prevalence, reducing the burden on prevalence matching.
- ``BIOCHEM_PRIOR_WALL_DECAY_ND`` (default ``0.006``) — boundary-layer decay
  length ``lambda_w`` for ``exp(-sdf_nd / lambda_w)``. Smaller = tighter wall
  layer.
- ``BIOCHEM_PRIOR_MIN_FLOOR`` (default ``1e-4``) — small additive floor inside
  the SDF gate so every output channel inherits the wall localisation gate
  even when local kinematics give zero physics signal.
- ``BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS`` (default ``0``) — after wall-local
  gating, dilate each prior channel by repeated **per-node max over graph
  neighbors** (one iteration per hop). Use ``2``–``3`` so thrombus-shaped
  high-risk regions extend a short graph distance from the wall into the
  lumen without smearing into the full bulk. Set ``0`` to disable.

Deprecated env vars (silently ignored — kept for backward compat with old run
configurations and tests that still set them):

- ``BIOCHEM_PRIOR_BULK_SCALE`` — replaced by smooth SDF gate above.
- ``BIOCHEM_PRIOR_W_LOW_SHEAR``, ``BIOCHEM_PRIOR_W_SEP``,
  ``BIOCHEM_PRIOR_W_GRAD`` — superseded by per-physics-component weights.
- ``BIOCHEM_PRIOR_GRAD_QUANTILE``, ``BIOCHEM_PRIOR_GRAD_TEMP`` — the v2 prior
  uses ``|sgt|`` as a fixed physical reference and per-graph ``max``
  normalisation, both of which avoid quantile-driven noise on small graphs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from src.utils.rheology import compute_shear_rate

if TYPE_CHECKING:
    from src.config import BiochemConfig


def _max_neighbor_dilate_1d(v: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Per-node max over self and neighbors (one undirected graph hop when edges are bidirectional)."""
    flat = v.reshape(-1).contiguous()
    row = edge_index[0]
    col = edge_index[1]
    msg = torch.full_like(flat, float("-inf"))
    msg.scatter_reduce_(0, row, flat[col], reduce="amax", include_self=False)
    return torch.maximum(flat, msg).view(v.shape)


def _thrombus_corona_hop_count() -> int:
    """Graph hops to dilate wall-adjacent clot risk (thrombus a few elements into the lumen)."""
    try:
        return max(0, min(12, int(os.environ.get("BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS", "0"))))
    except ValueError:
        return 0


def _apply_thrombus_corona_dilation(
    cols: list[torch.Tensor],
    edge_index: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    hops = _thrombus_corona_hop_count()
    if hops <= 0 or edge_index.numel() == 0:
        return cols
    ei = edge_index.to(device=device)
    out_cols: list[torch.Tensor] = []
    for c in cols:
        flat = c.reshape(-1).contiguous()
        for _ in range(hops):
            flat = _max_neighbor_dilate_1d(flat, ei)
        out_cols.append(flat.reshape(c.shape).clamp(0.0, 1.0))
    return out_cols


def shear_rate_si(data, u_nd: torch.Tensor, v_nd: torch.Tensor, props: dict) -> torch.Tensor:
    """Physical shear rate [1/s] from ND velocities and graph sparse gradients."""
    u = u_nd.reshape(-1).to(dtype=torch.float32)
    v = v_nd.reshape(-1).to(dtype=torch.float32)
    du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)
    gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6)
    u_ref = props["u_ref"].to(device=u.device, dtype=torch.float32).reshape(-1)
    d_bar = props["d_bar"].to(device=u.device, dtype=torch.float32).reshape(-1)
    d_safe = torch.clamp(d_bar, min=1e-8)
    return gamma_dot_nd * (u_ref / d_safe)


def _safe_max_norm(field: torch.Tensor) -> torch.Tensor:
    """Per-graph max normalisation with a detached reference (gradient-safe)."""
    ref = field.detach().abs().max().clamp(min=1e-8)
    return (field / ref).clamp(0.0, 1.0)


def clot_prior_features(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: "BiochemConfig",
    props: dict,
    n_features: int = 2,
) -> torch.Tensor:
    """Return ``[N, n_features]`` physics-derived clot-risk features.

    See the module docstring for the full physical motivation.
    """
    n_features = max(1, int(n_features))
    u = u_nd.reshape(-1).to(dtype=torch.float32)
    v = v_nd.reshape(-1).to(dtype=torch.float32)

    gamma_si = shear_rate_si(data, u, v, props)

    # Streamwise shear gradient [1/(m*s)] — same construction as the
    # ``biochem_wall_residual`` ``dshear_ds_phys`` quantity, so the prior and
    # the live wall residual see the same separation cue.
    gx = torch.sparse.mm(data.G_x, gamma_si.unsqueeze(1)).squeeze(1)
    gy = torch.sparse.mm(data.G_y, gamma_si.unsqueeze(1)).squeeze(1)

    d_bar = props["d_bar"].to(device=u.device, dtype=torch.float32).reshape(-1)
    d_safe = torch.clamp(d_bar, min=1e-8)
    u_ref = props["u_ref"].to(device=u.device, dtype=torch.float32).reshape(-1)
    u_ref_safe = torch.clamp(u_ref, min=1e-8)

    vel_mag_nd = torch.sqrt(u * u + v * v) + 1e-8
    u_dir = u / vel_mag_nd
    v_dir = v / vel_mag_nd
    ds_stream = u_dir * gx + v_dir * gy
    dshear_ds_phys = ds_stream / d_safe

    # COMSOL-aligned soft gates (identical form to BiochemPhysicsKernels.biochem_wall_residual).
    T_ls = max(float(bio_cfg.soft_step_T_low_shear) * float(bio_cfg.soft_step_T_scale), 1e-6)
    T_gr = max(float(bio_cfg.soft_step_T_grad) * float(bio_cfg.soft_step_T_scale), 1e-6)
    is_low_shear = torch.sigmoid(((float(bio_cfg.lss) - gamma_si) / T_ls).clamp(-50.0, 50.0))
    is_separation = torch.sigmoid(((float(bio_cfg.sgt) - dshear_ds_phys) / T_gr).clamp(-50.0, 50.0))

    # 1. Pathological flux: gated by separation, scaled by |dshear|/|sgt|.
    #    The reference |sgt| (COMSOL threshold) keeps the field roughly O(1) at
    #    onset, with a hard cap of 5x against extreme outliers.
    sgt_ref = max(abs(float(bio_cfg.sgt)), 1e-6)
    flux_path = (is_separation * (torch.abs(dshear_ds_phys) / sgt_ref)).clamp(0.0, 5.0)

    # 2. Stagnation flux: gated by low shear, dose-weighted by local residence.
    #    Residence = exp(-|u|/u_ref) ∈ (0, 1]; ≈ 1 at near-zero velocity,
    #    decays for fast flow. Beta controls how aggressively slow-flow zones
    #    are amplified.
    beta_res = max(float(os.environ.get("BIOCHEM_PRIOR_RESIDENCE_BOOST", "0.5")), 0.0)
    vel_mag_si = vel_mag_nd * u_ref
    residence = torch.exp(-(vel_mag_si / u_ref_safe).clamp(min=0.0, max=50.0))
    flux_stag = is_low_shear * (1.0 + beta_res * residence)

    # Per-graph max normalisation: each component is rescaled into [0, 1] by
    # its own detached max. Robust to graph size and to absolute scale
    # differences between path (∝ |dshear|/|sgt|) and stag (∝ rate).
    path_norm = _safe_max_norm(flux_path)
    stag_norm = _safe_max_norm(flux_stag)
    stag_power = max(float(os.environ.get("BIOCHEM_PRIOR_STAGNATION_POWER", "1.5")), 1e-6)
    stag_norm = stag_norm.pow(stag_power)

    # Component scalers (independent multipliers in [0, 1]; 0 disables branch).
    w_path = min(max(float(os.environ.get("BIOCHEM_PRIOR_W_PATHOLOGICAL", "1.0")), 0.0), 1.0)
    w_stag = min(max(float(os.environ.get("BIOCHEM_PRIOR_W_STAGNATION", "0.25")), 0.0), 1.0)
    path_w = (w_path * path_norm).clamp(0.0, 1.0)
    stag_w = (w_stag * stag_norm).clamp(0.0, 1.0)

    # Probabilistic-OR union: flux_total = 1 - (1 - path_w)(1 - stag_w).
    # Properties (used by tests + bio_encoder semantics):
    #   * flux_total ∈ [0, 1]
    #   * flux_total >= path_w  (so col 0 >= col 1 always)
    #   * flux_total >= stag_w
    #   * Reduces to path_w when stag_w = 0 and vice-versa
    flux_total = (path_w + stag_w - path_w * stag_w).clamp(0.0, 1.0)
    total_power = max(float(os.environ.get("BIOCHEM_PRIOR_TOTAL_POWER", "1.5")), 1e-6)
    flux_total = flux_total.pow(total_power)

    # Smooth SDF wall proximity. BIO_X_SCHEMA channel 2 is ``sdf_nd`` (distance
    # to nearest wall, normalised by d_bar; wall = 0, lumen interior > 0).
    # ``exp(-sdf_nd / lambda_w)`` gives 1 at the wall and decays into the bulk.
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] > 2:
        sdf_nd = data.x[:, 2].to(device=u.device, dtype=torch.float32).clamp(min=0.0)
    elif hasattr(data, "mask_wall"):
        # Fallback when no SDF channel: collapse to a hard wall step
        # (1 at the wall, 0 elsewhere) so behaviour degrades gracefully.
        wall_soft = data.mask_wall.view(-1).to(device=u.device, dtype=torch.float32).clamp(0.0, 1.0)
        sdf_nd = (1.0 - wall_soft).clamp(min=0.0)
    else:
        sdf_nd = torch.zeros_like(u)

    lambda_w = max(float(os.environ.get("BIOCHEM_PRIOR_WALL_DECAY_ND", "0.006")), 1e-4)
    wall_proximity = torch.exp(-sdf_nd / lambda_w).clamp(0.0, 1.0)

    # Min floor: small additive constant inside the SDF gate so every column
    # carries the wall-proximity signature even when the kinematics produce
    # zero physics signal (e.g. anchor batches with negligible separation).
    min_floor = max(float(os.environ.get("BIOCHEM_PRIOR_MIN_FLOOR", "1e-4")), 0.0)

    # Column 0: full union, wall-localised.
    col_full = ((flux_total + min_floor) * wall_proximity).clamp(0.0, 1.0)

    # Column 1: pathological subset, wall-localised. Because
    # ``flux_total >= path_w``, col_full >= col_path elementwise before clamp,
    # and clamps to 1.0 preserve that ordering at saturation.
    path_channel = path_w.pow(total_power)
    col_path = ((path_channel + min_floor) * wall_proximity).clamp(0.0, 1.0)

    # Extra columns inherit only the wall proximity gate so the encoder still
    # gets a clean boundary-layer mask in slots beyond the two physics
    # channels (and tests asserting wall > far across all columns hold).
    pad_template = (min_floor * wall_proximity).clamp(0.0, 1.0)

    cols = [col_full]
    if n_features >= 2:
        cols.append(col_path)
    while len(cols) < n_features:
        cols.append(pad_template)

    if (
        hasattr(data, "edge_index")
        and torch.is_tensor(data.edge_index)
        and data.edge_index.dim() == 2
        and data.edge_index.shape[1] > 0
    ):
        cols = _apply_thrombus_corona_dilation(cols, data.edge_index, device=u.device)

    return torch.stack(cols[:n_features], dim=1)


def clot_prior_score_flat(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: "BiochemConfig",
    props: dict,
) -> torch.Tensor:
    """Single-channel prior ``[N]`` (column 0 of ``clot_prior_features``)."""
    return clot_prior_features(data, u_nd, v_nd, bio_cfg, props, n_features=1).squeeze(1)


def kinematics_clot_prior_score(
    kernels, data, u_nd: torch.Tensor, v_nd: torch.Tensor, props: dict
) -> torch.Tensor:
    """Backward-compatible wrapper using ``BiochemPhysicsKernels.cfg``."""
    return clot_prior_score_flat(data, u_nd, v_nd, kernels.cfg, props)
