"""Phase-3 physics wall-clot model: the COMSOL deposition law integrated on t=0 gates.

PHASE3_HANDOFF 1.2-1.5.  Nothing here is learned except a single global scalar
(``da_scale``) that sets the deposition rate level.

Inputs are deploy-legal under the Phase-3 bandaid: node positions, mesh connectivity,
``u_ref``/``d_bar``, the boundary/initial conditions, and the **GT velocity field at
t=0 only**.

The one substantive change from the previous stack: shear rate and its x-gradient are
computed with :mod:`src.core_physics.mls_gradient` instead of the packs' ``G_x``/``G_y``.
Audited against COMSOL's own ``spf.sr`` / ``d(spf.sr,x)`` on patient007:

    operator                  spearman(spf.sr)   spearman(d(spf.sr,x))
    packs' G_x / G_y                0.19                0.00
    MLS, 3 graph hops               0.998               0.990

Everything downstream -- both deposition gates -- was previously being evaluated on
noise.  See ``scripts/step0_mls_validate.py``.

Unit system here is COMSOL-native CGS, matching
``src/core_physics/comsol_surface_deposition.py`` and ``viscosity_mat_crit`` (2e7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.core_physics.mls_gradient import (
    build_mls_gradient,
    node_positions,
    shear_rate_2d,
)

M_TO_CM = 100.0
PER_M3_TO_PER_CM3 = 1.0e-6
PER_M2_TO_PER_CM2 = 1.0e-4


@dataclass
class T0Fields:
    sr: np.ndarray        # spf.sr [1/s]
    dsrx: np.ndarray      # d(spf.sr,x) [1/(s*cm)]
    gate_low: np.ndarray  # spf.sr < lss
    gate_sep: np.ndarray  # d(spf.sr,x) < sgt
    gate: np.ndarray      # (L/gamma_m)*|dsrx|*[sep] + [low]   -- the law's bracket prefactor


def t0_flow_fields(
    data, bio_cfg, *, hops: int = 3, time_index: int = 0, flow_source: str = "gt"
) -> T0Fields:
    """Shear rate and shear-gradient at ``time_index`` from the velocity field.

    ``flow_source='gt'`` is the Phase-3 bandaid (GT flow at t=0).  ``'pred'`` uses the
    deployable kinematic model's ``u0_pred``/``v0_pred`` and is the Phase-5 arm -- the
    delta between the two IS the deployability gap (PHASE3_HANDOFF 4a).
    """
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])           # m/s
    d_bar = float(data.d_bar.reshape(-1)[0])           # m
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    if flow_source == "pred":
        if getattr(data, "u0_pred", None) is None:
            raise ValueError("pack has no u0_pred (deployable flow unavailable)")
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        u = data.y[time_index, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[time_index, :, 1].detach().cpu().numpy().astype(np.float64)
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)   # 1/s
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)                                   # 1/(s*cm)

    lss = float(bio_cfg.lss)
    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM             # 1/(s*m) -> 1/(s*cm)
    g_low = (sr < lss).astype(np.float64)
    g_sep = (dsrx < sgt_cgs).astype(np.float64)
    L_cm = float(bio_cfg.L_char) * M_TO_CM
    gate = g_sep * (L_cm / float(bio_cfg.gamma_m)) * np.abs(dsrx) + g_low
    return T0Fields(sr=sr, dsrx=dsrx, gate_low=g_low, gate_sep=g_sep, gate=gate)


def wall_platelet_constants(data, bio_cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(rp, ap)`` at the wall in CGS [plt/cm^3], read from the t=0 initial condition.

    PHASE3_HANDOFF 1.3 / 26.16: both are spatially flat (CV 0.3% / 10%) and vary 0.2%
    across the cohort, so this is an initial condition, not a learned field.
    """
    names = data.y_channel_names.split(",")
    scales = bio_cfg.get_species_scales(device="cpu")
    rp_nd = torch.expm1(data.y[0, :, names.index("RP_log1p_nd")].clamp(-10, 8)).numpy()
    ap_nd = torch.expm1(data.y[0, :, names.index("AP_log1p_nd")].clamp(-10, 8)).numpy()
    rp = rp_nd * float(scales[0]) * PER_M3_TO_PER_CM3
    ap = ap_nd * float(scales[1]) * PER_M3_TO_PER_CM3
    return rp, ap


def graded_gate(
    fields: T0Fields,
    bio_cfg,
    *,
    mode: str = "hard",
    tau_low: float = 0.25,
    tau_sep: float = 0.25,
) -> np.ndarray:
    """The law's bracket prefactor, either COMSOL's hard step or a graded surrogate.

    WHY GRADE A LAW WHOSE GATE IS PROVABLY A HARD STEP.  COMSOL's gate is evaluated on
    the *current* shear at every step; this model freezes it at t=0.  A node sitting just
    below ``lss`` at t=0 is the one most likely to leave the gate as the clot narrows the
    lumen and accelerates the flow, while a node deep inside a stagnation zone stays
    gated for the whole run.  So the correct t=0 surrogate for the *time-averaged* gate is
    not the t=0 indicator, it is a decreasing function of the margin.  ``tau_*`` are the
    margins in units of the thresholds themselves (``temp = tau * lss``), i.e.
    dimensionless, so they transfer across vessels.

    ``mode='hard'`` reproduces ``t0_flow_fields``'s gate exactly.
    """
    lss = float(bio_cfg.lss)
    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM
    L_cm = float(bio_cfg.L_char) * M_TO_CM
    coef = L_cm / float(bio_cfg.gamma_m)
    def soft(x, thresh, tau, scale):
        t = max(tau * scale, 1e-12)
        return 1.0 / (1.0 + np.exp(np.clip((x - thresh) / t, -50, 50)))

    if mode == "hard":
        g_low, g_sep = fields.gate_low, fields.gate_sep
    elif mode == "sigmoid":
        g_low = soft(fields.sr, lss, tau_low, lss)
        g_sep = soft(fields.dsrx, sgt_cgs, tau_sep, abs(sgt_cgs))
    elif mode == "sigmoid_low":
        # Grade only the stagnation branch: the separation branch already carries a
        # magnitude through |dsrx|, so it is not the one that flashes.
        g_low = soft(fields.sr, lss, tau_low, lss)
        g_sep = fields.gate_sep
    else:
        raise ValueError(f"unknown gate mode {mode!r}")
    return g_sep * coef * np.abs(fields.dsrx) + g_low


def integrate_mat_trajectory(
    data,
    bio_cfg,
    gate: np.ndarray,
    *,
    da_scale: float = 40.0,
    blockage=None,
    species=None,
    ap_boost=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the surface ODEs, returning ``Mat`` at EVERY timestep.

    ``da_scale`` defaults to 40, not the 100 the final-mask sweep settled on.  Every value
    above ~50 gives a bit-identical committed set (docs/PHASE3_RESULTS.md 3), so the mask
    metric could not distinguish them -- but the growth CURVE can: 100 -> 40 takes
    ``curve_l1`` from 0.2998 to 0.1018 at an unchanged deploy score (13.1).

    ``species`` -- ``(rp, ap)`` in CGS, each ``[N]`` (frozen) or ``[T, N]`` (time-varying).
        Defaults to the t=0 constants.  Passing GT trajectories makes this a CHEMISTRY
        oracle, the analogue of the time-varying-gate flow oracle in 13.4.
    ``ap_boost`` -- optional ``(mat, step) -> multiplier [N]``, the thrombin coupling:
        committed nodes generate thrombin, thrombin activates platelets, and ``k_as`` is
        12x ``k_rs``.  This is the mechanism the ad-hoc graph-growth term stands in for.

    ``blockage`` -- optional callable ``(mat, gate0) -> gate`` applied every step, used by
    the shear-redistribution arm to let the growing clot close its own gates.

    Returns ``(traj [T, N], t [T])`` in COMSOL model units.
    """
    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)
    if species is None:
        rp, ap = wall_platelet_constants(data, bio_cfg)
    else:
        rp, ap = species
    rp = np.asarray(rp, dtype=np.float64)
    ap = np.asarray(ap, dtype=np.float64)
    t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gate0 = gate
    n = gate0.shape[0]
    mas = np.zeros(n)
    mat = np.zeros(n)
    traj = np.zeros((len(t), n))
    gate_s = float(bio_cfg.surface_time_gate_s)
    slope = float(bio_cfg.surface_time_gate_slope)
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        g = gate0 if blockage is None else blockage(mat, gate0, i)
        rp_i = rp[i] if rp.ndim == 2 else rp
        ap_i = ap[i] if ap.ndim == 2 else ap
        if ap_boost is not None:
            ap_i = ap_i * ap_boost(mat, i)
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        dep = sat * (k_rs * rp_i + k_as * ap_i)
        auto = (mas / minf) * k_aa * ap_i
        mas = mas + h * da * g * dep * step2t
        mat = mat + h * da * g * (dep + auto) * step2t
        traj[i + 1] = mat
    return traj, t


def first_crossing(traj: np.ndarray, thresh: float) -> np.ndarray:
    """[T,N] -> per-node index of the first crossing of ``thresh``, or -1 if never."""
    hot = traj >= thresh
    any_hot = hot.any(axis=0)
    idx = np.where(any_hot, hot.argmax(axis=0), -1)
    return idx


def integrate_mat(
    data,
    bio_cfg,
    fields: T0Fields,
    *,
    da_scale: float = 1.0,
    wall_only: bool = True,
) -> np.ndarray:
    """Integrate the COMSOL surface ODEs with the t=0 gates held fixed.

    ``dMas/dt = Da*gate*Sat*(k_rs*rp + k_as*ap)*step2t``
    ``dMat/dt = Da*gate*(Sat*(k_rs*rp + k_as*ap) + (Mas/Minf)*k_aa*ap)*step2t``
    ``Sat = 1 - Mas/Minf``   (verified against the exported ``Sat(M)`` column, rel 1.8e-12)

    Returns ``Mat`` at the final time in COMSOL model units (compare to
    ``viscosity_mat_crit`` = 2e7).
    """
    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)

    rp, ap = wall_platelet_constants(data, bio_cfg)
    t = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gate = fields.gate.copy()
    if wall_only:
        gate = gate * data.mask_wall.reshape(-1).bool().cpu().numpy()

    n = gate.shape[0]
    mas = np.zeros(n)
    mat = np.zeros(n)
    gate_s = float(bio_cfg.surface_time_gate_s)
    slope = float(bio_cfg.surface_time_gate_slope)
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        dep = sat * (k_rs * rp + k_as * ap)
        auto = (mas / minf) * k_aa * ap
        mas = mas + h * da * gate * dep * step2t
        mat = mat + h * da * gate * (dep + auto) * step2t
    return mat


def predict_phi(
    data,
    bio_cfg,
    *,
    mode: str = "ode",
    hops: int = 3,
    da_scale: float = 1.0,
    time_index: int = 0,
) -> tuple[torch.Tensor, T0Fields, np.ndarray | None]:
    """Binary wall-clot prediction ``phi_pred`` [N] plus the intermediates."""
    fields = t0_flow_fields(data, bio_cfg, hops=hops, time_index=time_index)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    if mode == "gate":
        pred = (fields.gate > 0) & wall
        return torch.tensor(pred.astype(np.float32)), fields, None
    mat = integrate_mat(data, bio_cfg, fields, da_scale=da_scale)
    pred = (mat >= float(bio_cfg.viscosity_mat_crit)) & wall
    return torch.tensor(pred.astype(np.float32)), fields, mat
