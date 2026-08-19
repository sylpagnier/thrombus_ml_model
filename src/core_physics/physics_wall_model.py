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


def gate_from_shear(
    sr: np.ndarray, dsrx: np.ndarray, bio_cfg, *, wall: np.ndarray | None = None
) -> np.ndarray:
    """The COMSOL deposition law's bracket prefactor from a shear field.

    ``gate = [dsrx < sgt] * (L_char/gamma_m) * |dsrx|  +  [sr < lss]``

    Single source of truth: every arm that re-evaluates the gate on a *different* flow
    field -- the GT-flow oracle, the corrector rollout, the frozen t=0 fields -- must use
    the same expression, or their comparison measures the transcription and not the flow.
    ``sr`` in 1/s, ``dsrx`` in 1/(s*cm), both COMSOL-native CGS.
    """
    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM
    coef = float(bio_cfg.L_char) * M_TO_CM / float(bio_cfg.gamma_m)
    g = (dsrx < sgt_cgs) * coef * np.abs(dsrx) + (sr < float(bio_cfg.lss))
    return g if wall is None else g * wall


def gt_flow_gate_series(
    data, bio_cfg, *, hops: int = 3, wall: np.ndarray | None = None
) -> np.ndarray:
    """``[T, N]`` gate recomputed from the GT velocity at EVERY timestep -- an ORACLE.

    Upper bound on any evolving-flow model, learned or not: zero flow error. Illegal as a
    model, decisive as a ceiling. Prefer ``outputs/wall_species_cache/<v>.npz``'s
    ``sr_t``/``dsrx_t`` when only wall nodes are needed -- this recomputes MLS gradients at
    all T timesteps and is minutes per vessel.
    """
    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    nt = int(data.y.shape[0])
    out = np.zeros((nt, int(data.num_nodes)))
    for ti in range(nt):
        u = data.y[ti, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[ti, :, 1].detach().cpu().numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        out[ti] = gate_from_shear(sr, (Dx @ sr) / (d_bar * M_TO_CM), bio_cfg, wall=wall)
    return out


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
        # MLS on predicted velocity, not the kinematics shear head.  On patient005 the
        # cached head has wall corr 0.17 vs GT MLS (median 54 1/s against 193) and
        # dsrx from that field never trips sgt, so the wall gate is empty.  MLS-on-u0
        # keeps wall corr 0.82.  ``sr0_pred`` stays on the pack for a later head.
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    else:
        u = data.y[time_index, :, 0].detach().cpu().numpy().astype(np.float64)
        v = data.y[time_index, :, 1].detach().cpu().numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
    dsrx = (Dx @ sr) / (d_bar * M_TO_CM)                                   # 1/(s*cm)

    sgt_cgs = float(bio_cfg.sgt) / M_TO_CM             # 1/(s*m) -> 1/(s*cm)
    return T0Fields(
        sr=sr, dsrx=dsrx,
        gate_low=(sr < float(bio_cfg.lss)).astype(np.float64),
        gate_sep=(dsrx < sgt_cgs).astype(np.float64),
        gate=gate_from_shear(sr, dsrx, bio_cfg),
    )


#: Washout coefficient, DIMENSIONLESS (it multiplies ``sr`` [1/s] to make a rate).  Fit as a
#: single global scalar on WALL_COHORT_V2_TRAIN by ``scripts/diag_mat_washout.py``; 16 of 19
#: vessels pick this value under leave-one-vessel-out, and the LOO gain is 0.310 -> 0.442
#: against the in-sample 0.464, so almost none of it is the fit reading its own answer.
WASHOUT_LAMBDA = 1.54e-6


def washout_step(mat: np.ndarray, source: np.ndarray, h: float, decay: np.ndarray):
    """One step of ``dMat/dt = source - decay*Mat``, unconditionally stable and positive.

    Backward-Euler in the removal term only:  ``mat <- (mat + h*source) / (1 + h*decay)``.

    NOT explicit Euler, and the difference is not cosmetic.  The stored timestep is 150 s and
    ``decay = lambda*sr`` reaches ~1e-2 1/s where the separation branch fires on fast flow, so
    ``h*decay`` passes 2 and an explicit update oscillates and then diverges on exactly the
    high-shear nodes this term exists to suppress.  Shared by the model and by
    ``scripts/diag_mat_washout.py`` so the fitted ``lambda`` transfers between them instead of
    silently meaning two different things.
    """
    return (mat + h * source) / (1.0 + h * np.asarray(decay, dtype=np.float64))


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
    da_scale_auto: float | None = None,
    blockage=None,
    species=None,
    ap_boost=None,
    ap_closure=None,
    washout: float = 0.0,
    washout_sr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the surface ODEs, returning ``Mat`` at EVERY timestep.

    ``da_scale`` defaults to 40, not the 100 the final-mask sweep settled on.  Every value
    above ~50 gives a bit-identical committed set (docs/PHASE3_RESULTS.md 3), so the mask
    metric could not distinguish them -- but the growth CURVE can: 100 -> 40 takes
    ``curve_l1`` from 0.2998 to 0.1018 at an unchanged deploy score (13.1).

    ``da_scale_auto`` -- separate rate scalar for the AUTOCATALYTIC term.  ``None`` means
        "same as ``da_scale``", which is bit-identical to the one-scalar model.
        WHY THIS EXISTS: COMSOL's own export never supported one scalar.  Refitting the two
        terms independently across 19 TRAIN vessels (``scripts/diag_damkohler_cohort.py``)
        gives ``A_s/Da`` median 20.7 and ``A_a/Da`` median 67.6 -- a ratio of **3.07**,
        positive in 15/19 vessels, and the same signature COMSOL's own numbers carry
        (``d(Mas,t)/J0_Mas`` = 25.8 against ``d(Mat,t)/J0_Mat`` = 145.6).  The single
        ``da_scale`` was absorbing the smaller of the two.  This matters for TIMING and not
        for the mask: the autocatalytic term is what decides how long a node idles below
        ``crit`` before it runs away, so the ratio sets the delay between the first
        deposition and the commitment that the score sees.

    ``species`` -- ``(rp, ap)`` in CGS, each ``[N]`` (frozen) or ``[T, N]`` (time-varying).
        Defaults to the t=0 constants.  Passing GT trajectories makes this a CHEMISTRY
        oracle, the analogue of the time-varying-gate flow oracle in 13.4.
    ``ap_boost`` -- optional ``(mat, step) -> multiplier [N]``, the thrombin coupling:
        committed nodes generate thrombin, thrombin activates platelets, and ``k_as`` is
        12x ``k_rs``.  This is the mechanism the ad-hoc graph-growth term stands in for.
    ``ap_closure`` -- optional ``(gate, sat, mas, mat) -> multiplier [N]``, the wall-AP
        Damkohler balance (:mod:`src.core_physics.ap_closure`).  Applied AFTER ``ap_boost``
        and evaluated on the rollout's OWN surface state, so it is self-consistent.  This
        is what breaks the flash: with ``ap`` frozen and uniform, every ``gate == 1`` node
        integrates the identical ODE and they all cross ``crit`` in the same step.
        Leaving it ``None`` reproduces the frozen-``ap`` trajectory bit-for-bit.

    ``blockage`` -- optional callable ``(mat, gate0) -> gate`` applied every step, used by
    the shear-redistribution arm to let the growing clot close its own gates.

    ``washout`` -- dimensionless coefficient on the REMOVAL term ``- washout*sr*Mat``, with
        ``washout_sr`` the per-node shear rate [1/s] (normally ``T0Fields.sr``).  ``0.0``
        reproduces the accumulate-only trajectory bit-for-bit.  ``washout_sr`` may be
        static ``[N]``, time-varying ``[T, N]``, or a callable ``(mat, step) -> [N]`` so a
        wake/blockage can update the sink on the same committed state the source sees.

        THIS IS THE ONE STRUCTURAL TERM THE LAW WAS MISSING, and it is missing because the
        repo treats ``Mat`` as a surface coverage.  It is not: in the ``.mph`` it is a
        *Transport of Diluted Species* DOMAIN concentration on ``tds2``, with convection
        enabled (nonconservative form, Do Carmo and Galeao crosswind stabilisation), sourced
        at the wall by the ``J0_Mat`` flux.  Material deposited at the wall therefore sits in
        the near-wall fluid and is carried off by the flow; accumulating it forever, as this
        function did, has no removal channel and no steady state.

        WHY IT MATTERS MORE THAN ANY RATE SCALAR.  Handed a perfect oracle -- GT ``RP``,
        ``AP``, ``M``, ``Mas``, ``sr`` and ``d(sr,x)`` at every timestep -- the accumulate-only
        ODE still ranks GT ``Mat`` at only 0.31 on live wall nodes, and is ANTI-correlated on
        5 of 19 train vessels (``scripts/diag_local_ode_closure.py``).  No input model and no
        choice of ``da_scale`` can fix that, because the deficit is in the equation.  The
        removal term takes the same oracle to 0.464 in-sample and 0.442 leave-one-vessel-out.

        WHY IT IS PROPORTIONAL TO ``sr`` AND NOT A BARE LIFETIME.  The gate has two branches.
        The stagnation branch fires where ``sr < lss``, so those nodes deposit AND retain.
        The separation branch fires on ``d(sr,x) < sgt``, which happens at reattachment points
        where ``sr`` itself is large -- those nodes deposit and are immediately scoured.  With
        no removal the model ranks the second group far too high, and since that branch is the
        one carrying a magnitude (``(L/gamma_m)*|dsrx|`` reaches ~1.5) it dominates the
        predicted ordering.  Measured against the two cheaper stories: a bare lifetime
        ``-lam*Mat`` reaches 0.431 and pure saturation ``J0*(1-Mat/Msat)`` reaches 0.310,
        i.e. exactly nothing.  Saturation is dead; flow-proportional removal beats a bare
        lifetime by 0.033, which is a thin margin on 19 vessels -- the strong claim here is
        that removal exists, and the ``sr`` scaling is the better of two live options and the
        only one with a mechanism in the model tree.

    Returns ``(traj [T, N], t [T])`` in COMSOL model units.
    """
    k_rs = float(bio_cfg.k_rs) * M_TO_CM
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2
    da = float(bio_cfg.surface_damkohler) * float(da_scale)
    da_a = da if da_scale_auto is None else float(bio_cfg.surface_damkohler) * float(da_scale_auto)
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
    lam = float(washout)
    if lam != 0.0 and washout_sr is None:
        raise ValueError("washout != 0 needs washout_sr (the per-node shear rate)")
    wsr_static = None
    if lam != 0.0 and not callable(washout_sr):
        wsr_arr = np.asarray(washout_sr, dtype=np.float64)
        if wsr_arr.ndim not in (1, 2):
            raise ValueError("washout_sr must be [N], [T, N], or a callable")
        wsr_static = wsr_arr
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        step2t = 1.0 / (1.0 + np.exp(-np.clip((t[i] - gate_s) * slope, -50, 50)))
        g = gate0 if blockage is None else blockage(mat, gate0, i)
        rp_i = rp[i] if rp.ndim == 2 else rp
        ap_i = ap[i] if ap.ndim == 2 else ap
        if ap_boost is not None:
            ap_i = ap_i * ap_boost(mat, i)
        sat = np.clip(1.0 - mas / minf, 0.0, 1.0)
        if ap_closure is not None:
            ap_i = ap_i * ap_closure(g, sat, mas, mat)
        dep = sat * (k_rs * rp_i + k_as * ap_i)
        auto = (mas / minf) * k_aa * ap_i
        mas = mas + h * da * g * dep * step2t
        src = g * (da * dep + da_a * auto) * step2t
        if lam == 0.0:
            mat = mat + h * src
        else:
            if callable(washout_sr):
                sr_i = np.asarray(washout_sr(mat, i), dtype=np.float64).reshape(-1)
            elif wsr_static.ndim == 2:
                sr_i = wsr_static[min(i, wsr_static.shape[0] - 1)].reshape(-1)
            else:
                sr_i = wsr_static.reshape(-1)
            mat = washout_step(mat, src, h, lam * np.abs(sr_i))
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


# ---------------------------------------------------------------------------
# FLOW-COUPLED ARM (isolated; nothing above this line changes behaviour)
# ---------------------------------------------------------------------------
#
# The shipped arms freeze the gate at t=0.  COMSOL re-evaluates it on the CURRENT flow, and
# the GT-flow oracle shows that is worth wall F1 0.8405 -> 0.8953 and count floor
# 0.0424 -> 0.0254.  This arm tries to recover that deployably by letting the model's own
# clot reroute its own flow through ``LocalKinematicCorrector``.
#
# An earlier evaluation of exactly this idea (``scripts/diag_corrector_rollout.py``) scored a
# clean negative and concluded the corrector had a SIGN error learned from isolated-clot
# training patches.  That was wrong on three counts, each measured:
#
#   * the corrector LOWERS wake shear like GT does (-0.063 against GT -0.194 at the physical
#     Delta-mu) and RAISES the low-shear open fraction (+0.047 against GT +0.053).  Right
#     sign -- ``scripts/diag_corrector_sign.py``;
#   * it was driven at ``delta_mu = 3.0`` Pa.s, the stale clamp in
#     ``corrector_max_delta_mu_si``, against a measured GT median of **0.68** at committed
#     wall nodes.  The patch factory's real training range is (0.1, 10.0) Pa.s, so 0.68 is
#     comfortably inside it and 3.0 over-drives by 4.4x;
#   * its ODE-ignition-only mask was compared against a static mask that also carries 6-hop
#     graph growth.  Like-for-like the static baseline is 73.7, not 81.5, so the reported
#     "shrink to 73.3" was -0.4 nodes -- ``scripts/diag_corrector_mask_accounting.py``.
#
# What actually limits it is the BOOTSTRAP: the loop cannot start until nodes have already
# crossed ``viscosity_mat_crit``, while GT's clot has been growing under continuously-opening
# gates since t=0.  Handed GT's occlusion the corrector recovers 88% of the oracle's F1 gain
# (``scripts/diag_corrector_ceiling.py``), so ``seed_ramp`` below feeds it the model's OWN
# t=0 predicted mask instead of waiting.  That uses no GT and is deploy-legal.


@dataclass
class CorrectorArm:
    """Configuration for the flow-coupled arm.  ``seed_ramp=0`` disables seeding."""

    corrector: object                 # loaded LocalKinematicCorrector
    phys_cfg: object
    device: object = None
    delta_mu: float = 0.68            # measured GT median at committed wall nodes
    every: int = 10                   # rollout steps between corrector calls
    num_hops: int = 5                 # corrector's receptive field
    # Swept on WALL_COHORT_V2_TRAIN (scripts/sweep_corrector_arm.py, 26 vessels, GT t=0 flow,
    # delta_mu 0.68).  The response is broad and shallow, and the ORIGINAL GUESS OF 2.0 WAS
    # NEARLY THE WORST POINT ON IT:
    #
    #     ramp   0.00    0.50    1.00    1.50    2.00    3.00
    #     score 0.7659  0.7689  0.7600  0.7568  0.7512  0.7445     (static baseline 0.7489)
    #
    # Anything in [0, 1] beats static by ~+0.017; the 0.50-vs-0.00 gap (+0.003) is inside the
    # noise this knob can produce, so read the plateau, not the argmax.
    seed_ramp: float = 0.50           # fraction of the predicted mask seeded per unit time
    front_admission: bool = True
    relax: float = 2.0                # shear-admission multiple of lss, as shipped
    grow_hops: int = 6                # graph-growth hops, as shipped
    da_scale: float = 40.0


def _wall_adjacency(data):
    import scipy.sparse as sp

    ei = data.edge_index.detach().cpu().numpy()
    n = int(data.num_nodes)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def predicted_seed_mask(data, bio_cfg, fields, *, relax=2.0, grow_hops=6, adj=None):
    """The shipped t=0 prediction: both gates, then shear-admitted graph growth."""
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    A = _wall_adjacency(data) if adj is None else adj
    cur = (fields.gate > 0) & wall
    adm = (fields.sr < float(bio_cfg.lss) * float(relax)) & wall
    for _ in range(int(grow_hops)):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur, adm, A


def corrector_blockage(data, bio_cfg, fields, arm: CorrectorArm, *, hops: int = 3,
                       flow_source: str = "gt"):
    """A ``blockage`` callable for :func:`integrate_mat_trajectory`.

    Every ``arm.every`` steps: current occlusion -> per-node viscosity bump -> corrector ->
    MLS gradients -> ``sr``/``dsrx`` -> both gates.  Occluded nodes keep at least their t=0
    gate (hysteresis), and with ``front_admission`` a shear-admitted neighbour of committed
    tissue is gated outright -- the time-resolved analogue of the shipped 6-hop growth.

    ``flow_source`` selects the BASE flow the corrector bends: ``'gt'`` is the Phase-3
    bandaid (COMSOL velocity at t=0), ``'pred'`` is ``u0_pred``/``v0_pred`` from RGP-DEQ and
    is the only fully deploy-legal setting -- the corrector patches a predicted base field,
    so no GT velocity enters the rollout at any time.  Pass ``fields`` computed from the
    same source or the gate and the base flow disagree.
    """
    import torch as _torch

    from src.inference.corrector_coupling import couple_flow_with_corrector

    device = arm.device or _torch.device("cpu")
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    crit = float(bio_cfg.viscosity_mat_crit)
    nt = int(data.t.reshape(-1).shape[0])

    seed, adm, A = predicted_seed_mask(data, bio_cfg, fields,
                                       relax=arm.relax, grow_hops=arm.grow_hops)
    order = np.argsort(-(fields.gate * seed))
    n_seed = int(seed.sum())

    pos = node_positions(data)
    ei = data.edge_index.detach().cpu().numpy()
    Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
    u_ref = float(data.u_ref.reshape(-1)[0])
    d_bar = float(data.d_bar.reshape(-1)[0])
    if flow_source == "pred":
        if getattr(data, "u0_pred", None) is None:
            raise ValueError("flow_source='pred' needs u0_pred (deployable base flow absent)")
        u0 = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v0 = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        u0 = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
        v0 = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
    u0_t = _torch.tensor(u0, dtype=_torch.float32, device=device)
    v0_t = _torch.tensor(v0, dtype=_torch.float32, device=device)

    class _DevView:
        """Only the three tensors the corrector reads, on device.

        ``Data.to(device)`` moves the pack in place and then breaks the CPU-side ODE
        integrator, which calls ``.numpy()`` on ``data.y``.
        """

        def __init__(self, dd):
            self.x = dd.x.to(device)
            self.edge_index = dd.edge_index.to(device)
            self.num_nodes = int(dd.num_nodes)

    dev_view = _DevView(data)
    state = {"gate": None, "last": -(10 ** 9), "calls": 0}

    def _flow_gate(occ, gate0):
        if not occ.any():
            return gate0
        delta = _torch.tensor(occ.astype(np.float32) * float(arm.delta_mu), device=device)
        with _torch.no_grad():
            uu, vv, _ = couple_flow_with_corrector(
                dev_view, u0_t, v0_t, delta, corrector=arm.corrector,
                phys_cfg=arm.phys_cfg, device=device, num_hops=int(arm.num_hops))
        un = uu.detach().cpu().numpy().astype(np.float64)
        vn = vv.detach().cpu().numpy().astype(np.float64)
        sr = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * (u_ref / d_bar)
        dsx = (Dx @ sr) / (d_bar * M_TO_CM)
        g = gate_from_shear(sr, dsx, bio_cfg, wall=wall)
        state["calls"] += 1
        return np.where(occ, np.maximum(g, gate0), g)

    def blockage(mat, gate0, i):
        if state["gate"] is not None and i - state["last"] < arm.every:
            return state["gate"]
        occ = mat >= crit
        if arm.seed_ramp > 0.0 and n_seed:
            k = int(np.clip(n_seed * (i / max(nt - 1, 1)) * float(arm.seed_ramp), 0, n_seed))
            s = np.zeros(len(wall), dtype=bool)
            s[order[:k]] = True
            occ = (s & seed) | occ
        g = _flow_gate(occ, gate0)
        if arm.front_admission:
            g = g.copy()
            adj = (np.asarray(A @ (mat >= crit).astype(np.int8)).reshape(-1) > 0) & adm
            g[adj] = np.maximum(g[adj], 1.0)
        state["gate"], state["last"] = g, i
        return g

    blockage.state = state
    blockage.seed_mask = seed
    return blockage


def predict_corrector(data, bio_cfg, arm: CorrectorArm, *, hops: int = 3,
                      flow_source: str = "gt", ap_closure=True):
    """Flow-coupled wall-clot mask.  Returns ``(mask [N], onset [N], fields, calls)``."""
    from src.core_physics.ap_closure import SHIPPED, make_rollout_hook

    fields = t0_flow_fields(data, bio_cfg, hops=hops, flow_source=flow_source)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    blk = corrector_blockage(data, bio_cfg, fields, arm, hops=hops, flow_source=flow_source)
    hook = make_rollout_hook(SHIPPED, bio_cfg, fields.sr) if ap_closure else None
    traj, _ = integrate_mat_trajectory(data, bio_cfg, fields.gate * wall,
                                       da_scale=arm.da_scale, blockage=blk,
                                       ap_closure=hook)
    onset = first_crossing(traj, float(bio_cfg.viscosity_mat_crit))
    mask = (onset >= 0) & wall
    return mask, np.where(wall, onset, -1), fields, blk.state["calls"]


def predict_phi(
    data,
    bio_cfg,
    *,
    mode: str = "ode",
    hops: int = 3,
    da_scale: float = 1.0,
    time_index: int = 0,
    arm: "CorrectorArm | None" = None,
    flow_source: str = "gt",
) -> tuple[torch.Tensor, T0Fields, np.ndarray | None]:
    """Binary wall-clot prediction ``phi_pred`` [N] plus the intermediates.

    ``mode='corrector'`` runs the flow-coupled arm and requires ``arm``.  The other modes
    are unchanged.
    """
    if mode == "corrector":
        if arm is None:
            raise ValueError("mode='corrector' requires a CorrectorArm")
        mask, _, fields, _ = predict_corrector(data, bio_cfg, arm, hops=hops,
                                               flow_source=flow_source)
        return torch.tensor(mask.astype(np.float32)), fields, None
    fields = t0_flow_fields(data, bio_cfg, hops=hops, time_index=time_index)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    if mode == "gate":
        pred = (fields.gate > 0) & wall
        return torch.tensor(pred.astype(np.float32)), fields, None
    mat = integrate_mat(data, bio_cfg, fields, da_scale=da_scale)
    pred = (mat >= float(bio_cfg.viscosity_mat_crit)) & wall
    return torch.tensor(pred.astype(np.float32)), fields, mat
