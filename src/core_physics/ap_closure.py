"""Wall-AP closure: a quasi-steady Damkohler balance for activated platelets at the wall.

PHASE6_HANDOFF 2.1.  The rollout holds ``ap`` at its t=0 value, which is spatially uniform
(CV 0.0000).  Every node whose gate is exactly 1 then has an *identical* ODE and they all
cross ``Mat >= crit`` in the same step -- the flash.  COMSOL's own fields say ``ap``
develops CV 0.07-0.31 and falls to 1.2% of inlet where deposition is heavy, and that
missing suppression is what spreads GT's onsets.

The closure is a wall Damkohler balance -- adhesion consumption against shear-driven
renewal:

    ap_i(t) / ap0_i = 1 / (1 + C * consumption_i(t) / sr_i^q)

READ THIS BEFORE USING IT.  ``Sat = 1 - Mas/Minf`` and ``k_aa == k_as`` in the config, so
the handoff's ``consumption = gate*(Sat + Mas/Minf)*k_as`` is **identically** ``gate*k_as``
until ``Mas`` overshoots ``Minf`` -- measured pooled R2 on TRAIN differs in the 4th decimal
(0.7233 vs 0.7235).  With ``gate`` and ``sr`` frozen at t=0 the closure is therefore a
*static spatial multiplier*, not a depletion feedback.  That is still exactly what breaks
the flash (identically-gated nodes get different rates, ordered by shear, which is the
sign 2 measured) but it must not be described as a feedback.  ``KERNELS`` keeps the
genuinely time-varying variants alongside so a rollout can select between them on DEV.

The exponent is fit, not assumed: ``q = 1`` beats Leveque's diffusive-boundary-layer 1/3
by a wide margin on TRAIN (R2 0.748 vs 0.509 in the p007 fit, 0.748 vs 0.508 refit on
TRAIN).  A renewal rate linear in shear is a stirred-replenishment balance, not a
diffusive one; do not call it Leveque.

Constants are fit by ``scripts/fit_ap_closure.py`` on WALL_COHORT_V2_TRAIN only.  The
patient007 values (C=68) in the handoff came from the SEALED vessel's raw export and are
not usable (6.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4
SR_FLOOR = 1.0e-3          # 1/s -- below this "renewal by shear" is not a mechanism

#: consumption kernels, all returning [cm/s].  ``gate`` is the law's bracket prefactor,
#: ``sat`` the clipped availability, ``mas_f``/``mat_f`` the surface loads over ``Minf``.
KERNELS = ("handoff", "static", "unit_gate", "clip1_gate", "sat_only", "sat_plus_mat",
           "mat_linear")

#: ``mat_linear``'s slope, chosen on FIT by WINDOW STABILITY -- see ``consumption``.
MAT_COEF_DEFAULT = 0.3


def consumption(kernel: str, gate, sat, mas_f, mat_f, k_as: float, k_aa: float,
                mat_coef: float = MAT_COEF_DEFAULT):
    """The wall AP sink [cm/s] under one of ``KERNELS``.

    ``unit_gate``/``clip1_gate`` exist because the separation branch of the gate is an
    *amplification factor* on the reaction rate (``(L/gamma_m)*|dsrx|``, which reaches ~1.5)
    rather than a physical area, and it is not obvious the AP sink should inherit it.  It
    is measurable rather than arguable: over the full gated set, at the deployed C, the
    full-gate kernel makes onset ORDERING worse than the bare gate (rho -0.839 -> -0.748),
    because it suppresses ap hardest exactly where the gate is strongest and so cancels the
    gate's own signal.  (On the gate==1 subset, where the gate carries no ordering at all,
    it is the only thing supplying one -- so the two sets disagree and both matter.)

    ``mat_linear`` -- ``gate * k_as * (1 + mat_coef * Mat/Minf)`` -- is the kernel this
    module recommends, and ``mat_coef`` was picked by a criterion that does not look at R2
    at all.  A correctly-specified kernel must recover the SAME ``C`` whichever slice of
    the horizon it is fitted on; a misspecified one absorbs the drift into ``C``.  Refit on
    three disjoint windows of the horizon, 19 TRAIN vessels
    (``scripts/fit_ap_closure.py`` section A3 reprints this):

        mat_coef            C[0-25%]  C[25-60%]  C[60-100%]   drift   pooled R2
        0.0  (= static)        141.2      243.1       377.4   2.67x     0.7235
        0.1                    124.9      173.8       205.4   1.64x     0.7748
        0.3  (the default)     101.3      110.9       106.2   1.09x     0.7842
        1.0  (~sat_plus_mat)    61.0       49.1        39.4   1.55x     0.7679

    The static kernel's ``C`` moves 2.7x depending on which part of the run you weight --
    which is most of why this repo has held both ``C = 68`` (patient007) and ``C = 250``
    (pooled TRAIN) and thought they disagreed.  They are the same measurement under
    different weighting.  At ``mat_coef = 0.3`` the drift collapses to 1.14x and the pooled
    R2 is at its plateau.  Physically: the AP sink grows with the mature deposit but
    sub-linearly, as an aging clot buries its own reactive surface.
    """
    g = np.asarray(gate, dtype=np.float64)
    if kernel == "handoff":
        return g * (sat + mas_f) * k_as
    if kernel == "static":
        return g * k_as * np.ones_like(np.asarray(sat, dtype=np.float64))
    if kernel == "unit_gate":
        return (g > 0).astype(np.float64) * k_as * np.ones_like(np.asarray(sat, dtype=np.float64))
    if kernel == "clip1_gate":
        return np.minimum(g, 1.0) * k_as * np.ones_like(np.asarray(sat, dtype=np.float64))
    if kernel == "sat_only":
        return g * sat * k_as
    if kernel == "sat_plus_mat":
        return g * (sat * k_as + mat_f * k_aa)
    if kernel == "mat_linear":
        return g * k_as * (1.0 + mat_coef * mat_f)
    raise ValueError(f"unknown ap-closure kernel {kernel!r}")


def build_smoother(edge_index_local: np.ndarray, n: int, hops: int):
    """Row-stochastic mesh averaging, applied ``hops`` times.  ``hops=0`` -> identity.

    MEASURED, AND IT DOES NOT HELP -- keep it that way in your expectations.  ``ap`` is a
    transport field (neighbour correlation 0.993) and the local closure's residual still
    correlates 0.926 with its own neighbour mean, so smoothing the sink looks like the
    obvious zero-parameter stand-in for the missing transport.  It is not: pooled R2 on
    TRAIN falls monotonically, 0.7235 (0 hops) -> 0.7063 -> 0.6962 -> 0.6878 (4 hops).
    The non-locality is advective/upstream, not a diffusive smear.  This lives on as the
    baseline that result rests on, and as the reason a graph model here needs directed or
    flow-aware message passing rather than isotropic diffusion.

    WHY MESH AND NOT KD-TREE.  A spatial KD-tree would join nodes on OPPOSITE walls
    wherever the vessel narrows -- precisely the stenoses -- so the stencil must follow
    connectivity.
    """
    if hops <= 0:
        return lambda v: v
    ei = np.asarray(edge_index_local)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    A.setdiag(1.0)
    deg = np.asarray(A.sum(1)).reshape(-1)
    P = sp.diags(1.0 / np.maximum(deg, 1.0)) @ A

    def apply(v):
        out = np.asarray(v, dtype=np.float64)
        for _ in range(hops):
            out = P @ out
        return out

    return apply


@dataclass
class ApClosure:
    """Calibrated wall-AP closure.  ``C=0`` reproduces the frozen-``ap`` model exactly."""

    C: float = 0.0
    q: float = 1.0
    kernel: str = "static"
    smooth_hops: int = 0
    mat_coef: float = MAT_COEF_DEFAULT

    def multiplier(self, gate, sr, sat, mas_f, mat_f, k_as, k_aa, smoother=None):
        """``ap/ap0`` in [0, 1], shape ``[N]``."""
        if self.C == 0.0:
            return np.ones_like(np.asarray(gate, dtype=np.float64))
        c = consumption(self.kernel, gate, sat, mas_f, mat_f, k_as, k_aa,
                        mat_coef=self.mat_coef)
        x = c / np.power(np.maximum(np.asarray(sr, dtype=np.float64), SR_FLOOR), self.q)
        if smoother is not None:
            x = smoother(x)
        return 1.0 / (1.0 + self.C * x)


#: THE SHIPPED CONFIGURATION.  Kernel and ``C`` selected on DEV only, ``C`` anchored by a
#: least-squares fit on FIT vessels (``scripts/eval_ap_closure_protocol.py``); ``da_scale``
#: 40 is the pre-existing value and won on DEV both with and without the closure.
#:
#: WHAT IT BUYS, on the growth-count metric that measures the actual objective
#: (``scripts/eval_growth_count.py``):
#:
#:     growth_l1   train 0.1316 -> 0.1224      SEALED 0.1158 -> 0.1078
#:
#: -7% of total error on both sets, same sign, zero learned parameters, and the final mask
#: is bit-identical (the mask comes from the gates plus graph growth; the ODE only supplies
#: timing).  Under the older time-resolved overlap score the identical arm read -0.0001 --
#: that score is discontinuous in commit time and could not see it (PHASE6_RESULTS 15.3).
SHIPPED = ApClosure(C=62.42, q=1.0, kernel="static")
SHIPPED_DA_SCALE = 40.0


def fit_C(ratio: np.ndarray, x: np.ndarray, *, lo: float = -4.0, hi: float = 6.0) -> float:
    """Least-squares ``C`` on ``ap/ap0`` itself.  1-D, so grid + refine is exact enough.

    Fitting the LINEARISED form ``ap0/ap - 1 = C x`` has a closed form but weights the most
    depleted nodes -- the ones that decide onset -- by ``1/ap^2``, and on TRAIN the two
    estimators disagree by 10x (258.7 vs 26.5 at q=1).  This one optimises the quantity the
    rollout actually multiplies by.
    """
    best = (float("nan"), np.inf)
    for _ in range(4):
        for c in np.logspace(lo, hi, 121):
            sse = float(((ratio - 1.0 / (1.0 + c * x)) ** 2).sum())
            if sse < best[1]:
                best = (float(c), sse)
        span = (hi - lo) / 12.0
        lo, hi = np.log10(best[0]) - span, np.log10(best[0]) + span
    return best[0]


def make_rollout_hook(closure: ApClosure, bio_cfg, sr: np.ndarray, smoother=None):
    """``(gate, sat, mas, mat) -> ap/ap0`` for ``integrate_mat_trajectory(ap_closure=...)``.

    ``mas``/``mat`` are the ROLLOUT's own surface state in CGS, so the closure is
    self-consistent: whatever the model has deposited is what it charges against ``ap``.
    """
    k_as = float(bio_cfg.k_as) * M_TO_CM
    k_aa = float(bio_cfg.k_aa) * M_TO_CM
    minf = float(bio_cfg.Minf) * PER_M2_TO_PER_CM2

    def hook(gate, sat, mas, mat):
        return closure.multiplier(gate, sr, sat, mas / minf, mat / minf, k_as, k_aa,
                                  smoother=smoother)

    return hook
