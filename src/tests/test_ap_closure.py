"""Guards for the wall-AP closure (PHASE6_HANDOFF 2.1 / 5.1-5.2).

Every assertion below corresponds to a way this wiring could become a silent no-op or a
silent lie, which is the standing rule for the guard files:

  * the closure not reaching the ODE at all (the hook is optional, so a typo in the kwarg
    name would just... do nothing, and the rollout would still look plausible);
  * the closure changing the committed SET rather than only its timing (kill criterion 9);
  * the closure failing to break the flash, which is the entire reason it exists;
  * the ``handoff``/``static`` kernel identity silently ceasing to hold, which would mean
    ``Sat + Mas/Minf != 1`` and the module docstring's central algebraic claim is wrong;
  * the sign flipping, i.e. ap being suppressed MORE at high shear -- that is the retracted
    graded gate's error (3) reappearing inside a different operator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import BiochemConfig
from src.core_physics.ap_closure import (
    ApClosure,
    build_smoother,
    consumption,
    fit_C,
    make_rollout_hook,
)
from src.core_physics.physics_wall_model import first_crossing, integrate_mat_trajectory

M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


class _Pack:
    """The three attributes ``integrate_mat_trajectory`` reads."""

    def __init__(self, n, nt=201, t_final=30000.0):
        import torch

        self.t = torch.linspace(0.0, t_final, nt).reshape(-1, 1)
        names = ("u_nd,v_nd,p_nd,mu_eff_nd,RP_log1p_nd,AP_log1p_nd,APR_log1p_nd,"
                 "APS_log1p_nd,PT_log1p_nd,T_log1p_nd,AT_log1p_nd,FG_log1p_nd,"
                 "FI_log1p_nd,M_log1p_nd,Mas_log1p_nd,Mat_log1p_nd")
        self.y_channel_names = names
        y = torch.zeros(1, n, 16)
        # Spatially UNIFORM inlet species, which is what the packs actually carry at t=0
        # (CV 0.0000) and what makes every gate==1 node integrate the identical ODE.
        # The ND encoding is ``nd = cgs / 2.5e14``; the real packs read RP_log1p_nd ~1.0e-6
        # and AP_log1p_nd ~5e-8, so these are the observed magnitudes, not round numbers.
        y[0, :, 4] = float(np.log1p(2.5e8 / 2.5e14))      # rp = 2.5e8 plt/cm^3
        y[0, :, 5] = float(np.log1p(1.25e7 / 2.5e14))     # ap = c_AP0 = 1.25e7 plt/cm^3
        self.y = y


def _bio():
    return BiochemConfig(phase="biochem")


# --------------------------------------------------------------- the algebraic identity

def test_handoff_and_static_kernels_are_the_same_thing_below_saturation():
    """``Sat + Mas/Minf == 1`` exactly, so the handoff's 'consumption' is static.

    If this ever fails, the closure has become a genuine depletion feedback and the
    module docstring (and every 'near-static' claim built on it) needs rewriting.
    """
    bio = _bio()
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    gate = np.array([1.0, 1.0, 0.5, 2.0])
    mas_f = np.array([0.0, 0.3, 0.9, 0.999])           # all strictly below Minf
    sat = np.clip(1.0 - mas_f, 0.0, 1.0)
    mat_f = np.array([0.0, 5.0, 20.0, 50.0])
    a = consumption("handoff", gate, sat, mas_f, mat_f, k_as, k_aa)
    b = consumption("static", gate, sat, mas_f, mat_f, k_as, k_aa)
    assert np.allclose(a, b, rtol=1e-12)


def test_handoff_and_static_diverge_only_once_mas_overshoots_minf():
    bio = _bio()
    k_as, k_aa = float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM
    gate = np.ones(1)
    mas_f = np.array([1.2])                            # COMSOL drives Sat to -0.195
    sat = np.clip(1.0 - mas_f, 0.0, 1.0)
    a = consumption("handoff", gate, sat, mas_f, np.zeros(1), k_as, k_aa)
    b = consumption("static", gate, sat, mas_f, np.zeros(1), k_as, k_aa)
    assert a[0] > b[0]


# ---------------------------------------------------------------------- the multiplier

def test_multiplier_is_a_suppression_bounded_by_one():
    bio = _bio()
    cl = ApClosure(C=100.0, q=1.0, kernel="static")
    m = cl.multiplier(np.array([1.0, 1.0, 1.0]), np.array([0.1, 5.0, 25.0]),
                      np.ones(3), np.zeros(3), np.zeros(3),
                      float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM)
    assert np.all(m > 0.0) and np.all(m <= 1.0)


def test_higher_shear_suppresses_ap_LESS():
    """The sign 2 measured, and the opposite of the retracted graded gate (3).

    Inside the gated band higher shear ignites EARLIER because shear renews the activated
    platelet supply.  If this assertion inverts, the closure has acquired the graded gate's
    backwards ordering inside a different operator and will spread onsets the wrong way.
    """
    bio = _bio()
    cl = ApClosure(C=100.0, q=1.0, kernel="static")
    sr = np.array([0.5, 2.0, 8.0, 24.0])
    m = cl.multiplier(np.ones(4), sr, np.ones(4), np.zeros(4), np.zeros(4),
                      float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM)
    assert np.all(np.diff(m) > 0.0)


def test_C_zero_is_exactly_the_identity():
    cl = ApClosure(C=0.0)
    m = cl.multiplier(np.ones(5), np.linspace(1, 50, 5), np.ones(5), np.zeros(5),
                      np.zeros(5), 4.5e-2, 4.5e-2)
    assert np.array_equal(m, np.ones(5))


def test_mat_linear_reduces_to_static_at_zero_slope():
    bio = _bio()
    k_as, k_aa = float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM
    args = (np.ones(4), np.ones(4), np.zeros(4), np.array([0.0, 1.0, 10.0, 50.0]))
    a = consumption("mat_linear", *args, k_as, k_aa, mat_coef=0.0)
    b = consumption("static", *args, k_as, k_aa)
    assert np.allclose(a, b)


def test_mat_linear_sink_grows_with_the_mature_deposit():
    bio = _bio()
    c = consumption("mat_linear", np.ones(3), np.ones(3), np.zeros(3),
                    np.array([0.0, 10.0, 50.0]),
                    float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM, mat_coef=0.3)
    assert c[0] < c[1] < c[2]


def test_unknown_kernel_raises_rather_than_silently_defaulting():
    with pytest.raises(ValueError):
        consumption("no_such_kernel", np.ones(2), np.ones(2), np.zeros(2), np.zeros(2),
                    4.5e-2, 4.5e-2)


# --------------------------------------------------------------------- the rollout hook

def test_no_closure_reproduces_the_frozen_ap_trajectory_bit_for_bit():
    """The kwarg is optional, so this is the guard against it never being wired at all."""
    bio = _bio()
    d = _Pack(64)
    gate = np.ones(64)
    a, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0)
    b, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0, ap_closure=None)
    assert np.array_equal(a, b)


def test_the_hook_actually_changes_the_trajectory():
    """A closure that reaches the ODE must move Mat.  Catches a silently ignored kwarg."""
    bio = _bio()
    d = _Pack(64)
    gate = np.ones(64)
    sr = np.linspace(0.5, 24.0, 64)
    hook = make_rollout_hook(ApClosure(C=100.0, q=1.0, kernel="static"), bio, sr)
    a, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0)
    b, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0, ap_closure=hook)
    assert not np.allclose(a, b)
    assert np.all(b[-1] <= a[-1] + 1e-9)          # it can only slow deposition down


def test_the_closure_breaks_the_flash():
    """THE point of the whole exercise.

    With ``ap`` frozen and uniform, every ``gate == 1`` node has an identical ODE and they
    all cross ``crit`` in the same step (onset spread 0.000 of the horizon).  Grading ``ap``
    by shear must produce more than one distinct onset.
    """
    bio = _bio()
    d = _Pack(64)
    gate = np.ones(64)
    sr = np.linspace(0.5, 24.0, 64)
    crit = float(bio.viscosity_mat_crit)

    flat, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0)
    on_flat = first_crossing(flat, crit)
    assert len(np.unique(on_flat[on_flat >= 0])) == 1, "the flash is supposed to be a flash"

    hook = make_rollout_hook(ApClosure(C=100.0, q=1.0, kernel="static"), bio, sr)
    graded, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0, ap_closure=hook)
    on_graded = first_crossing(graded, crit)
    assert len(np.unique(on_graded[on_graded >= 0])) > 1


def test_onset_order_follows_shear_the_right_way_round():
    """Higher shear must ignite EARLIER, on an identically-gated set (2, and 3's retraction)."""
    bio = _bio()
    d = _Pack(64)
    sr = np.linspace(0.5, 24.0, 64)
    hook = make_rollout_hook(ApClosure(C=100.0, q=1.0, kernel="static"), bio, sr)
    traj, _ = integrate_mat_trajectory(d, bio, np.ones(64), da_scale=40.0, ap_closure=hook)
    on = first_crossing(traj, float(bio.viscosity_mat_crit))
    ok = on >= 0
    assert ok.sum() > 8
    # rank correlation of shear with onset index must be negative
    ra = np.argsort(np.argsort(sr[ok])).astype(float)
    rb = np.argsort(np.argsort(on[ok])).astype(float)
    assert float(np.corrcoef(ra, rb)[0, 1]) < 0.0


def test_ungated_nodes_are_untouched():
    """gate == 0 -> no reaction, so the closure must not invent one (mask invariance, 9)."""
    bio = _bio()
    d = _Pack(32)
    gate = np.zeros(32)
    hook = make_rollout_hook(ApClosure(C=1e4, q=1.0, kernel="static"), bio, np.full(32, 5.0))
    traj, _ = integrate_mat_trajectory(d, bio, gate, da_scale=40.0, ap_closure=hook)
    assert np.array_equal(traj, np.zeros_like(traj))


# -------------------------------------------------------------- the two-scalar Damkohler

def test_da_scale_auto_none_is_bit_identical():
    """The second scalar is opt-in; `None` must not perturb the shipped one-scalar model."""
    bio = _bio()
    d = _Pack(48)
    g = np.ones(48)
    a, _ = integrate_mat_trajectory(d, bio, g, da_scale=40.0)
    b, _ = integrate_mat_trajectory(d, bio, g, da_scale=40.0, da_scale_auto=None)
    c, _ = integrate_mat_trajectory(d, bio, g, da_scale=40.0, da_scale_auto=40.0)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_a_larger_autocatalytic_scale_only_accelerates_after_mas_exists():
    """A_a multiplies ``(Mas/Minf)*k_aa*ap``, which is exactly zero until Mas > 0.

    So the first step must be untouched and every later step strictly faster.  If the first
    step moves, A_a has leaked into the fresh-deposition term and the two scalars are not
    separated after all.
    """
    bio = _bio()
    d = _Pack(48)
    g = np.ones(48)
    a, _ = integrate_mat_trajectory(d, bio, g, da_scale=40.0)
    b, _ = integrate_mat_trajectory(d, bio, g, da_scale=40.0, da_scale_auto=120.0)
    assert np.array_equal(a[1], b[1])
    assert np.all(b[-1] > a[-1])


def test_the_autocatalytic_scale_does_not_touch_the_surface_load():
    """``Mas`` obeys A_s alone; only ``Mat`` sees A_a.  Guards the two-equation split."""
    bio = _bio()
    d = _Pack(48)
    g = np.ones(48)
    crit = float(bio.viscosity_mat_crit)
    # Mas is not returned, so probe it through the saturation it causes: with a huge A_a the
    # Mat curve must run away while the FIRST crossing ordering stays shear-free (gate flat).
    on_a = first_crossing(integrate_mat_trajectory(d, bio, g, da_scale=40.0)[0], crit)
    on_b = first_crossing(
        integrate_mat_trajectory(d, bio, g, da_scale=40.0, da_scale_auto=400.0)[0], crit)
    assert np.all(on_b <= on_a)
    assert len(np.unique(on_b[on_b >= 0])) == 1      # still a flash: A_a alone cannot spread


# -------------------------------------------------------------- the SHIPPED configuration

def test_shipped_config_is_the_one_that_was_measured():
    """Pin the deployed constants.

    ``SHIPPED`` is what ``scripts/predict_wall_clot.py --temporal`` deploys, and its
    -7% growth_l1 on train AND SEALED was measured at exactly these values.  Changing them
    silently would invalidate that number while every test still passed.
    """
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE

    assert SHIPPED.kernel == "static"
    assert SHIPPED.q == 1.0
    assert abs(SHIPPED.C - 62.42) < 1e-6
    assert SHIPPED_DA_SCALE == 40.0
    assert SHIPPED.smooth_hops == 0        # 3.4 measured smoothing to be harmful


def test_shipped_closure_suppresses_and_orders_correctly():
    """The deployed operator must still be a bounded suppression with the measured sign."""
    from src.core_physics.ap_closure import SHIPPED

    bio = _bio()
    sr = np.array([0.5, 2.0, 8.0, 24.0])
    m = SHIPPED.multiplier(np.ones(4), sr, np.ones(4), np.zeros(4), np.zeros(4),
                           float(bio.k_as) * M_TO_CM, float(bio.k_aa) * M_TO_CM)
    assert np.all((m > 0.0) & (m <= 1.0))
    assert np.all(np.diff(m) > 0.0)        # higher shear -> less suppression -> earlier


# ------------------------------------------------------------------------ the smoother

def test_smoother_zero_hops_is_the_identity():
    ei = np.array([[0, 1, 2], [1, 2, 3]])
    sm = build_smoother(ei, 4, 0)
    v = np.array([1.0, 5.0, -2.0, 7.0])
    assert np.array_equal(sm(v), v)


def test_smoother_preserves_constants_and_contracts_range():
    """Row-stochastic: a constant field is a fixed point, and nothing is amplified."""
    ei = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    sm = build_smoother(ei, 4, 3)
    assert np.allclose(sm(np.full(4, 2.5)), 2.5)
    v = np.array([0.0, 10.0, 0.0, 0.0])
    assert np.ptp(sm(v)) < np.ptp(v)


# ------------------------------------------------------------------------------ the fit

def test_fit_C_recovers_a_planted_constant():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.001, 0.5, 4000)
    for c_true in (5.0, 68.0, 250.0):
        ratio = 1.0 / (1.0 + c_true * x)
        assert fit_C(ratio, x) == pytest.approx(c_true, rel=0.02)
