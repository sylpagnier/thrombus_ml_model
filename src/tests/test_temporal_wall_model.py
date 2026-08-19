"""Guards for the temporal arms of the physics wall model.

The final-mask sweep could not distinguish ``da_scale`` 50 from 1000 -- every value above
~50 gives a bit-identical committed set (docs/PHASE3_RESULTS.md §3). The growth CURVE can:
``da_scale`` 100 -> 40 halves ``curve_l1`` at unchanged deploy score. So the timing
machinery is only meaningful if these invariants hold, and none of them is visible to the
mask metric.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import BiochemConfig
from src.core_physics.physics_wall_model import (
    T0Fields,
    first_crossing,
    gate_from_shear,
    graded_gate,
    integrate_mat_trajectory,
)
from src.core_physics.temporal_metrics import curve_l1, onset_metrics, spearman


def _fields(sr, dsrx, bio):
    return T0Fields(
        sr=sr, dsrx=dsrx,
        gate_low=(sr < float(bio.lss)).astype(np.float64),
        gate_sep=(dsrx < float(bio.sgt) / 100.0).astype(np.float64),
        gate=None,
    )


def test_hard_mode_reproduces_the_law_bracket_exactly():
    bio = BiochemConfig(phase="biochem")
    sr = np.array([1.0, 10.0, 24.9, 25.1, 100.0])
    dsrx = np.array([-1e4, -800.0, -100.0, 0.0, 500.0])
    f = _fields(sr, dsrx, bio)
    g = graded_gate(f, bio, mode="hard")
    L_cm = float(bio.L_char) * 100.0
    ref = f.gate_sep * (L_cm / float(bio.gamma_m)) * np.abs(dsrx) + f.gate_low
    assert np.allclose(g, ref)


def test_gate_from_shear_matches_the_graded_hard_gate():
    """The one gate expression every evolving-flow arm shares.

    The GT-flow oracle, the corrector rollout and the frozen t=0 fields each used to carry
    their own transcription of the law's bracket. If they drift apart, an ablation between
    them measures the transcription rather than the flow, so pin them to one function.
    """
    bio = BiochemConfig(phase="biochem")
    sr = np.array([1.0, 10.0, 24.9, 25.1, 100.0])
    dsrx = np.array([-1e4, -800.0, -100.0, 0.0, 500.0])
    assert np.allclose(gate_from_shear(sr, dsrx, bio),
                       graded_gate(_fields(sr, dsrx, bio), bio, mode="hard"))
    wall = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    assert np.allclose(gate_from_shear(sr, dsrx, bio, wall=wall),
                       gate_from_shear(sr, dsrx, bio) * wall)


def test_graded_gate_is_monotone_in_the_margin():
    """Deeper inside the stagnation zone must never ignite slower than borderline."""
    bio = BiochemConfig(phase="biochem")
    sr = np.linspace(1.0, 120.0, 40)
    dsrx = np.zeros_like(sr)
    g = graded_gate(_fields(sr, dsrx, bio), bio, mode="sigmoid_low", tau_low=0.1)
    assert np.all(np.diff(g) <= 1e-12), "graded gate must decrease with shear"
    assert g[0] > 0.99 and g[-1] < 0.01


def test_graded_gate_recovers_the_hard_step_as_tau_goes_to_zero():
    bio = BiochemConfig(phase="biochem")
    sr = np.array([5.0, 24.0, 26.0, 90.0])
    f = _fields(sr, np.zeros_like(sr), bio)
    g = graded_gate(f, bio, mode="sigmoid_low", tau_low=1e-4)
    assert np.allclose(g, graded_gate(f, bio, mode="hard"), atol=1e-6)


def _toy_pack(gate_vals, nt=51, horizon=30000.0):
    n = len(gate_vals)
    d = type("P", (), {})()
    d.t = torch.tensor(np.linspace(0.0, horizon, nt), dtype=torch.float64)
    d.mask_wall = torch.ones(n, dtype=torch.bool)
    names = ["RP_log1p_nd", "AP_log1p_nd"]
    d.y_channel_names = ",".join(names)
    y = torch.zeros(nt, n, 2, dtype=torch.float32)
    y[:, :, 0] = float(np.expm1(0.0) + 1e-6)
    d.y = torch.zeros(nt, n, 2)
    d.y[:, :, 0] = 1.0e-6      # RP nd -> 2.5e14 * 1e6 * 1e-6 plt/cm^3 scale handled in model
    d.y[:, :, 1] = 5.0e-8
    return d


def test_a_uniform_gate_ignites_every_node_in_the_same_step():
    """The flash this work exists to fix, pinned so the diagnosis stays reproducible."""
    bio = BiochemConfig(phase="biochem")
    gate = np.ones(20)
    d = _toy_pack(gate)
    traj, t = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    assert (idx >= 0).all()
    assert idx.min() == idx.max(), "identical gates must give identical ignition times"


def test_a_graded_gate_spreads_ignition_times():
    bio = BiochemConfig(phase="biochem")
    gate = np.linspace(0.15, 1.0, 20)
    d = _toy_pack(gate)
    traj, t = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    lit = idx[idx >= 0]
    assert lit.max() > lit.min(), "a graded gate must produce a spread of onset times"
    # strongest gate ignites first
    assert idx[-1] <= idx[0] or idx[0] < 0


def test_blockage_callable_is_applied_and_can_stop_growth():
    bio = BiochemConfig(phase="biochem")
    gate = np.ones(10)
    d = _toy_pack(gate)
    free, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
    shut, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0,
                                       blockage=lambda mat, g0, i: g0 * 0.0)
    assert float(free[-1].max()) > 0.0
    assert float(shut[-1].max()) == pytest.approx(0.0)


def test_washout_defaults_off_and_is_bit_identical():
    """``washout=0`` must reproduce the accumulate-only trajectory exactly, not approximately.

    Every Phase 3-7 number was produced by the accumulate-only rollout, so if the default path
    moves even in the last bit those results stop being comparable.
    """
    bio = BiochemConfig(phase="biochem")
    gate = np.linspace(0.2, 1.0, 12)
    d = _toy_pack(gate)
    a, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
    b, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=0.0,
                                    washout_sr=np.full(12, 50.0))
    assert np.array_equal(a, b)


def test_washout_needs_its_shear_field():
    bio = BiochemConfig(phase="biochem")
    gate = np.ones(5)
    with pytest.raises(ValueError, match="washout_sr"):
        integrate_mat_trajectory(_toy_pack(gate), bio, gate, washout=1e-6)


def test_washout_callable_matches_a_static_field():
    """A frozen callable must be bit-identical to the static [N] path -- the wake hook's baseline."""
    bio = BiochemConfig(phase="biochem")
    gate = np.linspace(0.3, 1.0, 8)
    d = _toy_pack(gate)
    sr = np.linspace(10.0, 80.0, 8)
    a, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=1e-5, washout_sr=sr)
    b, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=1e-5,
                                    washout_sr=lambda mat, i: sr)
    assert np.array_equal(a, b)


def test_washout_2d_uses_the_step_row():
    """Later high shear must scavenge more than a freeze of the first-step shear."""
    bio = BiochemConfig(phase="biochem")
    gate = np.ones(5)
    d = _toy_pack(gate, nt=21)
    n_t = int(d.t.shape[0])
    sr_lo = np.full(5, 10.0)
    sr_hi = np.full(5, 400.0)
    frozen, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=1e-5,
                                         washout_sr=sr_lo)
    stacked = np.vstack([np.tile(sr_lo, (n_t // 2, 1)),
                         np.tile(sr_hi, (n_t - n_t // 2, 1))])
    evolved, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=1e-5,
                                          washout_sr=stacked)
    assert float(evolved[-1].mean()) < float(frozen[-1].mean())


def test_washout_removes_more_where_shear_is_higher():
    """The whole point of the term: it must be a SHEAR-ordered sink, not a uniform decay."""
    bio = BiochemConfig(phase="biochem")
    gate = np.ones(4)
    d = _toy_pack(gate)
    sr = np.array([1.0, 10.0, 100.0, 1000.0])
    traj, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0, washout=1e-5,
                                       washout_sr=sr)
    fin = traj[-1]
    assert np.all(np.diff(fin) < 0), "higher shear must retain less Mat"
    plain, _ = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
    assert np.all(fin <= plain[-1] + 1e-12), "removal can only reduce Mat"


def test_washout_step_is_stable_where_explicit_euler_would_diverge():
    """``h*decay >> 1`` happens on real packs: h is 150 s and lambda*sr reaches ~1e-2 1/s."""
    from src.core_physics.physics_wall_model import washout_step

    mat = np.array([1.0e8, 1.0e8])
    decay = np.array([1.0e-2, 1.0])          # h*decay = 1.5 and 150
    out = washout_step(mat, np.zeros(2), 150.0, decay)
    assert np.all(out >= 0.0) and np.all(out < mat), "must decay monotonically, never overshoot"
    # An explicit step would have gone negative and then oscillated.
    assert np.all(mat - 150.0 * decay * mat < 0.0)


def test_washout_step_reaches_the_analytic_steady_state():
    """``dMat/dt = S - k*Mat`` must relax to ``S/k``, which is the ordering the term implies."""
    from src.core_physics.physics_wall_model import washout_step

    src, decay = np.array([2.0]), np.array([1e-3])
    mat = np.zeros(1)
    for _ in range(4000):
        mat = washout_step(mat, src, 10.0, decay)
    assert mat[0] == pytest.approx(float(src[0] / decay[0]), rel=1e-6)


def test_curve_l1_is_zero_for_identical_onset_distributions():
    t = np.linspace(0, 30000, 61)
    wall = np.ones(30, dtype=bool)
    idx = np.arange(30) % 60
    assert curve_l1(idx, idx.copy(), t, wall) == pytest.approx(0.0, abs=1e-12)


def test_onset_metrics_flag_a_flash_against_a_spread_gt():
    t = np.linspace(0, 30000, 61)
    wall = np.ones(40, dtype=bool)
    flash = np.full(40, 10)
    spread = np.arange(40)
    m = onset_metrics(flash, spread, t, wall)
    assert m["spread_model"] == pytest.approx(0.0)
    assert m["spread_ratio"] == pytest.approx(0.0)
    assert curve_l1(flash, spread, t, wall) > 0.1


def test_spearman_handles_degenerate_input():
    assert np.isnan(spearman(np.ones(5), np.arange(5)))


def test_pred_flow_mls_on_velocity_ignores_a_cached_shear_head():
    """The kinematics shear head is cached but not the deployable sr.

    On patient005 the head has wall corr 0.17 vs GT MLS and never trips sgt; MLS-on-u0
    keeps wall corr 0.82.  A flat cached head must not flatten the gate.
    """
    from src.core_physics.physics_wall_model import t0_flow_fields

    bio = BiochemConfig(phase="biochem")
    n = 8
    d = type("P", (), {})()
    d.x = torch.zeros(n, 2, dtype=torch.float64)
    d.x[:, 0] = torch.arange(n, dtype=torch.float64)
    a = np.arange(n - 1)
    ei = np.stack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])])
    d.edge_index = torch.tensor(ei, dtype=torch.long)
    d.u_ref = torch.tensor([1.0])
    d.d_bar = torch.tensor([1.0])
    d.u0_pred = torch.linspace(0.0, 8.0, n)
    d.v0_pred = torch.zeros(n)
    d.sr0_pred = torch.full((n,), 5.0)
    f = t0_flow_fields(d, bio, hops=2, flow_source="pred")
    assert not np.allclose(f.sr, 5.0)
