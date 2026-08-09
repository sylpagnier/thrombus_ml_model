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
