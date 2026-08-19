"""Pins for the PHASE10 additions: the advection operator and the readout constraints.

See `docs/PHASE10_V4.md`.  These are pure-function tests -- no weights, no cache, no CUDA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.calibration import RULES, apply_rule, rule_grid  # noqa: E402
from src.clot_ml.transport import (  # noqa: E402
    advect_source, residence_time, transport_fields, upwind_operator,
)


def _line_mesh(n=25, h=1.0):
    """A 1-D chain along +x: pos, edge_index, and a uniform rightward flow."""
    pos = np.stack([np.arange(n) * h, np.zeros(n)], axis=1).astype(np.float64)
    src = np.arange(n - 1)
    ei = np.stack([src, src + 1])
    u = np.ones(n)
    v = np.zeros(n)
    return pos, ei, u, v


def test_upwind_operator_only_carries_flux_downstream():
    """With flow along +x, node i may send to i+1 and never to i-1."""
    pos, ei, u, v = _line_mesh()
    F, out = upwind_operator(pos, ei, u, v)
    A = F.toarray()
    assert np.all(np.triu(A, 1) >= 0)
    # nothing flows upstream: the strictly-lower triangle is empty
    assert np.allclose(np.tril(A), 0.0)
    # the outlet node is the only one with no outflow
    assert out[-1] == pytest.approx(0.0)
    assert np.all(out[:-1] > 0)


def test_advected_source_accumulates_along_the_characteristic():
    """`u.grad(C) = S` with a source only at the inlet is constant downstream of it.

    This is the property the whole off-wall argument rests on: with `D = 0` the field at a
    node is the source integrated along the BACKWARD characteristic, so a node downstream
    of a source sees it and a node upstream does not.
    """
    pos, ei, u, v = _line_mesh()
    S = np.zeros(len(u))
    S[5] = 1.0
    C = advect_source(pos, ei, u, v, S, horizon=1e9)
    assert np.all(C[:5] < 1e-9)                 # upstream of the source: nothing
    assert C[6] > 0                             # downstream: carries it
    # and it does not decay, because there is no sink and no diffusion
    assert C[-1] == pytest.approx(C[6], rel=1e-6)


def test_residence_time_grows_downstream_and_is_capped_by_the_horizon():
    pos, ei, u, v = _line_mesh()
    tau = residence_time(pos, ei, u, v, horizon=1e9)
    assert np.all(np.diff(tau) > 0)
    short = residence_time(pos, ei, u, v, horizon=1.0)
    assert short[-1] < tau[-1]


def test_stagnant_flow_stays_finite():
    """No-slip drives `u -> 0` at the wall; the finite-horizon term must keep `C` bounded."""
    pos, ei, u, v = _line_mesh()
    C = advect_source(pos, ei, np.zeros_like(u), v, np.ones(len(u)), horizon=3.0)
    assert np.all(np.isfinite(C))
    assert np.all(C > 0)


def test_transport_fields_are_finite_and_named():
    pos, ei, u, v = _line_mesh()
    wall = np.zeros(len(u), dtype=bool)
    wall[:3] = True
    T = transport_fields(pos, ei, u, v, wall, np.linspace(1, 2, len(u)), horizon=10.0)
    assert set(T) == {"mat_adv", "tau", "mat_adv_n", "src_reach"}
    for k, arr in T.items():
        assert arr.shape == (len(u),), k
        assert np.all(np.isfinite(arr)), k


def test_calibration_rules_never_label_outside_their_domain():
    rng = np.random.default_rng(0)
    n = 200
    score = rng.random(n)
    phys = rng.random(n) > 0.7
    dom = np.zeros(n, dtype=bool)
    dom[:80] = True
    for name in RULES:
        for p in rule_grid(name)[:: max(len(rule_grid(name)) // 4, 1)]:
            m = apply_rule(name, score, dom, phys, p)
            assert m.dtype == bool
            assert not m[~dom].any(), name


def test_absolute_rule_reproduces_a_plain_threshold():
    """`absolute` is the control; it must be exactly the shipped readout."""
    rng = np.random.default_rng(1)
    score = rng.random(150)
    dom = np.ones(150, dtype=bool)
    for t in (0.1, 0.5, 0.9):
        assert np.array_equal(apply_rule("absolute", score, dom, dom, t), score >= t)


def test_commit_by_final_makes_the_last_mask_equal_the_set():
    """The committed set IS the prediction of the final mask (docs/PHASE10_V4.md 3)."""
    from eval_strict_temporal import series_masks

    rng = np.random.default_rng(2)
    n, T = 60, 7
    gm = rng.random(n) > 0.5
    P = np.sort(rng.random((T, n)), axis=0)          # monotone in time
    M = series_masks(gm, P, 0.8, commit_final=True)
    assert np.array_equal(M[-1], gm)
    # monotone: a node never un-clots
    assert np.all(M[1:] >= M[:-1])
    # and every committed node is in the set at every time
    assert np.all(M <= gm[None, :])


def test_commit_by_final_can_be_disabled():
    from eval_strict_temporal import series_masks

    gm = np.ones(5, dtype=bool)
    P = np.zeros((3, 5))                              # nothing ever crosses
    assert series_masks(gm, P, 0.5, commit_final=True)[-1].all()
    assert not series_masks(gm, P, 0.5, commit_final=False)[-1].any()


def test_v4_manifest_is_consistent_and_excludes_sealed():
    """The locked v4 artifact on disk: 9 members, SEALED unseen, feature width declared."""
    import json

    from src.clot_ml.features_v4 import V4_CHANNELS
    from src.core_physics.wall_cohort_splits import SEALED

    root = REPO / "outputs/clot_ml/locked/clot_gnn_v4"
    if not root.exists():
        pytest.skip("clot_gnn_v4 not promoted here")
    m = json.loads((root / "manifest.json").read_text())
    assert m["n_members"] == len(m["members"]) == 9
    for mem in m["members"]:
        assert (root / mem["file"]).exists(), mem["file"]
    assert not set(m["training_pool"]) & set(SEALED), "SEALED leaked into training"
    assert list(m["v4_channels"]) == list(V4_CHANNELS)
    norm = np.load(root / "feature_norm.npz", allow_pickle=True)
    cols = [str(c) for c in norm["cols"]]
    assert len(cols) == m["n_features"]
    assert cols[-1] == "phys_mask"
    assert cols[-14:-1] == list(V4_CHANNELS)
