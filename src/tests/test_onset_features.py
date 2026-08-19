"""Guards for the direct-onset feature builder.

These features exist because the ODE readout loses ordering its own inputs carry.  The
failure modes worth guarding are the silent ones: a feature column drifting out of sync
with ``FEATURE_NAMES`` (every fitted coefficient vector would then be mislabelled), the
committed-set reconstruction diverging from the shipped predictor's, and hop distance
quietly returning a constant, which would make the one lever that worked a no-op.
"""

from __future__ import annotations

import numpy as np

from src.config import BiochemConfig
from src.core_physics.onset_features import (
    FEATURE_NAMES,
    build_features,
    committed_set,
    hop_distance,
)


def _line_graph(n=12):
    """A path graph: 0-1-2-...-n-1, as an undirected edge list."""
    a = np.arange(n - 1)
    return np.stack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])])


def test_hop_distance_is_the_graph_distance_on_a_path():
    e = _line_graph(10)
    seed = np.zeros(10, dtype=bool)
    seed[0] = True
    d = hop_distance(seed, e, max_hops=6)
    assert d[0] == 0
    assert list(d[:7]) == [0, 1, 2, 3, 4, 5, 6]
    assert d[9] == 7          # beyond max_hops -> max_hops + 1


def test_hop_distance_is_not_constant_when_seeds_are_partial():
    """A constant hop field would make the propagation lever a silent no-op."""
    e = _line_graph(12)
    seed = np.zeros(12, dtype=bool)
    seed[0] = True
    assert np.ptp(hop_distance(seed, e)) > 0


def test_committed_set_contains_every_seed_and_only_grows():
    # A 20-node path with one seed at index 5: GROW=6 hops reaches 0..11 and no further,
    # so the growth radius itself is under test, not just "something grew".
    e = _line_graph(20)
    gate = np.zeros(20)
    gate[5] = 1.0
    sr = np.zeros(20)                       # everything admissible
    S = committed_set(gate, sr, e)
    assert S[5]
    assert np.array_equal(np.where(S)[0], np.arange(0, 12))
    tight = committed_set(gate, np.full(20, 1e6), e)
    assert tight.sum() == 1                 # nothing admissible -> seeds only


def test_feature_matrix_matches_the_declared_column_order():
    """FEATURE_NAMES is the contract every saved coefficient vector is read against."""
    bio = BiochemConfig(phase="biochem")
    n = 24
    rng = np.random.default_rng(0)
    z = {
        "sr0": rng.uniform(0.5, 200.0, n),
        "dsrx0": rng.uniform(-3000.0, 3000.0, n),
        "pos": rng.uniform(0.0, 1.0, (n, 2)),
        "wall_edges": _line_graph(n),
    }
    X, S = build_features(z, bio, C=50.0)
    assert X.shape == (n, len(FEATURE_NAMES))
    assert np.isfinite(X).all()
    assert S.dtype == bool and S.shape == (n,)


def test_gate_columns_reproduce_the_law_bracket():
    bio = BiochemConfig(phase="biochem")
    n = 16
    z = {
        "sr0": np.linspace(1.0, 100.0, n),
        "dsrx0": np.linspace(-2000.0, 2000.0, n),
        "pos": np.stack([np.linspace(0, 1, n), np.zeros(n)], 1),
        "wall_edges": _line_graph(n),
    }
    X, _ = build_features(z, bio, C=50.0)
    gate = X[:, FEATURE_NAMES.index("gate")]
    lo = X[:, FEATURE_NAMES.index("gate_low")]
    se = X[:, FEATURE_NAMES.index("gate_sep")]
    coef = float(bio.L_char) * 100.0 / float(bio.gamma_m)
    assert np.allclose(gate, se * coef * np.abs(z["dsrx0"]) + lo)


def test_ap_closure_feature_is_a_suppression():
    bio = BiochemConfig(phase="biochem")
    n = 16
    z = {
        "sr0": np.linspace(0.5, 24.0, n),
        "dsrx0": np.zeros(n),
        "pos": np.stack([np.linspace(0, 1, n), np.zeros(n)], 1),
        "wall_edges": _line_graph(n),
    }
    X, _ = build_features(z, bio, C=50.0)
    ap = X[:, FEATURE_NAMES.index("ap_closure")]
    assert np.all(ap > 0) and np.all(ap <= 1.0)
    assert np.all(np.diff(ap) > 0)          # higher shear -> less suppression (sign guard)
