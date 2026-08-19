"""Guards for the shipped lumen arm.

The lumen arm shipped for a long time at ``LUMEN_SPEED = 0.3``, where it scored **worse than
not running it at all** on the full mesh (0.7613 vs 0.7651 wall-only).  Nobody noticed because
every score in the phase passed ``wall_mask=wall``, which makes off-wall clot -- 17% of GT --
invisible.  These pin the retuned constants and the operator's contract so that cannot recur
silently.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.core_physics.physics_lumen_model import grow_into_lumen


def _line(n):
    a = np.arange(n - 1)
    ei = np.stack([np.concatenate([a, a + 1]), np.concatenate([a + 1, a])])
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def test_shipped_lumen_constants_are_the_swept_optimum():
    """PHASE6_RESULTS 21.1.  0.3 was measurably worse than not running the arm."""
    import scripts.predict_wall_clot as pw

    assert pw.LUMEN_HOPS == 2
    assert abs(pw.LUMEN_SPEED - 0.2) < 1e-9
    assert pw.GROW_HOPS == 20
    assert abs(pw.RELAX - 2.0) < 1e-9


def test_grow_into_lumen_never_returns_wall_nodes():
    """The arm exists to predict OFF-wall clot; returning wall nodes would double-count."""
    n = 10
    A = _line(n)
    wall = np.zeros(n, dtype=bool)
    wall[:4] = True
    seed = np.zeros(n, dtype=bool)
    seed[3] = True
    off = grow_into_lumen(seed, wall, A, np.zeros(n), np.zeros(n),
                          lumen_hops=3, speed_thresh=1.0)
    assert not (off & wall).any()
    assert not (off & seed).any()


def test_speed_threshold_gates_admission():
    """Only stagnant lumen is admissible -- the whole physical premise of the arm."""
    n = 8
    A = _line(n)
    wall = np.zeros(n, dtype=bool)
    wall[0] = True
    seed = np.zeros(n, dtype=bool)
    seed[0] = True
    fast = np.full(n, 0.9)
    slow = np.full(n, 0.05)
    assert grow_into_lumen(seed, wall, A, fast, np.zeros(n),
                           lumen_hops=3, speed_thresh=0.2).sum() == 0
    assert grow_into_lumen(seed, wall, A, slow, np.zeros(n),
                           lumen_hops=3, speed_thresh=0.2).sum() > 0


def test_hops_bound_the_reach():
    n = 12
    A = _line(n)
    wall = np.zeros(n, dtype=bool)
    wall[0] = True
    seed = np.zeros(n, dtype=bool)
    seed[0] = True
    spd = np.zeros(n)
    for hops in (1, 2, 3):
        off = grow_into_lumen(seed, wall, A, spd, np.zeros(n),
                              lumen_hops=hops, speed_thresh=1.0)
        assert int(off.sum()) == hops


def test_zero_hops_is_a_no_op():
    n = 6
    A = _line(n)
    wall = np.zeros(n, dtype=bool)
    seed = np.zeros(n, dtype=bool)
    seed[0] = True
    assert grow_into_lumen(seed, wall, A, np.zeros(n), np.zeros(n),
                           lumen_hops=0, speed_thresh=1.0).sum() == 0
