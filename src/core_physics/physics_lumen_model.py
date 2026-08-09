"""Phase-3 lumen (off-wall) arm -- the physics replacement for the learned lumen specialist.

The wall arm (:mod:`src.core_physics.physics_wall_model`) predicts where platelets commit
on the vessel wall.  Off-wall clot is 20.9% of all GT clot in the cohort, sits almost
entirely at 2-3 graph hops from the wall, and -- measured over 34 vessels -- **never**
nucleates away from committed wall tissue (0 orphans out of 890 off-wall clot nodes).
So the lumen arm is a propagation rule seeded by the wall arm, not an independent model.

It is genuinely clot, not a rheology artefact: on the patient007 domain export the COMSOL
gelation step ``mu1(Mat)`` is fully saturated (79 of a possible 80) at off-wall clot nodes,
fibrin ``mu2`` is identically zero, and clear-lumen nodes show ``d(spf.mu) = -3e-4``.

The rule, three scalars, all fit on ``WALL_COHORT_V2_TRAIN``:

    admit an off-wall node within ``lumen_hops`` of predicted wall clot when its t=0
    speed satisfies  ``speed_nd < speed_thresh``  and its shear ``sr < sr_max``

``speed_nd`` is ``|u|/u_ref``, already vessel-normalised, so the threshold transfers:
off-wall clot sits at median 0.25 against 1.12 for clear lumen.  Growth is barrier-limited
rather than distance-limited, which is why admission is re-tested at every hop.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                      shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def speed_nd(data) -> np.ndarray:
    """|u|/u_ref at t=0 -- dimensionless, hence comparable across vessels."""
    u = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
    v = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
    return np.hypot(u, v)


def speed_nd_pred(data) -> np.ndarray:
    """Deployable variant: the kinematic model's t=0 flow instead of the GT field."""
    u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
    v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
    return np.hypot(u, v)


def grow_into_lumen(
    wall_clot: np.ndarray,
    wall: np.ndarray,
    A: sp.csr_matrix,
    spd: np.ndarray,
    sr: np.ndarray,
    *,
    lumen_hops: int = 3,
    speed_thresh: float = 0.5,
    sr_max: float = np.inf,
) -> np.ndarray:
    """Off-wall clot mask, propagated from ``wall_clot`` into stagnant lumen."""
    admissible = (~wall) & (spd < speed_thresh) & (sr < sr_max)
    cur = wall_clot.copy()
    off = np.zeros_like(wall_clot)
    for _ in range(max(int(lumen_hops), 0)):
        nxt = ((A @ cur.astype(np.int8)) > 0) & admissible & ~cur
        if not nxt.any():
            break
        off = off | nxt
        cur = cur | nxt
    return off


def wall_normal_projection(pos: np.ndarray, wall: np.ndarray):
    """For every node: distance to the nearest wall node, and that wall node's index.

    Graph-hop dilation is the wrong operator for the lumen. Measured across the cohort,
    off-wall GT clot sits in a razor-thin shell at a near-constant NORMAL offset from the
    wall (patient032: all 120 nodes between 0.0459 and 0.0477), and each one's nearest
    wall node is always a committed one. Normalised by the mesh's median edge length that
    offset is ~1.7-1.8 on every vessel. So the lumen arm is a thickness in the wall-normal
    direction, and the pack's ``edge_index`` -- 64% of whose nodes are unreachable from
    the wall -- cannot express it.
    """
    from scipy.spatial import cKDTree

    idx = np.flatnonzero(wall)
    dist, which = cKDTree(pos[wall]).query(pos)
    return dist, idx[which]


def median_edge_length(pos: np.ndarray, edge_index: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], axis=1)))


def lumen_thickness_layer(
    wall_clot: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    spd: np.ndarray,
    *,
    thickness_edges: float = 1.8,
    speed_thresh: float = np.inf,
) -> np.ndarray:
    """Off-wall clot as a wall-normal thickness behind committed wall tissue.

    Three scalars in total for the arm: ``thickness_edges`` (in units of the mesh's median
    edge length, so it transfers across meshes), ``speed_thresh`` on the t=0 normalised
    speed, and the wall arm's own seed.
    """
    dist, owner = wall_normal_projection(pos, wall)
    h = median_edge_length(pos, edge_index)
    return ((~wall) & (dist < thickness_edges * h) & wall_clot[owner] & (spd < speed_thresh))


def radius_neighbors(pos: np.ndarray, radius: float):
    """Physical-radius neighbour lists (CSR-style) -- NOT graph hops.

    The packs' ``edge_index`` leaves 64% of nodes unreachable from the wall, so hop counts
    cannot express lumen geometry. Euclidean radius can, and off-wall clot is organised by
    physical offset (a shell 1.7-1.8 median edge lengths out from committed wall).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    return tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")


def autocatalytic_lumen(
    wall_clot: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    *,
    r_nuc: float = 1.8,
    expose_thresh: float = 0.4,
    n_steps: int = 6,
    spd: np.ndarray | None = None,
    speed_thresh: float = np.inf,
) -> np.ndarray:
    """Autocatalytic lumen growth with nucleation limited to a PHYSICAL radius.

    Mirrors the local structure of the COMSOL law rather than the neighbour-aggregating
    GraphSAGE the project retired (PHASE3_HANDOFF 1.1): a node ignites on its own local
    state, and the only thing its neighbours supply is exposure to existing clot.

    Per step, an uncommitted off-wall node commits when the committed FRACTION of the
    nodes inside its radius-``r_nuc`` ball reaches ``expose_thresh``.  That fraction is the
    brake: a node on an open-lumen frontier sees a small fraction and never ignites, while
    a node in a pocket that is already mostly clot does.  Without it the ball-based rule
    runs away and fills the lumen, exactly as the Da sweep does on the wall
    (docs/PHASE3_RESULTS.md 3).

    Scalars: ``r_nuc`` (in median edge lengths, so it transfers across meshes),
    ``expose_thresh``, ``n_steps``.
    """
    h = median_edge_length(pos, edge_index)
    M = radius_neighbors(pos, r_nuc * h).tocsr()
    M.data[:] = 1.0
    M.setdiag(0.0)
    M.eliminate_zeros()
    ball = np.asarray(M.sum(axis=1)).reshape(-1)          # neighbours per node
    ball = np.maximum(ball, 1.0)

    eligible = ~wall
    if spd is not None:
        eligible = eligible & (spd < speed_thresh)

    cur = wall_clot.copy()
    for _ in range(max(int(n_steps), 0)):
        exposure = np.asarray(M @ cur.astype(np.float64)).reshape(-1) / ball
        nxt = eligible & ~cur & (exposure >= expose_thresh)
        if not nxt.any():
            break
        cur = cur | nxt
    return cur & ~wall


def predict_compound(
    data,
    wall_clot: np.ndarray,
    sr: np.ndarray,
    *,
    lumen_hops: int = 3,
    speed_thresh: float = 0.5,
    sr_max: float = np.inf,
    flow: str = "gt",
) -> np.ndarray:
    """Full-mesh clot mask: the wall arm's prediction plus its lumen propagation."""
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    n = len(wall)
    A = adjacency(data.edge_index.detach().cpu().numpy(), n)
    spd = speed_nd_pred(data) if flow == "pred" else speed_nd(data)
    off = grow_into_lumen(wall_clot, wall, A, spd, sr, lumen_hops=lumen_hops,
                          speed_thresh=speed_thresh, sr_max=sr_max)
    return wall_clot | off
