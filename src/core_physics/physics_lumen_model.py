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

SUPERSEDED BY :func:`grow_into_lumen_by_mat` -- see docs/PHASE7_FINDINGS.md.  The premise
above ("a propagation rule seeded by the wall arm") is measurably wrong: a pure
thickness/dilation rule seeded on **perfect GT wall clot** peaks at off-wall F1 0.275.
Off-wall clot is a ``Mat`` MAGNITUDE problem, not a mask-propagation problem.
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
    u, v = uv_nd(data, flow="pred")
    return np.hypot(u, v)


def uv_nd(data, *, flow: str = "gt") -> tuple[np.ndarray, np.ndarray]:
    """t=0 (u, v) in the pack's non-dimensional units.

    ``flow='pred'`` is RGP-DEQ ``u0_pred``/``v0_pred`` (deployable).  ``flow='gt'`` is the
    COMSOL field at t=0 -- labels / oracles only, never a generalization claim.
    """
    if flow == "pred":
        if getattr(data, "u0_pred", None) is None:
            raise ValueError("pack has no u0_pred (deployable flow unavailable)")
        u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        return u, v
    u = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
    v = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)
    return u, v


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


def midside_nodes(pos: np.ndarray, edge_index: np.ndarray, *, tol: float = 0.05) -> np.ndarray:
    """The mesh's SECOND-ORDER (mid-edge) nodes -- ~3/4 of every pack, and M/Mas/Mat are
    not defined on them.

    The COMSOL meshes are quadratic triangles, so the exported node set is corner nodes
    PLUS one node at the midpoint of every element edge.  For a triangulation that is
    asymptotically 3 mid-side nodes per corner node, and the packs measure **0.742-0.746
    mid-side on all 19 train vessels** -- i.e. exactly 3/4, which is the confirmation that
    this is the P2 structure and not a coincidence of one meshing recipe.

    It matters because the velocity interface is quadratic while *Transport of Diluted
    Species* is linear, so ``M/Mas/Mat`` carry no value at a mid-side node while ``u, v, p``
    carry a normal one.  Anything that thresholds the species field therefore has to know
    which nodes can hold it:

    * off-wall (:data:`SHELL_SPECIES_LO`): the first off-wall node family, at ~1.0 median
      edge lengths, is entirely mid-side and receives ``Mat/Mat_owner`` of 0.000 against the
      next family's 0.155 on every vessel.  Admitting it is ~23 guaranteed false positives
      per vessel;
    * on the wall: **49.6% of wall nodes are mid-side**, and GT ``Mat`` is zero at 44.6% of
      those against 17.6% of corner wall nodes, so a rank correlation taken over all wall
      nodes is partly ranking a block of structural zeros (docs/PHASE7_FINDINGS.md 8.5).

    Detected from topology alone -- degree 2 and positioned at the midpoint of its two
    neighbours -- so unlike a length in mesh units this transfers to any mesh.  ``tol`` is a
    fraction of the median edge length.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    A = adjacency(np.asarray(edge_index), n)
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    out = np.zeros(n, dtype=bool)
    cand = np.flatnonzero(deg == 2)
    if len(cand) == 0:
        return out
    # deg == 2 exactly, so each candidate row of the CSR holds its two neighbours adjacently.
    start = A.indptr[cand]
    a, b = A.indices[start], A.indices[start + 1]
    err = np.linalg.norm(pos[cand] - 0.5 * (pos[a] + pos[b]), axis=1)
    out[cand[err < tol * median_edge_length(pos, edge_index)]] = True
    return out


def wall_normal_midside(pos: np.ndarray, wall: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """The empty near-wall family: mid-edge nodes of the WALL-NORMAL edges.

    These sit at ~1.0 median edge lengths and carry ``Mat`` identically zero -- they are the
    "phantom band" of docs/PHASE7_FINDINGS.md 8.  Being mid-side is *not* by itself what
    makes a node empty (the mid-side nodes lying along the first species row carry ``Mat``
    normally and hold 170 of the cohort's 493 off-wall GT clot nodes); being the mid-edge
    node of an edge that crosses *from* the wall is.  Identified with no length at all.
    """
    wall = np.asarray(wall, dtype=bool)
    A = adjacency(np.asarray(edge_index), len(wall))
    return midside_nodes(pos, edge_index) & ~wall & (A @ wall.astype(np.int8) > 0)


def resolve_offwall_shell(
    pos: np.ndarray,
    wall: np.ndarray,
    edge_index: np.ndarray,
    *,
    shell_lo: float | None = None,
    shell_hi: float | None = None,
) -> np.ndarray:
    """The off-wall shell, preferring the topological rule and falling back to the band.

    :func:`first_corner_shell` needs the mesh to be quadratic, because it navigates by the
    mid-edge family.  On a linear mesh there is no such family and it would select nothing,
    so detect that case and fall back to the calibrated distance band rather than silently
    returning an empty shell.  Explicit ``shell_lo``/``shell_hi`` always force the band.
    """
    if shell_lo is None and shell_hi is None:
        shell = first_corner_shell(pos, wall, edge_index)
        if shell.any():
            return shell
    dist, _ = wall_normal_projection(pos, wall)
    h = median_edge_length(pos, edge_index)
    lo = SHELL_SPECIES_LO if shell_lo is None else float(shell_lo)
    hi = SHELL_SPECIES_HI if shell_hi is None else float(shell_hi)
    return (~np.asarray(wall, dtype=bool)) & (dist > lo * h) & (dist < hi * h)


def topological_owner(pos: np.ndarray, wall: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Wall corner of the wall-normal P2 edge, or ``-1`` if the node is not on that bridge.

    Euclidean nearest-wall is often a mid-side wall node with structurally zero Mat
    (docs/PHASE7_FINDINGS.md §8.5) or a neighbour in a different element.  The wall-normal
    mid-side bridge names the element the shell node lives in: bridge -- wall corner --
    far-side corner.  Along-row mid-sides inherit either adjacent corner's owner.
    """
    wall = np.asarray(wall, dtype=bool)
    n = len(wall)
    A = adjacency(np.asarray(edge_index), n)
    ms = midside_nodes(pos, edge_index)
    bridge = wall_normal_midside(pos, wall, edge_index)
    owner = np.full(n, -1, dtype=np.int64)
    bidx = np.flatnonzero(bridge)
    for b in bidx:
        nbrs = A.indices[A.indptr[b]:A.indptr[b + 1]]
        w = nbrs[wall[nbrs]]
        far = nbrs[(~wall[nbrs]) & (~ms[nbrs])]
        if w.size == 0 or far.size == 0:
            continue
        owner[far] = int(w[0])
    along = np.flatnonzero(first_corner_shell(pos, wall, edge_index) & ms)
    for a in along:
        nbrs = A.indices[A.indptr[a]:A.indptr[a + 1]]
        ow = owner[nbrs]
        ow = ow[ow >= 0]
        if ow.size:
            owner[a] = int(ow[0])
    return owner


def p2_bridge_normal_speed(
    u: np.ndarray, v: np.ndarray, pos: np.ndarray, wall: np.ndarray, edge_index: np.ndarray,
) -> np.ndarray:
    """|u| at the wall-normal P2 midpoint -- the FEM face flux into the first species cell.

    No-slip makes the wall velocity zero, so Euclidean |u·n| at a wall owner is identically
    zero.  The mid-edge node of the wall-normal edge is the face the ``tds2`` flux actually
    crosses.  Shell nodes inherit the mean speed of neighbouring bridge nodes.
    """
    wall = np.asarray(wall, dtype=bool)
    n = len(wall)
    A = adjacency(np.asarray(edge_index), n)
    bridge = wall_normal_midside(pos, wall, edge_index)
    spd = np.hypot(np.asarray(u, dtype=np.float64).reshape(-1),
                   np.asarray(v, dtype=np.float64).reshape(-1))
    nbr_sum = np.asarray(A.astype(np.float64) @ (spd * bridge.astype(np.float64))).reshape(-1)
    nbr_n = np.asarray(A @ bridge.astype(np.int8)).reshape(-1).astype(np.float64)
    out = np.zeros(n, dtype=np.float64)
    ok = nbr_n > 0
    out[ok] = nbr_sum[ok] / nbr_n[ok]
    # Along-row mid-sides sit on the species row but do not touch the bridge.  Inherit
    # the face flux from the two adjacent corners so they are not scored at the floor.
    along = first_corner_shell(pos, wall, edge_index) & midside_nodes(pos, edge_index) & ~bridge
    inherit = np.asarray(A.astype(np.float64) @ out).reshape(-1)
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64)
    take = along & (deg > 0) & (out <= 0.0)
    out[take] = inherit[take] / deg[take]
    return out


def first_corner_shell(pos: np.ndarray, wall: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """The off-wall shell as a TOPOLOGICAL rule -- no length, nothing to recalibrate.

    Mesh-agnostic replacement for the ``SHELL_SPECIES_LO/HI`` bounds, built from the
    quadratic mesh's own layering rather than from a distance:

    1. :func:`wall_normal_midside` -- the empty family bridging the wall outward;
    2. the corner nodes on the far side of that bridge: the first species-carrying row;
    3. the mid-side nodes lying *along* that row (both neighbours in it).  These carry
       species and clot, so dropping them costs a third of the off-wall GT clot -- which is
       why "exclude all mid-side nodes" is the wrong rule (off-wall F1 0.429 against 0.530).

    Every step is a statement about element order and connectivity, so unlike a bound in
    median edge lengths there is nothing here to recalibrate on a customer mesh.
    """
    wall = np.asarray(wall, dtype=bool)
    A = adjacency(np.asarray(edge_index), len(wall))
    ms = midside_nodes(pos, edge_index)
    bridge = wall_normal_midside(pos, wall, edge_index)
    row = (~wall) & ~ms & (A @ bridge.astype(np.int8) > 0)
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    # A mid-side node ON the row has both of its two neighbours in the row.
    along = ms & ~wall & ~bridge & (deg == 2) & (A @ row.astype(np.int8) == 2)
    return row | along


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


# ---------------------------------------------------------------------------
# MAT-MAGNITUDE LUMEN ARM (docs/PHASE7_FINDINGS.md)
# ---------------------------------------------------------------------------
#
# In the production COMSOL model (``comsol_models/phase2_template_nowound.mph``) M/Mas/Mat
# are a 3-species *Transport of Diluted Species* DOMAIN field with ``D_M = D_Mas = D_Mat = 0``,
# advected by the Reacting Flow coupling, sourced only by the wall inward flux ``J0_Mat``.
# The clot label is ``mu1(Mat) > `` step, i.e. a threshold on that domain field.  So an
# off-wall node is clot exactly when the Mat that reached it exceeds ``viscosity_mat_crit``
# -- there is no separate "lumen mechanism" to model.
#
# Measured on WALL_COHORT_V2_TRAIN (12 vessels carrying off-wall GT clot):
#
#   * off-wall GT clot is one boundary-layer node row: normal offset 1.66-1.80 median edge
#     lengths, p50 ~= p90.  This is a MESH property (confirmed against the COMSOL field
#     plots), so the shell bound is expressed in mesh units and is NOT portable to a mesh
#     built with different boundary-layer settings;
#   * its nearest wall node is GT-committed 99.9% of the time;
#   * ``Mat_offwall / Mat_owner`` has median 0.146-0.172 on EVERY vessel -- one attenuation
#     constant, ~0.16, i.e. an off-wall node commits when ``Mat_owner > crit / 0.16``.
#
# With GT wall Mat this rule scores off-wall F1 0.780; the shipped speed-threshold arm
# scores 0.190 and a perfect-GT-seeded thickness rule cannot exceed 0.275.

MAT_ATTENUATION = 0.16        # Mat_offwall / Mat_owner, median over 12 train vessels
MAT_ATTENUATION_TUNED = 0.25  # best on the full-mesh deploy score (0.8588 vs 0.8324 at 0.16)
#
# The two differ because the deploy score is relaxed (2-hop) and precision-weighted (F0.5):
# 0.25 requires ``Mat_owner > 4*crit`` instead of ``6.25*crit``, buying recall at a
# precision cost the relaxed metric barely charges for.  0.16 is the MEASURED constant and
# is the one to quote as physics; 0.25 is a scoring-surface tune fitted on TRAIN.


# THE SHELL MUST EXCLUDE AN EMPTY NEAR-WALL NODE FAMILY.  Measured over 19 train vessels
# (``scripts/diag_offwall_mesh_portability.py``), off-wall nodes sit in bands at normal
# offsets 1.01 / 1.72 / 2.63 / 3.43 median edge lengths, ~one per wall node per band, and
# ``Mat`` is present on only every other one:
#
#     band [median edge lengths]   median Mat/Mat_owner   off-wall GT clot (cohort)
#     0.50 - 1.35                         0.000                    83
#     1.35 - 2.20                         0.154                   493
#     2.20 - 3.00                         0.000                     7
#
# Velocity and pressure are populated on every band, so this is specific to M/Mas/Mat --
# exactly the fields the clot label thresholds.  Scored with a GT-Mat oracle, the 0.5-1.35
# band is almost pure false positive and the original 0-2.1 shell spanned both:
#
#     shell                       off F1   precision   recall
#     0    - 2.1  (was shipped)   0.4091     0.428      0.484
#     0.5  - 1.35 (empty family)  0.0364     0.036      0.038
#     1.35 - 2.2  (species band)  0.5303     0.809      0.446   <- SHELL_SPECIES_*
#     topological (no length)     0.5614     0.849      0.470   <- first_corner_shell, DEFAULT
#
# THE DEFAULT IS THE TOPOLOGICAL RULE, because the bounds above are in mesh units and were
# the standing pre-deploy blocker (PHASE7_FINDINGS 5.1).  ``first_corner_shell`` reproduces
# this band with Jaccard 1.000 on 12 of 19 vessels (>= 0.84 on all), scores better, and has
# no constant to recalibrate.  The bounds are kept for reproducing the earlier numbers and
# because they are what the band was calibrated to.
#
# WHAT THE EMPTY FAMILY IS: the mid-edge nodes of the quadratic mesh's WALL-NORMAL edges
# (``wall_normal_midside``).  Being mid-side is not sufficient -- the mid-side nodes lying
# along the species row carry ``Mat`` normally and hold 170 of the 493 off-wall GT clot
# nodes, so excluding all mid-side nodes costs a third of the recall (0.429 vs 0.561).  Why
# the wall-crossing ones specifically are empty is not established; the measurement is
# stable on all 19 vessels.
#
# IT COSTS A LITTLE ON THE RELAXED SCORE.  ``full-mesh deploy_clot_score`` moves the other
# way, 0.8324 -> 0.8284 on the oracle arm (0.8588 -> 0.8492 tuned), because that metric is
# 2-hop relaxed and F0.5-weighted: the empty family sits within 2 hops of real clot, so it
# is charged almost nothing while adding recall.  Excluded anyway -- those nodes carry no
# species and so cannot be clot, which makes the 0.004 a measurement of how much the relaxed
# metric pays for noise rather than a reason to keep it.  Quote both.
SHELL_SPECIES_LO = 1.35
SHELL_SPECIES_HI = 2.2


def grow_into_lumen_by_mat(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    mat_crit: float,
    *,
    shell_lo: float | None = None,
    shell_hi: float | None = None,
    attenuation: float = MAT_ATTENUATION,
) -> np.ndarray:
    """Off-wall clot from the wall ``Mat`` MAGNITUDE, not from the wall mask.

    ``mat_wall`` is Mat per node in COMSOL model units (only the wall entries are read).
    An off-wall node in the shell commits when ``attenuation * Mat_owner >= mat_crit``.

    The shell comes from :func:`resolve_offwall_shell`: the **topological** rule when the
    mesh is quadratic (no length, nothing to recalibrate), else the calibrated distance band.
    Passing ``shell_lo`` / ``shell_hi`` forces the band, in units of the mesh's median edge
    length -- ``0.0, 2.1`` reproduces the original Phase-7 numbers.
    """
    _, owner = wall_normal_projection(pos, wall)
    shell = resolve_offwall_shell(pos, wall, edge_index,
                                  shell_lo=shell_lo, shell_hi=shell_hi)
    return shell & (float(attenuation) * np.asarray(mat_wall)[owner] >= float(mat_crit))


#: Reference non-dimensional speed at which :func:`grow_into_lumen_by_flux` reduces to
#: constant :data:`MAT_ATTENUATION`.  Equal to the shipped speed-arm threshold so the two
#: rules agree on the stagnant shell and only disagree where |u| and the wall-normal
#: component come apart.
FLUX_SPEED_REF = 0.2
FLUX_SPEED_FLOOR = 0.05


def wall_normal_speed(u: np.ndarray, v: np.ndarray, pos: np.ndarray,
                      owner: np.ndarray) -> np.ndarray:
    """|u · n| toward the nearest wall node -- the convective removal that sets first-cell Mat.

    Production ``tds2`` is D=0: a wall flux ``J`` into a cell with wall-normal speed ``u_n``
    equilibrates at ``Mat ~ J / u_n``.  Sliding along the wall (|u| large, ``u_n`` small)
    does not scour the first cell; the shipped |u| speed arm cannot see that.
    """
    pos = np.asarray(pos, dtype=np.float64)
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    dxy = pos - pos[owner]
    nrm = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]), 1e-12)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return np.abs((u * dxy[:, 0] + v * dxy[:, 1]) / nrm)


def grow_into_lumen_by_flux(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    mat_crit: float,
    u: np.ndarray,
    v: np.ndarray,
    *,
    attenuation: float = MAT_ATTENUATION,
    speed_ref: float = FLUX_SPEED_REF,
    speed_floor: float = FLUX_SPEED_FLOOR,
    shell_lo: float | None = None,
    shell_hi: float | None = None,
) -> np.ndarray:
    """Off-wall clot from wall-flux / wall-normal speed, not from |u| or a constant att.

    Constant :data:`MAT_ATTENUATION` is the measured median ``Mat_shell / Mat_owner``.
    That ratio is a residence time: ``Mat_shell ~ J / u_n``.  Replacing the constant with
    ``att * (u_ref / max(u_n, u_floor))`` recovers the median at ``u_n = u_ref`` and lets
    a slow wall-normal wake keep more of the wall load -- which is the ``tds2`` balance,
    and which |u| stagnation cannot express when flow is tangent to the wall.
    """
    _, owner = wall_normal_projection(pos, wall)
    shell = resolve_offwall_shell(pos, wall, edge_index,
                                  shell_lo=shell_lo, shell_hi=shell_hi)
    u_n = wall_normal_speed(u, v, pos, owner)
    att = float(attenuation) * (float(speed_ref) / np.maximum(u_n, float(speed_floor)))
    return shell & (att * np.asarray(mat_wall, dtype=np.float64)[owner] >= float(mat_crit))


def grow_into_lumen_by_convection(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    mat_crit: float,
    u: np.ndarray,
    v: np.ndarray,
    *,
    attenuation: float = MAT_ATTENUATION,
    steps: int = 4,
    shell_lo: float | None = None,
    shell_hi: float | None = None,
) -> np.ndarray:
    """Off-wall clot from D=0 upwind of wall Mat, then the measured shell attenuation.

    Nearest-wall owner is Euclidean; convection's owner is the upstream wall node the
    fluid actually came from.  Wall values are held (the ODE's job).  ``attenuation``
    is still applied because donor-cell copies the wall magnitude, while the first
    species cell in the ``.mph`` holds ~0.16 of it.
    """
    wall = np.asarray(wall, dtype=bool)
    seed = np.asarray(mat_wall, dtype=np.float64).copy()
    seed[~wall] = 0.0
    conv = convect_mat_from_wall(seed, wall, pos, edge_index, u, v, steps=steps)
    shell = resolve_offwall_shell(pos, wall, edge_index,
                                  shell_lo=shell_lo, shell_hi=shell_hi)
    return shell & (float(attenuation) * conv >= float(mat_crit))


def _best_upstream(n: int, rows: np.ndarray, cols: np.ndarray,
                   scores: np.ndarray) -> np.ndarray:
    """Per-node argmax of ``scores`` over incoming edges; ``-1`` if nothing is upstream."""
    arg = np.full(n, -1, dtype=np.int64)
    best = np.full(n, -np.inf)
    if rows.size == 0:
        return arg
    order = np.argsort(scores, kind="mergesort")
    r, c, s = rows[order], cols[order], scores[order]
    arg[r] = c
    best[r] = s
    arg[best <= 0.0] = -1
    return arg


def characteristic_origin(
    pos: np.ndarray,
    wall: np.ndarray,
    edge_index: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    max_hops: int = 12,
) -> np.ndarray:
    """Wall foot of the D=0 characteristic: walk upstream until a wall node.

    Production ``tds2`` is ``dC/dt + u·∇C = 0`` (D=0), so Mat at a shell node is Mat at
    the wall node the streamline came from, not at the Euclidean nearest wall node.
    No-slip is enforced (wall velocity zeroed) so the walk cannot leave the wall once
    it arrives.  ``-1`` means the walk never hit a wall within ``max_hops``.
    """
    wall = np.asarray(wall, dtype=bool)
    pos = np.asarray(pos, dtype=np.float64)
    n = len(wall)
    u = np.asarray(u, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(v, dtype=np.float64).reshape(-1).copy()
    u[wall] = 0.0
    v[wall] = 0.0
    origin = np.where(wall, np.arange(n), -1)
    idx = adjacency(np.asarray(edge_index), n).tocoo()
    dxy = pos[idx.col] - pos[idx.row]
    um = 0.5 * (u[idx.row] + u[idx.col])
    vm = 0.5 * (v[idx.row] + v[idx.col])
    # col is upstream of row when midpoint velocity points against (col - row)
    score = -(um * dxy[:, 0] + vm * dxy[:, 1])
    take = score > 0.0
    rows, cols, sc = idx.row[take], idx.col[take], score[take]
    live = ~wall
    for _ in range(max(int(max_hops), 0)):
        if not live.any():
            break
        arg = _best_upstream(n, rows, cols, sc)
        moving = live & (arg >= 0) & (origin[arg] >= 0)
        if not moving.any():
            break
        origin[moving] = origin[arg[moving]]
        live = live & (origin < 0)
    return origin


def tds2_mat_field(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    hops: int = 8,
    blend_nearest: bool = True,
) -> np.ndarray:
    """Volumetric Mat from D=0 upwind of the wall ODE field.

    Wall values are held (the surface law / ``da_scale = 1/h_cell`` already turned
    ``J0`` into a nodal ``dC/dt``).  Off-wall nodes take the Mat of their unique
    upstream neighbour -- the D=0 Riemann solver, not a max-over-neighbours flood.
    ``blend_nearest`` keeps the Euclidean owner as a floor so a streamline that
    misses the wall cannot erase the measured first-cell smear.
    """
    wall = np.asarray(wall, dtype=bool)
    mat_wall = np.asarray(mat_wall, dtype=np.float64)
    n = len(wall)
    C = np.zeros(n, dtype=np.float64)
    C[wall] = mat_wall[wall]
    pos = np.asarray(pos, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1).copy()
    v = np.asarray(v, dtype=np.float64).reshape(-1).copy()
    u[wall] = 0.0
    v[wall] = 0.0
    idx = adjacency(np.asarray(edge_index), n).tocoo()
    dxy = pos[idx.col] - pos[idx.row]
    um = 0.5 * (u[idx.row] + u[idx.col])
    vm = 0.5 * (v[idx.row] + v[idx.col])
    score = -(um * dxy[:, 0] + vm * dxy[:, 1])
    take = (score > 0.0) & ~wall[idx.row]
    rows, cols, sc = idx.row[take], idx.col[take], score[take]
    for _ in range(max(int(hops), 0)):
        arg = _best_upstream(n, rows, cols, sc)
        moving = ~wall & (arg >= 0)
        if not moving.any():
            break
        nxt = C.copy()
        nxt[moving] = C[arg[moving]]
        nxt[wall] = mat_wall[wall]
        if np.array_equal(nxt, C):
            break
        C = nxt
    if blend_nearest:
        _, owner = wall_normal_projection(pos, wall)
        C[~wall] = np.maximum(C[~wall], mat_wall[owner[~wall]])
    return C


def grow_into_lumen_by_tds2(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    mat_crit: float,
    u: np.ndarray,
    v: np.ndarray,
    *,
    attenuation: float = MAT_ATTENUATION,
    hops: int = 8,
    blend_nearest: bool = True,
    shell_lo: float | None = None,
    shell_hi: float | None = None,
) -> np.ndarray:
    """Off-wall clot from the ``tds2`` field, not from |u| or a nearest-owner lookup.

    The first-cell ratio ``attenuation`` is the measured FEM smear of a D=0 wall flux
    onto the first species row (docs/PHASE7_FINDINGS.md §3.2).  Convection decides
    *which* wall value that cell holds.
    """
    C = tds2_mat_field(mat_wall, wall, pos, edge_index, u, v,
                       hops=hops, blend_nearest=blend_nearest)
    shell = resolve_offwall_shell(pos, wall, edge_index,
                                  shell_lo=shell_lo, shell_hi=shell_hi)
    return shell & (float(attenuation) * C >= float(mat_crit))


def grow_into_lumen_by_first_cell(
    mat_wall: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    mat_crit: float,
    u: np.ndarray,
    v: np.ndarray,
    *,
    attenuation: float = MAT_ATTENUATION,
    speed_ref: float = FLUX_SPEED_REF,
    speed_floor: float = FLUX_SPEED_FLOOR,
    wall_clot: np.ndarray | None = None,
) -> np.ndarray:
    """Off-wall clot from the first-cell ``tds2`` balance on the P2 species row.

    Production first-cell Mat equilibrates at ``J / u_n`` where ``u_n`` is the flux through
    the wall-normal face, not |u| in the cell and not Euclidean |u·n| at a no-slip wall
    node.  ``attenuation * (speed_ref / u_n_bridge)`` is that residence form; the P2
    topological owner is the wall corner that actually faces the cell.  When ``wall_clot``
    is given, only committed wall tissue may source the cell -- the precision brake that
    Euclidean flux lacked on wall-only vessels.
    """
    owner = topological_owner(pos, wall, edge_index)
    shell = first_corner_shell(pos, wall, edge_index)
    u_n = p2_bridge_normal_speed(u, v, pos, wall, edge_index)
    att = float(attenuation) * (float(speed_ref) / np.maximum(u_n, float(speed_floor)))
    ok = (owner >= 0) & shell
    pred = np.zeros(len(wall), dtype=bool)
    if not ok.any():
        return pred
    keep = att[ok] * np.asarray(mat_wall, dtype=np.float64)[owner[ok]] >= float(mat_crit)
    if wall_clot is not None:
        keep = keep & np.asarray(wall_clot, dtype=bool)[owner[ok]]
    pred[ok] = keep
    return pred


def fill_grown_wall_mat(
    mat_wall: np.ndarray,
    wall_mask: np.ndarray,
    wall: np.ndarray,
    A: sp.csr_matrix,
    *,
    hops: int = 6,
) -> np.ndarray:
    """Give graph-grown wall nodes a Mat value so the lumen arm can see them.

    15-26% of the shipped wall mask arrives by shear-admitted graph growth rather than by
    ODE ignition, and those nodes carry ``Mat = 0``.  On the vessels that matter most for
    off-wall clot (p012/p041/p044) the MEDIAN model Mat at GT-clot wall nodes is exactly
    zero for that reason.  Propagate the neighbour maximum outward so a grown node inherits
    the magnitude of the ignited tissue it grew from, decayed once per hop.
    """
    out = np.asarray(mat_wall, dtype=np.float64).copy()
    need = wall_mask & wall & (out <= 0.0)
    if not need.any():
        return out
    idx = A.tocoo()
    for _ in range(int(hops)):
        cand = np.zeros_like(out)
        np.maximum.at(cand, idx.row, out[idx.col])
        upd = need & (cand > out)
        if not upd.any():
            break
        out[upd] = cand[upd]
        need = need & ~upd
    return out


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


def convect_mat_from_wall(
    mat: np.ndarray,
    wall: np.ndarray,
    pos: np.ndarray,
    edge_index: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    steps: int = 6,
) -> np.ndarray:
    """D=0 donor-cell transport of wall Mat into the lumen.

    Production ``tds2`` has ``D_Mat = 0`` and convection on: material deposited at the wall
    is carried with the flow, it does not diffuse.  :func:`grow_into_lumen_by_mat` replaces
    that with a constant attenuation from the nearest wall owner.  This pushes the wall
    field along the (t=0 or predicted) velocity instead, so a node in the wake of heavy
    deposition inherits that magnitude and a node upstream of it does not.

    Each step: a lumen node takes the max Mat of graph-neighbours that sit *upstream*
    (edge vector aligned with the midpoint velocity).  Wall values are held -- they are
    the ODE's job.  ``steps`` is a hop budget, not a fitted length.
    """
    mat = np.asarray(mat, dtype=np.float64).copy()
    wall = np.asarray(wall, dtype=bool)
    pos = np.asarray(pos, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n_steps = max(int(steps), 0)
    if n_steps == 0 or not (~wall).any():
        return mat
    idx = adjacency(np.asarray(edge_index), len(wall)).tocoo()
    dxy = pos[idx.col] - pos[idx.row]
    um = 0.5 * (u[idx.row] + u[idx.col])
    vm = 0.5 * (v[idx.row] + v[idx.col])
    # flow from col toward row  <=>  midpoint velocity points against (col - row)
    incoming = (um * dxy[:, 0] + vm * dxy[:, 1]) < 0.0
    take = incoming & ~wall[idx.row]
    rows = idx.row[take]
    cols = idx.col[take]
    for _ in range(n_steps):
        nxt = mat.copy()
        np.maximum.at(nxt, rows, mat[cols])
        mat = nxt
    return mat
