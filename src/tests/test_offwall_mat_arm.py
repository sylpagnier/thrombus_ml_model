"""Guards for the Mat-magnitude lumen arm (docs/PHASE7_FINDINGS.md 4).

Every assertion here corresponds to a way the arm could become a silent no-op or could
silently change the shipped prediction.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.core_physics.physics_lumen_model import (
    FLUX_SPEED_REF, MAT_ATTENUATION, MAT_ATTENUATION_TUNED, SHELL_SPECIES_HI,
    SHELL_SPECIES_LO, characteristic_origin, convect_mat_from_wall, fill_grown_wall_mat,
    first_corner_shell, grow_into_lumen_by_convection, grow_into_lumen_by_first_cell,
    grow_into_lumen_by_flux, grow_into_lumen_by_mat, grow_into_lumen_by_tds2,
    midside_nodes, p2_bridge_normal_speed, resolve_offwall_shell, tds2_mat_field,
    topological_owner, wall_normal_midside,
)

WIDE = dict(shell_lo=0.0, shell_hi=2.1)     # the original Phase-7 shell, kept as a control


def _quadratic_strip(n: int = 6, dy: float = 2.0):
    """A quadratic (P2) two-row strip: corner rows plus every mid-edge node.

    Layout, with the wall at ``y=0`` and the first off-wall corner row at ``y=dy``:

        wall corners        y=0     indices 0..n-1
        wall tangential ms  y=0     between consecutive wall corners
        wall-normal ms      y=dy/2  the EMPTY family -- one per wall corner
        off-wall corners    y=dy    the species row
        off-wall tangent ms y=dy    between consecutive off-wall corners

    Edges connect each mid-edge node to exactly the two corners it bisects, which is what
    :func:`midside_nodes` keys on. Median edge length is ``dy/2`` for ``dy=2``, i.e. 1.0.
    """
    P, wall_flag, e = [], [], []

    def add(p, is_wall):
        P.append(p)
        wall_flag.append(is_wall)
        return len(P) - 1

    lo = [add((float(i), 0.0), True) for i in range(n)]
    hi = [add((float(i), dy), False) for i in range(n)]
    for row, is_wall in ((lo, True), (hi, False)):           # tangential mid-edge nodes
        for i in range(n - 1):
            m = add(((P[row[i]][0] + P[row[i + 1]][0]) / 2.0, P[row[i]][1]), is_wall)
            e += [(row[i], m), (m, row[i + 1])]
    normal_ms = []
    for i in range(n):                                       # wall-normal mid-edge nodes
        m = add((float(i), dy / 2.0), False)
        normal_ms.append(m)
        e += [(lo[i], m), (m, hi[i])]
    pos = np.array(P, dtype=np.float64)
    return pos, np.array(wall_flag), np.array(e, dtype=np.int64).T, lo, hi, normal_ms


def _strip(offset: float = 1.0):
    """Two rows of nodes: y=0 is the wall, y=offset is the single off-wall shell row.

    Edge lengths are 1.0 along each row, so the mesh's median edge length is 1.0 and
    ``offset`` is directly in the units the shell bounds are expressed in.
    """
    n = 6
    pos = np.array([[float(i), 0.0] for i in range(n)]
                   + [[float(i), offset] for i in range(n)], dtype=np.float64)
    wall = np.array([True] * n + [False] * n)
    e = [(i, i + 1) for i in range(n - 1)] + [(n + i, n + i + 1) for i in range(n - 1)]
    e += [(i, n + i) for i in range(n)]
    ei = np.array(e, dtype=np.int64).T
    return pos, wall, ei


def test_attenuation_constants_are_pinned():
    """The measured constant and the score tune must not drift silently."""
    assert MAT_ATTENUATION == pytest.approx(0.16)
    assert MAT_ATTENUATION_TUNED == pytest.approx(0.25)


def test_admits_exactly_the_nodes_above_the_attenuated_threshold():
    pos, wall, ei = _strip()
    crit = 1.0
    mat = np.zeros(len(wall))
    mat[:3] = crit / MAT_ATTENUATION * 1.01     # three wall nodes clear it
    mat[3:6] = crit / MAT_ATTENUATION * 0.99    # three do not
    off = grow_into_lumen_by_mat(mat, wall, pos, ei, crit, **WIDE)
    assert not off[wall].any(), "must never label a wall node"
    assert off[6:9].all() and not off[9:12].any()


def test_is_a_magnitude_rule_not_a_mask_rule():
    """A wall node above ``crit`` but below ``crit/attenuation`` must NOT seed off-wall clot.

    This is the whole difference from the shipped dilation arm, which would admit it.
    """
    pos, wall, ei = _strip()
    crit = 1.0
    mat = np.full(len(wall), 0.0)
    mat[:6] = crit * 2.0                        # committed, but only 2x crit
    assert (mat[:6] > crit).all()
    off = grow_into_lumen_by_mat(mat, wall, pos, ei, crit, **WIDE)
    assert not off.any()


def test_shell_bound_excludes_the_far_row():
    pos, wall, ei = _strip(offset=5.0)           # off-wall row pushed out of every shell
    mat = np.zeros(len(wall))
    mat[:6] = 100.0
    assert not grow_into_lumen_by_mat(mat, wall, pos, ei, 1.0, **WIDE).any()
    assert not grow_into_lumen_by_mat(mat, wall, pos, ei, 1.0).any()


def test_band_fallback_excludes_the_empty_near_wall_family():
    """On a linear mesh the fallback band must still skip the empty near-wall family.

    Guards the correction in docs/PHASE7_FINDINGS.md 8: the old 0-2.1 bound spanned BOTH
    near-wall node families, and the nearer one is almost pure false positive (off-wall
    precision 0.036). A regression to a 0-lower-bound shell would silently reinstate it.
    """
    assert SHELL_SPECIES_LO == pytest.approx(1.35)
    assert SHELL_SPECIES_HI == pytest.approx(2.2)
    mat_hot = 100.0
    crit = 1.0
    for offset, admitted in ((1.01, False), (1.72, True), (2.63, False)):
        pos, wall, ei = _strip(offset=offset)
        mat = np.zeros(len(wall))
        mat[:6] = mat_hot
        off = grow_into_lumen_by_mat(mat, wall, pos, ei, crit)
        assert off[6:].all() == admitted and off[6:].any() == admitted, (
            f"row at {offset} median edge lengths: expected admitted={admitted}")
        # the control: the old bound wrongly admits the phantom band at 1.01
        assert grow_into_lumen_by_mat(mat, wall, pos, ei, crit, **WIDE)[6:].any() == (
            offset < 2.1)


def test_midside_detection_finds_every_mid_edge_node_and_no_corner():
    pos, wall, ei, lo, hi, normal_ms = _quadratic_strip()
    ms = midside_nodes(pos, ei)
    assert not ms[lo].any() and not ms[hi].any(), "corner nodes must never be mid-side"
    assert ms[normal_ms].all(), "the wall-normal mid-edge nodes must be detected"
    # 2 corner rows of 6 + 2 tangential runs of 5 + 6 normal = 12 corners, 16 mid-side
    assert int(ms.sum()) == 16


def test_wall_normal_midside_is_only_the_wall_crossing_family():
    """The empty family is the mid-edge nodes of edges LEAVING the wall -- not all mid-side.

    docs/PHASE7_FINDINGS.md 8.2: mid-side nodes lying along the species row carry Mat
    normally and hold 170 of the cohort's 493 off-wall GT clot nodes, so a rule that drops
    every mid-side node loses a third of the recall (off-wall F1 0.429 vs 0.561).
    """
    pos, wall, ei, lo, hi, normal_ms = _quadratic_strip()
    bridge = wall_normal_midside(pos, wall, ei)
    assert bridge[normal_ms].all()
    assert int(bridge.sum()) == len(normal_ms), "tangential mid-side nodes must not be here"
    ms = midside_nodes(pos, ei)
    assert (ms & ~wall & ~bridge).any(), "fixture must contain off-wall tangential mid-side"


def test_topological_shell_is_the_species_row_including_its_tangential_midsides():
    pos, wall, ei, lo, hi, normal_ms = _quadratic_strip()
    shell = first_corner_shell(pos, wall, ei)
    assert shell[hi].all(), "the first corner row is the shell"
    assert not shell[normal_ms].any(), "the empty wall-normal family is excluded"
    assert not shell[wall].any()
    tangential = midside_nodes(pos, ei) & ~wall & ~wall_normal_midside(pos, wall, ei)
    assert (shell & tangential).sum() == tangential.sum(), (
        "mid-side nodes along the species row carry Mat and must be kept")


def test_topological_shell_selects_no_length_and_survives_rescaling():
    """The point of the rule: no constant in mesh units, so a mesh rescale cannot move it."""
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    a = first_corner_shell(pos, wall, ei)
    b = first_corner_shell(pos * 37.0, wall, ei)          # same topology, different lengths
    assert np.array_equal(a, b)
    # and it is unaffected by a boundary layer of a different thickness
    pos2, wall2, ei2, _, hi2, _ = _quadratic_strip(dy=0.25)
    assert first_corner_shell(pos2, wall2, ei2)[hi2].all()


def test_shell_falls_back_to_the_band_on_a_linear_mesh():
    """A P1 mesh has no mid-edge family, so the topological rule would select nothing.

    Without the fallback the whole arm becomes a silent no-op on such a mesh.
    """
    pos, wall, ei = _strip(offset=1.72)
    assert not first_corner_shell(pos, wall, ei).any(), "no mid-side family to navigate"
    assert resolve_offwall_shell(pos, wall, ei)[6:].all()
    mat = np.zeros(len(wall))
    mat[:6] = 100.0
    assert grow_into_lumen_by_mat(mat, wall, pos, ei, 1.0)[6:].all()


def test_fill_propagates_magnitude_to_grown_wall_nodes():
    pos, wall, ei = _strip()
    n = len(wall)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    mat = np.zeros(n)
    mat[0] = 7.0                                # only node 0 ignited
    mask = wall.copy()                          # but the whole wall row is in the mask
    out = fill_grown_wall_mat(mat, mask, wall, A, hops=6)
    assert out[0] == 7.0, "an ignited node must keep its own value"
    assert (out[:6] == 7.0).all(), "grown nodes must inherit the ignited magnitude"
    assert (out[6:] == 0.0).all(), "off-wall nodes are not filled"


def test_fill_is_a_noop_when_nothing_is_missing():
    pos, wall, ei = _strip()
    n = len(wall)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    mat = np.arange(n, dtype=np.float64) + 1.0
    out = fill_grown_wall_mat(mat, wall, wall, A, hops=6)
    assert np.array_equal(out, mat)


def test_flux_matches_constant_att_at_the_reference_wall_normal_speed():
    """At u_n = FLUX_SPEED_REF the residence form is the measured constant att."""
    pos, wall, ei = _strip()
    crit = 1.0
    mat = np.zeros(len(wall))
    mat[:3] = crit / MAT_ATTENUATION * 1.01
    mat[3:6] = crit / MAT_ATTENUATION * 0.99
    u = np.zeros(len(wall))
    v = np.full(len(wall), FLUX_SPEED_REF)
    assert np.array_equal(
        grow_into_lumen_by_mat(mat, wall, pos, ei, crit, **WIDE),
        grow_into_lumen_by_flux(mat, wall, pos, ei, crit, u, v, **WIDE),
    )


def test_flux_keeps_mat_when_flow_is_tangent():
    """Sliding along the wall: |u| is large but u_n ~ 0, so the first cell is not scoured.

    Constant att (and the |u| speed arm) would reject this node; the tds2 balance must not.
    """
    pos, wall, ei = _strip()
    crit = 1.0
    mat = np.full(len(wall), 0.0)
    mat[:6] = 3.0       # below crit/att=6.25, above crit/att_floor ~ 1.56
    u, v = np.full(len(wall), 5.0), np.zeros(len(wall))
    assert not grow_into_lumen_by_mat(mat, wall, pos, ei, crit, **WIDE).any()
    off = grow_into_lumen_by_flux(mat, wall, pos, ei, crit, u, v, **WIDE)
    assert not off[wall].any()
    assert off[6:].all()


def test_convection_copies_downstream_and_not_upstream():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo] = 10.0
    u = np.zeros(n)
    down = convect_mat_from_wall(mat, wall, pos, ei, u, np.ones(n), steps=4)
    assert np.allclose(down[lo], 10.0), "wall values are the ODE's, convection must hold them"
    assert (down[hi] >= 10.0 - 1e-12).all(), "D=0 upwind must carry wall Mat into the species row"
    up = convect_mat_from_wall(mat, wall, pos, ei, u, -np.ones(n), steps=4)
    assert np.allclose(up[hi], 0.0), "upstream lumen must not inherit wall Mat"


def test_convection_lumen_arm_never_labels_the_wall():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo] = 100.0
    off = grow_into_lumen_by_convection(
        mat, wall, pos, ei, 1.0, np.zeros(n), np.ones(n), steps=4)
    assert not off[wall].any()
    assert off[hi].all()


def test_tds2_upwind_carries_the_upstream_wall_value_not_the_max():
    """D=0 copies the unique upstream neighbour.  Max-over-neighbours is the flood we already lost to."""
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo[0]] = 10.0
    mat[lo[-1]] = 99.0
    u = np.zeros(n)
    v = np.ones(n)
    C = tds2_mat_field(mat, wall, pos, ei, u, v, hops=4, blend_nearest=False)
    assert np.allclose(C[lo], mat[lo]), "no-slip wall is a BC, not a convected unknown"
    assert C[hi[0]] == pytest.approx(10.0)
    assert C[hi[-1]] == pytest.approx(99.0)
    back = tds2_mat_field(mat, wall, pos, ei, u, -v, hops=4, blend_nearest=False)
    assert np.allclose(back[hi], 0.0)


def test_tds2_characteristic_hits_the_wall_along_the_bridge():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    origin = characteristic_origin(pos, wall, ei, np.zeros(n), np.ones(n), max_hops=4)
    assert (origin[lo] == np.asarray(lo)).all()
    assert (origin[hi] >= 0).all(), "outward flow must connect the species row to the wall"
    assert origin[hi[0]] == lo[0]


def test_tds2_lumen_arm_never_labels_the_wall_and_uses_the_smear():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo] = 100.0
    off = grow_into_lumen_by_tds2(
        mat, wall, pos, ei, 1.0, np.zeros(n), np.ones(n), hops=4, blend_nearest=False)
    assert not off[wall].any()
    assert off[hi].all()
    # below crit/att the smear must not commit
    mat[lo] = 1.0
    assert not grow_into_lumen_by_tds2(
        mat, wall, pos, ei, 1.0, np.zeros(n), np.ones(n), hops=4, blend_nearest=False).any()


def test_topological_owner_is_the_wall_corner_across_the_p2_bridge():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    owner = topological_owner(pos, wall, ei)
    assert (owner[lo] < 0).all(), "wall nodes are not owned"
    assert (owner[hi] == np.asarray(lo)).all()


def test_first_cell_uses_bridge_speed_not_wall_no_slip():
    """No-slip wall velocity is zero; the FEM face flux lives on the P2 bridge."""
    pos, wall, ei, lo, hi, nms = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo] = 3.0       # below crit/att=6.25, above crit/att_floor ~ 1.56
    crit = 1.0
    u, v = np.zeros(n), np.zeros(n)
    v[nms] = 5.0        # fast wall-normal face: first cell is scoured
    assert not grow_into_lumen_by_first_cell(mat, wall, pos, ei, crit, u, v).any()
    v[nms] = 0.0        # stagnant face: first cell keeps the wall load
    off = grow_into_lumen_by_first_cell(mat, wall, pos, ei, crit, u, v)
    assert not off[wall].any()
    assert off[hi].all()
    spd = p2_bridge_normal_speed(np.zeros(n), np.ones(n), pos, wall, ei)
    assert (spd[hi] > 0.0).all()


def test_first_cell_committed_requires_the_owner_to_be_in_the_wall_mask():
    pos, wall, ei, lo, hi, _ = _quadratic_strip()
    n = len(wall)
    mat = np.zeros(n)
    mat[lo] = 100.0
    u, v = np.zeros(n), np.zeros(n)
    clot = np.zeros(n, dtype=bool)
    clot[lo[0]] = True
    off = grow_into_lumen_by_first_cell(
        mat, wall, pos, ei, 1.0, u, v, wall_clot=clot)
    assert off[hi[0]]
    assert not off[hi[1:]].any()
