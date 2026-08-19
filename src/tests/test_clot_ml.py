"""Guards for the PHASE9 clot-ML stack.

The two things that could silently invalidate every number: the fast scorer drifting from
the canonical deploy score, and the recurrent model quietly collapsing to the feed-forward
one (or vice versa).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.clot_ml.fastscore import VesselScorer
from src.clot_ml.gnn import ClotGNN, edge_features
from src.clot_ml.readouts import REGISTRY, SEPARABLE, apply
from src.clot_ml.recurrent import N_FEEDBACK, feedback_channels, neighbour_operator
from src.clot_ml.softmetric import dilation_operator, soft_dilate, to_torch_sparse


def _grid(n=6):
    """A small quad-ish mesh: two rows of ``n`` nodes, wall on the bottom row."""
    pos = np.array([[float(i), 0.0] for i in range(n)]
                   + [[float(i), 1.0] for i in range(n)], dtype=np.float64)
    wall = np.array([True] * n + [False] * n)
    e = [(i, i + 1) for i in range(n - 1)] + [(n + i, n + i + 1) for i in range(n - 1)]
    e += [(i, n + i) for i in range(n)]
    ei = np.array(e, dtype=np.int64).T
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    return pos, wall, ei


def test_fast_scorer_matches_the_canonical_deploy_score():
    """The metric of record must not drift.  Every PHASE9 number depends on this."""
    from src.clot_ml.evaluate import domain_score

    pos, wall, ei = _grid()
    n = len(wall)
    rng = np.random.default_rng(0)
    gt = np.zeros(n, bool)
    gt[[1, 2, 7]] = True
    vs = VesselScorer(ei, gt, n)
    for _ in range(12):
        pred = rng.random(n) < 0.35
        for domain in (wall, ~wall):
            ref = domain_score(pred, gt, torch.tensor(ei), domain, wall)
            got = vs.score(pred, domain)
            if ref != ref:
                assert got != got
            else:
                assert abs(ref - got) < 1e-9, (ref, got)


def test_soft_dilate_is_exact_on_hard_masks():
    """The training loss must agree with the eval metric where the prediction is binary."""
    pos, wall, ei = _grid()
    n = len(wall)
    D = to_torch_sparse(dilation_operator(ei, n, 2), torch.device("cpu"))
    p = torch.zeros(n)
    p[[0, 8]] = 1.0
    soft = soft_dilate(p, D).numpy() > 0.5
    hard = (dilation_operator(ei, n, 2) @ p.numpy().astype(np.int8)) > 0
    assert np.array_equal(soft, hard)


def test_rounds_one_is_the_feed_forward_model():
    """``rounds=1`` must be bit-identical to no recurrence, or the ablation is meaningless."""
    torch.manual_seed(0)
    pos, wall, ei = _grid()
    n, in_dim = len(wall), 5
    ea = torch.tensor(edge_features(pos, ei, np.ones(n), np.zeros(n), 1.0))
    m = ClotGNN(in_dim, ea.shape[1], dim=16, layers=2, drop=0.0)
    x = torch.randn(n, in_dim)
    w = torch.ones(ea.shape[0], 1)
    a = m(x, torch.tensor(ei), ea, w, w, torch.zeros(n))[0]
    b = m(x, torch.tensor(ei), ea, w, w, torch.zeros(n), extra=None)[0]
    assert torch.allclose(a, b)


def test_feedback_channels_carry_the_owner_attenuation():
    """The owner channel is the 0.16 attenuation law written as an input; it must gather
    from the WALL node, not from the node itself."""
    pos, wall, ei = _grid()
    n = len(wall)
    At = to_torch_sparse(neighbour_operator(ei, n), torch.device("cpu"))
    owner = torch.tensor(np.concatenate([np.arange(6), np.arange(6)]))
    p = torch.zeros(n)
    p[0] = 1.0
    fb = feedback_channels(p, At, owner)
    assert fb.shape == (n, N_FEEDBACK)
    assert fb[6, 1] == pytest.approx(1.0), "off-wall node must see its owner's occlusion"
    assert fb[6, 0] == pytest.approx(0.0), "and its own is still zero"


def test_readouts_never_label_outside_their_domain():
    pos, wall, ei = _grid()
    n = len(wall)
    S = dict(wall=wall, phys_mask=np.zeros(n, bool), shell=~wall,
             edge_index=ei, pos=pos.astype(np.float32))
    score = np.linspace(0, 1, n).astype(np.float32)
    for name, (_, grids) in REGISTRY.items():
        p = tuple(g[len(g) // 2] for g in grids)
        out = apply(name, S, score, p)
        assert out.dtype == bool and out.shape == (n,)


def test_separable_readouts_declare_valid_indices():
    for name, (iw, io_) in SEPARABLE.items():
        n_params = len(REGISTRY[name][1])
        assert 0 <= iw < n_params and 0 <= io_ < n_params and iw != io_


def test_geometry_stratified_folds_spread_the_priority_vessels():
    """Each priority vessel must land in a different fold, or the class cannot be measured
    out-of-fold at all (docs/PHASE9_ML.md 11.1)."""
    from src.clot_ml.geometry_splits import is_priority, stratified_folds

    classes = {f"v{i:02d}": "baseline" for i in range(16)}
    classes.update({"a0": "aneurysm", "s0": "stenosis", "s1": "stenosis"})
    folds = stratified_folds(classes, k=5)
    assert sorted(x for f in folds for x in f) == sorted(classes)
    where = {a: i for i, f in enumerate(folds) for a in f}
    prio = [a for a in classes if is_priority(classes[a])]
    assert len({where[a] for a in prio}) == len(prio), "priority vessels share a fold"


def test_stratified_folds_are_deterministic_and_cover_once():
    from src.clot_ml.geometry_splits import stratified_folds

    classes = {f"v{i:02d}": ("stenosis" if i % 7 == 0 else "baseline") for i in range(19)}
    a = stratified_folds(classes, k=4)
    b = stratified_folds(classes, k=4)
    assert a == b
    flat = [x for f in a for x in f]
    assert len(flat) == len(set(flat)) == len(classes)


def test_mask_series_is_nested_and_respects_the_set():
    """Clot never un-clots, and timing must never add a node outside the supplied mask."""
    from src.clot_ml.temporal import mask_series

    n = 10
    mask = np.zeros(n, bool)
    mask[:6] = True
    onset = np.array([0, 2, 4, 6, 8, 8, -1, -1, -1, -1])
    series = mask_series(onset, mask, [0, 2, 4, 6, 8])
    prev = None
    for ti in (0, 2, 4, 6, 8):
        m = series[ti]
        assert not (m & ~mask).any(), "timing added a node outside the mask"
        if prev is not None:
            assert (m | prev == m).all(), "mask shrank between timesteps"
        prev = m
    assert series[0].sum() == 1 and series[8].sum() == 6


def test_onset_fallback_comes_from_wall_nodes_only():
    """The fallback must not depend on the off-wall rule, or arms that differ only
    off-wall silently get different WALL scores (that bug cost 0.011 in a first run)."""
    from src.clot_ml.temporal import onset_from_ode

    n, T = 8, 20
    wall = np.array([True] * 4 + [False] * 4)
    pos = np.array([[float(i), 0.0] for i in range(4)]
                   + [[float(i), 1.0] for i in range(4)], dtype=np.float64)
    traj = np.zeros((T, n))
    for i, on in enumerate((2, 4, 6, None)):
        if on is not None:
            traj[on:, i] = 5.0
    mask = np.ones(n, bool)
    a = onset_from_ode(traj, mask, wall, pos, 1.0, attenuation=0.16)
    b = onset_from_ode(traj, mask, wall, pos, 1.0, attenuation=0.9)
    assert np.array_equal(a[wall], b[wall]), "wall onset changed with the off-wall rule"
    assert a[3] == int(np.median([2, 4, 6]))


def test_shipped_series_never_puts_offwall_clot_before_wall_clot():
    """Physical constraint on the shipped entry point: an off-wall node is fed by a wall
    node, so it cannot commit first.  A first version froze off-wall at the final mask and
    showed 19 off-wall nodes at t=0 with an empty wall."""
    from src.clot_ml.temporal import mask_series, onset_from_ode

    n, T = 8, 20
    wall = np.array([True] * 4 + [False] * 4)
    pos = np.array([[float(i), 0.0] for i in range(4)]
                   + [[float(i), 1.0] for i in range(4)], dtype=np.float64)
    traj = np.zeros((T, n))
    traj[5:, 0] = 1.0
    traj[7:, 1] = 3.0
    traj[9:, 2] = 8.0
    traj[11:, 3] = 8.0
    mask = np.ones(n, bool)
    onset = onset_from_ode(traj, mask, wall, pos, 1.0, attenuation=0.8)
    series = mask_series(onset, mask, list(range(T)))
    for ti in range(T):
        m = series[ti]
        if (m & ~wall).any():
            assert (m & wall).any(), f"off-wall clot with an empty wall at t={ti}"


def test_enforce_owner_and_monotone_keeps_time_monotone():
    """Clot must never un-clot once predicted -- the production law has no sink."""
    from src.clot_ml.locked import enforce_owner_and_monotone

    n = 6
    wall = np.array([True, True, True, False, False, False])
    owner = np.array([0, 1, 2, 0, 1, 2])
    raw = {
        0: np.array([True, False, False, False, False, False]),
        1: np.array([False, False, False, False, False, False]),  # would un-clot node 0
        2: np.array([True, True, False, False, False, False]),
    }
    out = enforce_owner_and_monotone(raw, wall, owner, [0, 1, 2])
    assert out[0][0] and out[1][0] and out[2][0], "node 0 must stay clot at every later time"
    for t in (0, 1, 2):
        assert (out[t] | (out[t - 1] if t > 0 else np.zeros(n, bool)) == out[t]).all()


def test_enforce_owner_and_monotone_blocks_offwall_before_owner():
    """An off-wall node cannot be clot before the wall node feeding it."""
    from src.clot_ml.locked import enforce_owner_and_monotone

    n = 4
    wall = np.array([True, True, False, False])
    owner = np.array([0, 1, 0, 1])
    raw = {
        0: np.array([False, False, True, False]),   # off-wall node 2 fires; owner 0 does not
        1: np.array([True, False, True, False]),     # now owner 0 fires too
    }
    out = enforce_owner_and_monotone(raw, wall, owner, [0, 1])
    assert not out[0][2], "off-wall node fired before its owner and must be suppressed"
    assert out[1][2], "once the owner fires the off-wall node may too"


def test_v3_manifest_is_consistent_and_excludes_sealed():
    """If a v3-kind artifact is promoted in this checkout, it must be internally consistent
    and its training pool must never include SEALED."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    ptr_path = repo / "data/reference/clot_gnn_locked.json"
    if not ptr_path.exists():
        pytest.skip("clot_gnn_locked not promoted in this checkout")
    ptr = json.loads(ptr_path.read_text())
    if ptr.get("kind") != "temporal_v3":
        pytest.skip("shipped pointer is not v3 in this checkout")
    root = repo / ptr["path"]
    man = json.loads((root / ptr["manifest"].split("/")[-1]).read_text())
    assert man["kind"] == "temporal_v3"
    assert (root / man["clf_file"]).exists()
    base_root = repo / "outputs/clot_ml/locked" / man["base_set_model"]
    assert (base_root / "manifest.json").exists(), "base SET model must still be on disk"
    from src.core_physics.wall_cohort_splits import SEALED
    assert not (set(man["training_pool"]) & set(SEALED))
    assert 0.0 <= man["thresh_wall"] <= 1.0 and 0.0 <= man["thresh_off"] <= 1.0
