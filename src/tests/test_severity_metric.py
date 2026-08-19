"""Property guards for `clot_severity_score`.

A metric that trains the model has to be defended like code: if it can be gamed, the model
will find the exploit.  Each test below corresponds to a way an earlier draft was wrong.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.clot_ml.severity_metric import (
    DEFAULT, LEGACY, SeverityConfig, SeverityScorer, dilation_operator, severity_components,
    soft_severity,
)
from src.clot_ml.softmetric import to_torch_sparse


def _chain(n=200):
    """A 1-D chain mesh: simple, and graph distance equals index distance."""
    ei = np.array([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]], dtype=np.int64)
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    return ei, n


def _scorer(gt_idx, n, ei, cfg=DEFAULT):
    gt = np.zeros(n, bool)
    gt[gt_idx] = True
    return SeverityScorer(ei, gt, n, cfg), gt


def test_legacy_config_reproduces_the_shipped_score():
    """With every tolerance at zero this must be the old metric, exactly."""
    from src.clot_ml.fastscore import VesselScorer

    ei, n = _chain()
    rng = np.random.default_rng(3)
    gt = np.zeros(n, bool)
    gt[40:60] = True
    old = VesselScorer(ei, gt, n)
    new = SeverityScorer(ei, gt, n, LEGACY)
    for _ in range(15):
        pred = rng.random(n) < 0.12
        assert new.score(pred) == pytest.approx(old.score(pred), abs=1e-9)


def test_missing_five_of_fifteen_beats_missing_fifty_of_one_fifty():
    """The whole point: the same RATE of miss is not the same severity."""
    ei, n = _chain(400)
    small, _ = _scorer(list(range(20, 35)), n, ei)          # 15 nodes
    big, _ = _scorer(list(range(100, 250)), n, ei)          # 150 nodes
    p_small = np.zeros(n, bool)
    p_small[20:30] = True                                   # found 10 of 15
    p_big = np.zeros(n, bool)
    p_big[100:200] = True                                   # found 100 of 150
    s_small = small.components(p_small)
    s_big = big.components(p_big)
    l_small = SeverityScorer(ei, small.gt, n, LEGACY).components(p_small)
    l_big = SeverityScorer(ei, big.gt, n, LEGACY).components(p_big)
    assert s_small["recall_eff"] > s_big["recall_eff"]
    assert s_small["score"] > s_big["score"]
    # The 2-hop relaxation already gives a MILD burden tolerance -- two dilated nodes are a
    # larger fraction of 15 than of 150 -- so the legacy gap is not zero.  What the new
    # config must do is widen it substantially and deliberately, not invent it.
    new_gap = s_small["recall_eff"] - s_big["recall_eff"]
    old_gap = l_small["recall_eff"] - l_big["recall_eff"]
    assert old_gap > 0.0
    assert new_gap > 2.0 * old_gap, (new_gap, old_gap)


def test_predicting_nothing_scores_zero_at_every_burden():
    """The grace must never open the empty-prediction hole (PHASE6_RESULTS 15.3, inverted)."""
    ei, n = _chain()
    for k in (2, 4, 15, 60):
        sc, _ = _scorer(list(range(10, 10 + k)), n, ei)
        assert sc.score(np.zeros(n, bool)) == 0.0


def test_true_positives_never_lower_and_false_positives_never_raise():
    ei, n = _chain()
    sc, gt = _scorer(list(range(50, 80)), n, ei)
    pred = np.zeros(n, bool)
    pred[50:60] = True
    base = sc.score(pred)
    for j in range(60, 80):                       # add real clot nodes
        pred[j] = True
        s = sc.score(pred)
        assert s >= base - 1e-9, (j, s, base)
        base = s
    base = sc.score(pred)
    for j in range(150, 170):                     # add far-away false positives
        pred[j] = True
        s = sc.score(pred)
        assert s <= base + 1e-9, (j, s, base)
        base = s


def test_spray_is_still_punished():
    """Precision grace is relative to the prediction, so flooding cannot buy recall."""
    ei, n = _chain()
    sc, _ = _scorer(list(range(50, 65)), n, ei)
    good = np.zeros(n, bool)
    good[50:65] = True
    flood = np.ones(n, bool)
    assert sc.score(flood) < 0.25 < sc.score(good)


def test_small_absolute_error_is_forgiven_but_not_free():
    ei, n = _chain()
    sc, _ = _scorer(list(range(50, 65)), n, ei)     # 15 nodes
    exact = np.zeros(n, bool)
    exact[50:65] = True
    miss3 = np.zeros(n, bool)
    miss3[50:62] = True
    assert sc.score(exact) == pytest.approx(1.0, abs=1e-9)
    assert 0.85 < sc.score(miss3) < sc.score(exact)


def test_empty_gt_grades_false_positive_volume():
    ei, n = _chain()
    sc = SeverityScorer(ei, np.zeros(n, bool), n)
    r0 = severity_components(np.zeros(n, bool), sc.gt, sc.D, None, DEFAULT)
    r5 = severity_components(np.array([i < 5 for i in range(n)]), sc.gt, sc.D, None, DEFAULT)
    assert r0["score"] == 1.0 and 0.5 < r5["score"] < 1.0


def test_soft_form_matches_the_hard_one_on_binary_input():
    """The training loss and the reported metric must be the same function."""
    ei, n = _chain(120)
    gt = np.zeros(n, bool)
    gt[30:50] = True
    D = dilation_operator(ei, n, 2)
    Dt = to_torch_sparse(D, torch.device("cpu"))
    gt_t = torch.tensor(gt.astype(np.float32))
    gt_dil = (torch.sparse.mm(Dt, gt_t.reshape(-1, 1)).reshape(-1) > 0).float()
    dom = torch.ones(n)
    for lo, hi in ((30, 50), (30, 45), (25, 55), (60, 70)):
        pred = np.zeros(n, bool)
        pred[lo:hi] = True
        hard = severity_components(pred, gt, D, None, DEFAULT)["score"]
        soft = float(soft_severity(torch.tensor(pred.astype(np.float32)), gt_t, Dt, dom,
                                   gt_dil, DEFAULT))
        assert soft == pytest.approx(hard, abs=2e-3), (lo, hi, hard, soft)


def test_config_is_frozen_and_serialisable():
    cfg = SeverityConfig()
    assert cfg.as_dict()["tau_abs"] == 5.0
    with pytest.raises(Exception):
        cfg.tau_abs = 9.0  # type: ignore[misc]


def test_locked_ensemble_manifest_is_consistent():
    """The named artifact must stay loadable and self-describing.

    Generation-aware: a ``gnn_ensemble`` manifest (v1/v2) lists ``members`` directly; a
    ``temporal_v3`` manifest instead names a ``base_set_model`` whose OWN manifest carries
    those members, plus a standalone classifier file.  Only the base model is checked here
    for member/feature-norm shape; ``test_v3_manifest_is_consistent_and_excludes_sealed``
    in ``test_clot_ml.py`` covers the v3-specific fields.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    ptr = repo / "data/reference/clot_gnn_locked.json"
    if not ptr.exists():
        pytest.skip("clot_gnn_locked not promoted in this checkout")
    p = json.loads(ptr.read_text())
    kind = p.get("kind", "gnn_ensemble")
    root = repo / p["path"]
    man = json.loads((root / p["manifest"].split("/")[-1]).read_text())
    assert man["name"] == p["name"]

    from src.core_physics.wall_cohort_splits import SEALED

    if kind == "temporal_v3":
        assert (root / man["clf_file"]).exists()
        base_root = repo / "outputs/clot_ml/locked" / man["base_set_model"]
        base_man = json.loads((base_root / "manifest.json").read_text())
        assert base_man["n_members"] == len(base_man["members"]) > 0
        for m in base_man["members"]:
            assert (base_root / m["file"]).exists(), m["file"]
        assert (base_root / base_man["feature_norm"]).exists()
        trained_on = set(man.get("training_pool") or [])
    else:
        assert man["n_members"] == len(man["members"]) > 0
        for m in man["members"]:
            assert (root / m["file"]).exists(), m["file"]
        assert (root / man["feature_norm"]).exists()
        # v1 used "fit_anchors" (DEV excluded too, by an old confounded split); v2 trains on
        # the full eligible pool -- FIT+DEV together -- under "training_pool".
        trained_on = set(man.get("fit_anchors") or man.get("training_pool") or [])

    assert trained_on, "manifest must declare what it trained on"
    assert not (trained_on & set(SEALED))


def test_geometry_classifier_reproduces_the_designated_class():
    """The measured classifier must agree with the human designation where width is usable."""
    import torch as _t
    from pathlib import Path

    from src.clot_ml.geometry_class import USER_DESIGNATED, classify, width_stats

    repo = Path(__file__).resolve().parents[2]
    seen = 0
    for anchor, expected in USER_DESIGNATED.items():
        p = repo / f"data/processed/graphs_biochem_anchors/{anchor}.pt"
        if not p.exists():
            continue
        d = _t.load(p, map_location="cpu", weights_only=False)
        s = width_stats(d)
        if not s.get("usable"):
            continue
        seen += 1
        assert classify(s, anchor) == expected, (anchor, s)
    assert seen >= 4, "expected the designated vessels to be present and measurable"
