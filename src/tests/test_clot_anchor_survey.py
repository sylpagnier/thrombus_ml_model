"""Cross-anchor survey: where COMSOL clots form and kinematic cues at onset vs t_final."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.core_physics.clot_anchor_survey import (
    aggregate_separability,
    discover_anchor_paths,
    format_survey_table,
    survey_all_anchors,
    survey_anchor_graph,
)

REPO = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO / "data" / "processed" / "graphs_biochem_anchors"


def _require_anchors() -> list[Path]:
    paths = discover_anchor_paths(ANCHOR_DIR)
    if not paths:
        pytest.skip(f"no anchor graphs under {ANCHOR_DIR}")
    return paths


@pytest.fixture(scope="module")
def anchor_surveys():
    for key, val in (
        ("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1"),
        ("BIOCHEM_PRIOR_DGAMMA_DX_THRESH", "800"),
        ("BIOCHEM_PRIOR_NORM_MASK", "adjacent"),
    ):
        os.environ[key] = val
    return survey_all_anchors(ANCHOR_DIR)


def test_anchor_survey_runs_on_all_graphs(anchor_surveys):
    paths = _require_anchors()
    assert len(anchor_surveys) == len(paths)
    for s in anchor_surveys:
        assert s.n_nodes > 0
        assert s.n_times > 0


def test_print_anchor_clot_survey_table(anchor_surveys):
    """Human-readable report (run with ``pytest -s``)."""
    _require_anchors()
    table = format_survey_table(anchor_surveys)
    agg = aggregate_separability(anchor_surveys)
    assert "Anchor clot survey" in table
    assert agg.get("n_anchors", 0) >= 0
    # Uncomment to inspect: pytest -s -k test_print_anchor_clot_survey_table
    print("\n" + table)
    print(f"aggregate: {agg}")


def test_clots_are_near_wall_by_sdf(anchor_surveys):
    """High-μ clots should sit near the wall geometrically (low sdf_nd), mesh flags vary."""
    active = [s for s in anchor_surveys if s.n_clot_strict_tfinal >= 10]
    if len(active) < 3:
        pytest.skip("need several anchors with >=10 strict clots")
    near_frac = sum(1 for s in active if s.pct_clot_near_wall_sdf_tfinal >= 50.0)
    assert near_frac >= len(active) // 2, (
        f"expected most anchors with clots near wall (sdf<=0.02); "
        f"{near_frac}/{len(active)} passed"
    )


def test_patient007_dgamma_dx_separates_clots(anchor_surveys):
    s = next((x for x in anchor_surveys if x.stem == "patient007"), None)
    if s is None or s.n_clot_strict_tfinal < 5:
        pytest.skip("patient007 missing or too few clots")
    assert s.dgamma_dx_clot_mean_tfinal < s.dgamma_dx_non_mean_tfinal
    assert s.n_clot_strict_t0 <= max(10, int(0.15 * s.n_clot_strict_tfinal))


def test_majority_anchors_negative_dx_delta(anchor_surveys):
    agg = aggregate_separability(anchor_surveys)
    if agg.get("n_anchors", 0) < 2:
        pytest.skip("need multiple anchors with clots")
    n_neg = int(agg.get("n_anchors_dx_negative", 0))
    assert n_neg >= max(1, agg["n_anchors"] // 2), (
        f"expected most anchors with clot dγ/dx < non-clot; agg={agg}"
    )


def test_suggested_dx_threshold_below_comsol_plot_scale(anchor_surveys):
    """Mesh dγ/dx is O(1–50); threshold 800 silences the gate (see diagnostic)."""
    vals = [
        s.suggested_dx_thresh_p10
        for s in anchor_surveys
        if s.n_clot_strict_tfinal >= 5 and s.suggested_dx_thresh_p10 == s.suggested_dx_thresh_p10
    ]
    if not vals:
        pytest.skip("no calibrated p10 values (need clot∩adjacent nodes)")
    med = float(torch.tensor(vals).median())
    assert med < 200.0, (
        f"median p10 adverse dγ/dx={med:.2f}; use ~{med:.0f} not 800 for PRIOR_DGAMMA_DX_THRESH"
    )


def test_patient007_survey_smoke(monkeypatch):
    path = ANCHOR_DIR / "patient007.pt"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    data = torch.load(path, map_location="cpu", weights_only=False)
    s = survey_anchor_graph(data, stem="patient007")
    assert s.n_clot_strict_tfinal >= 0
    if s.n_clot_strict_tfinal >= 5:
        assert s.dgamma_dx_clot_mean_tfinal < s.dgamma_dx_non_mean_tfinal
