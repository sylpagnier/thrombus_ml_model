"""T0 mu physics oracle (GT flow + species, no GT mu input)."""

from pathlib import Path

import pytest
import torch

from src.core_physics.t0_mu_physics import (
    comsol_sr_sidecar_available,
    eval_anchor_t0_mu,
    predict_mu_si_at_time,
    t0_physics_env,
)


@pytest.fixture
def anchor_graph():
    paths = list(Path("data/processed/graphs_biochem_anchors").glob("patient*.pt"))
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(str(paths[0]), map_location="cpu", weights_only=False)


def test_t0_bulk_mu_matches_gt_at_t0(anchor_graph):
    """M=1 Carreau + comsol_sr should match GT bulk spf.mu at t=0."""
    anchor = "patient007"
    graph = Path("data/processed/graphs_biochem_anchors/patient007.pt")
    if not graph.is_file():
        pytest.skip("patient007 not available")
    if not comsol_sr_sidecar_available(anchor):
        pytest.skip("patient007 sr sidecar not available")

    report = eval_anchor_t0_mu(graph, times=[0], gamma_mode="comsol_sr")
    t0 = report.times[0]
    assert report.pass_gates["bulk_mu_t0"]
    assert 0.98 <= t0["ratio_median_bulk"] <= 1.02
    assert t0["pearson_all"] >= 0.99


def test_t0_does_not_read_mu_channel_for_prediction(anchor_graph):
    """Corrupting GT mu channel must not change mu_pred."""
    from src.config import BiochemConfig, PhysicsConfig

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = anchor_graph
    device = torch.device("cpu")
    anchor = "patient007" if Path("data/processed/graphs_biochem_anchors/patient007.pt").is_file() else "patient002"

    with t0_physics_env(anchor, gamma_mode="max"):
        step_a = predict_mu_si_at_time(data, 0, phys, bio, device, gamma_mode="max")
        y_corrupt = data.y.clone()
        y_corrupt[:, :, 3] = 999.0
        data_corrupt = data
        data_corrupt.y = y_corrupt
        step_b = predict_mu_si_at_time(data_corrupt, 0, phys, bio, device, gamma_mode="max")
        data.y = anchor_graph.y
    assert torch.allclose(step_a.mu_pred_si, step_b.mu_pred_si)
