"""Mat channel decode for COMSOL mu1(Mat) gelation step."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.core_physics.clot_phi_simple import (
    mat_si_for_gelation_from_log1p,
    mu1_comsol_from_mat_si,
    species_log1p_nd_to_si,
)
from src.core_physics.t0_mu_physics import (
    clot_phi_binary_from_mu_growth,
    gt_clot_phi_at_time,
    gt_mu_anchor_cap_si,
    load_debug_sidecar,
    predict_mu_si_at_time,
    predict_mu_si_from_graph_species_legs,
    t0_physics_env,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def test_mat_gelation_decode_removes_surface_scale_factor():
    """Graph ND encodes log1p(raw/Minf); legacy decode was raw*surface_scale."""
    bio = BiochemConfig(phase="biochem")
    raw = torch.tensor([0.0, 5.9067e4, 2.5454986e7, 3.0e7])
    nd = torch.log1p(raw / bio.Minf)
    legacy = torch.expm1(nd) * bio.Minf * bio.surface_scale
    fixed = mat_si_for_gelation_from_log1p(nd, bio)
    assert torch.allclose(fixed, raw, rtol=1e-4, atol=1.0)
    ratio = (legacy / fixed.clamp(min=1.0)).median()
    assert abs(float(ratio.item()) - bio.surface_scale) / bio.surface_scale < 0.01


def test_species_log1p_mat_channel_uses_gelation_units():
    bio = BiochemConfig(phase="biochem")
    raw_mat = 5.9067e4
    sp = torch.zeros(4, 12)
    sp[:, 11] = torch.log1p(torch.tensor(raw_mat / bio.Minf))
    si = species_log1p_nd_to_si(sp, bio)
    assert abs(float(si[0, 11].item()) - raw_mat) / raw_mat < 1e-4
    # M/Mas still use legacy surface_scale decode (deposition channels).
    legacy_m = float(torch.expm1(sp[0, 9]) * bio.Minf * bio.surface_scale)
    assert si[0, 9].item() == pytest.approx(legacy_m)


@pytest.fixture
def patient007_assets():
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph.is_file():
        pytest.skip("patient007 graph missing")
    debug = load_debug_sidecar("patient007", root=root)
    if debug is None:
        pytest.skip("patient007 debug sidecar missing; run build_comsol_debug_sidecar.py")
    return root, graph, debug


def test_mat_matches_comsol_export(patient007_assets):
    root, graph_path, debug = patient007_assets
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    t = min(53, int(data.y.shape[0]) - 1)
    mat_graph = mat_si_for_gelation_from_log1p(data.y[t, :, 15], bio)
    mat_comsol = debug["mat_si"][t].reshape(-1)
    growth = gt_growth_commit_mask_at_time(data, t, phys, torch.device("cpu"))
    active = mat_comsol > 1.0
    ac = mat_graph.float() - mat_graph.float().mean()
    bc = mat_comsol.float() - mat_comsol.float().mean()
    den = ac.pow(2).sum().sqrt() * bc.pow(2).sum().sqrt()
    pearson = float((ac * bc).sum().item() / den.clamp(min=1e-12).item())
    assert pearson >= 0.99
    if bool(growth.any().item()):
        ratio_g = float((mat_comsol / mat_graph.clamp(min=1e-8))[growth].median().item())
        assert 0.98 <= ratio_g <= 1.02
    if bool(active.any().item()):
        ratio_a = float((mat_comsol / mat_graph.clamp(min=1e-8))[active].median().item())
        assert 0.98 <= ratio_a <= 1.02


def test_mu1_from_graph_mat_matches_comsol(patient007_assets):
    root, graph_path, debug = patient007_assets
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    t = min(53, int(data.y.shape[0]) - 1)
    with t0_physics_env("patient007", gamma_mode="comsol_sr"):
        _, py_mu1, _ = predict_mu_si_from_graph_species_legs(
            data, t, phys, bio, debug, device, mat_source="graph", fi_source="graph"
        )
    comsol_mu1 = debug["mu1"][t].reshape(-1)
    frac_diff = float((py_mu1 != comsol_mu1).float().mean().item())
    assert frac_diff < 0.02


def test_t0_clot_f1_improves_with_mat_fix(patient007_assets):
    """Graph species + COMSOL sr should approach oracle Mat/FI clot F1."""
    root, graph_path, _debug = patient007_assets
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    t = min(53, int(data.y.shape[0]) - 1)
    with t0_physics_env("patient007", gamma_mode="comsol_sr"):
        step = predict_mu_si_at_time(data, t, phys, bio, device, gamma_mode="comsol_sr")
    anchor = gt_mu_anchor_cap_si(data, phys, device)
    phi_gt = gt_clot_phi_at_time(data, t, phys, device)
    phi_pred = clot_phi_binary_from_mu_growth(step.mu_pred_si, anchor, phys)
    mask = torch.ones(phi_pred.numel(), dtype=torch.bool)
    f1 = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)["clot_f1"]
    assert f1 >= 0.65


def test_mu1_step_threshold_at_comsol_crit():
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    below = torch.tensor([crit * 0.5])
    above = torch.tensor([crit * 1.5])
    mu_below = mu1_comsol_from_mat_si(below, bio, bio.mu_ratio_max)
    mu_above = mu1_comsol_from_mat_si(above, bio, bio.mu_ratio_max)
    assert float(mu_below.item()) == pytest.approx(1.0, rel=0.01)
    assert float(mu_above.item()) == pytest.approx(float(bio.mu_ratio_max), rel=0.01)
