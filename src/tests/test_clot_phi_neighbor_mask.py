"""Neighbor-based clot-phi supervision mask vs GT clot nodes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_simple import (
    _wall_mask_from_data,
    cap_mu_eff_si,
    clot_phi_center_exclude_frac,
    clot_phi_thresh_si,
    gt_mu_anchor_cap_si,
    neighbor_supervision_mask,
    supervision_region_mask,
)
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.utils.paths import get_project_root

REPO = get_project_root()
ANCHOR = REPO / VesselConfig(phase="biochem_anchors").graph_output_dir / "patient007.pt"


@pytest.fixture
def patient007():
    if not ANCHOR.is_file():
        pytest.skip(f"missing anchor graph: {ANCHOR}")
    return torch.load(ANCHOR, map_location="cpu", weights_only=False)


def test_center_exclude_frac_env_default(monkeypatch):
    monkeypatch.delenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", raising=False)
    assert clot_phi_center_exclude_frac() == pytest.approx(0.10)
    monkeypatch.setenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.25")
    assert clot_phi_center_exclude_frac() == pytest.approx(0.25)


def test_neighbor_mask_includes_all_wall_nodes(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_SHEAR_MIN_FRAC", "0")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    region = supervision_region_mask(patient007, device, mu_cap, phys)
    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    assert int((wall & ~region).sum()) == 0, "every wall node must be in supervision mask"


def test_dgamma_slice_raises_mask_precision_on_patient007(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.10")
    monkeypatch.setenv("CLOT_PHI_SHEAR_MIN_FRAC", "0")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "0")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    base = supervision_region_mask(patient007, device, mu_cap, phys)
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_REF_TIME", "0")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_WALL_MIN_SI", "100")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_OFFWALL_PCT", "80")
    sliced = supervision_region_mask(patient007, device, mu_cap, phys)
    base_pos = int((clot & base).sum())
    slice_pos = int((clot & sliced).sum())
    base_prec = base_pos / max(int(base.sum()), 1)
    slice_prec = slice_pos / max(int(sliced.sum()), 1)
    assert slice_prec > base_prec + 0.2
    assert slice_pos >= int(0.9 * base_pos)


def test_shear_filter_keeps_wall_and_trims_low_shear_offwall(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.10")
    monkeypatch.setenv("CLOT_PHI_SHEAR_MIN_FRAC", "0.5")
    monkeypatch.setenv("CLOT_PHI_SHEAR_WALL_EXEMPT", "1")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    base = supervision_region_mask(patient007, device, mu_cap, phys)
    monkeypatch.setenv("CLOT_PHI_SHEAR_MIN_FRAC", "0")
    full = supervision_region_mask(patient007, device, mu_cap, phys)
    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    assert int((wall & ~base).sum()) == 0
    assert int(base.sum()) < int(full.sum())
    assert int((clot & base).sum()) >= int((clot & full).sum()) * 0.9


def test_neighbor_mask_covers_all_gt_clot_nodes(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_CLOT_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_CLOT_TOUCH_HOPS", "1")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    region = supervision_region_mask(patient007, device, mu_cap, phys)
    from src.core_physics.clot_phi_simple import sdf_nd_from_data

    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    sdf = sdf_nd_from_data(patient007, device, int(patient007.num_nodes))
    lumen = (~wall) & (sdf > 1e-8)
    inner = torch.zeros_like(clot)
    if int(lumen.sum()) > 0:
        inner = (~wall) & (sdf >= torch.quantile(sdf[lumen], 0.90))
    outer_clot = clot & ~inner
    assert int((outer_clot & region).sum()) == int(outer_clot.sum()), (
        "every non-centerline GT clot must lie in neighbor support"
    )


def test_neighbor_mask_excludes_inflow_far_from_clot(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CLOT_TOUCH_HOPS", "1")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    region = neighbor_supervision_mask(patient007, device, clot)
    wall = patient007.mask_wall.view(-1).bool()
    near = clot.clone()
    row, col = patient007.edge_index
    active = near.clone()
    near[row[active[col]]] = True
    near[col[active[row]]] = True
    inflow = region & ~wall & ~near
    assert int(inflow.sum()) == 0, "off-wall mask nodes must be within 1 hop of a clot seed"


def test_neighbor_mask_no_halo_from_excluded_inner_clot(patient007, monkeypatch):
    """1-hop band must not inherit from centerline-excluded clot seeds."""
    monkeypatch.setenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.10")
    monkeypatch.setenv("CLOT_PHI_CLOT_TOUCH_HOPS", "1")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    region = neighbor_supervision_mask(patient007, device, clot)
    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    from src.core_physics.clot_phi_simple import sdf_nd_from_data

    sdf = sdf_nd_from_data(patient007, device, int(patient007.num_nodes))
    lumen = (~wall) & (sdf > 1e-8)
    inner = (~wall) & (sdf >= torch.quantile(sdf[lumen], 0.90))
    excluded_seed = inner & clot
    row, col = patient007.edge_index
    for i in excluded_seed.nonzero(as_tuple=True)[0].tolist():
        nbr = torch.cat([col[row == i], row[col == i]]).unique()
        halo = nbr[(~wall[nbr]) & (~clot[nbr])]
        assert int((region[halo]).sum()) == 0, f"halo of excluded inner clot {i} must stay out of mask"


def test_neighbor_mask_excludes_centerline_interior(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CENTER_EXCLUDE_FRAC", "0.10")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    region = neighbor_supervision_mask(patient007, device, clot)
    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    from src.core_physics.clot_phi_simple import sdf_nd_from_data

    sdf = sdf_nd_from_data(patient007, device, int(patient007.num_nodes))
    lumen_all = (~wall) & (sdf > 1e-8)
    if int(lumen_all.sum()) > 0:
        thr = torch.quantile(sdf[lumen_all], 0.90)
        clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
        inner = (~wall) & (sdf >= thr)
        inner_in_mask = region & inner
        assert int(inner_in_mask.sum()) == 0, "top 10% interior lumen nodes should be excluded from mask"


def test_neighbor_mask_has_fewer_inflow_nodes_than_sdf_shell(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CLOT_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_CLOT_TOUCH_HOPS", "1")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    ti = int(patient007.y.shape[0]) - 1
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(patient007.y[ti][:, STATE_CHANNEL_MU_EFF_ND]))
    clot = gt_growth_commit_mask_at_time(patient007, ti, phys, device)
    region = neighbor_supervision_mask(patient007, device, clot)
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "sdf")
    from src.core_physics.clot_phi_simple import sdf_supervision_mask

    sdf = sdf_supervision_mask(patient007, device)
    wall = patient007.mask_wall.view(-1).bool()
    touch = clot.clone()
    row, col = patient007.edge_index
    active = touch.clone()
    touch[row[active[col]]] = True
    touch[col[active[row]]] = True
    inflow_neighbor = int((region & ~wall & ~touch).sum())
    inflow_sdf = int((sdf & ~wall & ~touch).sum())
    assert inflow_neighbor <= inflow_sdf
    from src.core_physics.clot_phi_simple import sdf_nd_from_data

    wall = _wall_mask_from_data(patient007, device, int(patient007.num_nodes))
    sdf = sdf_nd_from_data(patient007, device, int(patient007.num_nodes))
    lumen = (~wall) & (sdf > 1e-8)
    inner = torch.zeros_like(clot)
    if int(lumen.sum()) > 0:
        inner = (~wall) & (sdf >= torch.quantile(sdf[lumen], 0.90))
    outer_clot = clot & ~inner
    assert int((outer_clot & region).sum()) == int(outer_clot.sum())
