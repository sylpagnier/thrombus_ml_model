"""Tests for customer geometry inbox + timeline synthesis."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.data_gen.lib.customer_geometry_import import (
    CustomerGeometryError,
    apply_re_target,
    copy_into_inbox,
    ensure_inbox,
    list_inbox,
    load_customer_geometry,
    synthesize_deploy_timeline,
)
from src.utils.channel_schema import BIO_Y_SCHEMA
from src.utils.paths import get_project_root

REPO = get_project_root()
DEMO_PT = REPO / "data" / "phase_comparison_test" / "graphs_biochem" / "vessel_0.pt"


def test_ensure_and_list_inbox(tmp_path: Path) -> None:
    d = ensure_inbox(tmp_path)
    assert d.is_dir()
    assert d.name == "customer_geometries"
    assert list_inbox(tmp_path) == []


def test_copy_into_inbox(tmp_path: Path) -> None:
    if not DEMO_PT.is_file():
        pytest.skip("demo vessel_0.pt missing")
    dest = copy_into_inbox(DEMO_PT, root=tmp_path)
    assert dest.is_file()
    assert dest.parent == ensure_inbox(tmp_path)
    files = list_inbox(tmp_path)
    assert len(files) == 1
    assert files[0].name == DEMO_PT.name


def test_copy_rejects_stl(tmp_path: Path) -> None:
    bad = tmp_path / "shape.stl"
    bad.write_text("solid", encoding="utf-8")
    with pytest.raises(CustomerGeometryError, match="Unsupported"):
        copy_into_inbox(bad, root=tmp_path)


def test_synthesize_deploy_timeline_shapes() -> None:
    n = 40
    data = Data(
        x=torch.randn(n, 18),
        y=torch.randn(n, 5),
        edge_index=torch.randint(0, n, (2, 80)),
        mask_inlet=torch.zeros(n, dtype=torch.bool),
        mask_outlet=torch.zeros(n, dtype=torch.bool),
        mask_wall=torch.zeros(n, dtype=torch.bool),
        d_bar=torch.tensor([0.02]),
        u_ref=torch.tensor([0.08]),
    )
    data.mask_inlet[0] = True
    data.mask_outlet[-1] = True
    data.mask_wall[1:5] = True
    out = synthesize_deploy_timeline(data, t_final_s=90000.0, n_steps=30)
    assert out.y.shape == (30, n, 16)
    assert out.t.shape == (30,)
    assert float(out.t[-1].item()) == pytest.approx(90000.0)
    assert getattr(out, "y_schema", None) == BIO_Y_SCHEMA


def test_apply_re_target_scales_u_ref() -> None:
    n = 10
    data = Data(
        x=torch.randn(n, 18),
        y=torch.zeros(2, n, 16),
        t=torch.tensor([0.0, 100.0]),
        edge_index=torch.randint(0, n, (2, 20)),
        mask_inlet=torch.zeros(n, dtype=torch.bool),
        mask_outlet=torch.zeros(n, dtype=torch.bool),
        mask_wall=torch.ones(n, dtype=torch.bool),
        d_bar=torch.tensor([0.02]),
        u_ref=torch.tensor([0.05]),
        u_inlet_bc=torch.ones(n, 1) * 0.5,
    )
    data.mask_inlet[0] = True
    data.mask_outlet[-1] = True
    out = apply_re_target(data, 450.0)
    assert float(out.u_ref.item()) > 0.0
    # Re=900 should roughly double u_ref vs Re=450 for same d_bar
    out2 = apply_re_target(data, 900.0)
    assert float(out2.u_ref.item()) == pytest.approx(2.0 * float(out.u_ref.item()), rel=1e-4)


@pytest.mark.skipif(not DEMO_PT.is_file(), reason="demo vessel_0.pt missing")
def test_load_customer_pt_demo() -> None:
    data = load_customer_geometry(DEMO_PT, re_target=450.0, t_final_s=8000.0, n_steps=20)
    assert data.y.shape[0] == 20
    assert data.y.shape[-1] == 16
    assert int(data.mask_wall.bool().sum().item()) > 0
    assert float(data.t[-1].item()) == pytest.approx(8000.0)


def test_mesh_without_sidecar_errors(tmp_path: Path) -> None:
    msh = tmp_path / "lonely.msh"
    # Minimal invalid msh is fine; loader fails on missing json first
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(CustomerGeometryError, match="sidecar"):
        load_customer_geometry(msh, re_target=450.0)
