"""Tests for biochem COMSOL boundary mask expression fallbacks (no LiveLink)."""

from __future__ import annotations

import numpy as np

from src.data_gen.lib.biochem_comsol_mesh_export import (
    boundary_mask_expr_candidates,
    discover_boundary_mask_exprs,
    write_boundary_txt_from_axis_extents,
)


class _FakeSelection:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeSelectionRoot:
    def __init__(self, items: dict[str, str]) -> None:
        self._items = {k: _FakeSelection(v) for k, v in items.items()}

    def tags(self) -> list[str]:
        return list(self._items.keys())

    def get(self, tag: str) -> _FakeSelection:
        return self._items[tag]


class _FakeModelJava:
    def __init__(self, selections: dict[str, str]) -> None:
        self._selections = _FakeSelectionRoot(selections)

    def selection(self) -> _FakeSelectionRoot:
        return self._selections


def test_discover_boundary_mask_exprs_explicit_inlet_outlet_wall_tags():
    model = _FakeModelJava({"inlet": "inlet", "outlet": "outlet", "wall": "wall"})
    found = discover_boundary_mask_exprs(model)
    assert found == {
        "inlet": "inlet(x,y)",
        "outlet": "outlet(x,y)",
        "wall": "wall(x,y)",
    }
    assert boundary_mask_expr_candidates(model, "inlet")[0] == "inlet(x,y)"


def test_discover_boundary_mask_exprs_from_legacy_box_tags():
    model = _FakeModelJava(
        {
            "box1": "inlet",
            "box2": "outlet",
            "dif1": "wall",
            "imp1": "import",
        }
    )
    found = discover_boundary_mask_exprs(model)
    assert found == {
        "inlet": "box1(x,y)",
        "outlet": "box2(x,y)",
        "wall": "dif1(x,y)",
    }
    assert boundary_mask_expr_candidates(model, "inlet")[0] == "box1(x,y)"
    assert "inlet(x,y)" in boundary_mask_expr_candidates(model, "inlet")


def test_write_boundary_txt_from_axis_extents(tmp_path):
    # Channel along x: inlet at x=0, outlet at x=1, walls at y=0 and y=1.
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    write_boundary_txt_from_axis_extents(coords, tmp_path, "ch", force=True)
    inlet = (tmp_path / "ch_inlet.txt").read_text(encoding="utf-8")
    assert "0 0 0.0000000000 0.0000000000" in inlet or "0 0 0 0" in inlet
    assert (tmp_path / "ch_outlet.txt").is_file()
    assert (tmp_path / "ch_wall.txt").is_file()
