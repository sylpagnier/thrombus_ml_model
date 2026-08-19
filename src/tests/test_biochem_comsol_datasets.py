"""Tests for COMSOL dataset discovery (no LiveLink)."""

from __future__ import annotations

from src.data_gen.lib.biochem_comsol_datasets import (
    _boundary_dataset_score,
    list_comsol_datasets,
    resolve_boundary_datasets,
    resolve_solution_dataset,
    sample_coords_from_named_selection,
)


class _FakeDataset:
    def __init__(self, label: str, solution: str | None = None) -> None:
        self._label = label
        self._solution = solution

    def label(self) -> str:
        return self._label

    def getString(self, prop: str) -> str:
        if prop in ("solution", "sol"):
            return self._solution or ""
        return ""


class _FakeDatasetRoot:
    def __init__(self, items: dict[str, _FakeDataset]) -> None:
        self._items = items

    def tags(self) -> list[str]:
        return list(self._items.keys())

    def get(self, tag: str) -> _FakeDataset:
        return self._items[tag]


class _FakeResult:
    def __init__(self, datasets: dict[str, _FakeDataset]) -> None:
        self._datasets = _FakeDatasetRoot(datasets)

    def dataset(self) -> _FakeDatasetRoot:
        return self._datasets


class _FakeModelJava:
    def __init__(self, datasets: dict[str, _FakeDataset]) -> None:
        self._result = _FakeResult(datasets)

    def result(self) -> _FakeResult:
        return self._result


def test_resolve_solution_dataset_prefers_sol1_link():
    model = _FakeModelJava(
        {
            "dset1": _FakeDataset("Study 1 (fluid + biochemistry)/Solution 1 (sol1)", "sol1"),
            "dset2": _FakeDataset("Study 2 (only fluid)/Soluzione 2 (sol2)", "sol2"),
            "dset3": _FakeDataset("Inlet"),
            "dset4": _FakeDataset("Wall"),
        }
    )
    assert resolve_solution_dataset(model, "sol1") == "dset1"
    assert resolve_solution_dataset(model, "sol2") == "dset2"


def test_boundary_dataset_score_prefers_inlet_over_edg():
    assert _boundary_dataset_score("inlet", "Inlet", "dset5") > _boundary_dataset_score("inlet", "Inlet", "edg1")
    assert _boundary_dataset_score("inlet", "Inlet", "edg1") == 90
    assert _boundary_dataset_score("inlet", "edge only", "edg9") < 0


def test_resolve_boundary_datasets_edg_labels_only():
    model = _FakeModelJava(
        {
            "dset1": _FakeDataset("Study 1/Solution 1", "sol1"),
            "edg1": _FakeDataset("Inlet"),
            "edg2": _FakeDataset("Outlet"),
            "edg3": _FakeDataset("Wall"),
        }
    )
    assert resolve_boundary_datasets(model) == {
        "inlet": "edg1",
        "outlet": "edg2",
        "wall": "edg3",
    }


def test_boundary_dataset_score_template_box_selections():
    assert _boundary_dataset_score("inlet", "inlet", "box1") >= 95
    assert _boundary_dataset_score("wall", "wall", "dif1") >= 95


def test_resolve_boundary_datasets_by_label():
    model = _FakeModelJava(
        {
            "dset1": _FakeDataset("Study 1/Solution 1 (sol1)", "sol1"),
            "edg1": _FakeDataset("Inlet"),
            "dset2": _FakeDataset("Inlet"),
            "dset3": _FakeDataset("Outlet"),
            "dset4": _FakeDataset("Wall"),
        }
    )
    got = resolve_boundary_datasets(model)
    assert got == {"inlet": "dset2", "outlet": "dset3", "wall": "dset4"}


def test_list_comsol_datasets():
    model = _FakeModelJava({"dset1": _FakeDataset("Wall")})
    rows = list_comsol_datasets(model)
    assert rows[0]["tag"] == "dset1"
    assert rows[0]["label"] == "Wall"


def test_sample_coords_from_named_selection_edge2d_and_cleanup(monkeypatch):
    import numpy as np

    created: list[tuple[str, str]] = []
    removed: list[str] = []

    class _Sel:
        def __init__(self) -> None:
            self.named_tag = None

        def named(self, tag: str) -> None:
            self.named_tag = tag

    class _Ds:
        def __init__(self) -> None:
            self.props: dict[str, str] = {}
            self._sel = _Sel()

        def set(self, key: str, val: str) -> None:
            self.props[key] = val

        def selection(self) -> _Sel:
            return self._sel

    class _DsRoot:
        def __init__(self) -> None:
            self.last: _Ds | None = None

        def create(self, tag: str, dtype: str) -> _Ds:
            created.append((tag, dtype))
            self.last = _Ds()
            return self.last

        def remove(self, tag: str) -> None:
            removed.append(tag)

    class _Result:
        def __init__(self) -> None:
            self._ds = _DsRoot()

        def dataset(self) -> _DsRoot:
            return self._ds

        def numerical(self):
            raise AssertionError("Eval fallback should not run when Edge2D succeeds")

    class _Model:
        def __init__(self) -> None:
            self._r = _Result()

        def result(self) -> _Result:
            return self._r

    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_datasets.sample_coords_from_dataset",
        lambda _m, _tag, *, edim=1, exprs=("x", "y"): np.array([[1.0, 2.0]], dtype=np.float64),
    )
    model = _Model()
    xy = sample_coords_from_named_selection(model, "sel1", parent_dataset="dset1")
    assert xy.shape == (1, 2)
    assert float(xy[0, 0]) == 1.0 and float(xy[0, 1]) == 2.0
    assert created[0][1] == "Edge2D"
    assert model._r._ds.last is not None
    assert model._r._ds.last.props["data"] == "dset1"
    assert model._r._ds.last._sel.named_tag == "sel1"
    assert "py_wound_sel" in removed
