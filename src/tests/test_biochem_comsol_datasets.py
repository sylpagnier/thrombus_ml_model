"""Tests for COMSOL dataset discovery (no LiveLink)."""

from __future__ import annotations

from src.data_gen.lib.biochem_comsol_datasets import (
    _boundary_dataset_score,
    list_comsol_datasets,
    resolve_boundary_datasets,
    resolve_solution_dataset,
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
    assert _boundary_dataset_score("inlet", "Inlet", "edg1") < 0


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
