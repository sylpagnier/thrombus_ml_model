"""Tests for COMSOL mph Export node helpers (no LiveLink)."""

from __future__ import annotations

from pathlib import Path

from src.data_gen.lib.biochem_comsol_mph_export import (
    discover_export_tags,
    ensure_biochem_extract_dirs,
    resolve_export_tags,
    use_mph_result_exports,
    _normalize_boundary_export,
    _validate_domain_export,
)


class _FakeExportNode:
    def __init__(
        self,
        *,
        node_type: str = "Data",
        label: str = "",
        expr_count: int = 2,
        dataset: str = "",
    ) -> None:
        self._node_type = node_type
        self._label = label
        self._expr_count = expr_count
        self._dataset = dataset

    def getType(self) -> str:
        return self._node_type

    def label(self) -> str:
        return self._label

    def name(self) -> str:
        return self._label

    def getStringArray(self, key: str) -> list[str]:
        if key == "expr":
            return [f"e{i}" for i in range(self._expr_count)]
        return []

    def getString(self, key: str) -> str:
        if key == "data":
            return self._dataset
        return ""


class _FakeExportRoot:
    def __init__(self, nodes: dict[str, _FakeExportNode]) -> None:
        self._nodes = nodes

    def tags(self) -> list[str]:
        return list(self._nodes.keys())

    def get(self, tag: str) -> _FakeExportNode:
        return self._nodes[tag]


class _FakeResultForExport:
    def __init__(self, nodes: dict[str, _FakeExportNode]) -> None:
        self._export = _FakeExportRoot(nodes)

    def export(self) -> _FakeExportRoot:
        return self._export


class _FakeModelForExport:
    def __init__(self, nodes: dict[str, _FakeExportNode]) -> None:
        self._result = _FakeResultForExport(nodes)

    def result(self) -> _FakeResultForExport:
        return self._result


def test_ensure_biochem_extract_dirs(tmp_path):
    raw = tmp_path / "raw" / "biochem_anchors"
    label = tmp_path / "processed" / "cfd_results_biochem"
    proc = tmp_path / "processed" / "graphs_biochem_anchors"
    ensure_biochem_extract_dirs(raw, label, proc)
    assert raw.is_dir() and label.is_dir() and proc.is_dir()


def test_resolve_export_tags_defaults():
    tags = resolve_export_tags()
    assert tags["domain"] == "sol_data"
    assert tags["inlet"] == "inlet_nodes"


def test_normalize_boundary_export(tmp_path):
    src = tmp_path / "raw.txt"
    src.write_text("% x y\n1.0 2.0 3.0\n", encoding="utf-8")
    dest = tmp_path / "out.txt"
    _normalize_boundary_export(src, dest)
    text = dest.read_text(encoding="utf-8")
    assert "0 0 2.0000000000 3.0000000000" in text


def test_validate_domain_export_header(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("% x y u @ t=0\n0 0 1 2 3\n", encoding="utf-8")
    _validate_domain_export(p)


def test_use_mph_exports_default_on(monkeypatch):
    monkeypatch.delenv("BIOCHEM_COMSOL_USE_MPH_EXPORTS", raising=False)
    assert use_mph_result_exports() is True
    monkeypatch.setenv("BIOCHEM_COMSOL_USE_MPH_EXPORTS", "0")
    assert use_mph_result_exports() is False


def test_discover_export_tags_phase2_data_nodes():
    model = _FakeModelForExport(
        {
            "anim1": _FakeExportNode(node_type="Animation"),
            "data1": _FakeExportNode(expr_count=24, dataset="dset1"),
            "data2": _FakeExportNode(expr_count=2, dataset="edg1"),
            "data3": _FakeExportNode(expr_count=2, dataset="edg2"),
            "data4": _FakeExportNode(expr_count=2, dataset="edg3"),
        }
    )
    got = discover_export_tags(model)
    assert got["domain"] == "data1"
    assert got["inlet"] == "data2"
    assert got["outlet"] == "data3"
    assert got["wall"] == "data4"


def test_resolve_export_tags_uses_discovery(monkeypatch):
    monkeypatch.delenv("BIOCHEM_COMSOL_EXPORT_DOMAIN", raising=False)
    model = _FakeModelForExport(
        {
            "data1": _FakeExportNode(expr_count=20, dataset="dset1"),
            "data2": _FakeExportNode(expr_count=2, dataset="edg1"),
            "data3": _FakeExportNode(expr_count=2, dataset="edg2"),
            "data4": _FakeExportNode(expr_count=2, dataset="edg3"),
        }
    )
    tags = resolve_export_tags(model)
    assert tags["domain"] == "data1"
    assert tags["inlet"] == "data2"
