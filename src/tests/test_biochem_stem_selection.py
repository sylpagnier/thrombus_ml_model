"""Tests for multi-stem COMSOL extract selection parsing."""

from __future__ import annotations

from src.data_gen.lib.biochem_comsol_auto_export import resolve_stem_selection


def _table(n: int) -> list[str]:
    return [f"patient{i:03d}" for i in range(1, n + 1)]


def test_resolve_indices_and_ranges():
    table = _table(11)
    assert resolve_stem_selection("5", table) == ["patient005"]
    assert resolve_stem_selection("5,8,9", table) == ["patient005", "patient008", "patient009"]
    assert resolve_stem_selection("8-10", table) == ["patient008", "patient009", "patient010"]
    assert resolve_stem_selection("5, 8-10", table) == [
        "patient005",
        "patient008",
        "patient009",
        "patient010",
    ]


def test_resolve_patient_names():
    table = _table(7)
    assert resolve_stem_selection("patient007", table) == ["patient007"]
    assert resolve_stem_selection("patient005,patient007", table) == ["patient005", "patient007"]
