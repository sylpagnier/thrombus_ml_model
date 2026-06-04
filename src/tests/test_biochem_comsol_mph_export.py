"""Tests for COMSOL mph Export node helpers (no LiveLink)."""

from __future__ import annotations

from pathlib import Path

from src.data_gen.lib.biochem_comsol_mph_export import (
    ensure_biochem_extract_dirs,
    resolve_export_tags,
    use_mph_result_exports,
    _normalize_boundary_export,
    _validate_domain_export,
)


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
