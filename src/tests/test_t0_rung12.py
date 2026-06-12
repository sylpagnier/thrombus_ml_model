"""Rung 1/2 COMSOL oracle gates (patient007)."""

from pathlib import Path

import pytest

from src.utils.paths import get_project_root


@pytest.fixture
def rung12_eval():
    path = get_project_root() / "outputs/biochem/clot_trigger/t0_rung12_eval.json"
    if not path.is_file():
        pytest.skip("run go_t0_rung12.ps1 first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_rung1_mu_gates(rung12_eval):
    gates = rung12_eval["gates"]
    assert gates["rung1_bulk_t0"]
    assert gates["rung1_growth_t_last"]
    assert gates["rung1_pearson_growth_t_last"]


def test_rung2_proxy_bulk_and_clot(rung12_eval):
    gates = rung12_eval["gates"]
    assert gates["rung2_bulk_t0"]
    assert gates["rung2_clot_f1_matches_rung1"]
    assert gates["rung1_clot_f1_nuc_t_last"]
