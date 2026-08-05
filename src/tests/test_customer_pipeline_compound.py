"""Customer deploy defaults to canonical compound model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.inference.customer_pipeline import (
    DEFAULT_COMPOUND_LEG,
    _compound_deploy_overrides,
    _compound_mode_requested,
    _resolve_offwall_ckpt,
    _resolve_wall_ckpt,
)
from src.biochem_gnn.compound_deploy import load_compound_manifest


@pytest.fixture(autouse=True)
def _clean_customer_env(monkeypatch):
    for key in (
        "CUSTOMER_COMPOUND",
        "CUSTOMER_WALL_CKPT",
        "CUSTOMER_OFFWALL_CKPT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_compound_mode_on_by_default():
    assert _compound_mode_requested() is True


def test_compound_mode_opt_out():
    os.environ["CUSTOMER_COMPOUND"] = "0"
    assert _compound_mode_requested() is False


def test_resolve_wall_ckpt_uses_locked_default():
    p = _resolve_wall_ckpt()
    assert p.name == "species_gnn_best.pth"
    assert "locked" in p.as_posix()


def test_resolve_offwall_ckpt_uses_compound_growth():
    p = _resolve_offwall_ckpt()
    assert p is not None
    assert p.name == "compound_growth_best.pth"
    assert p.is_file()


def test_resolve_offwall_ckpt_wall_only_when_opt_out():
    os.environ["CUSTOMER_COMPOUND"] = "0"
    assert _resolve_offwall_ckpt() is None


def test_compound_deploy_overrides_match_manifest():
    ov = _compound_deploy_overrides()
    deploy = (load_compound_manifest().get("deploy") or {})
    assert ov["SPECIES_TWO_MODEL_MODE"] == "1"
    assert ov["SPECIES_TWO_MODEL_ROUTE"] == str(deploy.get("SPECIES_TWO_MODEL_ROUTE", "frontier"))
    assert ov["SPECIES_TWO_MODEL_FRONTIER_HOPS"] == str(deploy.get("SPECIES_TWO_MODEL_FRONTIER_HOPS", "1"))
    assert ov["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] == "1"


def test_default_compound_leg_label():
    assert DEFAULT_COMPOUND_LEG == "WC_v8_compound_front_h1"
