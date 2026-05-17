"""Tests for ``BIOCHEM_COMPLEXITY_STEP=2`` preset and data-only μ anchor wiring."""
from __future__ import annotations

import os

import pytest


def test_complexity_step2_sets_mu_and_tbptt(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_complexity_step2_env

    for key in (
        "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT",
        "BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS",
        "BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT",
        "BIOCHEM_TBPTT_MAX_WINDOW",
        "BIOCHEM_TBPTT_WINDOW_CURRICULUM",
        "BIOCHEM_TBPTT_ANCHOR_RANDOM_START",
        "BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "2")
    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)

    _apply_biochem_complexity_step2_env()

    assert float(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT"]) == pytest.approx(8.0)
    assert os.environ["BIOCHEM_TBPTT_ANCHOR_END_BIAS"].strip().lower() in ("1", "true", "yes", "on")
    assert int(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS"]) == 8
    assert float(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_MULT"]) == pytest.approx(1.5)
    assert int(os.environ["BIOCHEM_TBPTT_MAX_WINDOW"]) == 12
    assert os.environ["BIOCHEM_TBPTT_WINDOW_CURRICULUM"].strip().lower() == "1"


def test_complexity_step2_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_complexity_step2_env

    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "phase2")
    monkeypatch.setenv("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", "0.25")
    monkeypatch.setenv("BIOCHEM_TBPTT_MAX_WINDOW", "7")
    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)

    _apply_biochem_complexity_step2_env()

    assert float(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT"]) == pytest.approx(0.25)
    assert int(os.environ["BIOCHEM_TBPTT_MAX_WINDOW"]) == 7


def test_complexity_step2_skipped_when_stock_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_complexity_step2_env

    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "2")
    monkeypatch.setenv("BIOCHEM_STOCK_DEFAULTS", "1")
    monkeypatch.delenv("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", raising=False)

    _apply_biochem_complexity_step2_env()

    assert "BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT" not in os.environ
