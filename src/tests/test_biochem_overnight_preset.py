"""Tests for ``BIOCHEM_PRESET=overnight_step2`` (long-run same-tier data-only step 2)."""
from __future__ import annotations

import os

import pytest


def test_overnight_preset_overrides_after_pycharm(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", "overnight_step2")
    for key in (
        "BIOCHEM_TEACHER_EPOCHS",
        "BIOCHEM_TEACHER_TF_WARMUP_EPOCHS",
        "BIOCHEM_DEBUG",
        "BIOCHEM_DATA_ONLY_PHYS_TEMP",
        "BIOCHEM_TBPTT_MAX_WINDOW",
        "BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS",
    ):
        monkeypatch.delenv(key, raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert int(os.environ["BIOCHEM_TEACHER_EPOCHS"]) == 60
    assert int(os.environ["BIOCHEM_TEACHER_TF_WARMUP_EPOCHS"]) == 60
    assert os.environ["BIOCHEM_DEBUG"].strip().lower() in ("1", "true", "yes", "on")
    assert os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"].strip().lower() in ("1", "true", "yes", "on")
    assert int(os.environ["BIOCHEM_TBPTT_MAX_WINDOW"]) == 14
    assert int(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_EARLY_EPOCHS"]) == 16


@pytest.mark.parametrize("alias", ("comprehensive_step2", "step2_overnight"))
def test_overnight_preset_aliases(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", alias)
    monkeypatch.delenv("BIOCHEM_TEACHER_EPOCHS", raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert int(os.environ["BIOCHEM_TEACHER_EPOCHS"]) == 60


def test_overnight_preset_skipped_when_stock_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.setenv("BIOCHEM_STOCK_DEFAULTS", "1")
    monkeypatch.setenv("BIOCHEM_PRESET", "overnight_step2")
    monkeypatch.delenv("BIOCHEM_TEACHER_EPOCHS", raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert "BIOCHEM_TEACHER_EPOCHS" not in os.environ
