"""Complexity step 3 (full multitask) and compact step-2.5 (phys-temp) preset tests."""
from __future__ import annotations

import os

import pytest


def test_complexity_step3_forces_multitask_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_complexity_step3_env

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "phase3")
    monkeypatch.setenv("BIOCHEM_LOSS_DATA_ONLY", "1")
    monkeypatch.setenv("BIOCHEM_DATA_ONLY_PHYS_TEMP", "1")

    _apply_biochem_complexity_step3_env()

    assert os.environ["BIOCHEM_LOSS_DATA_ONLY"].strip().lower() in ("0", "false", "no", "off")
    assert os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"].strip().lower() in ("0", "false", "no", "off")


def test_complexity_step3_skipped_when_stock_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_complexity_step3_env

    monkeypatch.setenv("BIOCHEM_STOCK_DEFAULTS", "1")
    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "3")
    monkeypatch.setenv("BIOCHEM_LOSS_DATA_ONLY", "1")

    _apply_biochem_complexity_step3_env()

    assert os.environ["BIOCHEM_LOSS_DATA_ONLY"] == "1"


@pytest.mark.parametrize("alias", ("step2p5", "phys_temp_only", "compact_step2p5"))
def test_step2p5_preset_enables_phys_temp(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", alias)
    monkeypatch.delenv("BIOCHEM_DATA_ONLY_PHYS_TEMP", raising=False)
    monkeypatch.delenv("BIOCHEM_COMSOL_TEMPORAL_WEIGHT", raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"].strip().lower() in ("1", "true", "yes", "on")
    assert os.environ["BIOCHEM_LOSS_DATA_ONLY"].strip().lower() in ("1", "true", "yes", "on")
    assert float(os.environ["BIOCHEM_COMSOL_TEMPORAL_WEIGHT"]) == pytest.approx(0.02)


def test_step3_after_pycharm_when_complexity_step3(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.delenv("BIOCHEM_PRESET", raising=False)
    monkeypatch.setenv("BIOCHEM_COMPLEXITY_STEP", "3")
    monkeypatch.delenv("BIOCHEM_LOSS_DATA_ONLY", raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert os.environ["BIOCHEM_LOSS_DATA_ONLY"].strip().lower() in ("0", "false", "no", "off")


def test_thrombus_corona_preset_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", "thrombus_corona")
    monkeypatch.delenv("BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS", raising=False)
    monkeypatch.delenv("BIOCHEM_GELATION_PRIOR_GATE", raising=False)
    monkeypatch.delenv("BIOCHEM_PRIOR_WALL_DECAY_ND", raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert int(os.environ["BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS"]) == 3
    assert os.environ["BIOCHEM_GELATION_PRIOR_GATE"].strip().lower() in ("1", "true", "yes", "on")
    assert os.environ["BIOCHEM_DATA_ONLY_PHYS_TEMP"].strip().lower() in ("1", "true", "yes", "on")
    assert os.environ["BIOCHEM_STOP_AFTER_TEACHER"].strip().lower() in ("0", "false", "no", "off")
    assert float(os.environ["BIOCHEM_PRIOR_WALL_DECAY_ND"]) == pytest.approx(0.01)
