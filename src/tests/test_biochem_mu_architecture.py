"""Architecture defaults for COMSOL μ (late TBPTT, rheology cap, loss scale)."""
from __future__ import annotations

import os

import pytest
import torch


def test_mu_best_practice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_mu_best_practice_env

    for key in (
        "BIOCHEM_TEACHER_MU_RATIO_MAX",
        "BIOCHEM_MU_SI_HUBER_DELTA",
        "BIOCHEM_TBPTT_ANCHOR_END_BIAS",
        "BIOCHEM_MU_LOG_ANCHOR_WEIGHT",
    ):
        monkeypatch.delenv(key, raising=False)

    _apply_biochem_mu_best_practice_env(only_if_missing=True)

    assert float(os.environ["BIOCHEM_TEACHER_MU_RATIO_MAX"]) == pytest.approx(80.0)
    assert float(os.environ["BIOCHEM_MU_SI_HUBER_DELTA"]) == pytest.approx(0.25)
    assert os.environ["BIOCHEM_TBPTT_ANCHOR_END_BIAS"].strip().lower() in ("1", "true", "yes", "on")
    assert float(os.environ["BIOCHEM_MU_LOG_ANCHOR_WEIGHT"]) == pytest.approx(2.0)


def test_tbptt_end_bias_start_idx() -> None:
    from src.training.train_biochem_corrector import _resolve_tbptt_anchor_start_idx

    os.environ["BIOCHEM_TBPTT_ANCHOR_END_BIAS"] = "1"
    os.environ["BIOCHEM_TBPTT_ANCHOR_RANDOM_START"] = "0"
    device = torch.device("cpu")
    assert _resolve_tbptt_anchor_start_idx(60, 8, device) == 52
    os.environ.pop("BIOCHEM_TBPTT_ANCHOR_END_BIAS", None)


def test_comprehensive_mu_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_pycharm_biochem_optimal_defaults

    monkeypatch.delenv("BIOCHEM_STOCK_DEFAULTS", raising=False)
    monkeypatch.setenv("BIOCHEM_PRESET", "comprehensive_mu")
    for key in ("BIOCHEM_TEACHER_EPOCHS", "BIOCHEM_TEACHER_MU_RATIO_MAX", "BIOCHEM_STOP_AFTER_TEACHER"):
        monkeypatch.delenv(key, raising=False)

    _apply_pycharm_biochem_optimal_defaults()

    assert int(os.environ["BIOCHEM_TEACHER_EPOCHS"]) == 36
    assert float(os.environ["BIOCHEM_TEACHER_MU_RATIO_MAX"]) == pytest.approx(80.0)
    assert os.environ["BIOCHEM_STOP_AFTER_TEACHER"].strip() == "0"
    assert float(os.environ["BIOCHEM_TEACHER_FORCE_MIN"]) == pytest.approx(0.2)
