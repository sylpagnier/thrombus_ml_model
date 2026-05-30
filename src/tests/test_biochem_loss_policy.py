"""Loss policy gates for frozen-kin viscosity teacher path."""
from __future__ import annotations

import os

import pytest

from src.training.biochem_loss_policy import (
    biochem_legacy_losses_enabled,
    check_deprecated_preset,
    validate_isolate_key,
)


def test_approved_isolate_allowed() -> None:
    validate_isolate_key("MU_LOG")
    validate_isolate_key("DATA_BIO")
    validate_isolate_key("PASSIVE")


def test_deprecated_isolate_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOCHEM_LEGACY_LOSSES", raising=False)
    with pytest.raises(ValueError, match="MU_LOG_WALL"):
        validate_isolate_key("MU_LOG_WALL")


def test_deprecated_isolate_allowed_with_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOCHEM_LEGACY_LOSSES", "1")
    validate_isolate_key("MU_LOG_WALL")


def test_deprecated_preset_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIOCHEM_LEGACY_LOSSES", raising=False)
    assert check_deprecated_preset("sweep_wall_overcomp") is False


def test_deprecated_preset_allowed_with_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOCHEM_LEGACY_LOSSES", "1")
    assert check_deprecated_preset("thrombus_corona") is True
    assert biochem_legacy_losses_enabled()
