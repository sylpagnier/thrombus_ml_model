"""Supervised data leash overrides after MU-isolate presets."""
from __future__ import annotations

import os

import pytest


def test_supervised_leash_clears_isolate_after_sentinel_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import (
        _apply_biochem_preset_sweep_wall_sentinel_if_requested,
        _apply_biochem_supervised_data_leash_after_presets,
    )

    monkeypatch.setenv("BIOCHEM_PRESET", "sweep_wall_sentinel")
    monkeypatch.setenv("BIOCHEM_SUPERVISED_DATA_LEASH", "1")
    monkeypatch.delenv("BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT", raising=False)

    _apply_biochem_preset_sweep_wall_sentinel_if_requested()
    assert os.environ.get("BIOCHEM_LOSS_ISOLATE") == "MU_LOG"
    assert os.environ.get("BIOCHEM_DETACH_MACRO_STATE") == "1"

    _apply_biochem_supervised_data_leash_after_presets()

    assert "BIOCHEM_LOSS_ISOLATE" not in os.environ
    assert os.environ["BIOCHEM_LOSS_DATA_ONLY"] == "1"
    assert os.environ["BIOCHEM_DETACH_MACRO_STATE"] == "0"
    assert float(os.environ["BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT"]) == pytest.approx(2.0)


def test_bulk_fluid_surgical_fix_after_sentinel_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import (
        _apply_biochem_bulk_fluid_surgical_fix_after_presets,
        _apply_biochem_preset_sweep_wall_sentinel_if_requested,
    )

    monkeypatch.setenv("BIOCHEM_PRESET", "sweep_wall_sentinel")
    monkeypatch.setenv("BIOCHEM_BULK_FLUID_SURGICAL_FIX", "1")
    monkeypatch.setenv("BIOCHEM_DELTA_MU_LOG_CLIP_BULK", "0.05")
    monkeypatch.setenv("BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR", "0.0")

    _apply_biochem_preset_sweep_wall_sentinel_if_requested()
    assert os.environ.get("BIOCHEM_USE_BIO_GATE_SUPPRESSOR") == "0"

    _apply_biochem_bulk_fluid_surgical_fix_after_presets()

    assert os.environ["BIOCHEM_USE_BIO_GATE_SUPPRESSOR"] == "1"
    assert float(os.environ["BIOCHEM_DELTA_MU_LOG_CLIP_BULK"]) == pytest.approx(0.05)
    assert float(os.environ["BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR"]) == pytest.approx(0.0)


def test_mu_gate_hard_threshold_after_sentinel_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import (
        _apply_biochem_mu_gate_hard_threshold_after_presets,
        _apply_biochem_preset_sweep_wall_sentinel_if_requested,
    )

    monkeypatch.setenv("BIOCHEM_PRESET", "sweep_wall_sentinel")
    monkeypatch.setenv("BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH", "0.15")

    _apply_biochem_preset_sweep_wall_sentinel_if_requested()
    assert float(os.environ.get("BIOCHEM_TRIGGER_GATE_MIN", "1")) == pytest.approx(0.0)
    assert float(os.environ.get("BIOCHEM_WALL_GATE_MIN", "1")) == pytest.approx(0.0)

    _apply_biochem_mu_gate_hard_threshold_after_presets()
    assert os.environ["BIOCHEM_TRIGGER_GATE_MIN"] == "0"
    assert os.environ["BIOCHEM_WALL_GATE_MIN"] == "0"
    assert os.environ["BIOCHEM_MU_SOFT_GATE_SCOPE"] == "wall_only"
    assert float(os.environ["BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH"]) == pytest.approx(0.15)


def test_mu_soft_gate_scope_wall_only_skips_bulk_clot_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.architecture.gnode_biochem import _apply_mu_gate_soft_threshold, _mu_soft_gate_scope

    monkeypatch.setenv("BIOCHEM_MU_SOFT_GATE_SCOPE", "wall_only")
    assert _mu_soft_gate_scope() == "wall_only"
    monkeypatch.setenv("BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH", "0.15")
    monkeypatch.setenv("BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS", "20.0")
    g_high = torch.tensor([[0.90]], dtype=torch.float32)
    # wall_only: bulk gate would stay 0.90; soft only when caller applies it
    assert float(_apply_mu_gate_soft_threshold(g_high).item()) == pytest.approx(0.90 * 0.9999, rel=0.01)


def test_mu_gate_soft_threshold_preserves_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from src.architecture.gnode_biochem import _apply_mu_gate_soft_threshold

    monkeypatch.setenv("BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH", "0.15")
    monkeypatch.setenv("BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS", "20.0")
    g = torch.tensor([[0.10]], dtype=torch.float32, requires_grad=True)
    out = _apply_mu_gate_soft_threshold(g)
    out.sum().backward()
    assert g.grad is not None
    assert float(g.grad.abs().sum()) > 0.0
    assert float(out.item()) < 0.05


def test_supervised_leash_noop_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.train_biochem_corrector import _apply_biochem_supervised_data_leash_after_presets

    monkeypatch.delenv("BIOCHEM_SUPERVISED_DATA_LEASH", raising=False)
    monkeypatch.setenv("BIOCHEM_LOSS_ISOLATE", "MU_LOG")
    _apply_biochem_supervised_data_leash_after_presets()
    assert os.environ.get("BIOCHEM_LOSS_ISOLATE") == "MU_LOG"
