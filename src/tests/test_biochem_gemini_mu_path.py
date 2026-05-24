"""Tests for Gemini/additive μ residual path env toggles."""

from __future__ import annotations

import os

import pytest
import torch

from src.architecture import gnode_biochem as gb


def test_gemini_additive_and_symmetric_bulk_clip(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_GEMINI_FIX", "1")
    monkeypatch.setenv("BIOCHEM_DELTA_MU_LOG_CLIP_BULK", "0.05")
    assert gb._biochem_mu_gemini_fix_enabled()
    assert gb._biochem_mu_additive_delta_enabled()
    assert gb._biochem_delta_mu_symmetric_bulk_clip()

    bulk_raw = torch.tensor([[-5.0], [0.02], [3.0]])
    bulk_clip = 0.05
    bulk = torch.clamp(bulk_raw, min=-bulk_clip, max=bulk_clip)
    assert float(bulk.min()) == pytest.approx(-bulk_clip, abs=1e-6)
    assert float(bulk.max()) == pytest.approx(bulk_clip, abs=1e-6)

    gate = torch.tensor([[0.0], [1.0], [0.5]])
    tail = torch.tensor([[1.0], [2.0], [0.5]])
    mixed = bulk + (gate * tail)
    legacy = ((1.0 - gate) * bulk) + (gate * tail)
    assert not torch.allclose(mixed, legacy)
    assert float(mixed[0].item()) == pytest.approx(float(bulk[0].item()))


def test_simple_log_residual_disables_explicit_gelation(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_SIMPLE_LOG_RESIDUAL", "1")
    assert gb._biochem_mu_simple_log_residual_enabled()
    assert gb._biochem_mu_disable_explicit_gelation()


def test_carreau_only_disables_gelation(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_CARREAU_ONLY", "1")
    assert gb._biochem_mu_carreau_only_enabled()
    assert gb._biochem_mu_disable_explicit_gelation()


def test_viz_health_score_penalizes_mu2_flood():
    from src.training.train_biochem_corrector import _viz_health_score

    healthy = _viz_health_score(
        {
            "t0_speed_mean": 0.9,
            "final_mu2_mean": 2.0,
            "final_clot_frac": 0.01,
            "final_gate_mean_all": 0.4,
            "final_mu1_mean": 0.2,
        },
        mu_log_mae_final=0.5,
    )
    flooded = _viz_health_score(
        {
            "t0_speed_mean": 0.1,
            "final_mu2_mean": 60.0,
            "final_clot_frac": 0.95,
            "final_gate_mean_all": 0.9,
            "final_mu1_mean": 0.0,
        },
        mu_log_mae_final=0.5,
    )
    assert flooded > healthy


def test_archive_checkpoint_dir_writes(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import src.training.train_biochem_corrector as train_mod

    model_dir = tmp_path / "global"
    model_dir.mkdir()
    archive = tmp_path / "leg_G0"
    archive.mkdir()
    teacher = MagicMock()
    teacher.state_dict.return_value = {"w": torch.ones(1)}

    monkeypatch.setenv("BIOCHEM_ARCHIVE_CHECKPOINT_DIR", str(archive))
    monkeypatch.setattr(train_mod, "_biochem_env_truthy", lambda _k, default=True: default)
    train_mod._persist_biochem_teacher_checkpoints(
        model_dir,
        teacher,
        teacher_best_mu_score=-0.4,
        best_epoch=3,
        run_note="health10h_G0",
        val_mu_log_mae_high_mu=0.42,
        best_high_epoch=3,
        best_high_state={"w": torch.ones(1)},
    )
    assert (archive / train_mod.BIOCHEM_TEACHER_BEST_HIGH_MU_CKPT_NAME).is_file()
    assert (archive / train_mod.BIOCHEM_TEACHER_LAST_CKPT_NAME).is_file()
