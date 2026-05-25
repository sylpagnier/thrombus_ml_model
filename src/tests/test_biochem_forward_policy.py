"""Checkpoint forward_policy snapshot/restore for biochem μ rollout."""

from __future__ import annotations

import os

from src.architecture import gnode_biochem as gb


def test_snapshot_forward_policy_k10a_shape(monkeypatch):
    monkeypatch.setenv("BIOCHEM_USE_DELTA_MU_HEAD", "1")
    monkeypatch.setenv("BIOCHEM_USE_SPLIT_MU_HEAD", "0")
    monkeypatch.setenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", "1")
    monkeypatch.setenv("BIOCHEM_MU_IC_STEADY_KIN", "1")
    monkeypatch.delenv("BIOCHEM_MU_ADDITIVE_DELTA", raising=False)
    fp = gb.snapshot_biochem_forward_policy()
    assert int(fp["schema"]) == 1
    assert fp["use_delta_mu_head"] is True
    assert fp["use_split_mu_head"] is False
    assert fp["mu_ic_steady_kin"] is True
    assert fp["mu_disable_explicit_gelation"] is True
    assert fp["mu_additive_delta"] is False


def test_apply_forward_policy_roundtrip(monkeypatch):
    policy = {
        "schema": 1,
        "use_delta_mu_head": True,
        "use_split_mu_head": True,
        "use_wall_delta_head": False,
        "mu_disable_explicit_gelation": True,
        "mu_simple_log_residual": False,
        "mu_ic_steady_kin": True,
        "mu_additive_delta": True,
        "mu_gemini_fix": False,
        "gelation_prior_gate": True,
        "mu_disable_mu1": False,
        "mu_disable_mu2": False,
        "delta_mu_symmetric_bulk_clip": False,
        "use_clot_nucleation_growth": False,
        "use_bio_gate_suppressor": False,
        "delta_mu_log_clip_bulk": "0.05",
    }
    for key in (
        "BIOCHEM_USE_DELTA_MU_HEAD",
        "BIOCHEM_USE_SPLIT_MU_HEAD",
        "BIOCHEM_MU_IC_STEADY_KIN",
        "BIOCHEM_MU_ADDITIVE_DELTA",
        "BIOCHEM_MU_DISABLE_EXPLICIT_GELATION",
    ):
        monkeypatch.delenv(key, raising=False)
    gb.apply_biochem_forward_policy(policy, quiet=True)
    assert gb._biochem_delta_mu_head_enabled()
    assert gb._biochem_split_mu_regime_head_enabled()
    assert gb._biochem_mu_ic_steady_kin_enabled()
    assert gb._biochem_mu_additive_delta_enabled()
    assert gb._biochem_mu_disable_explicit_gelation()
    assert os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP_BULK") == "0.05"


def test_forward_policy_from_nested_model_config():
    meta = {
        "model_config": {
            "schema": 1,
            "latent_dim": 256,
            "forward_policy": {"schema": 1, "mu_ic_steady_kin": True},
        }
    }
    fp = gb.biochem_forward_policy_from_checkpoint_meta(meta)
    assert fp is not None
    assert fp["mu_ic_steady_kin"] is True


def test_k10d_simple_env(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_K10D_SIMPLE", "1")
    assert gb._biochem_mu_k10d_simple_enabled()


def test_model_config_snapshot_includes_forward_policy(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("BIOCHEM_MU_IC_STEADY_KIN", "1")
    model = MagicMock()
    model.latent_dim = 128
    model.max_inner_iters = 10
    model.bio_encoder_prior_dim = 2
    model.num_fourier_freqs = 16
    model.use_siren_decoder = False
    model.gnode_layers = 1
    model.use_hard_bcs = True
    cfg = gb.snapshot_biochem_model_config(model)
    assert "forward_policy" in cfg
    assert cfg["forward_policy"]["mu_ic_steady_kin"] is True
