"""Kinematics ``model_config`` snapshot / resolve round-trip."""
from __future__ import annotations

import json

from src.architecture.ginodeq import GINO_DEQ
from src.architecture.kinematics_model_config import (
    KINEMATICS_MODEL_CONFIG_SCHEMA,
    build_gino_deq_from_ctor,
    infer_wss_fuse_from_state_dict,
    kinematics_reference_path,
    load_kinematics_reference_record,
    resolve_gino_deq_ctor_kwargs,
    snapshot_gino_deq_model_config,
)
from src.config import PhysicsConfig


def test_snapshot_matches_training_defaults():
    phys = PhysicsConfig(phase="kinematics")
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=256,
        max_iters=25,
        num_fourier_freqs=16,
        phys_cfg=phys,
        use_hard_bcs=True,
        use_siren_decoder=True,
        use_width_priors=True,
    )
    cfg = snapshot_gino_deq_model_config(model)
    assert cfg["schema"] == KINEMATICS_MODEL_CONFIG_SCHEMA
    assert cfg["latent_dim"] == 256
    assert cfg["num_fourier_freqs"] == 16
    assert cfg["use_siren_decoder"] is True


def test_reference_json_loads_and_resolves():
    ref = load_kinematics_reference_record()
    assert ref is not None, f"missing {kinematics_reference_path()}"
    ctor = resolve_gino_deq_ctor_kwargs(ref, {})
    assert ctor["latent_dim"] == 256
    assert ctor["num_fourier_freqs"] == 16
    mc = ref["model_config"]
    assert mc["use_hard_bcs"] is True


def test_infer_wss_fuse_from_wss_decoder_input_width():
    import torch

    fused = {"wss_decoder.0.linear.parametrizations.weight.original": torch.zeros(256, 260)}
    assert infer_wss_fuse_from_state_dict(fused, latent_dim=256) is True
    legacy = {"wss_decoder.0.linear.parametrizations.weight.original": torch.zeros(256, 256)}
    assert infer_wss_fuse_from_state_dict(legacy, latent_dim=256) is False


def test_resolve_wss_fuse_from_state_dict_when_not_in_model_config():
    import torch

    meta = {
        "model_config": {
            "schema": KINEMATICS_MODEL_CONFIG_SCHEMA,
            "latent_dim": 256,
            "num_fourier_freqs": 16,
            "use_siren_decoder": True,
            "use_width_priors": True,
            "use_hard_bcs": True,
        }
    }
    state = {"wss_decoder.0.linear.parametrizations.weight.original": torch.zeros(256, 260)}
    ctor = resolve_gino_deq_ctor_kwargs(meta, state)
    assert ctor["wss_fuse"] is True
    phys = PhysicsConfig(phase="kinematics")
    model = build_gino_deq_from_ctor(phys, ctor)
    assert model.wss_fuse is True


def test_reference_json_is_valid_document():
    path = kinematics_reference_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["source_run_id"] == "20260426T184600Z"
    assert raw["best_checkpoint"]["epoch"] == 84
