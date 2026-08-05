"""Unit tests for offwall growth compound-val static helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.train_offwall_growth import _band_static_to_device


def test_band_static_to_device_moves_tensors_only():
    cpu = torch.device("cpu")
    src = {
        "base_feats": torch.zeros(4, 3),
        "node_idx": torch.arange(4),
        "n_band": 4,
        "flow_cols": (1, 5),
    }
    out = _band_static_to_device(src, cpu)
    assert out["base_feats"].device.type == "cpu"
    assert out["n_band"] == 4
    assert out["flow_cols"] == (1, 5)
    assert int(out["node_idx"].numel()) == 4


def test_train_feat_source_band_is_default():
    import argparse
    from src.training import train_offwall_growth as mod

    # Ensure CLI default stays on band (eval-matched features).
    ap = argparse.ArgumentParser()
    # Re-parse only the relevant flag by invoking main's parser construction is heavy;
    # assert the default string embedded in help/choices path via module source contract.
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert '--train-feat-source"' in src or "--train-feat-source" in src
    assert 'default="band"' in src


def test_compound_val_requires_band_static_key():
    """Regression: full-graph static silently zeros patient001 clot F1."""
    import pytest
    from src.training.train_offwall_growth import eval_wall_only_deploy_floor

    with pytest.raises(RuntimeError, match="band_static"):
        # Missing band_static must fail loud (do not fall back to full graph).
        eval_wall_only_deploy_floor(
            wall_ckpt=__import__("pathlib").Path("missing.pth"),
            val_pack={"data": None, "base_feats_global": torch.zeros(1, 1)},
            phys=None,
            bio=None,
            device=torch.device("cpu"),
        )
