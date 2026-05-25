"""Viz health metrics and effective gelation terms (K0/S0 ablation alignment)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import torch

from src.architecture import gnode_biochem as gb
from src.config import STATE_CHANNEL_MU_EFF_ND


def _mock_biochem_model():
    model = MagicMock()
    model.mu1_sigmoid = lambda x: torch.ones_like(x) * 2.0
    model.mu2_sigmoid = lambda x: torch.ones_like(x) * 80.0
    model.species_log_nd_to_si = lambda sp: sp
    return model


def test_explicit_gelation_terms_zero_when_disable_explicit_gelation(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", "1")
    model = _mock_biochem_model()
    fi = torch.tensor([[100.0]])
    mat = torch.tensor([[100.0]])
    mu1, mu2 = gb.biochem_explicit_gelation_terms(model, fi, mat)
    assert float(mu1.sum()) == 0.0
    assert float(mu2.sum()) == 0.0


def test_explicit_gelation_respects_disable_mu2(monkeypatch):
    monkeypatch.delenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", raising=False)
    monkeypatch.delenv("BIOCHEM_MU_SIMPLE_LOG_RESIDUAL", raising=False)
    monkeypatch.setenv("BIOCHEM_MU_DISABLE_MU2", "1")
    model = _mock_biochem_model()
    fi = torch.tensor([[50.0]])
    mat = torch.tensor([[1.0]])
    mu1, mu2 = gb.biochem_explicit_gelation_terms(model, fi, mat)
    assert float(mu1.sum()) == pytest.approx(2.0)
    assert float(mu2.sum()) == 0.0


def test_slice_viz_health_mu2_zero_under_disable_explicit_gelation(monkeypatch):
    from src.training.train_biochem_corrector import _compute_slice_viz_health_metrics

    monkeypatch.setenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", "1")
    model = _mock_biochem_model()
    phys = MagicMock()
    phys.mu_inf = 0.001
    phys.viscosity_nd_to_si = lambda x: x * 0.001
    kernels = MagicMock()
    kernels.core.cfg = phys

    n = 4
    pred = torch.zeros(n, 16)
    pred[:, 0] = 0.5
    pred[:, 1] = 0.1
    pred[:, STATE_CHANNEL_MU_EFF_ND] = 1.0
    pred[:, 12] = 8.0
    y = pred.clone()
    data = MagicMock()
    data.num_nodes = n

    monkeypatch.setattr(
        "src.training.train_biochem_corrector.biochem_truth_node_mask",
        lambda *a, **k: torch.zeros(n, dtype=torch.bool),
    )
    out = _compute_slice_viz_health_metrics(
        pred, y, data, model, kernels, torch.device("cpu")
    )
    assert out["mu2_mean"] == pytest.approx(0.0)
    assert out["mu1_mean"] == pytest.approx(0.0)
    assert out["clot_frac"] == pytest.approx(0.0)


def test_clot_frac_uses_rollout_mu_eff_when_explicit_gelation_on(monkeypatch):
    from src.training.train_biochem_corrector import _compute_slice_viz_health_metrics

    monkeypatch.delenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", raising=False)
    monkeypatch.delenv("BIOCHEM_MU_SIMPLE_LOG_RESIDUAL", raising=False)
    monkeypatch.delenv("BIOCHEM_VIZ_CLOT_FRAC_USE_MU2", raising=False)
    monkeypatch.setenv("BIOCHEM_TEACHER_MU_RATIO_MAX", "80.0")
    model = _mock_biochem_model()
    phys = MagicMock()
    phys.mu_inf = 0.001
    phys.viscosity_nd_to_si = lambda x: x * phys.mu_inf
    kernels = MagicMock()
    kernels.core.cfg = phys

    n = 100
    pred = torch.zeros(n, 16)
    pred[:, STATE_CHANNEL_MU_EFF_ND] = 40.0
    pred[0, STATE_CHANNEL_MU_EFF_ND] = 120.0
    pred[:, 12] = 8.0
    y = pred.clone()
    data = MagicMock()
    data.num_nodes = n

    monkeypatch.setattr(
        "src.training.train_biochem_corrector.biochem_truth_node_mask",
        lambda *a, **k: torch.zeros(n, dtype=torch.bool),
    )
    out = _compute_slice_viz_health_metrics(
        pred, y, data, model, kernels, torch.device("cpu")
    )
    assert out["mu2_mean"] == pytest.approx(80.0)
    assert out["clot_frac"] == pytest.approx(0.01)


def test_clot_frac_legacy_mu2_when_env_set(monkeypatch):
    from src.training.train_biochem_corrector import _compute_slice_viz_health_metrics

    monkeypatch.delenv("BIOCHEM_MU_DISABLE_EXPLICIT_GELATION", raising=False)
    monkeypatch.setenv("BIOCHEM_VIZ_CLOT_FRAC_USE_MU2", "1")
    model = _mock_biochem_model()
    phys = MagicMock()
    phys.mu_inf = 0.001
    phys.viscosity_nd_to_si = lambda x: x * phys.mu_inf
    kernels = MagicMock()
    kernels.core.cfg = phys

    n = 4
    pred = torch.zeros(n, 16)
    pred[:, STATE_CHANNEL_MU_EFF_ND] = 40.0
    pred[:, 12] = 8.0
    y = pred.clone()
    data = MagicMock()
    data.num_nodes = n

    monkeypatch.setattr(
        "src.training.train_biochem_corrector.biochem_truth_node_mask",
        lambda *a, **k: torch.zeros(n, dtype=torch.bool),
    )
    out = _compute_slice_viz_health_metrics(
        pred, y, data, model, kernels, torch.device("cpu")
    )
    assert out["clot_frac"] == pytest.approx(1.0)


def test_mu_ic_steady_kin_env_flag(monkeypatch):
    monkeypatch.delenv("BIOCHEM_MU_IC_STEADY_KIN", raising=False)
    assert not gb._biochem_mu_ic_steady_kin_enabled()
    monkeypatch.setenv("BIOCHEM_MU_IC_STEADY_KIN", "1")
    assert gb._biochem_mu_ic_steady_kin_enabled()


def test_rollout_mu_eff_si_numpy():
    from src.evaluation.visualize_pipeline import _rollout_mu_eff_si_numpy

    phys = MagicMock()
    phys.viscosity_nd_to_si = lambda x: (x * 2.0).squeeze(-1)
    pred = __import__("numpy").zeros((3, 16))
    pred[:, STATE_CHANNEL_MU_EFF_ND] = [1.0, 2.0, 3.0]
    out = _rollout_mu_eff_si_numpy(phys, pred)
    assert list(out) == pytest.approx([2.0, 4.0, 6.0])


def test_viz_mu_si_clim_fixed_comsol_defaults(monkeypatch):
    import numpy as np

    from src.evaluation.visualize_pipeline import _viz_mu_si_clim

    monkeypatch.delenv("VIZ_MU_CLIM", raising=False)
    monkeypatch.delenv("VIZ_MU_VMIN", raising=False)
    monkeypatch.delenv("VIZ_MU_VMAX", raising=False)
    arr = np.array([0.0, 5.0])
    assert _viz_mu_si_clim(arr) == pytest.approx((0.04, 0.10))


def test_viz_mu_si_clim_auto_from_data(monkeypatch):
    import numpy as np

    from src.evaluation.visualize_pipeline import _viz_mu_si_clim

    monkeypatch.setenv("VIZ_MU_CLIM", "auto")
    a = np.array([1.0, 3.0])
    b = np.array([0.5, 4.0])
    assert _viz_mu_si_clim(a, b) == pytest.approx((0.5, 4.0))
