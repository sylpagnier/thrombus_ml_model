"""Regression tests for Tier 1 / Tier 2 predictor training helpers."""

from types import SimpleNamespace

import torch
import torch.nn as nn

import src.training.train_t1_predictor as t1_mod
import src.training.train_t2_predictor as t2_mod
from src.utils.metrics import DynamicLossWeighter


class _GraphStub:
    def __init__(self, is_anchor: bool):
        self.is_anchor = torch.tensor([is_anchor], dtype=torch.bool)


class _DummyModel:
    def __init__(self, pred, jac):
        self._pred = pred
        self._jac = jac

    def __call__(self, *_args, **_kwargs):
        return self._pred, self._jac


class _DummyKernels:
    def _get_geometric_props(self, data):
        return {"n": data.y.shape[0]}

    def _compute_derivatives(self, field, props):
        _ = field
        n = int(props["n"])
        return torch.zeros((n, 2, 1), dtype=torch.float32)


class _SimpleWeighter:
    def __call__(self, losses, scales=None):
        scales = scales or [1.0] * len(losses)
        out = torch.tensor(0.0, dtype=torch.float32)
        for loss, scale in zip(losses, scales):
            out = out + (loss * float(scale))
        return out


def test_assert_tier2_train_split_validation():
    ok_train = [_GraphStub(True), _GraphStub(False)]
    ok_val = [_GraphStub(True)]
    t2_mod._assert_tier2_train_split(ok_train, ok_val)

    try:
        t2_mod._assert_tier2_train_split([], ok_val)
        assert False, "expected empty train split to fail"
    except ValueError as e:
        assert "train_data is empty" in str(e)

    try:
        t2_mod._assert_tier2_train_split([_GraphStub(True)], [_GraphStub(True)])
        assert False, "expected no physics-only split to fail"
    except ValueError as e:
        assert "no physics-only graphs" in str(e)

    try:
        t2_mod._assert_tier2_train_split([_GraphStub(False)], [_GraphStub(True)])
        assert False, "expected no anchor split to fail"
    except ValueError as e:
        assert "no anchor" in str(e)


def test_tier2_dynamic_loss_weighter_precision_floor():
    floor = 0.8
    lw = t2_mod._tier2_dynamic_loss_weighter(device="cpu", mom_precision_floor=floor)
    expected_max_lv = -torch.log(torch.tensor(floor))
    assert torch.allclose(lw.per_task_max_log_var, expected_max_lv.view(1), atol=1e-6)


def test_compute_step_loss_t2_distillation_and_coupled(monkeypatch):
    def _fake_terms(*_args, **_kwargs):
        return {
            "l_wss": torch.tensor(0.6),
            "l_data_kine": torch.tensor(0.1),
            "l_data_mu": torch.tensor(0.2),
            "l_mom": torch.tensor(0.5),
            "l_cont": torch.tensor(0.25),
            "l_bc": torch.tensor(0.3),
            "l_io": torch.tensor(0.4),
            "l_rheo": torch.tensor(0.7),
        }

    monkeypatch.setattr(t2_mod, "compute_kinematics_physics_terms", _fake_terms)

    n = 4
    model = _DummyModel(pred=torch.zeros((n, 5), dtype=torch.float32), jac=torch.tensor(0.5))
    data = SimpleNamespace(y=torch.zeros((n, 5), dtype=torch.float32))

    loss_d, metrics_d = t2_mod.compute_step_loss(
        model=model,
        data=data,
        kernels=_DummyKernels(),
        loss_weighter=_SimpleWeighter(),
        current_solver="picard",
        lambda_phys=0.25,
        device="cpu",
        is_distillation=True,
        carreau_n=0.7,
    )
    assert abs(float(loss_d.item()) - 17.55) < 1e-6
    assert metrics_d["L_mom"] == 0.0

    loss_c, metrics_c = t2_mod.compute_step_loss(
        model=model,
        data=data,
        kernels=_DummyKernels(),
        loss_weighter=_SimpleWeighter(),
        current_solver="anderson",
        lambda_phys=0.25,
        device="cpu",
        is_distillation=False,
        carreau_n=0.7,
        tier2_kine_p_weight=1.35,
        coupled_io_scale=6.0,
    )
    assert abs(float(loss_c.item()) - 70.775) < 1e-5
    assert abs(metrics_c["L_mom"] - 0.5) < 1e-6


def test_compute_step_loss_t1_no_anchor_pgrad_is_zero(monkeypatch):
    def _fake_terms(*_args, **_kwargs):
        return {
            "l_wss": torch.tensor(0.5),
            "l_data_kine": torch.tensor(0.2),
            "l_mom": torch.tensor(0.4),
            "l_cont": torch.tensor(0.3),
            "l_bc": torch.tensor(0.1),
            "l_io": torch.tensor(0.2),
        }

    monkeypatch.setattr(t1_mod, "compute_kinematics_physics_terms", _fake_terms)
    monkeypatch.setattr(
        t1_mod,
        "anchor_node_mask",
        lambda data: torch.zeros(data.y.shape[0], dtype=torch.bool),
    )

    n = 3
    pred = torch.zeros((n, 5), dtype=torch.float32)
    jac = torch.tensor(0.4)
    model = _DummyModel(pred=pred, jac=jac)
    data = SimpleNamespace(
        y=torch.zeros((n, 5), dtype=torch.float32),
        x=torch.zeros((n, 2), dtype=torch.float32),
    )
    # Dynamic PDE weighting with log_var=0 matches fixed-sum momentum+continuity when λ_phys is shared.
    lw = DynamicLossWeighter(num_losses=2)

    loss, metrics = t1_mod.compute_step_loss(
        model=model,
        data=data,
        kernels=_DummyKernels(),
        loss_weighter=lw,
        current_solver="anderson",
        lambda_phys=0.5,
        device="cpu",
    )

    assert abs(float(loss.item()) - 107.39) < 1e-6
    assert metrics["L_pgrad"] == 0.0
    assert metrics["A_nodes"] == 0


def test_load_tier1_bootstrap_adapts_encoder_width(tmp_path):
    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(5, 4, bias=False))

    ckpt_path = tmp_path / "tier1_best_physics.pth"
    state = {"encoder.0.weight": torch.ones((4, 3), dtype=torch.float32)}
    torch.save(state, ckpt_path)

    model = _TinyModel()
    ok = t2_mod._load_tier1_bootstrap(model, ckpt_path, device="cpu")
    assert ok is True
    w = model.encoder[0].weight.detach()
    assert torch.allclose(w[:, :3], torch.ones((4, 3), dtype=torch.float32), atol=1e-6)
    assert torch.allclose(w[:, 3:], torch.zeros((4, 2), dtype=torch.float32), atol=1e-6)
