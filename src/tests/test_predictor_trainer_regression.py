"""Regression tests for unified kinematics predictor training helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

import src.training.train_kinematics_predictor as kin_mod


class _SimpleGraph:
    def __init__(self, is_anchor: bool = False):
        self.is_anchor = torch.tensor([is_anchor], dtype=torch.bool)

    def clone(self):
        return _SimpleGraph(bool(self.is_anchor.any().item()))

    def to(self, _device):
        return self


class _DummyModel:
    def __init__(self, pred, jac):
        self._pred = pred
        self._jac = jac

    def __call__(self, *_args, **_kwargs):
        return self._pred, self._jac


class _DummyKernels:
    def __init__(self):
        self.cfg = SimpleNamespace(mu_viscosity_nd_scale=0.0035)
        self.mu_0_nd = 1.0

    def _get_geometric_props(self, _data):
        return {"n": 1}

    def _compute_derivatives(self, field, _props):
        n = int(field.shape[0])
        return torch.zeros((n, 2, 1), dtype=torch.float32)


class _SimpleWeighter:
    def __call__(self, losses, scales=None):
        scales = scales or [1.0] * len(losses)
        total = torch.tensor(0.0, dtype=torch.float32)
        for loss, scale in zip(losses, scales):
            total = total + (loss * float(scale))
        return total


def test_compute_step_loss_respects_curriculum_stage_logic(monkeypatch):
    def _fake_terms(*_args, **_kwargs):
        return {
            "l_wss": torch.tensor(2.0),
            "l_data_kine": torch.tensor(1.0),
            "l_data_mu": torch.tensor(3.0),
            "l_mom": torch.tensor(0.5),
            "l_cont": torch.tensor(0.25),
            "l_bc": torch.tensor(0.1),
            "l_io": torch.tensor(0.2),
        }

    monkeypatch.setattr(kin_mod, "compute_kinematics_physics_terms", _fake_terms)

    n = 4
    model = _DummyModel(pred=torch.zeros((n, 5), dtype=torch.float32), jac=torch.tensor(0.1))
    data = SimpleNamespace(y=torch.zeros((n, 5), dtype=torch.float32))
    kernels = _DummyKernels()
    weighter = _SimpleWeighter()

    _, m1 = kin_mod.compute_step_loss(
        model=model,
        data=data,
        kernels=kernels,
        loss_weighter=weighter,
        solver="anderson",
        device="cpu",
        stage=1,
        current_n=1.0,
        current_mu_0=0.0035,
        weight_data_base=500.0,
        weight_mu_base=10.0,
        weight_wss_base=10.0,
    )
    assert m1["L_data"] > 0.0
    assert m1["L_mu"] > 0.0
    assert abs(kernels.mu_0_nd - 1.0) < 1e-6

    _, m2 = kin_mod.compute_step_loss(
        model=model,
        data=data,
        kernels=kernels,
        loss_weighter=weighter,
        solver="anderson",
        device="cpu",
        stage=2,
        current_n=0.8,
        current_mu_0=0.02,
        weight_data_base=500.0,
        weight_mu_base=10.0,
        weight_wss_base=10.0,
    )
    assert m2["L_data"] == 0.0
    assert m2["L_mu"] > 0.0
    assert abs(kernels.mu_0_nd - (0.02 / 0.0035)) < 1e-6


def test_fast_forward_curriculum_three_epochs(monkeypatch):
    stage_calls = []
    freeze_states = []
    optimizer_kinds = []
    loaded = []

    class _FakeModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            _ = (args, kwargs)
            self.w = nn.Parameter(torch.tensor(1.0))

        def forward(self, data, **_kwargs):
            n = 2
            pred = torch.zeros((n, 5), dtype=torch.float32) + self.w * 0.0
            _ = data
            return pred, torch.tensor(0.0)

    class _FakeKernels:
        def __init__(self, phys_cfg):
            _ = phys_cfg
            self.cfg = SimpleNamespace(mu_viscosity_nd_scale=0.0035)
            self.mu_0_nd = 1.0

    class _FakeLossWeighter(nn.Module):
        def __init__(self, num_losses):
            super().__init__()
            _ = num_losses
            self.p = nn.Parameter(torch.tensor(0.0))

        def forward(self, losses):
            return losses[0] + losses[1] + self.p * 0.0

        def requires_grad_(self, flag=True):
            freeze_states.append(bool(flag))
            return super().requires_grad_(flag)

    class _FakeAdam:
        def __init__(self, *_args, **_kwargs):
            optimizer_kinds.append("adam")
            # Mirror real torch optimizers that expose a mutable state dict.
            # This keeps the regression test focused on real stage/curriculum behavior.
            self.state = {}

        def zero_grad(self):
            return None

        def step(self):
            return None

    class _FakeLBFGS:
        def __init__(self, *_args, **_kwargs):
            optimizer_kinds.append("lbfgs")

        def zero_grad(self):
            return None

        def step(self, closure):
            return closure()

    class _FakeScheduler:
        def __init__(self, *_args, **_kwargs):
            return None

        def step(self):
            return None

    def _fake_load_dataset(phase, target_n=None):
        loaded.append((phase, target_n))
        return [_SimpleGraph(False), _SimpleGraph(False)]

    def _fake_split(dataset, seed=42, train_ratio=0.9):
        _ = (seed, train_ratio)
        return {"train": dataset, "val": dataset, "n_anchors": 0, "n_physics": len(dataset)}

    def _fake_compute_step_loss(
        _model,
        _data,
        _kernels,
        _loss_weighter,
        _solver,
        _device,
        stage,
        current_n,
        current_mu_0,
        weight_data_base,
        weight_mu_base,
        weight_wss_base,
    ):
        _ = (weight_data_base, weight_mu_base, weight_wss_base)
        stage_calls.append((stage, current_n, current_mu_0))
        return torch.tensor(1.0, requires_grad=True), {
            "L_total": 1.0,
            "L_data": 0.0,
            "L_mu": 0.0,
            "L_mom": 0.0,
            "L_cont": 0.0,
            "L_bc": 0.0,
            "L_io": 0.0,
            "L_wss": 0.0,
            "L_pgrad": 0.0,
            "L_jac": 0.0,
            "C_weighted_pde": 0.0,
            "C_data_kine": 0.0,
            "C_data_mu": 0.0,
            "C_bc": 0.0,
            "C_io": 0.0,
            "C_pgrad": 0.0,
            "C_wss": 0.0,
            "C_jac": 0.0,
        }

    monkeypatch.setattr(kin_mod, "GINO_DEQ", _FakeModel)
    monkeypatch.setattr(kin_mod, "PhysicsKernels", _FakeKernels)
    monkeypatch.setattr(kin_mod, "DynamicLossWeighter", _FakeLossWeighter)
    monkeypatch.setattr(kin_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(kin_mod, "split_anchor_physics", _fake_split)
    monkeypatch.setattr(kin_mod, "compute_step_loss", _fake_compute_step_loss)
    monkeypatch.setattr(
        kin_mod,
        "quantify_performance",
        lambda *_args, **_kwargs: {"rel_l2": 0.1, "continuity": 0.001},
    )
    monkeypatch.setattr(kin_mod, "DataLoader", lambda data, **_kwargs: list(data))
    monkeypatch.setattr(kin_mod.optim, "AdamW", _FakeAdam)
    monkeypatch.setattr(kin_mod.optim, "LBFGS", _FakeLBFGS)
    monkeypatch.setattr(kin_mod, "LinearLR", _FakeScheduler)
    monkeypatch.setattr(kin_mod, "CosineAnnealingLR", _FakeScheduler)
    monkeypatch.setattr(kin_mod, "SequentialLR", _FakeScheduler)
    monkeypatch.setattr(kin_mod.os, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kin_mod.torch, "save", lambda *_args, **_kwargs: None)

    kin_mod.train_kinematics(
        epochs=3,
        adam_epochs=2,
        stage1_end_epoch=1,
        stage2_end_epoch=2,
    )

    assert [s for s, _, _ in stage_calls] == [1, 1, 2, 2, 3, 3]
    assert loaded == [("kinematics", "newtonian"), ("kinematics", "carreau")]
    assert False in freeze_states  # Stage 2 freezes Kendall weighter
    assert True in freeze_states   # Stage 1/3 unfreezes it
    assert "lbfgs" in optimizer_kinds


def test_prune_kine_training_artifacts_keeps_only_latest_three(tmp_path):
    out = tmp_path / "outputs" / "kinematics"
    out.mkdir(parents=True)
    for epoch in range(1, 6):
        (out / f"kinematics_ckpt_{epoch}.pth").write_text("ckpt", encoding="utf-8")
        (out / f"kinematics_state_{epoch}.pth").write_text("state", encoding="utf-8")
    (out / "kinematics_ckpt_latest.pth").write_text("latest", encoding="utf-8")
    (out / "kinematics_state_latest.pth").write_text("latest", encoding="utf-8")
    (out / "kinematics_best.pth").write_text("best", encoding="utf-8")

    removed = kin_mod._prune_kine_training_artifacts(out, keep=3)

    assert removed == 4
    assert sorted(p.name for p in out.glob("kinematics_ckpt_*.pth")) == [
        "kinematics_ckpt_3.pth",
        "kinematics_ckpt_4.pth",
        "kinematics_ckpt_5.pth",
        "kinematics_ckpt_latest.pth",
    ]
    assert sorted(p.name for p in out.glob("kinematics_state_*.pth")) == [
        "kinematics_state_3.pth",
        "kinematics_state_4.pth",
        "kinematics_state_5.pth",
        "kinematics_state_latest.pth",
    ]
    assert (out / "kinematics_best.pth").exists()
