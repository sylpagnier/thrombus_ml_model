"""Wall-only vel-decay: lumen Mat must not be washed by alpha*speed."""

from __future__ import annotations

import pytest
import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.core_physics.species_pushforward_continuous import (
    apply_velocity_decay,
    continuous_vel_decay_wall_only,
    resolve_vel_decay_speed,
)


def test_resolve_vel_decay_speed_wall_only_zeros_lumen():
    cfg = PushforwardConfig(vel_decay_wall_only=True, species_scope="mat", channels=(11,))
    with use_pushforward_config(cfg):
        assert continuous_vel_decay_wall_only()
        spd = torch.tensor([0.0, 0.8, 0.5])
        wall = torch.tensor([True, False, False])
        out = resolve_vel_decay_speed(spd, wall)
    assert float(out[0]) == 0.0
    assert float(out[1]) == 0.0
    assert float(out[2]) == 0.0


def test_resolve_vel_decay_speed_legacy_full_band():
    cfg = PushforwardConfig(vel_decay_wall_only=False, species_scope="mat", channels=(11,))
    with use_pushforward_config(cfg):
        assert not continuous_vel_decay_wall_only()
        spd = torch.tensor([0.0, 0.8, 0.5])
        wall = torch.tensor([True, False, False])
        out = resolve_vel_decay_speed(spd, wall)
    assert torch.allclose(out, spd)


def test_apply_velocity_decay_wall_only_preserves_lumen_mat():
    cfg = PushforwardConfig(vel_decay_wall_only=True, species_scope="mat", channels=(11,))
    with use_pushforward_config(cfg):
        st = torch.tensor([[1.0], [1.0], [1.0]])
        spd = torch.tensor([0.0, 0.9, 0.9])
        wall = torch.tensor([True, False, False])
        alphas = (torch.tensor(0.7), torch.tensor(0.7))
        out = apply_velocity_decay(st, spd, alphas, wall_mask=wall)
    assert torch.allclose(out, st)


def test_apply_velocity_decay_legacy_wipes_fast_lumen():
    cfg = PushforwardConfig(vel_decay_wall_only=False, species_scope="mat", channels=(11,))
    with use_pushforward_config(cfg):
        st = torch.tensor([[1.0], [1.0]])
        spd = torch.tensor([0.0, 0.9])
        wall = torch.tensor([True, False])
        alphas = (torch.tensor(0.7), torch.tensor(0.7))
        out = apply_velocity_decay(st, spd, alphas, wall_mask=wall)
    assert float(out[0, 0]) == pytest.approx(1.0)
    assert float(out[1, 0]) < 0.5


def test_wall_only_fail_closed_without_mask():
    cfg = PushforwardConfig(vel_decay_wall_only=True, species_scope="mat", channels=(11,))
    with use_pushforward_config(cfg):
        spd = torch.tensor([0.5, 0.9])
        out = resolve_vel_decay_speed(spd, wall_mask=None)
    assert torch.allclose(out, torch.zeros_like(spd))
