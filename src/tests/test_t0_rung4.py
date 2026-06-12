"""Rung 4 species override wiring."""

from __future__ import annotations

import os

import pytest
import torch

from src.core_physics.t0_mu_physics import resolve_t0_species_log_nd
from src.core_physics.t0_rung_config import t0_rung4_env, t0_teacher_gt_kine_env
from src.utils.paths import get_project_root


def test_resolve_t0_species_from_series():
    n, c_sp = 5, 12
    t_steps = 3
    data = type("G", (), {})()
    data.y = torch.randn(t_steps, n, 16)
    series = torch.randn(t_steps, n, 16)
    device = torch.device("cpu")
    pred = resolve_t0_species_log_nd(data, 1, device, pred_species_series=series)
    assert torch.allclose(pred, series[1, :, 4:16])
    gt = resolve_t0_species_log_nd(data, 1, device, pred_species_series=None)
    assert torch.allclose(gt, data.y[1, :, 4:16])


def test_t0_teacher_gt_kine_env_sets_flag():
    prev = os.environ.get("BIOCHEM_GT_KINE_VEL")
    with t0_teacher_gt_kine_env():
        assert os.environ.get("BIOCHEM_GT_KINE_VEL") == "1"
    if prev is None:
        os.environ.pop("BIOCHEM_GT_KINE_VEL", None)
    else:
        os.environ["BIOCHEM_GT_KINE_VEL"] = prev


def test_t0_rung4_env_metadata():
    with t0_rung4_env(teacher_ckpt="outputs/biochem/biochem_teacher_last.pth") as cfg:
        assert cfg["rung"] == "4"
        assert cfg["flow_source"] == "gt"
        assert cfg["species_source"] == "teacher"


@pytest.mark.skipif(
    not (get_project_root() / "outputs/biochem/biochem_teacher_last.pth").is_file()
    and not (get_project_root() / "outputs/biochem/biochem_teacher_best_high_mu.pth").is_file(),
    reason="no biochem teacher ckpt",
)
def test_rollout_t0_pred_species_shape():
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph.is_file():
        pytest.skip("patient007 missing")
    from src.core_physics.t0_rung_config import resolve_default_teacher_ckpt, rollout_t0_pred_species_series

    data = torch.load(graph, map_location="cpu", weights_only=False)
    ckpt = resolve_default_teacher_ckpt()
    series = rollout_t0_pred_species_series(data, ckpt, torch.device("cpu"))
    assert series.shape[0] == data.y.shape[0]
    assert series.shape[1] == data.y.shape[1]
