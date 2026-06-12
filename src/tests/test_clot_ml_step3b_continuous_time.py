"""F6: Step 3b continuous-time pair / tau unit tests."""

from __future__ import annotations

import torch

from src.config import BiochemConfig
from src.core_physics.clot_continuous_time import (
    extrapolated_t_out_max,
    growth_u_from_t_frac,
    macro_tau_at_index,
    rollout_time_indices,
    time_frac_for_rollout,
)
from src.core_physics.clot_temporal_growth_rules import _progressive_frac_from_growth_u
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.core_physics.clot_forecast import iter_forecast_pairs


class _TinyGraph:
    def __init__(self, n_steps: int = 5):
        self.y = torch.zeros(n_steps, 4, 16)
        self.t = torch.linspace(0.0, 40.0, n_steps)
        self.num_nodes = 4


def test_iter_forecast_pairs_extrap_beyond_comsol_window():
    pairs = iter_forecast_pairs(5, t_out_max=8)
    assert pairs
    assert pairs[-1][1] == 8
    assert pairs[0] == (0, 1)
    assert all(t_out > 4 for t_in, t_out in pairs if t_out > 4)


def test_macro_tau_virtual_index_exceeds_one():
    g = _TinyGraph(5)
    bio = BiochemConfig(phase="biochem")
    tau_end = macro_tau_at_index(g, 4, bio_cfg=bio)
    tau_extrap = macro_tau_at_index(g, 7, bio_cfg=bio)
    assert tau_end <= 1.0 + 1e-5
    assert tau_extrap > tau_end


def test_extrapolated_rollout_indices_scale_with_sim_end():
    g = _TinyGraph(10)
    n_in = len(rollout_time_indices(g, sim_end_scale=1.0))
    n_ex = len(rollout_time_indices(g, sim_end_scale=1.5))
    assert n_ex > n_in
    assert extrapolated_t_out_max(g, sim_end_scale=1.5) > 9


def test_time_frac_macro_tau_env(monkeypatch):
    g = _TinyGraph(5)
    monkeypatch.setenv("CLOT_ML_USE_MACRO_TAU", "1")
    frac = time_frac_for_rollout(g, 2, bio_cfg=BiochemConfig(phase="biochem"), clamp_unit=False)
    assert 0.0 < frac < 1.0


def test_growth_u_extrap_extends_past_comsol_end():
    u_in = growth_u_from_t_frac(1.0, 0.4, extrap=True, sim_end_scale=5.0)
    u_ex = growth_u_from_t_frac(3.0, 0.4, extrap=True, sim_end_scale=5.0)
    assert abs(u_in - 1.0) < 1e-5
    assert u_ex > u_in


def test_progressive_frac_increases_in_extrap_segment():
    from src.core_physics.clot_temporal_growth_rules import TemporalGrowthRuleConfig

    cfg = TemporalGrowthRuleConfig(
        name="test_prog",
        kind="progressive_topk",
        start_frac=0.05,
        end_frac=0.22,
        power=1.5,
    )
    f_in = _progressive_frac_from_growth_u(cfg, 1.0, extrap=True, sim_end_scale=5.0)
    f_ex = _progressive_frac_from_growth_u(cfg, 2.0, extrap=True, sim_end_scale=5.0)
    assert f_ex > f_in


def test_rule_rollout_extrap_grows_commit_frac(monkeypatch):
    import os
    from pathlib import Path

    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache, rollout_temporal_phi

    root = Path(__file__).resolve().parents[2]
    graph = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    step0 = root / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    if not graph.is_file() or not step0.is_file():
        return

    monkeypatch.setenv("CLOT_ML_USE_MACRO_TAU", "1")
    monkeypatch.setenv("CLOT_ML_CONTINUOUS_EXTRAP", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    monkeypatch.setenv("CLOT_PHI_KINE_CKPT", str(root / "outputs/kinematics/kinematics_best.pth"))

    import torch

    device = torch.device("cpu")
    data = torch.load(graph, map_location=device, weights_only=False)
    rule_cfg = load_step0_coef_json(step0).to_rule_config(name="test_extrap")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    reset_temporal_kinematics_cache()
    n = int(data.y.shape[0])
    tf = n - 1
    phi_in = rollout_temporal_phi(data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=1.0)
    reset_temporal_kinematics_cache()
    phi_ex = rollout_temporal_phi(data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=2.0)
    t_max = extrapolated_t_out_max(data, sim_end_scale=2.0)
    frac_in = float((phi_in[tf] > 0.5).float().mean().item())
    frac_ex = float((phi_ex[t_max] > 0.5).float().mean().item())
    assert frac_ex > frac_in + 1e-4
