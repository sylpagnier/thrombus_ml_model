"""Wall-gen gate rule + FP geography classification."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.fp_geography import classify_fp_geography
from src.evaluation.seed_growth_diagnostics import (
    WALL_GEN_FLOOR_CLOT_F1,
    passes_wall_gen_gate,
    seed_growth_diagnostic_panel,
)
from src.training.train_species_pushforward_continuous import select_checkpoint_score


def _row(**over):
    base = {
        "deploy_eval_t": 1.0,
        "loss": 1.0,
        "deploy_clot_f1": 0.37,
        "val_state_f1": 0.5,
        "val_mat_f1": 0.5,
        "val_growth_f1": 0.5,
        "val_growth_mat_f1": 0.5,
        "val_init_f1": 0.5,
        "val_clot_phi_f1": 0.5,
        "val_pred_delta": 0.0,
    }
    base.update(over)
    return base


def test_gate_rejects_precision_mirage_low_mass():
    """fh=2 style: score high, mass starved -> never promote."""
    ok, reason = passes_wall_gen_gate(
        {
            "deploy_clot_f1": 0.308,
            "deploy_clot_score": 0.529,
            "deploy_clot_mass_ratio": 0.18,
            "deploy_clot_fn": 90.0,
        }
    )
    assert ok is False
    assert "mass_starve" in reason


def test_gate_accepts_floor_band():
    ok, reason = passes_wall_gen_gate(
        {
            "deploy_clot_f1": WALL_GEN_FLOOR_CLOT_F1,
            "deploy_clot_score": 0.35,
            "deploy_clot_mass_ratio": 1.11,
            "deploy_clot_fn": 67.0,
            "deploy_clot_fp": 79.0,
        }
    )
    assert ok is True
    assert reason.startswith("ok")


def test_gate_rejects_fn_rise():
    ok, reason = passes_wall_gen_gate(
        {
            "deploy_clot_f1": 0.40,
            "deploy_clot_score": 0.40,
            "deploy_clot_mass_ratio": 1.0,
            "deploy_clot_fn": 95.0,
        }
    )
    assert ok is False
    assert "fn_rose" in reason


def test_select_hard_min_rejects_starvation():
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.31),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.53,
        deploy_mat_f1=0.17,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
        deploy_clot_mass_ratio=0.18,
        select_clot_f1_weight=0.75,
        select_clot_score_weight=0.15,
        select_mat_f1_weight=0.10,
        select_mass_hard_min=0.5,
        select_mass_hard_max=1.5,
    )
    assert mode == "deploy_only_mass_reject"
    assert score < -1e11


def test_select_f1_primary_prefers_higher_f1_over_score():
    """Locked gate: F1 primary beats score mirage when mass is sane."""
    base = dict(
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
        deploy_clot_mass_ratio=1.1,
        select_clot_f1_weight=0.75,
        select_clot_score_weight=0.15,
        select_mat_f1_weight=0.10,
        select_mass_hard_min=0.5,
        select_mass_hard_max=1.5,
    )
    high_f1, mode_a = select_checkpoint_score(
        _row(deploy_clot_f1=0.42),
        deploy_clot_score=0.30,
        deploy_mat_f1=0.20,
        **base,
    )
    high_score, mode_b = select_checkpoint_score(
        _row(deploy_clot_f1=0.30),
        deploy_clot_score=0.55,
        deploy_mat_f1=0.20,
        **base,
    )
    assert mode_a == mode_b == "deploy_only"
    assert high_f1 > high_score


def test_select_fn_hard_max_rejects():
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.40, deploy_clot_fn=95.0),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.40,
        deploy_mat_f1=0.30,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
        deploy_clot_mass_ratio=1.0,
        select_clot_f1_weight=0.75,
        select_fn_hard_max=80.0,
        select_mass_hard_min=0.5,
        select_mass_hard_max=1.5,
    )
    assert mode == "deploy_only_fn_reject"
    assert score < -1e11


def test_fp_geography_distant_recommends_physfp():
    # Line graph 0-1-2-3-4-5; GT at 0; FP at 5 => distant
    edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    phi_gt = np.zeros(6)
    phi_gt[0] = 1.0
    phi_pred = np.zeros(6)
    phi_pred[0] = 1.0
    phi_pred[5] = 1.0
    s = classify_fp_geography(phi_pred, phi_gt, edge, adjacent_max_hops=2)
    assert s["n_distant_fp"] == 1
    assert s["recommend_leg"] == "physfp"
    assert s["mode"] == "distant"


def test_fp_geography_adjacent_recommends_cloop():
    edge = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    phi_gt = np.array([1.0, 1.0, 0.0, 0.0])
    phi_pred = np.array([1.0, 1.0, 1.0, 0.0])  # FP at hop1 from GT
    s = classify_fp_geography(phi_pred, phi_gt, edge, adjacent_max_hops=2)
    assert s["n_adjacent_fp"] == 1
    assert s["recommend_leg"] == "cloop"
    assert s["mode"] == "adjacent"


def test_wg_prec_physfp_and_cloop_leg_specs():
    from src.architecture.pushforward_config import PushforwardConfig
    from src.architecture.runtime_config import BiochemRuntimeConfig
    from src.biochem_gnn.mat_growth_simple import (
        get_mat_growth_config_kwargs,
        get_mat_growth_runtime_kwargs,
        mat_growth_leg_spec,
    )

    phys = mat_growth_leg_spec("WG_prec_physfp")
    cloop = mat_growth_leg_spec("WG_prec_cloop")
    base = mat_growth_leg_spec("WG_prec_iter")

    assert phys.config_kwargs.get("physical_fp_gating") is True
    assert int(phys.config_kwargs.get("frontier_hops") or 0) == 0
    assert float(phys.config_kwargs.get("seed_aux_weight") or 0) == 0.0
    assert float(phys.config_kwargs.get("underpred_weight") or 1) == float(
        base.config_kwargs.get("underpred_weight") or 1
    )
    assert float(phys.runtime_kwargs.get("select_clot_f1_weight") or 0) >= 0.7
    assert float(phys.runtime_kwargs.get("select_mass_hard_min") or 0) >= 0.5
    assert float(phys.runtime_kwargs.get("select_fn_hard_max") or 0) >= 80.0

    assert cloop.config_kwargs.get("physical_fp_gating") in (False, None)
    assert float(cloop.config_kwargs.get("closed_loop_init") or 0) >= 0.8
    assert int(cloop.config_kwargs.get("tbptt_tail") or 0) >= 12
    assert cloop.config_kwargs.get("scheduled_sampling") is False

    cfg = PushforwardConfig.from_meta({"config_kwargs": get_mat_growth_config_kwargs("WG_prec_physfp")})
    rt = BiochemRuntimeConfig.from_meta(
        {"runtime_kwargs": get_mat_growth_runtime_kwargs("WG_prec_physfp")}
    )
    assert cfg.physical_fp_gating is True
    assert rt.scoring.select_clot_f1_weight >= 0.7
    assert rt.scoring.select_mass_hard_min >= 0.5


def test_panel_includes_gate():
    panel = seed_growth_diagnostic_panel(
        {
            "deploy_clot_f1": 0.37,
            "deploy_clot_score": 0.35,
            "deploy_clot_mass_ratio": 1.11,
            "deploy_clot_fn": 67,
            "deploy_clot_fp": 79,
            "mat_seed_prec": 1.0,
            "mat_seed_count": 1.0,
            "mat_front_speed_ratio": 0.5,
        }
    )
    assert "gate_ok" in panel
    assert panel["gate_ok"] is True
