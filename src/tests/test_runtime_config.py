"""BiochemRuntimeConfig / PushforwardConfig composition smoke tests."""

from __future__ import annotations

from src.architecture.pushforward_config import get_active_config
from src.architecture.runtime_config import (
    BiochemRuntimeConfig,
    get_active_runtime,
    split_legacy_runtime_env,
    use_biochem_runtime,
    validate_runtime_kwargs,
)
from src.biochem_gnn.config import apply_train_recipe_env, build_train_recipe_configs
from src.biochem_gnn.mat_growth_simple import get_mat_growth_runtime_kwargs, mat_growth_leg_spec
from src.core_physics.species_pushforward_continuous import (
    continuous_dual_head,
    continuous_final_state_all_band,
    deploy_eval_use_full_timeline,
    deploy_horizon_aux_all_packs,
    deploy_horizon_aux_cap_steps,
    deploy_horizon_steps,
    train_deploy_eval_flow_source,
)
from src.evaluation.clot_relaxed_metrics import species_continuous_clout_score_mode
from src.inference.corrector_coupling import corrector_coupling_enabled


def test_runtime_kwargs_validate_and_build():
    kw = {
        "corrector_coupling": True,
        "rollout_vel_source": "coupled",
        "clout_score_mode": "guiding",
        "phi_loss_weight": 20.0,
    }
    validate_runtime_kwargs(kw)
    rt = BiochemRuntimeConfig.from_kwargs(kw)
    assert rt.coupling.corrector_coupling is True
    assert rt.rollout.rollout_vel_source == "coupled"
    assert rt.scoring.clout_score_mode == "guiding"
    assert rt.gelation.phi_loss_weight == 20.0


def test_split_legacy_runtime_env():
    cfg, rem = split_legacy_runtime_env(
        {
            "BIOCHEM_CORRECTOR_COUPLING": "1",
            "SPECIES_ROLLOUT_VEL_SOURCE": "coupled",
            "SOME_UNKNOWN_KNOB": "x",
        }
    )
    assert cfg["corrector_coupling"] is True
    assert cfg["rollout_vel_source"] == "coupled"
    assert rem == {"SOME_UNKNOWN_KNOB": "x"}


def test_wg_sweep_v3_runtime_is_typed():
    spec = mat_growth_leg_spec("WG_sweep_v3_01")
    assert spec.env_overrides == {}
    kw = get_mat_growth_runtime_kwargs("WG_sweep_v3_01")
    validate_runtime_kwargs(kw)
    rt = BiochemRuntimeConfig.from_kwargs(kw)
    assert rt.coupling.corrector_coupling is True
    assert rt.rollout.rollout_vel_source == "coupled"
    assert rt.rollout.dynamic_occlusion is True
    assert rt.gelation.viscosity_calib is True


def test_active_runtime_overrides_helpers():
    rt = BiochemRuntimeConfig.from_kwargs(
        {
            "corrector_coupling": False,
            "clout_score_mode": "relaxed_prec_floor",
        }
    )
    with use_biochem_runtime(rt):
        assert corrector_coupling_enabled() is False
        assert species_continuous_clout_score_mode() == "relaxed_prec_floor"


def test_build_train_recipe_configs_dual_head_unroll():
    pf, rt = build_train_recipe_configs()
    assert pf.dual_head is True
    assert pf.unroll == 10
    assert rt.rollout.deploy_eval_full is True
    assert rt.rollout.deploy_horizon_all_packs is True
    assert rt.rollout.deploy_horizon_aux_cap == 72


def test_apply_train_recipe_env_binds_active_config(monkeypatch):
    """Typed bind must drive continuous_dual_head without relying solely on env."""
    apply_train_recipe_env(force=True)
    # Clear env bridge so helpers must read the active PushforwardConfig.
    monkeypatch.delenv("SPECIES_CONTINUOUS_DUAL_HEAD", raising=False)
    monkeypatch.delenv("SPECIES_PUSHFORWARD_UNROLL", raising=False)
    monkeypatch.delenv("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND", raising=False)
    monkeypatch.delenv("SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL", raising=False)
    monkeypatch.delenv("SPECIES_DEPLOY_HORIZON_ALL_PACKS", raising=False)
    monkeypatch.delenv("SPECIES_DEPLOY_HORIZON_AUX_CAP", raising=False)
    pf = get_active_config()
    rt = get_active_runtime()
    assert pf is not None and pf.dual_head is True
    assert pf.unroll == 10
    assert rt is not None
    assert continuous_dual_head() is True
    assert continuous_final_state_all_band() is True
    assert deploy_eval_use_full_timeline() is True
    assert deploy_horizon_aux_all_packs() is True
    assert deploy_horizon_aux_cap_steps() == 72
    assert train_deploy_eval_flow_source() == "auto"


def test_deploy_horizon_helpers_prefer_runtime():
    rt = BiochemRuntimeConfig.from_kwargs(
        {
            "deploy_horizon": 27,
            "deploy_eval_full": False,
            "deploy_horizon_all_packs": False,
            "deploy_horizon_aux_cap": 40,
            "train_deploy_eval_flow": "kinematics",
        }
    )
    with use_biochem_runtime(rt):
        assert deploy_horizon_steps() == 27
        assert deploy_eval_use_full_timeline() is False
        assert deploy_horizon_aux_all_packs() is False
        assert deploy_horizon_aux_cap_steps() == 40
        assert train_deploy_eval_flow_source() == "kinematics"


def test_wc_v7_leg_typed_with_empty_residuals(monkeypatch):
    from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env, mat_growth_leg_spec
    from src.core_physics.species_viscosity_calibration import viscosity_calibration_enabled
    from src.evaluation.clot_relaxed_metrics import clot_guide_f_beta

    spec = mat_growth_leg_spec("WC_v7_clot_phi_mse")
    assert spec.env_overrides == {}
    apply_mat_growth_leg_env("WC_v7_clot_phi_mse", force=True)
    for key in (
        "SPECIES_CONTINUOUS_DUAL_HEAD",
        "SPECIES_VISCOSITY_CALIB",
        "SPECIES_DYNAMIC_OCCLUSION",
        "CLOT_GUIDE_F_BETA",
    ):
        monkeypatch.delenv(key, raising=False)
    pf = get_active_config()
    rt = get_active_runtime()
    assert pf is not None and pf.dual_head is True
    assert pf.physics_readout is True
    assert rt is not None
    assert continuous_dual_head() is True
    assert viscosity_calibration_enabled() is True
    assert abs(clot_guide_f_beta() - float(rt.scoring.guide_f_beta)) < 1e-9
