"""Mat-only pushforward scope + single-head state dim safety."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from src.biochem_gnn.config import GLOBAL_TRAIN_RECIPE
from src.biochem_gnn.mat_growth_simple import (
    MAT_GROWTH_SIMPLE_RECIPE,
    apply_mat_growth_simple_recipe_env,
    init_mat_single_from_fimat_ckpt,
    mat_growth_leg_spec,
)
from src.core_physics.species_pushforward_continuous import (
    SpeciesDualHeadContinuousGNN,
    build_continuous_gnn,
    continuous_feature_dim,
    load_continuous_bundle,
)
from src.training.biochem_species_scope import (
    MAT_CHANNEL,
    pushforward_state_bulk_indices,
    pushforward_state_dim,
)


def _write_dual_ckpt(dual: torch.nn.Module) -> str:
    p = Path(tempfile.gettempdir()) / "test_fimat_dual_ckpt.pth"
    torch.save(
        {
            "model_state": dual.state_dict(),
            "in_dim": int(dual.in_dim),
            "hidden": int(dual.hidden),
            "meta": {"dual_head": True, "arch": "sage"},
        },
        p,
    )
    return str(p)


def test_mat_scope_is_single_channel():
    os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = "mat"
    assert pushforward_state_bulk_indices() == [MAT_CHANNEL]
    assert pushforward_state_dim() == 1


def test_mat_growth_simple_recipe_knobs():
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime

    apply_mat_growth_simple_recipe_env(force=True)
    pf = get_active_config()
    rt = get_active_runtime()
    assert pf is not None and rt is not None
    assert pf.species_scope == "mat"
    assert pf.dual_head is False
    assert rt.scoring.clout_score_mode == "relaxed_prec_floor"
    assert float(pf.fp_weight) >= 16.0
    assert float(pf.score_clout_w) >= 0.75
    # Train flow block on COMSOL GT (avoid a second GINO-DEQ at pack build).
    assert pf.flow_feats_source == "gt"
    assert MAT_GROWTH_SIMPLE_RECIPE["SPECIES_FLOW_FEATS_SOURCE"] == "gt"
    # wall hops come from the global triangle6 recipe (applied first), not the mat-only overrides.
    assert int(rt.rollout.wall_hops) == int(GLOBAL_TRAIN_RECIPE["SPECIES_SNAPSHOT_WALL_HOPS"])


def test_mat_growth_leg_specs():
    a = mat_growth_leg_spec("A_random")
    b = mat_growth_leg_spec("B_backbone")
    c = mat_growth_leg_spec("C_geom")
    d = mat_growth_leg_spec("D_parity_single")
    e = mat_growth_leg_spec("E_dual_mat")
    f = mat_growth_leg_spec("F_single_fimat")
    g = mat_growth_leg_spec("G_dual_mat_neighbor_gate")
    h = mat_growth_leg_spec("H_dual_mat_crit_focus")
    i = mat_growth_leg_spec("I_dual_fimat_fi_aux")
    j = mat_growth_leg_spec("J_dual_mat_neighbor_crit")
    assert a.no_init and not b.no_init and c.no_init
    assert b.init_mode == "backbone"
    assert c.config_kwargs.get("geom_feats") is True
    assert d.init_mode == "mat_readout"
    assert e.config_kwargs.get("dual_head") is True
    assert f.config_kwargs.get("species_scope") == "fi_mat"
    assert g.config_kwargs.get("neighbor_commit_gate") is True
    assert h.config_kwargs.get("underpred_weight") == 5.0
    assert i.config_kwargs.get("channel_weight_fi") == 0.15
    assert j.config_kwargs.get("neighbor_commit_gate") is True
    assert j.config_kwargs.get("underpred_weight") == 5.0


def test_precision_sweep_leg_specs():
    """K/L/M/N flip exactly one in-training lever on the dual fi_mat baseline."""
    k = mat_growth_leg_spec("K_fimat_neighbor_gate")
    l = mat_growth_leg_spec("L_fimat_geom_rich")
    m = mat_growth_leg_spec("M_fimat_neighbor_geom_rich")
    n = mat_growth_leg_spec("N_mat_geom_rich")
    # K: neighbour gate kept, but on the dual fi_mat head (not Mat-only).
    assert k.config_kwargs.get("species_scope") == "fi_mat"
    assert k.config_kwargs.get("neighbor_commit_gate") is True
    assert "geom_feats_rich" not in k.config_kwargs
    # L: enriched geometry only.
    assert l.config_kwargs.get("species_scope") == "fi_mat"
    assert l.config_kwargs.get("geom_feats_rich") is True
    assert "neighbor_commit_gate" not in l.config_kwargs
    # M: both surviving levers.
    assert m.config_kwargs.get("neighbor_commit_gate") is True
    assert m.config_kwargs.get("geom_feats_rich") is True
    # N: rich geometry on the Mat-only control.
    assert n.config_kwargs.get("species_scope") == "mat"
    assert n.config_kwargs.get("geom_feats_rich") is True
    # O: N + G (mat scope, neighbour gate, rich geom).
    o = mat_growth_leg_spec("O_mat_neighbor_geom_rich")
    assert o.config_kwargs.get("species_scope") == "mat"
    assert o.config_kwargs.get("geom_feats_rich") is True
    assert o.config_kwargs.get("neighbor_commit_gate") is True


def test_precision_ladder_6h_leg_specs():
    """P/Q/R: pure-scope control + gate-precision levers, all Mat-only dual head."""
    p = mat_growth_leg_spec("P_mat_plain")
    q = mat_growth_leg_spec("Q_mat_gate_sharp_fp")
    r = mat_growth_leg_spec("R_mat_geom_gate_sharp_fp")
    # P: scope only -- no gate, no geom, no sharpening.
    assert p.config_kwargs.get("species_scope") == "mat"
    assert "neighbor_commit_gate" not in p.config_kwargs
    assert "geom_feats_rich" not in p.config_kwargs
    assert "gate_temp" not in p.config_kwargs
    # Q: gate + sharpening + spatial FP pressure (no geom).
    assert q.config_kwargs.get("neighbor_commit_gate") is True
    assert q.config_kwargs.get("gate_temp") == 0.5
    assert q.config_kwargs.get("spatial_loss_weight") == 3.0
    assert "geom_feats_rich" not in q.config_kwargs
    # R: Q + rich geometry (kitchen sink of survivors).
    assert r.config_kwargs.get("geom_feats_rich") is True
    assert r.config_kwargs.get("gate_temp") == 0.5
    assert r.config_kwargs.get("neighbor_commit_gate") is True


def test_nucleation_front_leg_specs():
    """U/V/S/T: SeedFrontMat pivot ladder, Mat-only, deployable (pred-state seed)."""
    u = mat_growth_leg_spec("U_mat_frontier_only")
    v = mat_growth_leg_spec("V_mat_frontier_geom")
    s = mat_growth_leg_spec("S_mat_frontier_nuc")
    t = mat_growth_leg_spec("T_mat_frontier_sharp")
    for leg in (u, v, s, t):
        assert leg.config_kwargs.get("species_scope") == "mat"
        assert leg.config_kwargs.get("frontier_hops") == 1
        assert float(leg.config_kwargs.get("nucleation_topk")) > 0.0
    # U: structural pivot only (no gate, no geom).
    assert "neighbor_commit_gate" not in u.config_kwargs
    assert "geom_feats_rich" not in u.config_kwargs
    # V: pivot + geom, no gate.
    assert v.config_kwargs.get("geom_feats_rich") is True
    assert "neighbor_commit_gate" not in v.config_kwargs
    # S: full SeedFrontMat_v0 (gate + geom).
    assert s.config_kwargs.get("neighbor_commit_gate") is True
    assert s.config_kwargs.get("geom_feats_rich") is True
    # T additionally sharpens the gate + spatial FP pressure.
    assert t.config_kwargs.get("gate_temp") == 0.5
    assert t.config_kwargs.get("spatial_loss_weight") == 3.0


def test_physical_guided_leg_specs():
    w = mat_growth_leg_spec("W_mat_flow_stagnation")
    x = mat_growth_leg_spec("X_mat_flow_seedfront")
    y = mat_growth_leg_spec("Y_mat_tight_seed")
    ab = mat_growth_leg_spec("AB_mat_gelation_aux")
    assert w.config_kwargs.get("flow_feats") is True
    assert x.config_kwargs.get("flow_feats") is True
    assert x.config_kwargs.get("frontier_hops") == 1
    assert float(y.config_kwargs.get("nucleation_topk")) == 0.02
    assert ab.config_kwargs.get("physics_readout") is True


def test_w_physics_triage_leg_specs():
    """WA-WJ: W base + one COMSOL-targeted channel each (physics triage ladder)."""
    wa = mat_growth_leg_spec("WA_mat_flow_neighbor_gate")
    wb = mat_growth_leg_spec("WB_mat_flow_geom_rich")
    wc = mat_growth_leg_spec("WC_mat_flow_dynamic")
    wd = mat_growth_leg_spec("WD_mat_flow_frontier")
    we = mat_growth_leg_spec("WE_mat_flow_thrombin")
    wf = mat_growth_leg_spec("WF_mat_flow_fg")
    wg = mat_growth_leg_spec("WG_mat_flow_neighbor_crit")
    wh = mat_growth_leg_spec("WH_mat_flow_gelation_light")
    wi = mat_growth_leg_spec("WI_mat_flow_neighbor_geom")
    wj = mat_growth_leg_spec("WJ_mat_flow_stack")
    for leg in (wa, wb, wc, wd, we, wf, wg, wh, wi, wj):
        assert leg.config_kwargs.get("flow_feats") is True
    assert wa.config_kwargs.get("neighbor_commit_gate") is True
    assert wb.config_kwargs.get("geom_feats_rich") is True
    assert wc.config_kwargs.get("flow_feats_dynamic") is True
    assert wd.config_kwargs.get("frontier_hops") == 1
    assert float(wd.config_kwargs.get("nucleation_topk")) == 0.0
    assert we.config_kwargs.get("channels") == (11, 5)
    assert wf.config_kwargs.get("channels") == (11, 7)
    assert wg.config_kwargs.get("underpred_weight") == 5.0
    assert wh.config_kwargs.get("physics_readout") is True
    assert float(wh.runtime_kwargs.get("phi_loss_weight")) == 0.25
    assert wi.config_kwargs.get("geom_feats_rich") is True
    assert wj.config_kwargs.get("flow_feats_dynamic") is True
    assert wj.config_kwargs.get("neighbor_commit_gate") is True


def test_eval_ckpt_recipe_is_deploy_faithful(monkeypatch):
    """Mat-growth eval must not inherit GT flow/species pins from training env."""
    from scripts.eval_mat_growth_simple import _apply_ckpt_recipe
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime

    monkeypatch.setenv("SPECIES_FLOW_FEATS_SOURCE", "gt")
    monkeypatch.setenv("SPECIES_ROLLOUT_PIN_OTHER", "gt")
    monkeypatch.setenv("SPECIES_ROLLOUT_IC_SOURCE", "gt")
    _apply_ckpt_recipe(
        {
            "pushforward_species_scope": "mat",
            "dual_head": True,
            "flow_feats": True,
            "flow_dynamic": True,
            "pushforward_species_channels": [11, 5],
            "runtime_kwargs": {
                "corrector_coupling": False,
                "closed_loop_coupling": False,
                "train_vel_source": "gt",
            },
        },
        label="mat_growth_simple",
    )
    pf = get_active_config()
    rt = get_active_runtime()
    assert pf is not None
    assert rt is not None
    assert pf.channels == (11, 5)
    assert pf.flow_feats_dynamic is True
    assert pf.flow_feats is True
    assert pf.dual_head is True
    assert pf.flow_feats_source == "auto"
    assert rt.rollout.deploy_faithful is True
    assert rt.rollout.rollout_pin_other == "rest"
    assert rt.rollout.rollout_ic_source == "resting"
    assert rt.rollout.rollout_vel_source == "kinematics"
    assert rt.coupling.corrector_coupling is True
    assert rt.coupling.closed_loop_coupling is True
    # Architecture must not be re-injected into env for typed readers.
    assert os.environ.get("SPECIES_FLOW_FEATS_SOURCE") is None
    assert os.environ.get("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS") in (None, "")


def test_eval_ckpt_recipe_applies_sparse_commit_overrides():
    from scripts.eval_mat_growth_simple import _apply_ckpt_recipe
    from src.architecture.pushforward_config import get_active_config

    _apply_ckpt_recipe(
        {
            "pushforward_species_scope": "mat",
            "dual_head": True,
            "config_kwargs": {
                "gate_temp": 1.0,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "mat_commit_thresh": -1.0,
            },
        },
        label="mat_growth_simple",
        pf_overrides={
            "gate_temp": 0.8,
            "frontier_hops": 2,
            "nucleation_topk": 0.05,
            "mat_commit_thresh": 1.5e-5,
        },
    )
    pf = get_active_config()
    assert pf is not None
    assert pf.gate_temp == 0.8
    assert pf.frontier_hops == 2
    assert pf.nucleation_topk == 0.05
    assert pf.mat_commit_thresh == 1.5e-5
    assert os.environ.get("SPECIES_FLOW_FEATS_DYNAMIC") in (None, "")


def test_wg_prec_seed_leg_specs():
    """Train-time sparse commitment legs: prec stack + frontier/topk, warm-start safe."""
    seed = mat_growth_leg_spec("WG_prec_seed")
    fh2 = mat_growth_leg_spec("WG_prec_seed_fh2")
    tk02 = mat_growth_leg_spec("WG_prec_seed_tk02")
    base = mat_growth_leg_spec("WG_prec_iter")

    for leg in (seed, fh2, tk02):
        assert leg.config_kwargs.get("species_scope") == "mat"
        assert leg.config_kwargs.get("flow_feats_drop_xy") is True
        assert leg.config_kwargs.get("geom_feats_rich") is True
        assert leg.config_kwargs.get("flux_stag_feat") is True
        assert float(leg.config_kwargs.get("frontier_hops", 0)) > 0
        assert float(leg.config_kwargs.get("nucleation_topk", 0.0)) > 0.0
        # Must not widen spatial_head vs prec_iter warm-start.
        assert leg.config_kwargs.get("neighbor_commit_gate") in (None, False)
        assert leg.init_ckpt == base.init_ckpt or "WG_prec_iter" in str(leg.init_ckpt)

    assert seed.config_kwargs.get("frontier_hops") == 1
    assert float(seed.config_kwargs.get("nucleation_topk")) == 0.05
    assert fh2.config_kwargs.get("frontier_hops") == 2
    assert float(tk02.config_kwargs.get("nucleation_topk")) == 0.02
    # Control: prec_iter itself still has sparse commit OFF.
    assert int(base.config_kwargs.get("frontier_hops", 0) or 0) == 0
    assert float(base.config_kwargs.get("nucleation_topk", 0.0) or 0.0) == 0.0


def test_wg_prec_seed_leg_binds_typed_sparse_commit():
    """apply_mat_growth_leg_env must put frontier/topk on the active PushforwardConfig."""
    from src.architecture.pushforward_config import PushforwardConfig, get_active_config
    from src.architecture.runtime_config import BiochemRuntimeConfig
    from src.biochem_gnn.config import _bind_typed_configs
    from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env
    from src.core_physics.species_pushforward_continuous import (
        continuous_frontier_hops,
        continuous_nucleation_topk,
    )

    try:
        apply_mat_growth_leg_env("WG_prec_seed", force=True)
        pf = get_active_config()
        assert pf is not None
        assert pf.frontier_hops == 1
        assert pf.nucleation_topk == 0.05
        assert continuous_frontier_hops() == 1
        assert continuous_nucleation_topk() == 0.05
        # No architecture env injection required for typed readers.
        assert os.environ.get("SPECIES_CONTINUOUS_FRONTIER_HOPS") in (None, "")
        assert os.environ.get("SPECIES_CONTINUOUS_NUCLEATION_TOPK") in (None, "")
    finally:
        # Restore defaults so later env-fallback tests are not polluted.
        _bind_typed_configs(PushforwardConfig(), BiochemRuntimeConfig())


def test_typed_frontier_mask_zeroes_far_deltas():
    """Physical check: under typed frontier_hops, deltas outside the predicted frontier are gated off."""
    from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
    from src.core_physics.species_pushforward_continuous import predict_continuous_step_delta

    cfg = PushforwardConfig(
        dual_head=True,
        species_scope="mat",
        frontier_hops=1,
        nucleation_topk=0.0,
        mat_commit_thresh=0.5,
        gate_temp=1.0,
    )
    with use_pushforward_config(cfg):
        in_dim = continuous_feature_dim(8)
        model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16)
        model.eval()
        # 5-node chain; only node 0 committed in predicted log_state.
        ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        base = torch.randn(5, 8)
        log_state = torch.zeros(5, 1)
        log_state[0, 0] = 1.0
        with torch.no_grad():
            delta = predict_continuous_step_delta(
                model, base, ei, log_state, training=False
            )
        d = delta.reshape(-1)
        # Nodes 3 and 4 are outside the 1-hop frontier of node 0 -> must be zeroed.
        assert float(d[3].abs()) < 1e-8
        assert float(d[4].abs()) < 1e-8


def test_typed_topk_seeds_when_frontier_empty():
    """Cold t0: with no committed mass, nucleation_topk allows the strongest gate logit to grow."""
    from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config

    cfg = PushforwardConfig(
        dual_head=True,
        species_scope="mat",
        frontier_hops=1,
        nucleation_topk=0.2,
        mat_commit_thresh=0.5,
    )
    with use_pushforward_config(cfg):
        model = SpeciesDualHeadContinuousGNN(continuous_feature_dim(8), hidden=16)
        ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        cold = torch.zeros(5, 1)
        logits = torch.tensor([[-9.0], [-9.0], [5.0], [-9.0], [-9.0]])
        mask = model._frontier_nucleation_mask(logits, cold, ei).reshape(-1).bool()
        assert bool(mask[2])
        assert not bool(mask[4])



def test_wg_sweep_v3_legs_use_wall_gen_coupled_baseline():
    """Phase1 v3 arms share FS_ab_coupled stack; 02+ are single-factor tweaks."""
    ctrl = mat_growth_leg_spec("WG_sweep_v3_01")
    geom = mat_growth_leg_spec("WG_sweep_v3_02")
    flux = mat_growth_leg_spec("WG_sweep_v3_03")
    mirror = mat_growth_leg_spec("WG_sweep_v3_04")
    noise_off = mat_growth_leg_spec("WG_sweep_v3_07")
    noise_hi = mat_growth_leg_spec("WG_sweep_v3_08")

    assert ctrl.config_kwargs.get("flow_feats_source") == "auto"
    assert ctrl.config_kwargs.get("flow_feats_drop_xy") is True
    assert ctrl.runtime_kwargs.get("train_vel_source") == "coupled"
    assert ctrl.runtime_kwargs.get("corrector_coupling") is True
    assert ctrl.runtime_kwargs.get("closed_loop_coupling") is True
    assert "wall_gen_baseline" in str(ctrl.init_ckpt).replace("\\", "/")

    assert geom.config_kwargs.get("geom_feats_rich") is True
    assert flux.config_kwargs.get("flux_stag_feat") is True
    assert mirror.runtime_kwargs.get("augment_mirror_y") is True
    assert float(noise_off.config_kwargs.get("teacher_noise")) == 0.0
    assert float(noise_hi.config_kwargs.get("teacher_noise")) == 0.04
    # Control matches coupled A/B architecture (not GT / kine-only).
    assert ctrl.config_kwargs.get("flow_feats_source") == mat_growth_leg_spec(
        "FS_ab_coupled"
    ).config_kwargs.get("flow_feats_source")


def test_wg_featfix_legs_match_v3_geom_flux_arms():
    """Featfix re-run mirrors phase1 v3 arms 02/03/05/06 on coupled wall-gen stack."""
    mapping = {
        "WG_featfix_01": "WG_sweep_v3_02",
        "WG_featfix_02": "WG_sweep_v3_03",
        "WG_featfix_03": "WG_sweep_v3_05",
        "WG_featfix_04": "WG_sweep_v3_06",
    }
    for fix, old in mapping.items():
        a = mat_growth_leg_spec(fix)
        b = mat_growth_leg_spec(old)
        assert a.config_kwargs == b.config_kwargs
        assert a.runtime_kwargs == b.runtime_kwargs
        assert "wall_gen_baseline" in str(a.init_ckpt).replace("\\", "/")


def test_wg_clotrich_nplus_matches_featfix_03_stack():
    """N+ LOAO reuses featfix_03 geom+flux stack; warm-start from featfix_03 (not wall_gen)."""
    from src.biochem_gnn.mat_growth_simple import (
        WALL_GEN_BATCH_1B_CHALLENGE,
        WALL_GEN_BATCH_1B_EXCLUDE,
        WALL_GEN_BATCH_1B_NEG_CONTROL,
        WALL_GEN_BATCH_1B_TRAIN,
        WALL_GEN_CLOT_RICH_ANCHORS,
        WG_FEATFIX_03_CKPT,
        wall_gen_clot_rich_train_anchors,
    )

    nplus = mat_growth_leg_spec("WG_clotrich_nplus")
    feat = mat_growth_leg_spec("WG_featfix_03")
    assert nplus.config_kwargs == feat.config_kwargs
    assert nplus.runtime_kwargs == feat.runtime_kwargs
    assert nplus.config_kwargs.get("geom_feats") is True
    assert nplus.config_kwargs.get("geom_feats_rich") is True
    assert nplus.config_kwargs.get("flux_stag_feat") is True
    assert nplus.config_kwargs.get("flow_feats_drop_xy") is True
    assert nplus.config_kwargs.get("flow_feats_source") == "auto"
    assert nplus.runtime_kwargs.get("train_vel_source") == "coupled"
    assert "WG_featfix_03" in str(nplus.init_ckpt).replace("\\", "/")
    assert nplus.init_ckpt == WG_FEATFIX_03_CKPT

    train = wall_gen_clot_rich_train_anchors(holdout="patient020")
    assert "patient020" not in train
    assert "patient002" not in train
    assert "patient023" not in train
    for a in WALL_GEN_BATCH_1B_NEG_CONTROL + WALL_GEN_BATCH_1B_EXCLUDE:
        assert a not in WALL_GEN_CLOT_RICH_ANCHORS
    for a in WALL_GEN_BATCH_1B_TRAIN + WALL_GEN_BATCH_1B_CHALLENGE:
        assert a in WALL_GEN_CLOT_RICH_ANCHORS
    for a in WALL_GEN_BATCH_1B_CHALLENGE:
        assert a not in train  # sealed challenge stays out of default N+ train
    for a in WALL_GEN_BATCH_1B_TRAIN:
        assert a in train
    assert len(train) == len(WALL_GEN_CLOT_RICH_ANCHORS) - 1 - len(WALL_GEN_BATCH_1B_CHALLENGE)
    assert set(train) | {"patient020"} | set(WALL_GEN_BATCH_1B_CHALLENGE) == set(
        WALL_GEN_CLOT_RICH_ANCHORS
    )

    # Opt-out keeps legacy "all clot-rich minus holdout" behaviour.
    train_all = wall_gen_clot_rich_train_anchors(
        holdout="patient020", exclude_sealed_challenge=False
    )
    assert set(train_all) | {"patient020"} == set(WALL_GEN_CLOT_RICH_ANCHORS)

    try:
        wall_gen_clot_rich_train_anchors(holdout="patient034")
        raise AssertionError("expected ValueError for non-clot-rich holdout")
    except ValueError:
        pass

    # Helper allows any clot-rich holdout; launcher blocks featfix_03 train-set leaks.
    assert "patient005" not in wall_gen_clot_rich_train_anchors(holdout="patient005")


def test_wg_clotrich_nplus_v2_mass_gate_and_light_ft():
    """v2 keeps featfix stack but enables mass-gated select + heads-only FT knobs."""
    v1 = mat_growth_leg_spec("WG_clotrich_nplus")
    v2 = mat_growth_leg_spec("WG_clotrich_nplus_v2")
    # Feature stack parity with v1 / featfix_03.
    for k in ("geom_feats", "geom_feats_rich", "flux_stag_feat", "flow_feats_drop_xy", "flow_feats_source"):
        assert v2.config_kwargs.get(k) == v1.config_kwargs.get(k)
    assert v2.config_kwargs.get("mature_fp_exempt") is False
    assert float(v2.config_kwargs.get("gate_fp_weight") or 0) > 0.0
    assert float(v2.config_kwargs.get("closed_loop_init") or 0) >= 0.55
    assert float(v2.config_kwargs.get("final_mass_penalty") or 0) > 0.0
    assert v2.config_kwargs.get("freeze_backbone") is True
    assert int(v2.runtime_kwargs.get("deploy_horizon") or 0) > 0
    assert v2.runtime_kwargs.get("deploy_horizon_all_packs") is False
    assert float(v2.runtime_kwargs.get("select_mass_soft_lambda") or 0) > 0.0
    assert float(v2.runtime_kwargs.get("select_mass_hard_max") or 0) >= 3.0
    assert float(v2.runtime_kwargs.get("select_mat_f1_weight") or 1) <= 0.15
    assert float(v2.runtime_kwargs.get("select_clot_score_weight") or 0) >= 0.85
    assert v2.runtime_kwargs.get("train_vel_source") == "coupled"


def test_wg_stenosis_subcohort_ft_flips_underpred_and_freezes_backbone():
    """s9: mirror of WG_prec_iter/v2 -- underpred UP not down, front-growth select, frozen trunk.

    Zero-shot diagnosis (WALL_MODEL_PLAN.md s9) found this cohort under-seeds (mass 0.653,
    front_speed 0.862, FN=44 vs FP=11 on patient043) -- the opposite failure from the vessels
    that motivated WG_prec_iter's precision tilt. This leg must invert that tilt, not repeat it.
    """
    leg = mat_growth_leg_spec("WG_stenosis_subcohort_ft")
    nplus = mat_growth_leg_spec("WG_clotrich_nplus")

    # Same feature stack / warm-start as N+ -- this is a fine-tune, not a new architecture.
    for k in ("geom_feats", "geom_feats_rich", "flux_stag_feat", "flow_feats_drop_xy", "flow_feats_source"):
        assert leg.config_kwargs.get(k) == nplus.config_kwargs.get(k)
    # Warm-starts FROM the N+ checkpoint's own *output* (not N+'s featfix_03 warm-start).
    from src.biochem_gnn.mat_growth_simple import WG_CLOTRICH_NPLUS_CKPT
    assert leg.init_ckpt == WG_CLOTRICH_NPLUS_CKPT
    assert nplus.init_ckpt != WG_CLOTRICH_NPLUS_CKPT

    # The core inversion: underpred weight >= fp weight (recall-tilted), not precision-tilted.
    underpred = float(leg.config_kwargs.get("underpred_weight") or 0)
    fp = float(leg.config_kwargs.get("fp_weight") or 0)
    assert underpred >= fp, f"expected underpred>=fp (recall tilt), got {underpred}/{fp}"
    assert underpred > 2.0, "must raise underpred above the N+ warm-start's baseline (2.0)"
    assert fp < 8.0, "must lower fp below the N+ warm-start's baseline (8.0)"

    # Light FT only: 5-vessel cohort, protect the warm-start's broader behaviour.
    assert leg.config_kwargs.get("freeze_backbone") is True

    # Selection: strict F1 primary, plus the two panels that target the diagnosed gap.
    assert float(leg.runtime_kwargs.get("select_clot_f1_weight") or 0) > 0.0
    assert float(leg.runtime_kwargs.get("select_front_speed_lambda") or 0) > 0.0
    assert float(leg.runtime_kwargs.get("select_fn_fp_lambda") or 0) > 0.0
    # Guardrail: must not be able to "win" by starving mass further than the zero-shot floor.
    assert float(leg.runtime_kwargs.get("select_mass_hard_min") or 0) > 0.0
    # v1 is the exact historical record of the run that produced deploy_clot_f1=0.522 (s9.9) --
    # no select_mass_hard_max. If this ever fires, v1 no longer matches that recorded run.
    assert not float(leg.runtime_kwargs.get("select_mass_hard_max") or 0)


def test_wg_stenosis_subcohort_ft_v2_fixes_every_v1_root_cause():
    """s9.9: v1 regressed (0.650 -> 0.522) by overshooting into over-seeding. v2 must fix all
    five diagnosed causes without disturbing v1's own spec (kept as the historical record)."""
    v1 = mat_growth_leg_spec("WG_stenosis_subcohort_ft")
    v2 = mat_growth_leg_spec("WG_stenosis_subcohort_ft_v2")

    # v1 itself must be untouched by v2's existence.
    assert float(v1.config_kwargs.get("underpred_weight")) == 4.0
    assert float(v1.config_kwargs.get("fp_weight")) == 4.0
    assert not float(v1.runtime_kwargs.get("select_mass_hard_max") or 0)

    # (1) Half the move, not full parity -- and still on the recall side, not past it.
    underpred = float(v2.config_kwargs.get("underpred_weight") or 0)
    fp = float(v2.config_kwargs.get("fp_weight") or 0)
    assert 2.0 < underpred < 4.0, f"expected a move short of v1's 4.0, got {underpred}"
    assert 4.0 < fp < 8.0, f"expected a move short of v1's 4.0, got {fp}"
    assert underpred < fp, "v2 should not reach v1's full 1:1 parity"

    # (2) Symmetric mass guard -- v1 had only the lower bound.
    assert float(v2.runtime_kwargs.get("select_mass_hard_min") or 0) > 0.0
    hard_max = float(v2.runtime_kwargs.get("select_mass_hard_max") or 0)
    assert hard_max > 0.0, "v1's exact gap: nothing rejected the 2.59 mass blow-up"
    assert hard_max < 2.0, "must actually catch a v1-scale blow-up (mass reached 2.59)"

    # (3) Training-time selection must grade under the SAME gate the final deploy eval uses.
    assert v2.env_overrides.get("CLOT_POCKET_GATE_PCT") == "25"

    # (4) Training windows must be able to start much later than the legacy ~66% cap.
    coverage = float(v2.config_kwargs.get("train_t0_coverage_frac") or 0)
    assert coverage > 0.66, "must exceed the legacy per-vessel formula's effective coverage"

    # (5) Selection must grade more than the single final point, and hard-floor the worst one.
    fracs = str(v2.runtime_kwargs.get("deploy_eval_time_fracs") or "")
    assert len(fracs.split(",")) >= 2, "must grade at least 2 points across the horizon"
    assert float(v2.runtime_kwargs.get("select_f1_min_hard_floor") or 0) > 0.0

    # Same warm start and feature stack as v1 -- this is a hyperparameter fix, not a rebuild.
    assert v2.init_ckpt == v1.init_ckpt
    for k in ("geom_feats", "geom_feats_rich", "flux_stag_feat", "flow_feats_drop_xy"):
        assert v2.config_kwargs.get(k) == v1.config_kwargs.get(k)
    assert v2.config_kwargs.get("freeze_backbone") is True


def test_wg_stenosis_subcohort_ft_v3_is_v2_plus_the_brake_and_nothing_else():
    """s9.11: v3 must be a clean single-mechanism A/B against v2 -- every training-relevant
    knob identical, ONLY the GT-relative time-resolved growth brake added. s9.10's onset
    probe showed the holdout needs MORE growth (recall 0.558, mass 0.674, precision 0.83),
    so v3 must NOT lower v2's recall pressure; the brake is silent below target anyway."""
    v2 = mat_growth_leg_spec("WG_stenosis_subcohort_ft_v2")
    v3 = mat_growth_leg_spec("WG_stenosis_subcohort_ft_v3")
    prec_iter = mat_growth_leg_spec("WG_prec_iter")

    # v2 had every brake at zero (this is the exact bug s9.10 diagnosed) -- lock that in so a
    # future edit to v2 can't silently "fix" it there instead of in a new leg.
    assert not float(v2.config_kwargs.get("step_mass_penalty") or 0)
    assert not float(v2.config_kwargs.get("final_mass_penalty") or 0)
    assert v2.config_kwargs.get("mature_fp_exempt") is True

    brake = {"step_mass_penalty", "step_prec_fp_penalty", "final_mass_penalty",
             "final_mass_target", "final_prec_fp_penalty", "mature_fp_exempt"}

    # THE core assertion: config differs from v2 ONLY by the brake. Anything else appearing
    # here means v3 stopped being an attributable A/B.
    diffs = {
        k for k in set(v2.config_kwargs) | set(v3.config_kwargs)
        if v2.config_kwargs.get(k) != v3.config_kwargs.get(k)
    }
    assert diffs <= brake, f"v3 diverges from v2 outside the brake: {sorted(diffs - brake)}"
    assert diffs, "v3 must actually add the brake"

    # Brake values are WG_prec_iter's own (borrowed, validated machinery -- not reinvented).
    for k in ("step_mass_penalty", "step_prec_fp_penalty", "final_mass_penalty",
              "final_mass_target", "final_prec_fp_penalty"):
        assert v3.config_kwargs.get(k) == prec_iter.config_kwargs.get(k), k
    assert v3.config_kwargs.get("mature_fp_exempt") is False

    # Recall pressure and freezing must match v2 exactly (s9.11: 4/5 vessels incl. the holdout
    # are under-grown, and the holdout's location is already right -- so no reason to change
    # either, and changing them would break attribution).
    assert v3.config_kwargs.get("underpred_weight") == v2.config_kwargs.get("underpred_weight") == 3.0
    assert v3.config_kwargs.get("fp_weight") == v2.config_kwargs.get("fp_weight") == 6.0
    assert v3.config_kwargs.get("freeze_backbone") is True

    # Carried forward from v2 unchanged.
    assert v3.env_overrides == v2.env_overrides
    assert v3.env_overrides.get("CLOT_POCKET_GATE_PCT") == "25"
    assert v3.init_ckpt == v2.init_ckpt
    for k in ("select_mass_hard_min", "select_mass_hard_max", "select_f1_min_hard_floor",
              "deploy_eval_time_fracs", "select_clot_f1_weight"):
        assert v3.runtime_kwargs.get(k) == v2.runtime_kwargs.get(k), k

    # Selection-only change (does not enter the training gradient, so it cannot confound the
    # brake A/B): symmetric terms replace the two confirmed-dead one-sided ones.
    assert float(v3.runtime_kwargs.get("select_front_speed_target_lambda") or 0) > 0.0
    assert float(v3.runtime_kwargs.get("select_fp_fn_imbalance_lambda") or 0) > 0.0
    assert not float(v3.runtime_kwargs.get("select_front_speed_lambda") or 0)
    assert not float(v3.runtime_kwargs.get("select_fn_fp_lambda") or 0)


def test_wg_stenosis_subcohort_ft_v4_restores_fp_weight_only():
    """s9.12: fp_weight splits every leg by observed mass -- 16.0 controls it (warm-start
    0.674, prec_iter 1.109), 4-6 blows up (v1 4.200, v2 ~4.02, v3 4.032). v1 cut it and
    v2/v3 inherited the cut. v4 restores it and changes NOTHING else, so v3-vs-v4 is a
    clean single-variable test of fp_weight itself."""
    v3 = mat_growth_leg_spec("WG_stenosis_subcohort_ft_v3")
    v4 = mat_growth_leg_spec("WG_stenosis_subcohort_ft_v4")
    nplus = mat_growth_leg_spec("WG_clotrich_nplus")
    prec_iter = mat_growth_leg_spec("WG_prec_iter")

    cfg_diff = {
        k for k in set(v3.config_kwargs) | set(v4.config_kwargs)
        if v3.config_kwargs.get(k) != v4.config_kwargs.get(k)
    }
    assert cfg_diff == {"fp_weight"}, f"v4 must differ from v3 by fp_weight alone, got {sorted(cfg_diff)}"
    assert v3.runtime_kwargs == v4.runtime_kwargs
    assert v3.env_overrides == v4.env_overrides
    assert v3.init_ckpt == v4.init_ckpt

    # 16.0 is the recipe baseline the warm-start and prec_iter actually train at -- NOT
    # PushforwardConfig's bare 8.0 dataclass default, which is what the s9.10 doc error said
    # and what caused v1 to design a "mild" cut that was really 2.7-4x.
    assert v4.config_kwargs.get("fp_weight") == 16.0
    assert float(MAT_GROWTH_SIMPLE_RECIPE["SPECIES_CONTINUOUS_FP_WEIGHT"]) == 16.0
    # Neither the warm-start nor prec_iter overrides it -- both inherit that 16.0 baseline.
    assert "fp_weight" not in nplus.config_kwargs
    assert "fp_weight" not in prec_iter.config_kwargs

    # v4 keeps the brake so its effect stays readable against v2 as well.
    assert v4.config_kwargs.get("step_mass_penalty") == 0.75
    assert v4.config_kwargs.get("mature_fp_exempt") is False


def test_wall_gen_stenosis_subcohort_anchors_and_helper():
    """s9 sub-cohort split is deliberately NOT the sealed WALL_GEN_BATCH_1B_* split."""
    from src.biochem_gnn.mat_growth_simple import (
        WALL_GEN_BATCH_1B_CHALLENGE,
        WALL_GEN_BATCH_1B_EXCLUDE,
        WALL_GEN_STENOSIS_SUBCOHORT,
        wall_gen_stenosis_subcohort_train_anchors,
    )

    assert set(WALL_GEN_STENOSIS_SUBCOHORT) == {
        "patient039", "patient040", "patient041", "patient042", "patient043", "patient044",
    }
    # The deliberate departures from the sealed split, spelled out so a future edit can't
    # silently re-converge the two without someone noticing in a diff.
    assert set(WALL_GEN_BATCH_1B_EXCLUDE) & set(WALL_GEN_STENOSIS_SUBCOHORT) == {"patient039"}
    assert set(WALL_GEN_BATCH_1B_CHALLENGE) <= set(WALL_GEN_STENOSIS_SUBCOHORT)

    train = wall_gen_stenosis_subcohort_train_anchors(holdout="patient043")
    assert "patient043" not in train
    assert set(train) == {"patient039", "patient040", "patient041", "patient042", "patient044"}

    try:
        wall_gen_stenosis_subcohort_train_anchors(holdout="patient020")
        raise AssertionError("expected ValueError for a holdout outside the sub-cohort")
    except ValueError:
        pass


def test_wg_prec_iter_binds_step_mass_on_small_cohort_stack():
    """Prec-iter: step+final mass/FP, no freeze, teacher_fp off, mass-gated select."""
    from src.biochem_gnn.mat_growth_simple import WALL_GEN_SMALL_TRAIN_ANCHORS

    leg = mat_growth_leg_spec("WG_prec_iter")
    feat = mat_growth_leg_spec("WG_featfix_03")
    for k in ("geom_feats", "geom_feats_rich", "flux_stag_feat", "flow_feats_drop_xy"):
        assert leg.config_kwargs.get(k) == feat.config_kwargs.get(k)
    assert leg.config_kwargs.get("mature_fp_exempt") is False
    assert float(leg.config_kwargs["teacher_fp_frac"]) == 0.0
    assert float(leg.config_kwargs.get("step_mass_penalty") or 0) > 0.0
    assert float(leg.config_kwargs.get("step_prec_fp_penalty") or 0) > 0.0
    assert float(leg.config_kwargs.get("final_mass_penalty") or 0) >= 1.0
    assert leg.config_kwargs.get("freeze_backbone") is False
    assert float(leg.runtime_kwargs.get("select_mass_soft_lambda") or 0) > 0.0
    assert float(leg.runtime_kwargs.get("select_mass_hard_max") or 0) >= 3.0
    assert set(WALL_GEN_SMALL_TRAIN_ANCHORS) == {
        "patient005",
        "patient006",
        "patient010",
    }


def test_wg_prec_mirror_and_sites_share_prec_loss():
    """Mirror/sites reuse prec_iter loss; mirror adds augment_mirror_y only."""
    base = mat_growth_leg_spec("WG_prec_iter")
    mirror = mat_growth_leg_spec("WG_prec_mirror")
    sites = mat_growth_leg_spec("WG_prec_sites")
    for k in (
        "step_mass_penalty",
        "step_prec_fp_penalty",
        "final_mass_penalty",
        "final_prec_fp_penalty",
        "teacher_fp_frac",
        "freeze_backbone",
        "geom_feats",
        "flux_stag_feat",
    ):
        assert mirror.config_kwargs.get(k) == base.config_kwargs.get(k)
        assert sites.config_kwargs.get(k) == base.config_kwargs.get(k)
    assert mirror.runtime_kwargs.get("augment_mirror_y") is True
    assert sites.runtime_kwargs.get("augment_mirror_y") in (None, False)
    assert float(sites.runtime_kwargs.get("select_mass_soft_lambda") or 0) > 0.0


def test_wg_prec_mid_and_ft_legs():
    """Mid shares prec loss; FT tightens mass/FP and warm-starts from prec_iter."""
    from src.biochem_gnn.mat_growth_simple import (
        WALL_GEN_MID_TRAIN_ANCHORS,
        WG_PREC_ITER_CKPT,
    )

    mid = mat_growth_leg_spec("WG_prec_mid")
    ft = mat_growth_leg_spec("WG_prec_ft")
    base = mat_growth_leg_spec("WG_prec_iter")
    assert mid.init_ckpt == WG_PREC_ITER_CKPT
    assert ft.init_ckpt == WG_PREC_ITER_CKPT
    assert mid.config_kwargs.get("step_mass_penalty") == base.config_kwargs.get(
        "step_mass_penalty"
    )
    assert float(ft.config_kwargs["step_mass_penalty"]) > float(
        base.config_kwargs["step_mass_penalty"]
    )
    assert float(ft.config_kwargs["final_mass_penalty"]) > float(
        base.config_kwargs["final_mass_penalty"]
    )
    assert set(WALL_GEN_MID_TRAIN_ANCHORS) >= {
        "patient005",
        "patient006",
        "patient010",
        "patient001",
        "patient007",
        "patient012",
    }


def test_wg_prec_loao_uses_tight_mass_and_hard_cap():
    """LOAO: tighter mass/FP than prec_iter; hard mass max <= 2.5 for spray abort."""
    base = mat_growth_leg_spec("WG_prec_iter")
    loao = mat_growth_leg_spec("WG_prec_loao")
    assert float(loao.config_kwargs["step_mass_penalty"]) > float(
        base.config_kwargs["step_mass_penalty"]
    )
    assert float(loao.config_kwargs["final_mass_penalty"]) >= 2.0
    assert float(loao.runtime_kwargs.get("select_mass_hard_max") or 99) <= 2.5
    assert loao.config_kwargs.get("freeze_backbone") is False
    assert loao.init_ckpt.endswith("WG_prec_iter/best.pth")


def test_wg_prec_front_raises_underpred_eases_fp_selects_front():
    """Front FT: underpred up, gate/step FP down, no hard mask/seed_aux; select front+FN."""
    from src.architecture.pushforward_config import PushforwardConfig
    from src.architecture.runtime_config import BiochemRuntimeConfig
    from src.biochem_gnn.mat_growth_simple import (
        WG_PREC_ITER_CKPT,
        get_mat_growth_config_kwargs,
        get_mat_growth_runtime_kwargs,
    )

    base = mat_growth_leg_spec("WG_prec_iter")
    front = mat_growth_leg_spec("WG_prec_front")
    assert front.init_ckpt == WG_PREC_ITER_CKPT
    assert float(front.config_kwargs["underpred_weight"]) > float(
        base.config_kwargs["underpred_weight"]
    )
    assert float(front.config_kwargs["gate_fp_weight"]) < float(
        base.config_kwargs["gate_fp_weight"]
    )
    assert float(front.config_kwargs["step_prec_fp_penalty"]) < float(
        base.config_kwargs["step_prec_fp_penalty"]
    )
    assert int(front.config_kwargs.get("frontier_hops") or 0) == 0
    assert float(front.config_kwargs.get("nucleation_topk") or 0.0) == 0.0
    assert float(front.config_kwargs.get("seed_aux_weight") or 0.0) == 0.0
    assert float(front.runtime_kwargs.get("select_front_speed_lambda") or 0) > 0.0
    assert float(front.runtime_kwargs.get("select_fn_fp_lambda") or 0) > 0.0
    assert float(front.runtime_kwargs.get("select_seed_prec_lambda") or 0) == 0.0
    # Typed path: kwargs must bind onto active dataclasses (env cleared).
    cfg = PushforwardConfig.from_meta({"config_kwargs": get_mat_growth_config_kwargs("WG_prec_front")})
    rt = BiochemRuntimeConfig.from_meta(
        {"runtime_kwargs": get_mat_growth_runtime_kwargs("WG_prec_front")}
    )
    assert cfg.underpred_weight == 3.0
    assert cfg.gate_fp_weight == 3.0
    assert cfg.seed_aux_weight == 0.0
    assert rt.scoring.select_front_speed_lambda == 0.10
    assert rt.scoring.select_fn_fp_lambda == 0.10


def test_from_meta_preserves_flux_stag_feat():
    """Flat meta + config_kwargs must round-trip flux_stag (eval rebuilds band width)."""
    from src.architecture.pushforward_config import PushforwardConfig
    from src.biochem_gnn.mat_growth_simple import get_mat_growth_config_kwargs

    meta = {
        "config_kwargs": get_mat_growth_config_kwargs("WG_featfix_02"),
        "flux_stag_feat": True,
        "flow_feats": True,
        "dual_head": True,
    }
    cfg = PushforwardConfig.from_meta(meta)
    assert cfg.flux_stag_feat is True
    assert cfg.geom_feats is False

    # Flat-only path (legacy ckpts without config_kwargs flux).
    cfg2 = PushforwardConfig.from_meta({"flux_stag_feat": True, "flow_feats": True})
    assert cfg2.flux_stag_feat is True


def test_wg_featfix_expected_in_dim_widens_vs_control():
    """Featfix arms must grow GNN in_dim vs FS_ab_coupled / v3 control (the prior bug)."""
    from dataclasses import fields, replace

    from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
    from src.biochem_gnn.mat_growth_simple import get_mat_growth_config_kwargs
    from src.core_physics.species_pushforward_continuous import continuous_feature_dim
    from src.core_physics.species_pushforward_gnn import (
        FLOW_FEATS_DIM,
        FLUX_STAG_DIM,
        GEOM_FEATS_RICH_DIM,
        band_extra_feature_dim,
    )

    field_names = {f.name for f in fields(PushforwardConfig)}

    def _cfg(leg: str) -> PushforwardConfig:
        kw = get_mat_growth_config_kwargs(leg)
        return replace(PushforwardConfig(), **{k: v for k, v in kw.items() if k in field_names})

    with use_pushforward_config(_cfg("WG_sweep_v3_01")):
        d0 = continuous_feature_dim(256)
        assert band_extra_feature_dim() == FLOW_FEATS_DIM
    with use_pushforward_config(_cfg("WG_featfix_01")):
        assert continuous_feature_dim(256) == d0 + GEOM_FEATS_RICH_DIM
    with use_pushforward_config(_cfg("WG_featfix_02")):
        assert continuous_feature_dim(256) == d0 + FLUX_STAG_DIM
    with use_pushforward_config(_cfg("WG_featfix_03")):
        assert continuous_feature_dim(256) == d0 + GEOM_FEATS_RICH_DIM + FLUX_STAG_DIM


def test_continuous_bundle_restores_flow_dynamic_and_channels(monkeypatch, tmp_path):
    """WC/WE eval must rebuild the same state width and dynamic-flow path as training."""
    from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config

    monkeypatch.delenv("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS", raising=False)
    monkeypatch.delenv("SPECIES_FLOW_FEATS_DYNAMIC", raising=False)
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    in_dim = continuous_feature_dim(8)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16, out_dim=2)
    ckpt = tmp_path / "mat_th_meta.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(model.in_dim),
            "hidden": int(model.hidden),
            "meta": {
                "dual_head": True,
                "pushforward_species_scope": "mat",
                "pushforward_species_channels": [11, 5],
                "flow_feats": True,
                "flow_dynamic": True,
            },
        },
        ckpt,
    )
    bundle = load_continuous_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    assert bundle.model.out_dim == 2
    # Architecture comes from typed meta -- not os.environ injection.
    assert os.environ.get("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS") in (None, "")
    assert os.environ.get("SPECIES_FLOW_FEATS_DYNAMIC") in (None, "")
    cfg = PushforwardConfig.from_meta(
        {
            "dual_head": True,
            "pushforward_species_scope": "mat",
            "pushforward_species_channels": [11, 5],
            "flow_feats": True,
            "flow_dynamic": True,
        }
    )
    assert cfg.channels == (11, 5)
    assert cfg.flow_feats_dynamic is True
    with use_pushforward_config(cfg):
        assert pushforward_state_bulk_indices() == [11, 5]


def test_continuous_bundle_restores_sparse_front_meta(monkeypatch, tmp_path):
    """Sparse-front deploy knobs are checkpoint metadata, not caller-side tribal knowledge."""
    from src.architecture.pushforward_config import PushforwardConfig

    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DUAL_HEAD", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE", "1")
    in_dim = continuous_feature_dim(8)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16)
    ckpt = tmp_path / "sparse_front_meta.pth"
    meta = {
        "dual_head": True,
        "pushforward_species_scope": "mat",
        "neighbor_commit_gate": True,
        "neighbor_commit_alpha": 0.7,
        "gate_temp": 0.5,
        "frontier_hops": 1,
        "nucleation_topk": 0.05,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(model.in_dim),
            "hidden": int(model.hidden),
            "meta": meta,
        },
        ckpt,
    )
    for k in (
        "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE",
        "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA",
        "SPECIES_CONTINUOUS_GATE_TEMP",
        "SPECIES_CONTINUOUS_FRONTIER_HOPS",
        "SPECIES_CONTINUOUS_NUCLEATION_TOPK",
    ):
        monkeypatch.delenv(k, raising=False)

    bundle = load_continuous_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    cfg = PushforwardConfig.from_meta(meta)
    assert cfg.species_scope == "mat"
    assert cfg.neighbor_commit_gate is True
    assert cfg.neighbor_commit_alpha == 0.7
    assert cfg.gate_temp == 0.5
    assert cfg.frontier_hops == 1
    assert cfg.nucleation_topk == 0.05
    # No architecture env injection from load.
    assert os.environ.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") in (None, "")


def test_init_load_skips_meta_env_when_requested(monkeypatch, tmp_path):
    """Warm-start from fi_mat ckpt must not overwrite an active Mat-only leg recipe."""
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DUAL_HEAD", "1")
    in_dim = continuous_feature_dim(8)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16, out_dim=1)
    ckpt = tmp_path / "fimat_init.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(in_dim),
            "hidden": 16,
            "meta": {
                "dual_head": True,
                "pushforward_species_scope": "fi_mat",
                "saturation_gate": True,
            },
        },
        ckpt,
    )
    bundle = load_continuous_bundle(ckpt, device=torch.device("cpu"), quiet=True, apply_meta_env=False)
    assert bundle is not None
    assert os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] == "mat"
    assert os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] == "1"


def test_backbone_warm_start_copies_conv_only():
    apply_mat_growth_simple_recipe_env(force=True)
    latent_dim = 8
    in_dim = continuous_feature_dim(latent_dim)
    dual = SpeciesDualHeadContinuousGNN(in_dim, hidden=16)
    single = build_continuous_gnn(in_dim, hidden=16)
    dev = torch.device("cpu")
    n = init_mat_single_from_fimat_ckpt(
        single,
        _write_dual_ckpt(dual),
        device=dev,
        mode="backbone",
        quiet=True,
    )
    assert n >= 9
    assert torch.allclose(single.conv1.lin_l.weight, dual.conv1.lin_l.weight)
    assert not torch.allclose(single.readout[2].weight, dual.magnitude_head[2].weight[:1])


def test_single_head_out_dim_matches_mat_scope():
    apply_mat_growth_simple_recipe_env(force=True)
    latent_dim = 8
    in_dim = continuous_feature_dim(latent_dim)
    model = build_continuous_gnn(in_dim, hidden=16)
    x = torch.randn(4, in_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    y = model(x, edge_index)
    assert y.shape == (4, 1)


def test_gelation_readout_embeds_mat_only_state():
    """Mat-only pushforward state must not assume fi_mat (STATE_DIM=2) in physics readout."""
    from src.core_physics.species_gelation_readout import band_log_state_to_species12

    os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = "mat"
    rest = torch.zeros(5, 12)
    rest[:, 4:8] = 0.1
    log_state = torch.tensor([0.2, 0.5, 0.8, 1.1, 0.3])
    sp12 = band_log_state_to_species12(log_state, rest)
    assert sp12.shape == (5, 12)
    assert torch.allclose(sp12[:, MAT_CHANNEL], log_state)
    assert torch.allclose(sp12[:, 4:8], rest[:, 4:8])
