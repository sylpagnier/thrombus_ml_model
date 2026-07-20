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
    apply_mat_growth_simple_recipe_env(force=True)
    assert os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] == "mat"
    assert os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] == "0"
    assert os.environ["SPECIES_CONTINUOUS_CLOUT_SCORE"] == "relaxed_prec_floor"
    assert float(os.environ["SPECIES_CONTINUOUS_FP_WEIGHT"]) >= 16.0
    assert float(os.environ["SPECIES_CONTINUOUS_SCORE_CLOUT_W"]) >= 0.75
    # Train flow block on COMSOL GT (avoid a second GINO-DEQ at pack build).
    assert os.environ["SPECIES_FLOW_FEATS_SOURCE"] == "gt"
    assert MAT_GROWTH_SIMPLE_RECIPE["SPECIES_FLOW_FEATS_SOURCE"] == "gt"
    # wall hops come from the global triangle6 recipe (applied first), not the mat-only overrides.
    assert os.environ["SPECIES_SNAPSHOT_WALL_HOPS"] == GLOBAL_TRAIN_RECIPE["SPECIES_SNAPSHOT_WALL_HOPS"]


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
    assert c.env_overrides.get("SPECIES_GEOM_FEATS") == "1"
    assert d.init_mode == "mat_readout"
    assert e.env_overrides.get("SPECIES_CONTINUOUS_DUAL_HEAD") == "1"
    assert f.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "fi_mat"
    assert g.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert h.env_overrides.get("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT") == "5.0"
    assert i.env_overrides.get("SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI") == "0.15"
    assert j.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert j.env_overrides.get("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT") == "5.0"


def test_precision_sweep_leg_specs():
    """K/L/M/N flip exactly one in-training lever on the dual fi_mat baseline."""
    k = mat_growth_leg_spec("K_fimat_neighbor_gate")
    l = mat_growth_leg_spec("L_fimat_geom_rich")
    m = mat_growth_leg_spec("M_fimat_neighbor_geom_rich")
    n = mat_growth_leg_spec("N_mat_geom_rich")
    # K: neighbour gate kept, but on the dual fi_mat head (not Mat-only).
    assert k.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "fi_mat"
    assert k.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert "SPECIES_GEOM_FEATS_RICH" not in k.env_overrides
    # L: enriched geometry only.
    assert l.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "fi_mat"
    assert l.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE" not in l.env_overrides
    # M: both surviving levers.
    assert m.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert m.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    # N: rich geometry on the Mat-only control.
    assert n.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "mat"
    assert n.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    # O: N + G (mat scope, neighbour gate, rich geom).
    o = mat_growth_leg_spec("O_mat_neighbor_geom_rich")
    assert o.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "mat"
    assert o.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert o.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"


def test_precision_ladder_6h_leg_specs():
    """P/Q/R: pure-scope control + gate-precision levers, all Mat-only dual head."""
    p = mat_growth_leg_spec("P_mat_plain")
    q = mat_growth_leg_spec("Q_mat_gate_sharp_fp")
    r = mat_growth_leg_spec("R_mat_geom_gate_sharp_fp")
    # P: scope only -- no gate, no geom, no sharpening.
    assert p.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "mat"
    assert "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE" not in p.env_overrides
    assert "SPECIES_GEOM_FEATS_RICH" not in p.env_overrides
    assert "SPECIES_CONTINUOUS_GATE_TEMP" not in p.env_overrides
    # Q: gate + sharpening + spatial FP pressure (no geom).
    assert q.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert q.env_overrides.get("SPECIES_CONTINUOUS_GATE_TEMP") == "0.5"
    assert q.env_overrides.get("SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT") == "3.0"
    assert "SPECIES_GEOM_FEATS_RICH" not in q.env_overrides
    # R: Q + rich geometry (kitchen sink of survivors).
    assert r.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert r.env_overrides.get("SPECIES_CONTINUOUS_GATE_TEMP") == "0.5"
    assert r.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"


def test_nucleation_front_leg_specs():
    """U/V/S/T: SeedFrontMat pivot ladder, Mat-only, deployable (pred-state seed)."""
    u = mat_growth_leg_spec("U_mat_frontier_only")
    v = mat_growth_leg_spec("V_mat_frontier_geom")
    s = mat_growth_leg_spec("S_mat_frontier_nuc")
    t = mat_growth_leg_spec("T_mat_frontier_sharp")
    for leg in (u, v, s, t):
        assert leg.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE") == "mat"
        assert leg.env_overrides.get("SPECIES_CONTINUOUS_FRONTIER_HOPS") == "1"
        assert float(leg.env_overrides.get("SPECIES_CONTINUOUS_NUCLEATION_TOPK")) > 0.0
    # U: structural pivot only (no gate, no geom).
    assert "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE" not in u.env_overrides
    assert "SPECIES_GEOM_FEATS_RICH" not in u.env_overrides
    # V: pivot + geom, no gate.
    assert v.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE" not in v.env_overrides
    # S: full SeedFrontMat_v0 (gate + geom).
    assert s.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert s.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    # T additionally sharpens the gate + spatial FP pressure.
    assert t.env_overrides.get("SPECIES_CONTINUOUS_GATE_TEMP") == "0.5"
    assert t.env_overrides.get("SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT") == "3.0"


def test_physical_guided_leg_specs():
    w = mat_growth_leg_spec("W_mat_flow_stagnation")
    x = mat_growth_leg_spec("X_mat_flow_seedfront")
    y = mat_growth_leg_spec("Y_mat_tight_seed")
    ab = mat_growth_leg_spec("AB_mat_gelation_aux")
    assert w.env_overrides.get("SPECIES_FLOW_FEATS") == "1"
    assert x.env_overrides.get("SPECIES_FLOW_FEATS") == "1"
    assert x.env_overrides.get("SPECIES_CONTINUOUS_FRONTIER_HOPS") == "1"
    assert float(y.env_overrides.get("SPECIES_CONTINUOUS_NUCLEATION_TOPK")) == 0.02
    assert ab.env_overrides.get("SPECIES_CONTINUOUS_PHYSICS_READOUT") == "1"


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
        assert leg.env_overrides.get("SPECIES_FLOW_FEATS") == "1"
    assert wa.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"
    assert wb.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert wc.env_overrides.get("SPECIES_FLOW_FEATS_DYNAMIC") == "1"
    assert wd.env_overrides.get("SPECIES_CONTINUOUS_FRONTIER_HOPS") == "1"
    assert float(wd.env_overrides.get("SPECIES_CONTINUOUS_NUCLEATION_TOPK")) == 0.0
    assert we.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS") == "11,5"
    assert wf.env_overrides.get("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS") == "11,7"
    assert wg.env_overrides.get("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT") == "5.0"
    assert wh.env_overrides.get("SPECIES_CONTINUOUS_PHYSICS_READOUT") == "1"
    assert float(wh.env_overrides.get("SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT")) == 0.25
    assert wi.env_overrides.get("SPECIES_GEOM_FEATS_RICH") == "1"
    assert wj.env_overrides.get("SPECIES_FLOW_FEATS_DYNAMIC") == "1"
    assert wj.env_overrides.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") == "1"


def test_eval_ckpt_recipe_is_deploy_faithful(monkeypatch):
    """Mat-growth eval must not inherit GT flow/species pins from training env."""
    from scripts.eval_mat_growth_simple import _apply_ckpt_recipe

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
        },
        label="mat_growth_simple",
    )
    assert os.environ.get("SPECIES_FLOW_FEATS_SOURCE") is None
    assert os.environ["SPECIES_ROLLOUT_DEPLOY_FAITHFUL"] == "1"
    assert os.environ["SPECIES_ROLLOUT_PIN_OTHER"] == "rest"
    assert os.environ["SPECIES_ROLLOUT_IC_SOURCE"] == "resting"
    assert os.environ["SPECIES_ROLLOUT_VEL_SOURCE"] == "kinematics"
    assert os.environ["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] == "11,5"
    assert os.environ["SPECIES_FLOW_FEATS_DYNAMIC"] == "1"
    os.environ.pop("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS", None)
    os.environ.pop("SPECIES_FLOW_FEATS_DYNAMIC", None)


def test_continuous_bundle_restores_flow_dynamic_and_channels(monkeypatch, tmp_path):
    """WC/WE eval must rebuild the same state width and dynamic-flow path as training."""
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
    assert os.environ["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] == "11,5"
    assert os.environ["SPECIES_FLOW_FEATS_DYNAMIC"] == "1"
    assert pushforward_state_bulk_indices() == [11, 5]
    assert bundle.model.out_dim == 2
    # load_continuous_bundle writes os.environ directly (not via monkeypatch).
    os.environ.pop("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS", None)
    os.environ.pop("SPECIES_FLOW_FEATS_DYNAMIC", None)


def test_continuous_bundle_restores_sparse_front_meta(monkeypatch, tmp_path):
    """Sparse-front deploy knobs are checkpoint metadata, not caller-side tribal knowledge."""
    monkeypatch.setenv("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE", "mat")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DUAL_HEAD", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE", "1")
    in_dim = continuous_feature_dim(8)
    model = SpeciesDualHeadContinuousGNN(in_dim, hidden=16)
    ckpt = tmp_path / "sparse_front_meta.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(model.in_dim),
            "hidden": int(model.hidden),
            "meta": {
                "dual_head": True,
                "pushforward_species_scope": "mat",
                "neighbor_commit_gate": True,
                "neighbor_commit_alpha": 0.7,
                "gate_temp": 0.5,
                "frontier_hops": 1,
                "nucleation_topk": 0.05,
            },
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
    assert os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] == "mat"
    assert os.environ["SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE"] == "1"
    assert os.environ["SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA"] == "0.7"
    assert os.environ["SPECIES_CONTINUOUS_GATE_TEMP"] == "0.5"
    assert os.environ["SPECIES_CONTINUOUS_FRONTIER_HOPS"] == "1"
    assert os.environ["SPECIES_CONTINUOUS_NUCLEATION_TOPK"] == "0.05"


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
