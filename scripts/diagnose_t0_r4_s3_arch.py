"""Architecture diagnostics for Rung4 s2/s3 localization (patient007 default).

Tests:
  1) Rank barrier: can max risk boost pull FN nodes into s0 top-8%?
  2) Oracle ceilings: risk boost vs direct gate patch vs species patch
  3) Trained s3: does tanh(logit) move s0 gate / commits?
  4) Temporal: FN node persistence across macro times

Usage::

    python scripts/diagnose_t0_r4_s3_arch.py
    python scripts/diagnose_t0_r4_s3_arch.py --anchor patient007 --s3-ckpt outputs/biochem/t0_r4_s3_temporal/best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time
from src.core_physics.t0_r4_s2_species import (
    _apply_loc_gate_residual,
    _apply_loc_risk_adjustment,
    _risk_n_at_time,
    _s0_gate_from_species,
    build_s2_features,
    build_s3_features,
    s3_feature_dim,
)
from src.core_physics.t0_r4_s3_temporal import load_s3_bundle, rollout_s3_species_series
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    _s0_spatial_weight,
    resting_species_log_nd,
    rollout_rung4_species_series,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.train_clot_phi_simple import _clot_metrics
from src.training.train_t0_r4_s2_species import _fn_fp_masks
from src.utils.paths import get_project_root


def _commits_at(data, t, phys, bio, device, pred_series):
    with t0_rung2_env():
        phi, _ = predict_clot_phi_at_time(
            data, t, phys, bio, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred_series,
        )
    return (phi.reshape(-1) >= 0.5).bool()


def _f1_at_t(data, t, phys, bio, device, pred_series):
    phi_gt = gt_clot_phi_at_time(data, t, phys, device).reshape(-1)
    phi_p = _commits_at(data, t, phys, bio, device, pred_series)
    m = _clot_metrics(phi_p.float(), phi_gt, torch.ones_like(phi_gt, dtype=torch.bool))
    return float(m["clot_f1"])


@torch.no_grad()
def _s0_context(data, t, phys, bio, device):
    commits_prev = None
    pred = data.y.clone().to(device=device)
    for ti in range(t):
        with t0_rung2_env():
            phi, _ = predict_clot_phi_at_time(
                data, ti, phys, bio, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
            )
        commits_prev = (phi.reshape(-1) >= 0.5).bool()
    elig = resolve_nucleation_eligibility(
        data, t, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
        use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
    ).reshape(-1).bool()
    s0_sp = _build_s0_deploy_species(data, t, device, bio, elig=elig, commits_prev=commits_prev)
    gate = _s0_gate_from_species(s0_sp, data, device, bio, elig)
    phi_gt = gt_clot_phi_at_time(data, t, phys, device).reshape(-1)
    fn, fp = _fn_fp_masks(s0_sp, data.y[t, :, 4:16], phi_gt, elig, gate)
    risk_n = _risk_n_at_time(data, t, device, bio, elig=elig)
    onset = float(_s0_onset_factor(float(macro_tau_at_index(data, t, bio_cfg=bio))))
    return {
        "t": t,
        "commits_prev": commits_prev,
        "elig": elig,
        "s0_sp": s0_sp,
        "gate": gate,
        "fn": fn,
        "fp": fp,
        "risk_n": risk_n,
        "onset": onset,
        "pred_prefix": pred,
    }


def _rank_rescue_stats(risk_n: torch.Tensor, elig: torch.Tensor, fn: torch.Tensor, loc_scale: float) -> dict:
    """How many FN nodes enter top-8% if boosted by max tanh (+1)?"""
    rn = risk_n.reshape(-1).float()
    e = elig.reshape(-1).bool()
    spatial_base = _s0_spatial_weight(rn, e)
    thr_q = spatial_base[e] > 0
    n_hot = int(thr_q.sum().item())
    max_boost = 1.0 + float(loc_scale) * 1.0
    rn_max = rn.clone()
    fn_b = fn.reshape(-1).bool()
    rn_max = torch.where(fn_b, rn * max_boost, rn_max)
    spatial_max = _s0_spatial_weight(rn_max, e)
    rescued = fn_b & (spatial_max > 0)
    return {
        "n_elig": int(e.sum().item()),
        "n_hot_top_frac": n_hot,
        "n_fn": int(fn_b.sum().item()),
        "n_fn_rescued_by_max_boost": int(rescued.sum().item()),
        "max_boost_factor": max_boost,
    }


@torch.no_grad()
def _species_from_gate(
    data,
    t: int,
    device,
    bio,
    *,
    elig: torch.Tensor,
    commits_prev,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Build species with explicit per-node gate (bypasses risk rank path)."""
    from src.core_physics.t0_rung4_ladder import (
        _log1p_nd_for_fi_si,
        _log1p_nd_for_mat_si,
        _s0_fi_mat_gain,
    )

    sp = resting_species_log_nd(data, device)
    gain = _s0_fi_mat_gain()
    fi_tgt = _log1p_nd_for_fi_si(float(bio.viscosity_fi_crit) * gain, bio, device)
    mat_tgt = _log1p_nd_for_mat_si(float(bio.viscosity_mat_crit) * gain, bio)
    g = gate.reshape(-1).clamp(0.0, 1.0)
    e = elig.reshape(-1).bool()
    g = torch.where(e, g, torch.zeros_like(g))
    fi_rest = sp[:, FI_SLICE_IDX]
    mat_rest = sp[:, MAT_SLICE_IDX]
    sp = sp.clone()
    sp[:, FI_SLICE_IDX] = fi_rest + g * (fi_tgt - fi_rest)
    sp[:, MAT_SLICE_IDX] = mat_rest + g * (mat_tgt - mat_rest)
    return sp


@torch.no_grad()
def _rollout_gate_oracle(data, phys, bio, device, *, mode: str, loc_scale: float = 1.5) -> torch.Tensor:
    out = data.y.clone().to(device=device)
    commits_prev = None
    rest = resting_species_log_nd(data, device)
    for t in range(int(data.y.shape[0])):
        ctx = _s0_context(data, t, phys, bio, device)
        elig, s0_sp, gate, fn, fp = ctx["elig"], ctx["s0_sp"], ctx["gate"], ctx["fn"], ctx["fp"]
        sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
        g = gate.clone()
        if mode == "gate_fn_fp":
            g[fn] = torch.clamp(g[fn] + float(loc_scale) * 0.5, 0.0, 1.0)
            g[fp] = torch.clamp(g[fp] - float(loc_scale) * 0.5, 0.0, 1.0)
        elif mode == "gate_fn_only":
            g[fn] = torch.clamp(g[fn] + float(loc_scale) * 0.5, 0.0, 1.0)
        elif mode == "risk_max_fn":
            risk = ctx["risk_n"]
            logit = torch.zeros_like(risk)
            logit[fn] = 10.0  # tanh ~ 1
            risk_adj = _apply_loc_risk_adjustment(
                risk, logit, elig, onset=ctx["onset"], loc_scale=loc_scale,
            )
            sp = _build_s0_deploy_species(
                data, t, device, bio, elig=elig, commits_prev=commits_prev,
                risk_n_override=risk_adj,
            )
            out[t, :, 4:16] = sp
            commits_prev = _commits_at(data, t, phys, bio, device, out)
            continue
        elif mode == "species_fn_fp":
            sp = s0_sp.clone()
            sp[fn] = sp_gt[fn]
            sp[fp, FI_SLICE_IDX] = rest[fp, FI_SLICE_IDX]
            sp[fp, MAT_SLICE_IDX] = rest[fp, MAT_SLICE_IDX]
            out[t, :, 4:16] = sp
            commits_prev = _commits_at(data, t, phys, bio, device, out)
            continue
        sp = _species_from_gate(
            data, t, device, bio, elig=elig, commits_prev=commits_prev, gate=g,
        )
        out[t, :, 4:16] = sp
        commits_prev = _commits_at(data, t, phys, bio, device, out)
    return out


@torch.no_grad()
def _s3_gate_delta_at_t(data, t, phys, bio, device, bundle) -> dict:
    from src.core_physics.t0_r4_s3_temporal import S3_ACTUATOR_GATE

    ctx = _s0_context(data, t, phys, bio, device)
    elig = ctx["elig"]
    s0_sp = ctx["s0_sp"]
    gate_s0 = ctx["gate"]
    n = int(data.num_nodes)
    h = bundle.model.init_hidden(n, device, torch.float32)
    phi_prev = None
    commits_prev = None
    for ti in range(t + 1):
        el = resolve_nucleation_eligibility(
            data, ti, device, phys, bio, commits_prev=commits_prev, growth_seed="pred",
            use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
        ).reshape(-1).bool()
        s0 = _build_s0_deploy_species(data, ti, device, bio, elig=el, commits_prev=commits_prev)
        g0 = _s0_gate_from_species(s0, data, device, bio, el)
        if bundle.in_dim >= s3_feature_dim():
            ft = build_s3_features(
                data, ti, device, bio, elig=el, s0_species=s0, s0_gate=g0,
                commits_prev=commits_prev, phi_prev=phi_prev,
            )
        else:
            ft = build_s2_features(data, ti, device, bio, elig=el, s0_species=s0, s0_gate=g0)
        logit, h = bundle.model.forward_step(ft, h, res_scale=bundle.res_scale)
        if ti < t:
            pred = data.y.clone().to(device=device)
            pred[ti, :, 4:16] = s0  # advance commits only
            with t0_rung2_env():
                phi, _ = predict_clot_phi_at_time(
                    data, ti, phys, bio, device,
                    gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=pred,
                )
            phi_prev = phi.reshape(-1).clamp(0.0, 1.0)
            commits_prev = (phi.reshape(-1) >= 0.5).bool()

    onset = ctx["onset"]
    tnh = torch.tanh(logit.reshape(-1) * onset)
    if getattr(bundle, "actuator", S3_ACTUATOR_GATE) == S3_ACTUATOR_GATE:
        gate_adj = _apply_loc_gate_residual(
            gate_s0, logit, elig, onset=onset, loc_scale=bundle.loc_scale
        )
    else:
        risk_adj = _apply_loc_risk_adjustment(
            ctx["risk_n"], logit, elig, onset=onset, loc_scale=bundle.loc_scale,
        )
        sp_risk = _build_s0_deploy_species(
            data, t, device, bio, elig=elig, commits_prev=ctx["commits_prev"],
            risk_n_override=risk_adj,
        )
        gate_adj = _s0_gate_from_species(sp_risk, data, device, bio, elig)
    fn, fp = ctx["fn"], ctx["fp"]
    hot_s0 = gate_s0 > 0.25
    hot_adj = gate_adj > 0.25
    return {
        "n_fn_flip_hot": int((fn & ~hot_s0 & hot_adj).sum().item()),
        "n_fp_flip_cold": int((fp & hot_s0 & ~hot_adj).sum().item()),
        "mean_tnh_fn": float(tnh[fn].mean().item()) if bool(fn.any()) else 0.0,
        "mean_tnh_fp": float(tnh[fp].mean().item()) if bool(fp.any()) else 0.0,
        "mean_dgate_fn": float((gate_adj - gate_s0)[fn].mean().item()) if bool(fn.any()) else 0.0,
        "mean_dgate_fp": float((gate_adj - gate_s0)[fp].mean().item()) if bool(fp.any()) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung4 s3 architecture diagnostics")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--s3-ckpt", default="outputs/biochem/t0_r4_s3_temporal/best.pth")
    ap.add_argument("--loc-scale", type=float, default=1.5)
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph, map_location=device, weights_only=False)
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    t_last = int(data.y.shape[0]) - 1

    print(f"[i] anchor={args.anchor} device={device} t_last={t_last}", flush=True)

    ctx = _s0_context(data, t_last, phys, bio, device)
    rank = _rank_rescue_stats(ctx["risk_n"], ctx["elig"], ctx["fn"], args.loc_scale)
    print("[1] rank barrier (s0 top-8% after max FN boost):", flush=True)
    print(f"    elig={rank['n_elig']} hot={rank['n_hot_top_frac']} fn={rank['n_fn']} "
          f"fn_rescued={rank['n_fn_rescued_by_max_boost']} boost={rank['max_boost_factor']:.2f}", flush=True)

    print("[2] oracle rollouts (final-t F1):", flush=True)
    modes = [
        ("s0", lambda: rollout_rung4_species_series(data, phys, bio, device, step="s0")),
        ("risk_max_fn", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="risk_max_fn", loc_scale=args.loc_scale)),
        ("gate_fn_only", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="gate_fn_only", loc_scale=args.loc_scale)),
        ("gate_fn_fp", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="gate_fn_fp", loc_scale=args.loc_scale)),
        ("species_fn_fp", lambda: _rollout_gate_oracle(data, phys, bio, device, mode="species_fn_fp")),
    ]
    for name, fn in modes:
        pred = fn()
        f1 = _f1_at_t(data, t_last, phys, bio, device, pred)
        print(f"    {name:16s} F1={f1:.3f}", flush=True)

    print("[3] FN persistence across times (s0, in E(t)):", flush=True)
    fn_sets: list[set[int]] = []
    for t in [7, 15, 22, 27, 40, t_last]:
        c = _s0_context(data, t, phys, bio, device)
        ids = set(torch.where(c["fn"])[0].tolist())
        fn_sets.append(ids)
        print(f"    t={t:2d} fn={len(ids)}", flush=True)
    if fn_sets:
        inter = set.intersection(*fn_sets)
        union = set.union(*fn_sets)
        print(f"    persistent FN (all times): {len(inter)} / union {len(union)}", flush=True)

    ckpt = root / args.s3_ckpt
    if ckpt.is_file():
        bundle = load_s3_bundle(ckpt, device=device, quiet=True)
        assert bundle is not None
        meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8")).get("meta", {})
        print(f"[4] trained s3 ({ckpt.name}, ep={meta.get('epoch', '?')}):", flush=True)
        pred_s3 = rollout_s3_species_series(data, phys, bio, device, bundle)
        f1_s3 = _f1_at_t(data, t_last, phys, bio, device, pred_s3)
        pred_s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
        f1_s0 = _f1_at_t(data, t_last, phys, bio, device, pred_s0)
        print(f"    rollout F1 s0={f1_s0:.3f} s3={f1_s3:.3f}", flush=True)
        for t in [27, 40, t_last]:
            d = _s3_gate_delta_at_t(data, t, phys, bio, device, bundle)
            print(
                f"    t={t} fn_flip_hot={d['n_fn_flip_hot']} fp_flip_cold={d['n_fp_flip_cold']} "
                f"tnh_fn={d['mean_tnh_fn']:.3f} tnh_fp={d['mean_tnh_fp']:.3f} "
                f"dgate_fn={d['mean_dgate_fn']:.4f} dgate_fp={d['mean_dgate_fp']:.4f}",
                flush=True,
            )
    else:
        print(f"[4] skip trained s3 (missing {ckpt})", flush=True)

    print("[OK] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
