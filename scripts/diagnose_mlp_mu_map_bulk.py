"""Diagnose Leg B v2 bulk mu offset (patient007-style).

Compares GT stored mu, Carreau(GT u,v), Carreau(pred u,v), committed v2 mu,
and clot-gate leakage at a chosen time index.

Usage (repo root):
  python scripts/diagnose_mlp_mu_map_bulk.py --anchor patient007
  python scripts/diagnose_mlp_mu_map_bulk.py --anchor patient007 --with-rollout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_mu_inject import (
    committed_mu_mesh_from_clot_model,
    mlp_mu_map_bulk_mode,
    mlp_mu_map_mask_mode,
    mu_map_carreau_baseline_si,
    resolve_clot_trigger_gate,
)
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    carreau_mu_si_from_uv,
    phi_gt_binary,
    sdf_nd_from_data,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.utils.rheology import compute_shear_rate


def _stats(name: str, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    v = x.reshape(-1)
    if mask is not None:
        v = v[mask.reshape(-1)]
    if v.numel() == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "p10": float(v.quantile(0.1)),
        "p50": float(v.quantile(0.5)),
        "p90": float(v.quantile(0.9)),
    }


def _gamma_dot_nd(data, u_nd, v_nd) -> torch.Tensor:
    u = u_nd.reshape(-1, 1).float()
    v = v_nd.reshape(-1, 1).float()
    du_dx = torch.sparse.mm(data.G_x, u)
    du_dy = torch.sparse.mm(data.G_y, u)
    dv_dx = torch.sparse.mm(data.G_x, v)
    dv_dy = torch.sparse.mm(data.G_y, v)
    return compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).reshape(-1)


def _load_clot_model(ckpt: Path, device: torch.device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    model = build_clot_phi_model(
        in_dim=int(cfg.get("in_dim", 6)), hidden=int(cfg.get("hidden", 32))
    ).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


def _maybe_rollout_pred(data, teacher_ckpt: Path, ti: int, device: torch.device) -> torch.Tensor | None:
    try:
        from src.architecture.gnode_biochem import GNODE_Phase3
        from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
        from src.utils.channel_schema import infer_missing_schema
    except ImportError:
        return None

    os.environ.setdefault("BIOCHEM_MLP_MU_MAP", "1")
    os.environ.setdefault("BIOCHEM_MLP_MU_MAP_MASK", "gt_clot")
    raw = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    model = GNODE_Phase3.from_checkpoint_blob(raw, device=device)
    model.eval()
    clot_ckpt = os.environ.get("BIOCHEM_MLP_CLOT_CKPT", "")
    if clot_ckpt:
        attach_clot_phi_injector_to_teacher(model, device, clot_ckpt)
    data = infer_missing_schema(data, phase_hint="biochem").to(device)
    bio = BiochemConfig(phase="biochem")
    t_si = bio.resolve_biochem_times(data, device)
    with torch.no_grad():
        pred, _ = model(data, t_si, detach_macro_state=True)
    idx = min(ti, int(pred.shape[0]) - 1)
    return pred[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose v2 bulk mu offset")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--time-index", type=int, default=-1, help="-1 = last COMSOL step")
    ap.add_argument("--clot-ckpt", default="outputs/biochem/clot_baseline/clot_phi_best.pth")
    ap.add_argument("--teacher-ckpt", default="outputs/biochem/clot_baseline/teacher_best_high_mu.pth")
    ap.add_argument("--with-rollout", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("BIOCHEM_MLP_MU_MAP", "1")
    os.environ.setdefault("BIOCHEM_MLP_MU_MAP_MASK", "gt_clot")
    os.environ.setdefault("BIOCHEM_MLP_MU_MAP_BULK", "cap_low_shear")
    os.environ.setdefault("BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND", "0.01")
    os.environ["BIOCHEM_MLP_CLOT_CKPT"] = str(ROOT / args.clot_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = ROOT / args.graph_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        print(f"[ERR] missing graph: {graph_path}")
        return 1
    clot_path = ROOT / args.clot_ckpt
    if not clot_path.is_file():
        print(f"[ERR] missing clot ckpt: {clot_path}")
        return 1

    data = torch.load(graph_path, map_location=device, weights_only=False)
    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    y = data.y[ti]

    mu_gt = phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).reshape(-1)
    from src.core_physics.clot_phi_mu_inject import resolve_mu_map_baselines_si

    mu_c_raw = carreau_mu_si_from_uv(data, y[:, 0], y[:, 1], phys)
    mu_c_gt, mu_c_mlp = resolve_mu_map_baselines_si(data, y[:, 0], y[:, 1], phys)
    gd_gt = _gamma_dot_nd(data, y[:, 0], y[:, 1])

    step = build_clot_phi_step(data, ti, phys, bio, device, u_nd_override=y[:, 0], v_nd_override=y[:, 1])
    gate = phi_gt_binary(step.mu_gt_cap, step.region, phys).bool()
    sdf = sdf_nd_from_data(data, device, int(data.num_nodes))
    lumen = (sdf > 0.002) & (~gate)
    bulk = ~gate

    clot_model, _ = _load_clot_model(clot_path, device)
    mu_out_gt_uv, mu_c_step, phi, mu_mlp = committed_mu_mesh_from_clot_model(
        clot_model,
        data,
        ti,
        u_nd=y[:, 0],
        v_nd=y[:, 1],
        species_log=y[:, 4:16],
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
    )

    pred_slice = None
    if args.with_rollout:
        t_ckpt = ROOT / args.teacher_ckpt
        if t_ckpt.is_file():
            pred_slice = _maybe_rollout_pred(data, t_ckpt, ti, device)
        else:
            print(f"[WARN] teacher ckpt missing for rollout: {t_ckpt}")

    rows = [
        _stats("GT stored mu (COMSOL panel)", mu_gt, bulk),
        _stats("GT stored mu lumen", mu_gt, lumen),
        _stats("Carreau raw(GT u,v) bulk", mu_c_raw, bulk),
        _stats("Carreau baseline(GT u,v) bulk", mu_c_gt, bulk),
        _stats("Carreau baseline(GT u,v) lumen", mu_c_gt, lumen),
        _stats("v2 committed(GT u,v) bulk", mu_out_gt_uv, bulk),
        _stats("v2 committed(GT u,v) lumen", mu_out_gt_uv, lumen),
        _stats("mu_mlp on clot gate", mu_mlp, gate),
        _stats("v2 committed on clot gate", mu_out_gt_uv, gate),
    ]

    diag = {
        "anchor": args.anchor,
        "time_index": ti,
        "mask_mode": mlp_mu_map_mask_mode(),
        "bulk_mode": mlp_mu_map_bulk_mode(),
        "phys_mu_inf_si": float(phys.mu_inf),
        "phys_mu_0_si": float(phys.mu_0),
        "viz_mu_clim_default": [0.04, 0.10],
        "n_clot_gate": int(gate.sum()),
        "n_bulk": int(bulk.sum()),
        "n_lumen": int(lumen.sum()),
        "gamma_dot_gt_p50": float(gd_gt.quantile(0.5)),
        "gamma_dot_gt_frac_floor": float((gd_gt <= 1.1e-4).float().mean()),
        "frac_carreau_gt_at_mu0": float((mu_c_gt >= phys.mu_0 * 0.99).float().mean()),
        "logMAE_carreau_gt_vs_gt_bulk": float(
            (torch.log(mu_c_gt[bulk].clamp(1e-8)) - torch.log(mu_gt[bulk].clamp(1e-8))).abs().mean()
        ),
        "logMAE_v2_vs_gt_bulk": float(
            (torch.log(mu_out_gt_uv[bulk].clamp(1e-8)) - torch.log(mu_gt[bulk].clamp(1e-8))).abs().mean()
        ),
        "bulk_stats": rows,
        "training_log_match": (
            "K8/K9/K98 class: uniform mu_eff ~0.05-0.06 vs COMSOL bulk ~0.04 on fixed clim "
            "(global Carreau low-shear plateau at mu_0, not GNODE mu head)"
        ),
    }

    if pred_slice is not None:
        mu_c_pr = carreau_mu_si_from_uv(data, pred_slice[:, 0], pred_slice[:, 1], phys)
        mu_out_pr, _, _, _ = committed_mu_mesh_from_clot_model(
            clot_model,
            data,
            ti,
            u_nd=pred_slice[:, 0],
            v_nd=pred_slice[:, 1],
            species_log=pred_slice[:, 4:16],
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
        )
        gd_pr = _gamma_dot_nd(data, pred_slice[:, 0], pred_slice[:, 1])
        diag["gamma_dot_pred_p50"] = float(gd_pr.quantile(0.5))
        diag["logMAE_carreau_pred_vs_gt_bulk"] = float(
            (torch.log(mu_c_pr[bulk].clamp(1e-8)) - torch.log(mu_gt[bulk].clamp(1e-8))).abs().mean()
        )
        diag["logMAE_v2_pred_vs_gt_bulk"] = float(
            (torch.log(mu_out_pr[bulk].clamp(1e-8)) - torch.log(mu_gt[bulk].clamp(1e-8))).abs().mean()
        )
        diag["bulk_stats"].extend(
            [
                _stats("Carreau(pred u,v) bulk", mu_c_pr, bulk),
                _stats("v2 committed(pred u,v) bulk", mu_out_pr, bulk),
                _stats("v2 committed(pred u,v) all", mu_out_pr, None),
            ]
        )
        gate_pr = resolve_clot_trigger_gate(
            phi,
            step.mu_c_si,
            mu_mlp,
            region=step.region,
            mu_gt_cap_si=step.mu_gt_cap,
            phys_cfg=phys,
        )
        leak = bulk & (gate_pr.reshape(-1) > 0.5)
        diag["bulk_clot_gate_leak_n"] = int(leak.sum())
        diag["bulk_clot_gate_leak_frac"] = float(leak.float().mean())

    out_path = Path(args.out) if args.out else ROOT / "outputs/biochem/mlp_mu_map_bulk_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(f"[i] anchor={args.anchor} ti={ti} mask={diag['mask_mode']}")
    print(f"[i] clot_gate={diag['n_clot_gate']} bulk={diag['n_bulk']} lumen={diag['n_lumen']}")
    print(f"[i] mu_inf={diag['phys_mu_inf_si']:.4f} mu_0={diag['phys_mu_0_si']:.4f}")
    print(f"[i] gamma_dot_gt p50={diag['gamma_dot_gt_p50']:.2e} frac@floor={diag['gamma_dot_gt_frac_floor']:.3f}")
    print(f"[i] Carreau(GTuv) at mu_0 on {diag['frac_carreau_gt_at_mu0']*100:.1f}% of nodes")
    for row in diag["bulk_stats"]:
        if row.get("n", 0) == 0:
            continue
        print(
            f"    {row['name']}: n={row['n']} p50={row['p50']:.4f} mean={row['mean']:.4f} "
            f"p10={row['p10']:.4f} p90={row['p90']:.4f}"
        )
    print(f"[i] logMAE Carreau(GT) vs GT bulk={diag['logMAE_carreau_gt_vs_gt_bulk']:.3f}")
    print(f"[i] logMAE v2(GT uv) vs GT bulk={diag['logMAE_v2_vs_gt_bulk']:.3f}")
    if pred_slice is not None:
        print(f"[i] logMAE v2(pred) vs GT bulk={diag['logMAE_v2_pred_vs_gt_bulk']:.3f}")
        print(f"[i] bulk gate leak n={diag.get('bulk_clot_gate_leak_n', 0)}")
    print(f"[i] wrote {out_path}")
    print(f"[i] note: {diag['training_log_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
