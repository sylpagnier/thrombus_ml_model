"""Deep audit: patient vs synthetic L2 kinematics error (features, masks, components)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.eval_kine_cross_cohort import (
    _apply_x_mode,
    _load_kinematics_model,
    _metrics,
    _node_mask,
    _steady_kine_targets,
)
from src.config import NodeFeat, PhysicsConfig, PredChannels
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.kinematics_paths import resolve_kinematics_anchor_graph
from src.utils.paths import data_root


def _interior_mask(data) -> torch.Tensor:
    kernels = PhysicsKernels(PhysicsConfig(phase="kinematics", rheology="carreau"))
    return kernels.fluid_interior_mask(data)


def _lumen_mask(data) -> torch.Tensor:
    sdf = data.x[:, NodeFeat.SDF].view(-1)
    mw = data.mask_wall.view(-1).bool()
    return (~mw) & (sdf > 0.02)


def audit_graph(model, data, phys, stem: str, tag: str, device) -> dict:
    de = _apply_x_mode(data, "native", phys, stem=stem).to(device)
    tgt = _steady_kine_targets(data, 0).to(device)
    masks = {
        "all": _node_mask(de),
        "interior": _interior_mask(de),
        "lumen_sdf02": _lumen_mask(de),
    }
    with torch.no_grad():
        pred = model(de, solver="anderson", anderson_beta=0.8)
        if isinstance(pred, tuple):
            pred = pred[0]

    x = de.x
    gt_uv = tgt[:, :2]
    out = {
        "tag": tag,
        "n": int(de.num_nodes),
        "gt_u_norm": float(gt_uv[:, 0].norm().item()),
        "gt_v_norm": float(gt_uv[:, 1].norm().item()),
        "gt_p_norm": float(tgt[:, 2].norm().item()),
        "gt_v_frac": float(gt_uv[:, 1].norm().item() / (gt_uv.norm().item() + 1e-8)),
        "width_max": float(x[:, NodeFeat.WIDTH_ND].max().item()),
        "width_d1_max": float(x[:, NodeFeat.WIDTH_D1].abs().max().item()),
        "wall_slip": float(pred[de.mask_wall.view(-1).bool(), :2].norm(dim=1).mean().item()),
    }
    for mname, mask in masks.items():
        m = _metrics(pred, tgt, mask)
        out[f"rel_{mname}"] = m["rel_l2_uvp"]
        out[f"ru_{mname}"] = m["rel_l2_u"]
        out[f"rv_{mname}"] = m["rel_l2_v"]
        out[f"rp_{mname}"] = m["rel_l2_p"]
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="kinematics", rheology="carreau")
    model, ckpt, ctor = _load_kinematics_model(device, rheology="carreau")
    model.eval()
    print(f"[i] ckpt={ckpt}")
    print(f"[i] ctor wss_fuse={ctor.get('wss_fuse')} bc_envelope={ctor.get('bc_envelope')}")
    print()

    dr = data_root()
    rows = []

    # synthetic L2 reference
    l2_dir = dr / "processed/graphs_kinematics/carreau"
    for pt in sorted(l2_dir.glob("vessel_*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if int(d.geometry_level.view(-1)[0]) != 2:
            continue
        if float(d.y[:, 0].abs().sum()) < 1e-3:
            continue
        rows.append(audit_graph(model, d, phys, pt.stem, f"synth_{pt.stem}", device))
        break

    for stem in sorted(["patient001", "patient002", "patient007"]):
        kpath = resolve_kinematics_anchor_graph(stem, rheology="carreau")
        if kpath is None:
            continue
        d = torch.load(kpath, map_location="cpu", weights_only=False)
        rows.append(audit_graph(model, d, phys, stem, stem, device))

    hdr = (
        f"{'tag':<18} {'rel_all':>8} {'rel_int':>8} {'rel_lum':>8} "
        f"{'rv_all':>7} {'gt_v%':>6} {'w_max':>6} {'w_d1':>6} {'wall_slip':>9}"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['tag']:<18} {r['rel_all']:8.4f} {r['rel_interior']:8.4f} {r['rel_lumen_sdf02']:8.4f} "
            f"{r['rv_all']:7.3f} {100*r['gt_v_frac']:5.1f}% {r['width_max']:6.2f} "
            f"{r['width_d1_max']:6.3f} {r['wall_slip']:9.4f}"
        )

    print()
    print("[i] Ablation: bc_envelope + env KINEMATICS_BC_ENVELOPE=1")
    import os

    os.environ["KINEMATICS_BC_ENVELOPE"] = "1"
    from src.architecture.kinematics_model_config import build_gino_deq_from_ctor

    model2 = build_gino_deq_from_ctor(phys, {**ctor, "bc_envelope": True}).to(device)
    _, state = __import__(
        "src.architecture.kinematics_model_config", fromlist=["kinematics_checkpoint_tensors"]
    ).kinematics_checkpoint_tensors(torch.load(ckpt, map_location=device, weights_only=False))
    model2.load_state_dict(state, strict=False)
    model2.eval()
    for stem in ["patient002", "patient007"]:
        kpath = resolve_kinematics_anchor_graph(stem, rheology="carreau")
        d = torch.load(kpath, map_location="cpu", weights_only=False)
        de = _apply_x_mode(d, "native", phys, stem=stem).to(device)
        tgt = _steady_kine_targets(d, 0).to(device)
        mask = _node_mask(de)
        with torch.no_grad():
            pred = model2(de, solver="anderson", anderson_beta=0.8)
            if isinstance(pred, tuple):
                pred = pred[0]
        m0 = _metrics(pred, tgt, mask)
        print(f"  {stem} bc_envelope=1 rel={m0['rel_l2_uvp']:.4f} (default {next(r for r in rows if r['tag']==stem)['rel_all']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
