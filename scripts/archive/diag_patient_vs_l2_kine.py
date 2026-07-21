"""Diagnose patient vs synthetic L2 kinematics rel_L2 gap (read-only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval_kine_cross_cohort import (  # noqa: E402
    _apply_x_mode,
    _load_kinematics_model,
    _metrics,
    _node_mask,
    _steady_kine_targets,
)
from src.config import NodeFeat, PhysicsConfig
from src.utils.kinematics_paths import resolve_kinematics_anchor_graph
from src.utils.paths import data_root


def _rel_l2(pred, tgt, mask):
    p = pred[mask, :3]
    t = tgt[mask, :3]
    return float((p - t).norm() / (t.norm() + 1e-8))


def _eval_graph(model, path: Path, *, x_mode: str, phys, time_index: int, label_source: str, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    data_eval = _apply_x_mode(data, x_mode, phys, stem=path.stem)
    data_eval = data_eval.to(device)
    if label_source == "kine_steady" and data.y.dim() == 2:
        tgt = data.y[:, :5]
    else:
        tgt = _steady_kine_targets(data, time_index)
    tgt = tgt.to(device)
    mask = _node_mask(data_eval)
    with torch.no_grad():
        pred = model(data_eval, solver="anderson", anderson_beta=0.8)
        if isinstance(pred, tuple):
            pred = pred[0]
    m = _metrics(pred, tgt, mask)
    x = data_eval.x
    return {
        "stem": path.stem,
        "path": str(path),
        "n_nodes": int(data.num_nodes),
        "y_shape": tuple(data.y.shape),
        "x_ch": int(x.shape[1]),
        "x_schema": str(getattr(data, "x_schema", "?")),
        "is_anchor_frac": float(data.is_anchor.float().mean()) if hasattr(data, "is_anchor") else 1.0,
        "uv_prior_max": float(x[:, NodeFeat.UV_PRIOR].abs().max().item()),
        "mu_prior_mean": float(x[:, NodeFeat.MU_PRIOR].mean().item()),
        "rheo_flag": float(x[:, NodeFeat.REST].mean().item()) if x.shape[1] > NodeFeat.REST.start else -1,
        "rel_l2": m["rel_l2_uvp"],
        "rel_u": m["rel_l2_u"],
        "rel_v": m["rel_l2_v"],
        "rel_p": m["rel_l2_p"],
        "wall_u_bio_t0": _wall_uv_max(data, time_index) if data.y.dim() == 3 else None,
    }


def _wall_uv_max(data, time_index: int):
    if not hasattr(data, "mask_wall"):
        return None
    y = data.y[time_index if time_index >= 0 else data.y.shape[0] + time_index, :, :2]
    w = data.mask_wall.view(-1).bool()
    return float(y[w].abs().max().item())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rheology", default="carreau", choices=("newtonian", "carreau"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dr = data_root()
    device = torch.device(args.device)
    phys = PhysicsConfig(phase="kinematics", rheology=args.rheology)
    model, ckpt, _ = _load_kinematics_model(device, rheology=args.rheology)

    print(f"[i] ckpt: {ckpt}")
    print(f"[i] rheology eval: {args.rheology}")
    print()

    print("=== Sidecar level (biochem_anchors JSON) ===")
    for jp in sorted((dr / "raw/biochem_anchors").glob("patient*.json")):
        meta = json.loads(jp.read_text(encoding="utf-8"))
        print(f"  {jp.stem}: level={meta.get('level', 'MISSING')}")

    print()
    print("=== Label path: biochem y[0] vs kine_anchor steady y ===")
    for stem in sorted((dr / "processed/graphs_biochem_anchors").glob("patient*.pt")):
        bio = torch.load(stem, map_location="cpu", weights_only=False)
        kin_p = resolve_kinematics_anchor_graph(stem.stem, rheology=args.rheology)
        line = f"  {stem.stem}: biochem_y={tuple(bio.y.shape)}"
        if kin_p.is_file():
            kin = torch.load(kin_p, map_location="cpu", weights_only=False)
            yb = bio.y[0, :, :3]
            yk = kin.y[:, :3]
            line += f" | kine_y={tuple(kin.y.shape)} label_diff={float((yb - yk).norm() / (yk.norm() + 1e-8)):.4f}"
            if hasattr(bio, "mask_wall"):
                wb = bio.mask_wall.view(-1).bool()
                line += f" wall|u| t0 bio={float(yb[wb, 0].abs().max()):.4e} kine={float(yk[kin.mask_wall, 0].abs().max()):.4e}"
        else:
            line += " | kine_anchor MISSING"
        print(line)

    print()
    print("=== Model rel_L2: same ckpt, different graph/label paths ===")
    rows = []
    # one L2 synthetic
    l2_dir = dr / f"processed/graphs_kinematics/{args.rheology}"
    for pt in sorted(l2_dir.glob("vessel_*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if hasattr(d, "geometry_level") and int(d.geometry_level.view(-1)[0].item()) == 2:
            rows.append(("synthetic_L2", pt, "biochem_default"))
            break

    for stem in ["patient002", "patient007"]:
        bio_p = dr / "processed/graphs_biochem_anchors" / f"{stem}.pt"
        kin_p = resolve_kinematics_anchor_graph(stem, rheology=args.rheology)
        if bio_p.is_file():
            rows.append((f"{stem}_biochem_graph", bio_p, "biochem_default"))
        if kin_p.is_file():
            rows.append((f"{stem}_kine_anchor", kin_p, "kine_steady"))

    model = model.to(device)
    for tag, path, label_src in rows:
        r = _eval_graph(model, path, x_mode="native", phys=phys, time_index=0, label_source=label_src, device=device)
        print(
            f"  {tag:24s} rel_L2={r['rel_l2']:.4f}  rel_u={r['rel_u']:.4f} rel_v={r['rel_v']:.4f} rel_p={r['rel_p']:.4f}"
            f"  n={r['n_nodes']} y={r['y_shape']} anchor_frac={r['is_anchor_frac']:.3f}"
        )
        if r["wall_u_bio_t0"] is not None:
            print(f"    wall|u| biochem t0={r['wall_u_bio_t0']:.4e} uv_prior_max={r['uv_prior_max']:.4f} rheo_flag={r['rheo_flag']:.2f}")

    print()
    print("=== Ablation: x_mode x rheology (patient biochem graphs) ===")
    for stem in ["patient002", "patient007"]:
        bio_p = dr / "processed/graphs_biochem_anchors" / f"{stem}.pt"
        for xm in ("native", "kine_layout"):
            for rheo in ("carreau", "newtonian"):
                phys_ab = PhysicsConfig(phase="kinematics", rheology=rheo)
                r = _eval_graph(
                    model, bio_p, x_mode=xm, phys=phys_ab, time_index=0,
                    label_source="biochem_default", device=device,
                )
                print(
                    f"  {stem} x={xm:12s} rheo={rheo:8s} "
                    f"rel={r['rel_l2']:.4f} ru={r['rel_u']:.4f} rv={r['rel_v']:.4f} rp={r['rel_p']:.4f}"
                )

    print()
    print("=== Valid synthetic L2 (carreau, first non-degenerate) ===")
    for pt in sorted(l2_dir.glob("vessel_*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        if int(d.geometry_level.view(-1)[0]) != 2:
            continue
        if float(d.y[:, 0].abs().sum()) < 1e-3:
            continue
        r = _eval_graph(
            model, pt, x_mode="native", phys=phys, time_index=0,
            label_source="kine_steady", device=device,
        )
        print(f"  {pt.stem} rel={r['rel_l2']:.4f} ru={r['rel_u']:.4f} rv={r['rel_v']:.4f} rp={r['rel_p']:.4f}")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
