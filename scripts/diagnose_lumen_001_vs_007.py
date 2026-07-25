"""Diagnose patient001 lumen miss vs patient007 (GT hops + optional Arm S pred).

Usage:
  python scripts/diagnose_lumen_001_vs_007.py
  python scripts/diagnose_lumen_001_vs_007.py --offwall-ckpt outputs/.../growth_frontier_ge2_prec/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    compute_hop_distances,
    deploy_eval_time_index,
    train_deploy_eval_flow_source,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _hop_hist(clot: np.ndarray, hops: np.ndarray) -> dict[str, int]:
    out = {}
    for h in range(0, 5):
        out[f"hop{h}"] = int((clot & (hops == h)).sum())
    out["hop_ge2"] = int((clot & (hops >= 2)).sum())
    out["hop_ge4"] = int((clot & (hops >= 4)).sum())
    out["n"] = int(clot.sum())
    return out


def _profile_anchor(
    *,
    anchor: str,
    root: Path,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    wall_ckpt: Path,
    offwall: Path | None,
) -> dict:
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hops = compute_hop_distances(data.edge_index, wall, n).detach().cpu().numpy()
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi_gt = gt_clot_phi_at_time(data, t_dep, phys, device).detach().cpu().numpy().reshape(-1)
    gt_clot = phi_gt >= 0.5
    gt_hist = _hop_hist(gt_clot, hops)

    # Spatial extent of GT lumen
    pos = data.x[:, :2].detach().cpu().numpy()
    lumen = gt_clot & (hops >= 2)
    extent = {}
    if lumen.any():
        xy = pos[lumen]
        extent = {
            "x_span": float(xy[:, 0].max() - xy[:, 0].min()),
            "y_span": float(xy[:, 1].max() - xy[:, 1].min()),
            "n_cc_est": None,
        }

    clear_offwall_model_cache()
    if offwall is not None:
        os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
        os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall).replace("\\", "/")
        os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
        os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = "2"
        mode = "compound"
    else:
        os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
        os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
        mode = "wall_only"

    bundle = load_species_gnn_rollout_bundle(wall_ckpt, device=device)
    static = prepare_species_gnn_rollout_static(data, device=device)
    traj = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device, flow_source="kinematics"
    )
    phi_pr = traj[t_dep].detach().cpu().numpy().reshape(-1)
    pr_clot = phi_pr >= 0.5
    pr_hist = _hop_hist(pr_clot, hops)
    tp = int((pr_clot & gt_clot & (hops >= 2)).sum())
    fp = int((pr_clot & ~gt_clot & (hops >= 2)).sum())
    fn = int((~pr_clot & gt_clot & (hops >= 2)).sum())
    clear_offwall_model_cache()

    return {
        "anchor": anchor,
        "n_nodes": n,
        "n_times": int(data.y.shape[0]),
        "t_deploy": t_dep,
        "mode": mode,
        "gt_hop": gt_hist,
        "pred_hop": pr_hist,
        "hop_ge2_tp": tp,
        "hop_ge2_fp": fp,
        "hop_ge2_fn": fn,
        "lumen_extent": extent,
        "wall_frac": float(wall.float().mean().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-ckpt", default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
    ap.add_argument(
        "--offwall-ckpt",
        default="outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth",
    )
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    wall = Path(args.wall_ckpt)
    if not wall.is_absolute():
        wall = root / wall
    off = Path(args.offwall_ckpt) if args.offwall_ckpt.strip() else None
    if off is not None and not off.is_absolute():
        off = root / off
    if off is not None and not off.is_file():
        print(f"[WARN] offwall ckpt missing ({off}); wall-only compare", flush=True)
        off = None

    payload = torch.load(wall, map_location="cpu", weights_only=False)
    _apply_ckpt_recipe(dict(payload.get("meta") or {}), label="diag_001")
    apply_mat_growth_leg_env(args.mat_leg, force=True)
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": train_deploy_eval_flow_source()})

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    print("=" * 72, flush=True)
    print("DIAG: patient001 vs patient007 lumen (deploy-time hop hist)", flush=True)
    print("=" * 72, flush=True)

    rows = []
    for anc in ("patient007", "patient001"):
        print(f"[i] profiling {anc}...", flush=True)
        rows.append(
            _profile_anchor(
                anchor=anc,
                root=root,
                phys=phys,
                bio=bio,
                device=device,
                wall_ckpt=wall,
                offwall=off,
            )
        )

    for r in rows:
        print(
            f"\n{r['anchor']} n={r['n_nodes']} t_dep={r['t_deploy']} mode={r['mode']}",
            flush=True,
        )
        print(f"  GT  hop: {r['gt_hop']}", flush=True)
        print(f"  Pred hop: {r['pred_hop']}", flush=True)
        print(
            f"  hop_ge2 TP/FP/FN = {r['hop_ge2_tp']}/{r['hop_ge2_fp']}/{r['hop_ge2_fn']}",
            flush=True,
        )
        if r["lumen_extent"]:
            print(f"  GT lumen extent: {r['lumen_extent']}", flush=True)

    a001 = next(r for r in rows if r["anchor"] == "patient001")
    a007 = next(r for r in rows if r["anchor"] == "patient007")
    notes = []
    if a001["gt_hop"]["hop_ge2"] > 10 and a001["pred_hop"]["hop_ge2"] <= 0:
        notes.append("001: rich GT lumen, zero pred hop_ge2 -> transfer/activation failure")
    if a007["pred_hop"]["hop_ge2"] > 0 and a001["pred_hop"]["hop_ge2"] <= 0:
        notes.append("007 fires lumen, 001 does not -> not a global dead specialist")
    if a001["gt_hop"]["hop_ge2"] > 0 and a007["gt_hop"]["hop_ge2"] > 0:
        ratio = a001["gt_hop"]["hop_ge2"] / max(a007["gt_hop"]["hop_ge2"], 1)
        notes.append(f"GT hop_ge2 count ratio 001/007 = {ratio:.2f}")
    e1 = a001.get("lumen_extent") or {}
    e7 = a007.get("lumen_extent") or {}
    if e1 and e7:
        notes.append(
            f"GT lumen x_span 001={e1.get('x_span', 0):.3f} vs 007={e7.get('x_span', 0):.3f}"
        )

    print("\n[i] Notes:", flush=True)
    for n in notes:
        print(f"  - {n}", flush=True)

    out = {
        "rows": rows,
        "notes": notes,
        "wall_ckpt": str(wall),
        "offwall_ckpt": None if off is None else str(off),
    }
    out_path = Path(args.out) if args.out.strip() else (
        root / "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/diag_001_vs_007.json"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
