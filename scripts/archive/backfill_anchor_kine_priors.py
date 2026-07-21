"""Backfill Carreau + GT t=0 priors on biochem anchor graphs (18ch ``data.x``).

Example:
    python scripts/backfill_anchor_kine_priors.py
    python scripts/backfill_anchor_kine_priors.py --stem patient007 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.node_feature_assembly import (
    apply_gt_flow_priors_to_kine_x,
    kinematics_uv_prior_max,
    refresh_kinematics_node_x_on_graph,
    resolve_anchor_kine_phys_cfg,
)
from src.utils.channel_schema import attach_patient_anchor_graph_metadata, infer_missing_schema


def _apply_gt_from_biochem_y(data) -> bool:
    if not hasattr(data, "y") or data.y is None or data.y.dim() != 3:
        return False
    for req in ("edge_index", "M_inv", "V", "W", "mask_wall"):
        if not hasattr(data, req) or getattr(data, req) is None:
            return False
    y0 = data.y[0]
    u = y0[:, 0]
    v = y0[:, 1]
    mu = y0[:, 3]
    data.x = apply_gt_flow_priors_to_kine_x(
        data.x,
        u_nd=u,
        v_nd=v,
        mu_nd=mu,
        mask_wall=data.mask_wall.view(-1).bool(),
        wall_normal=data.x[:, NodeFeat.WALL_NORMAL],
        edge_index=data.edge_index,
        M_inv=data.M_inv,
        V=data.V,
        W=data.W,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill kinematics priors on anchor graphs.")
    parser.add_argument("--stem", type=str, default="", help="Only this patient stem (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write .pt files.")
    parser.add_argument("--force", action="store_true", help="Rewrite even when priors look nonzero.")
    parser.add_argument(
        "--prior-mode",
        choices=("gt_flow", "analytic"),
        default="gt_flow",
        help="gt_flow = COMSOL t=0 u,v,mu in prior channels; analytic = Carreau Poiseuille only.",
    )
    args = parser.parse_args()

    anchor_dir = Path(VesselConfig(phase="biochem_anchors").graph_output_dir)
    if not anchor_dir.is_dir():
        raise SystemExit(f"[ERR] anchor dir missing: {anchor_dir}")

    paths = sorted(anchor_dir.glob("*.pt"))
    if args.stem:
        paths = [anchor_dir / f"{args.stem.strip()}.pt"]

    phys = resolve_anchor_kine_phys_cfg()
    n_ok = 0
    for path in paths:
        if not path.is_file():
            print(f"[WARN] skip missing {path.name}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        before = kinematics_uv_prior_max(data.x)
        refreshed = refresh_kinematics_node_x_on_graph(
            data,
            phys_cfg=phys,
            stem=path.stem,
            force=bool(args.force),
        )
        gt_applied = False
        if args.prior_mode == "gt_flow":
            gt_applied = _apply_gt_from_biochem_y(data)
        after = kinematics_uv_prior_max(data.x)
        rheo = float(data.x[:, NodeFeat.REST].mean().item()) if data.x.shape[1] > NodeFeat.REST.start else -1
        tag = "refresh" if (refreshed or gt_applied) else "skip"
        print(
            f"[{tag}] {path.name}: uv_prior {before:.4f}->{after:.4f} "
            f"rheo_mean={rheo:.2f} gt={int(gt_applied)}"
        )
        if (refreshed or gt_applied) and not args.dry_run:
            if hasattr(data, "x_biochem"):
                data = attach_patient_anchor_graph_metadata(data, mask_wall=getattr(data, "mask_wall", None))
            torch.save(data, path)
            n_ok += 1

    if args.dry_run:
        print(f"[i] dry-run complete ({len(paths)} graphs inspected)")
    else:
        print(f"[OK] wrote {n_ok} graph(s) under {anchor_dir}")


if __name__ == "__main__":
    main()
