"""EDA: speed distributions for GT lumen vs zero-GT spray vessels (WALL_ONLY context)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    compute_hop_distances,
    deploy_eval_time_index,
)
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    induced_subgraph,
    snapshot_wall_hops,
    wall_band_mask,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _stats(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"n": 0}
    xf = x.float()
    return {
        "n": int(x.numel()),
        "mean": float(xf.mean()),
        "p50": float(torch.quantile(xf, 0.5)),
        "p90": float(torch.quantile(xf, 0.9)),
        "p95": float(torch.quantile(xf, 0.95)),
        "max": float(xf.max()),
    }


@torch.no_grad()
def audit(anchor: str, data, phys, bio) -> dict:
    device = torch.device("cpu")
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hop = compute_hop_distances(data.edge_index.to(device), wall, n)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi = gt_clot_phi_at_time(data, t_dep, phys, device).reshape(-1)
    clot = phi >= 0.5
    lumen = (hop >= 2) & (~wall)
    lumen_gt = lumen & clot
    y = data.y[t_dep].to(dtype=torch.float32)
    speed = torch.linalg.vector_norm(y[:, 0:2], dim=-1)
    band_mask = wall_band_mask(data, device, wall_hops=snapshot_wall_hops())
    band_nodes, _, _ = induced_subgraph(band_mask, data.edge_index)
    on_band = torch.zeros(n, dtype=torch.bool)
    on_band[band_nodes.long()] = True
    band_lumen = lumen & on_band
    return {
        "anchor": anchor,
        "t_dep": t_dep,
        "n_lumen_gt": int(lumen_gt.sum()),
        "n_band_lumen": int(band_lumen.sum()),
        "speed_lumen_gt": _stats(speed[lumen_gt]),
        "speed_band_lumen": _stats(speed[band_lumen]),
        "speed_band_lumen_no_gt": _stats(speed[band_lumen & (~clot)]),
    }


def main() -> int:
    root = get_project_root()
    graph_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    phys, bio = PhysicsConfig(), BiochemConfig()
    groups = {
        "gt": ["patient001", "patient007"],
        "spray": ["patient002", "patient008"],
        "thin": ["patient006", "patient010"],
    }
    out = {"per_anchor": {}, "pooled": {}, "recommendation": {}}
    pools: dict[str, list[torch.Tensor]] = {
        "gt_lumen": [],
        "spray_band_no_gt": [],
        "thin_gt": [],
        "thin_band_no_gt": [],
    }
    for group, names in groups.items():
        for name in names:
            path = graph_dir / f"{name}.pt"
            data = torch.load(path, map_location="cpu", weights_only=False)
            row = audit(name, data, phys, bio)
            out["per_anchor"][name] = row
            print(
                f"[i] {name}: gt={row['n_lumen_gt']} band_lumen={row['n_band_lumen']} "
                f"gt_p90={row['speed_lumen_gt'].get('p90')} "
                f"band_nogt_p90={row['speed_band_lumen_no_gt'].get('p90')}",
                flush=True,
            )
            # rebuild tensors for pool from saved stats alone is lossy; recompute quickly
            device = torch.device("cpu")
            n = int(data.num_nodes)
            wall = _wall_mask_from_data(data, device, n)
            hop = compute_hop_distances(data.edge_index.to(device), wall, n)
            t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
            phi = gt_clot_phi_at_time(data, t_dep, phys, device).reshape(-1)
            clot = phi >= 0.5
            lumen = (hop >= 2) & (~wall)
            speed = torch.linalg.vector_norm(data.y[t_dep][:, 0:2].float(), dim=-1)
            band_mask = wall_band_mask(data, device, wall_hops=snapshot_wall_hops())
            band_nodes, _, _ = induced_subgraph(band_mask, data.edge_index)
            on_band = torch.zeros(n, dtype=torch.bool)
            on_band[band_nodes.long()] = True
            if group == "gt":
                pools["gt_lumen"].append(speed[lumen & clot])
            elif group == "spray":
                pools["spray_band_no_gt"].append(speed[lumen & on_band & (~clot)])
            else:
                pools["thin_gt"].append(speed[lumen & clot])
                pools["thin_band_no_gt"].append(speed[lumen & on_band & (~clot)])

    for k, parts in pools.items():
        xs = [p for p in parts if p.numel()]
        out["pooled"][k] = _stats(torch.cat(xs)) if xs else {"n": 0}

    gt = out["pooled"].get("gt_lumen") or {}
    sp = out["pooled"].get("spray_band_no_gt") or {}
    if gt.get("n", 0) and sp.get("n", 0):
        # High-speed wash helps when idle spray is faster than preserved GT (001-like).
        # Overlap is high when spray vessels are slow (002) while 001 GT is fast.
        out["recommendation"] = {
            "gt_p50": gt.get("p50"),
            "gt_p90": gt.get("p90"),
            "spray_idle_p50": sp.get("p50"),
            "spray_idle_p90": sp.get("p90"),
            "high_speed_wash_helps_001_risk": float(gt.get("p90", 0)) > 0.4,
            "spray_idle_slower_than_001_gt": float(sp.get("p90", 0)) < float(gt.get("p50", 1)),
            "note": (
                "If spray idle is slower than 001 GT, full-band high-speed washout "
                "cannot selectively kill 002 spray without hurting 001. Prefer FP loss / "
                "spray negatives / specialist confidence. If spray idle is faster, try mild "
                "lumen washout above ~gt_p50."
            ),
        }
        print(json.dumps(out["recommendation"], indent=2), flush=True)

    out_path = root / "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/eda_spray_vs_gt_speed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
