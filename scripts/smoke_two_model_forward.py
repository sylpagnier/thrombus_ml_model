"""Fast smoke: one-step two-model forward for wall + frontier routes.

Avoids full deploy rollout (too slow / VRAM-heavy for 4GB smoke).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.core_physics.species_pushforward_continuous import (
    clear_offwall_model_cache,
    load_continuous_bundle,
    predict_continuous_step_delta,
)
from src.core_physics.t0_device import require_cuda_device
from src.utils.paths import get_project_root


def _one_step(wall_ckpt: Path, growth_ckpt: Path, route: str, data, device) -> dict:
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_ckpt).replace("\\", "/")
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = route
    os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = "2"

    bundle = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(wall_ckpt)
    model = bundle.model
    n = int(data.num_nodes)
    # Tiny band: first 512 nodes (topology still valid via full edge_index).
    # Full-graph one-step is fine if VRAM allows; clamp for 4GB cards.
    use_n = min(n, 2048)
    node_idx = torch.arange(use_n, device=device)
    # Rebuild a tiny synthetic feature batch matching model.in_dim
    in_dim = int(model.in_dim)
    base_feats = torch.zeros(use_n, in_dim, device=device, dtype=torch.float32)
    # Prefer real positions when available
    if hasattr(data, "x") and data.x is not None:
        pos = data.x[:use_n, :2].to(device=device, dtype=torch.float32)
    else:
        pos = torch.zeros(use_n, 2, device=device)
    # Induced edges among first use_n nodes
    ei = data.edge_index.to(device)
    mask = (ei[0] < use_n) & (ei[1] < use_n)
    edge_index = ei[:, mask]
    wall_mask = torch.zeros(use_n, dtype=torch.bool, device=device)
    wall_mask[: max(1, use_n // 10)] = True
    out_dim = int(getattr(model, "out_dim", 1) or 1)
    log_state = torch.zeros(use_n, out_dim, device=device)
    # Seed one committed node so frontier route has a growth zone.
    log_state[0, 0] = 1.0

    with torch.no_grad():
        delta = predict_continuous_step_delta(
            model,
            base_feats,
            edge_index,
            log_state,
            training=False,
            pos_band=pos,
            wall_mask_band=wall_mask,
        )
    clear_offwall_model_cache()
    return {
        "route": route,
        "delta_shape": list(delta.shape),
        "delta_abs_mean": float(delta.abs().mean().item()),
        "nodes": use_n,
        "growth_ckpt": str(growth_ckpt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-ckpt", required=True)
    ap.add_argument("--growth-b", required=True)
    ap.add_argument("--growth-c", required=True)
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    apply_mat_growth_leg_env(args.mat_leg, force=True)

    wall = Path(args.wall_ckpt)
    if not wall.is_absolute():
        wall = root / wall
    gb = Path(args.growth_b)
    if not gb.is_absolute():
        gb = root / gb
    gc = Path(args.growth_c)
    if not gc.is_absolute():
        gc = root / gc
    for p in (wall, gb, gc):
        if not p.is_file():
            raise FileNotFoundError(p)

    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    data = torch.load(graph, map_location="cpu", weights_only=False)

    report = {
        "anchor": args.anchor,
        "frontier": _one_step(wall, gb, "frontier", data, device),
        "wall": _one_step(wall, gc, "wall", data, device),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] smoke micro-forward -> {out}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
