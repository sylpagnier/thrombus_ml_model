"""Eval wall-band species M0 vs s0 on one anchor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    rollout_rung4_phi_trajectory,
    rollout_rung4_species_series,
    species_log_mae_in_mask,
)
from src.core_physics.species_snapshot_gnn import wall_band_mask  # noqa: E402
from src.core_physics.wall_band_species_m0 import load_m0_bundle, rollout_m0_species_series  # noqa: E402
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _parse_times(raw: str, n_steps: int) -> list[int]:
    if not raw.strip():
        return [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    return sorted({max(0, min(n_steps - 1, int(x.strip()))) for x in raw.split(",") if x.strip()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval wall-band M0")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--times", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    ckpt = Path(args.ckpt) if args.ckpt.strip() else None
    bundle = load_m0_bundle(ckpt, device=device)
    if bundle is None:
        raise SystemExit("[ERR] missing M0 checkpoint")

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    times = _parse_times(args.times, int(data.y.shape[0]))
    band = wall_band_mask(data, device, wall_hops=bundle.wall_hops)

    t0 = time.perf_counter()
    pred_m0 = rollout_m0_species_series(data, bundle, phys_cfg=phys, bio_cfg=bio, device=device)
    pred_s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
    print(f"[i] rollout {time.perf_counter() - t0:.1f}s", flush=True)

    from src.core_physics.wall_band_species_m0 import rollout_m0_phi_trajectory

    phi_m0 = rollout_m0_phi_trajectory(data, bundle, phys_cfg=phys, bio_cfg=bio, device=device)
    phi_s0 = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s0")

    clot_rows = []
    species_rows = []
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device).reshape(-1)
        mask = torch.ones(phi_gt.numel(), device=device, dtype=torch.bool)
        m0m = _clot_metrics(phi_m0[int(t)].reshape(-1), phi_gt, mask)
        s0m = _clot_metrics(phi_s0[int(t)].reshape(-1), phi_gt, mask)
        clot_rows.append({
            "time": int(t),
            "m0_f1": float(m0m["clot_f1"]),
            "s0_f1": float(s0m["clot_f1"]),
            "m0_pred_pos": float(m0m["pred_pos_frac"]),
            "s0_pred_pos": float(s0m["pred_pos_frac"]),
        })
        species_rows.append({
            "time": int(t),
            "m0_band": species_log_mae_in_mask(pred_m0, data, t, band, device),
            "s0_band": species_log_mae_in_mask(pred_s0, data, t, band, device),
        })

    t_last = times[-1]
    health = compute_rung4_rollout_health(phi_m0, data, phys, bio, device, times=times)

    payload = {
        "anchor": args.anchor,
        "channel_set": bundle.channel_set,
        "channel_names": bundle.channel_names,
        "clot": clot_rows,
        "species_mae": species_rows,
        "rollout_health": {k: v for k, v in health.items() if k != "timeline"},
        "t_last": {
            "m0_f1": clot_rows[-1]["m0_f1"],
            "s0_f1": clot_rows[-1]["s0_f1"],
            "delta_vs_s0": clot_rows[-1]["m0_f1"] - clot_rows[-1]["s0_f1"],
        },
    }

    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/clot_trigger/wall_band_m0_{bundle.channel_set}_{args.anchor}.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out}")
    print(
        f"[i] M0 F1={payload['t_last']['m0_f1']:.3f} "
        f"s0 F1={payload['t_last']['s0_f1']:.3f} "
        f"delta={payload['t_last']['delta_vs_s0']:+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
