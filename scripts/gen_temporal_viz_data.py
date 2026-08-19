"""Build the JSON payload for a time-resolved clot-growth visualization.

Uses the ODE-integrated model (gates frozen at t=0, Mat/Mas integrated through the real
COMSOL surface ODE) -- the only component in this project with an actual time axis. The
deployed gate+growth heuristic has none; it emits a single final mask.

diag_ignition_timing.py already found this integration ignites far faster than GT
(most nodes cross threshold by ~20% of the horizon vs GT's onset spread of 70-90%), so
this payload includes the FULL-RESOLUTION committed-node count curve (all 201 steps),
not just the snapshot frames, so that mismatch is visible rather than just claimed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import node_positions, t0_flow_fields, integrate_mat_trajectory, graded_gate  # noqa: E402
from src.core_physics.shear_redistribution import build_crosssection_operator, sdf_nd, make_blockage  # noqa: E402
from src.core_physics.thrombin_field import make_thrombin_solver, make_ap_boost  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
VESSELS = ["patient043", "patient044", "patient014", "patient001", "patient007", "patient013"]
N_FRAMES = 13
MAX_BG_POINTS = 1800


def main() -> None:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    out = {}
    for anchor in VESSELS:
        d = torch.load(DIR / f"{anchor}.pt", map_location="cpu", weights_only=False)
        pos = node_positions(d)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        g0 = graded_gate(f, bio, mode="hard") * wall
        
        ts, _ = make_thrombin_solver(d, bio, pos, f.sr, wash_coef=0.0, wall=wall)
        B = build_crosssection_operator(pos, sdf_nd(d), wall, radius_mult=0.30)
        blk = make_blockage(f, bio, B, wall, every=5, feedback="wake", wake=8.0)
        boost = make_ap_boost(ts, bio, gain=4.0, every=5)
        
        traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=40.0, blockage=blk, ap_boost=boost)
        crit = float(bio.viscosity_mat_crit)
        model_hot = (traj >= crit) & wall[None, :]           # [T, N]

        gt_hot = np.zeros((len(t), n), dtype=bool)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5
        gt_hot = gt_hot & wall[None, :]

        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)
        wall_idx = np.where(wall)[0]

        out[anchor] = {
            "t_final": float(t[-1]),
            "n_wall": int(wall.sum()),
            "bg": [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in bg],
            "wall_pos": [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in wall_idx],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt": [[bool(x) for x in gt_hot[i][wall_idx]] for i in frame_idx],
            "frame_model": [[bool(x) for x in model_hot[i][wall_idx]] for i in frame_idx],
            "count_t": [round(float(x), 1) for x in t],
            "count_gt": [int(gt_hot[i].sum()) for i in range(len(t))],
            "count_model": [int(model_hot[i].sum()) for i in range(len(t))],
        }
        print(f"{anchor}: wall={wall.sum()} bg={len(bg)} frames={N_FRAMES} "
              f"gt_final={out[anchor]['count_gt'][-1]} model_final={out[anchor]['count_model'][-1]}")

    out_path = Path("outputs/temporal_viz_data.json")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
