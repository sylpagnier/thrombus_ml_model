"""Is arm 2 inert because the mechanism is wrong, or because phi is too small to bite?

Shear redistribution moved curve_l1 only 0.300 -> 0.283.  Either the occluded fraction
never gets large enough to close a gate, or the feedback genuinely does not matter.  This
measures phi directly, and counts how many wall nodes actually change gate state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, local_half_width, sdf_nd,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    for a in ("patient043", "patient007", "patient014", "patient001"):
        d = torch.load(DIR / f"{a}.pt", map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        pos = node_positions(d)
        sdf = sdf_nd(d)
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        gate = graded_gate(f, bio, mode="hard") * wall
        traj, t = integrate_mat_trajectory(d, bio, gate, da_scale=100.0)
        final = traj[-1] >= float(bio.viscosity_mat_crit)

        # GT clot at the final time, including OFF-WALL nodes
        gt = gt_clot_phi_at_time(d, len(t) - 1, phys, device=torch.device("cpu")).numpy() > 0.5
        r0 = local_half_width(pos, sdf, wall)
        print("\n=== %s ===  wall %d  model-clot %d  GT-clot(all nodes) %d (off-wall %d)"
              % (a, wall.sum(), final.sum(), gt.sum(), (gt & ~wall).sum()))
        print("  local half-width nd: median %.3f   median edge %.4f  -> ~%.1f cells deep"
              % (np.median(r0[wall]), np.median(np.linalg.norm(
                  pos[d.edge_index.numpy()[0]] - pos[d.edge_index.numpy()[1]], axis=1)),
                 np.median(r0[wall]) / max(np.median(np.linalg.norm(
                     pos[d.edge_index.numpy()[0]] - pos[d.edge_index.numpy()[1]], axis=1)), 1e-9)))
        for rm in (0.3, 0.5, 1.0):
            B = build_crosssection_operator(pos, sdf, wall, radius_mult=rm)
            phi_model = np.asarray(B @ final.astype(float)).reshape(-1)[wall]
            phi_gt = np.asarray(B @ gt.astype(float)).reshape(-1)[wall]
            for p in (2.0, 3.0):
                amp_m = (1 - np.clip(phi_model, 0, 0.85)) ** -p
                flip = ((f.sr[wall] < bio.lss) & (f.sr[wall] * amp_m >= bio.lss)).sum()
                print("   rm=%.1f p=%.0f | phi model med %.3f p90 %.3f | phi GT med %.3f p90 %.3f"
                      " | shear amp med %.2f | gates closed %d"
                      % (rm, p, np.median(phi_model), np.percentile(phi_model, 90),
                         np.median(phi_gt), np.percentile(phi_gt, 90),
                         np.median(amp_m), flip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
