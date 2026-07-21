"""Analyze hop-2 off-wall prediction counts for orig10 compound A/B/C.

Purpose:
  - Answer whether a given arm actually predicts off-wall nodes at BFS hop=2
    from the wall at the deploy evaluation time.

Outputs:
  - Print per-hop counts for off-wall predictions (hop 0..max_hop).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
)  # noqa: E402
from src.core_physics.species_pushforward_continuous import deploy_eval_time_index  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    compute_hop_distances,
    train_deploy_eval_flow_source,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _set_two_model_env(*, arm: str, wall_ckpt: Path, offwall_ckpt: Path | None, route: str, frontier_hops: int) -> str:
    # Save old env and then set only what's required for two-model.
    if arm == "A":
        os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
        os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
        os.environ.pop("SPECIES_TWO_MODEL_ROUTE", None)
        os.environ.pop("SPECIES_TWO_MODEL_FRONTIER_HOPS", None)
        return "single-model"

    if offwall_ckpt is None:
        raise ValueError("offwall_ckpt required for arms B/C")

    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall_ckpt).replace("\\", "/")
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = route
    os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = str(int(frontier_hops))
    return f"two-model route={route} hops={frontier_hops}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Hop-2 off-wall prediction analyzer")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--arm", default="A", choices=("A", "B", "C"))
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--run-root", default="outputs/biochem/offwall_model/wc_v7_compound_abc_orig10_9h")
    ap.add_argument("--wall-ckpt", default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    run_root = root / args.run_root
    wall_ckpt = root / args.wall_ckpt

    growth_b = run_root / "growth_B_frontier_blurring_prec/best.pth"
    growth_c = run_root / "growth_C_offwall_blurring_prec/best.pth"

    if not wall_ckpt.is_file():
        raise SystemExit(f"[ERR] missing wall ckpt: {wall_ckpt}")
    if args.arm == "B" and not growth_b.is_file():
        raise SystemExit(f"[ERR] missing arm B ckpt: {growth_b}")
    if args.arm == "C" and not growth_c.is_file():
        raise SystemExit(f"[ERR] missing arm C ckpt: {growth_c}")

    # Load deploy graph.
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    n_nodes = int(data.num_nodes)

    # Apply primary ckpt recipe + mat-leg env overrides.
    payload = torch.load(wall_ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple")
    apply_mat_growth_leg_env(args.mat_leg, force=True)

    # Two-model environment.
    if args.arm == "B":
        two_note = _set_two_model_env(
            arm="B",
            wall_ckpt=wall_ckpt,
            offwall_ckpt=growth_b,
            route="frontier",
            frontier_hops=2,
        )
    elif args.arm == "C":
        two_note = _set_two_model_env(
            arm="C",
            wall_ckpt=wall_ckpt,
            offwall_ckpt=growth_c,
            route="wall",
            frontier_hops=2,
        )
    else:
        two_note = _set_two_model_env(
            arm="A",
            wall_ckpt=wall_ckpt,
            offwall_ckpt=None,
            route="frontier",
            frontier_hops=2,
        )

    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": args.flow if args.flow != "kinematics" else flow_eval})

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    # Deploy rollout.
    bundle = load_species_gnn_rollout_bundle(wall_ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] could not load species bundle: {wall_ckpt}")
    static = prepare_species_gnn_rollout_static(data, device=device)

    phi_pred_traj = rollout_species_gnn_phi_trajectory(
        data,
        bundle,
        static,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        flow_source=args.flow,
    )

    t_eval = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi_pred = phi_pred_traj[t_eval].reshape(-1)
    pred_pos = phi_pred >= 0.5

    wall_mask_full = _wall_mask_from_data(data, device, n_nodes)
    offwall = ~wall_mask_full.bool()
    hop_distances = compute_hop_distances(data.edge_index, wall_mask_full, n_nodes)

    hop2_off_pred = int((pred_pos & offwall & (hop_distances == 2)).sum().item())
    print(
        f"[{args.arm}] anchor={args.anchor} t_eval={t_eval} {two_note} | offwall hop2 pred nodes={hop2_off_pred}",
        flush=True,
    )

    # Per-hop counts (off-wall predictions only).
    max_h = max(0, int(args.max_hops))
    for h in range(0, max_h + 1):
        nh = int((pred_pos & offwall & (hop_distances == h)).sum().item())
        print(f"  hop{h}: {nh}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

