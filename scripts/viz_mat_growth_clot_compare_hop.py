"""Comparative mat-growth clot viz: GT | model-A | model-B, hop-colored clots.

Row 0 = ground truth, row 1 = model 1 prediction, row 2 = model 2 prediction.
Clot nodes are colored by hop distance from the wall (not viscosity / 0-1 phi).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_mat_growth_simple import _apply_ckpt_recipe  # noqa: E402
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_gnn_ladder_viz import (  # noqa: E402
    clot_hop_legend,
    ladder_viz_times,
    scatter_clot_hop_panel,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    compute_hop_distances,
    train_deploy_eval_flow_source,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _resolve(root: Path, raw: str) -> Path:
    p = Path(raw.strip())
    if not p.is_absolute():
        p = root / p
    return p


def _configure_two_model(*, offwall: Path | None, route: str, frontier_hops: int) -> str:
    clear_offwall_model_cache()
    if offwall is None:
        os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
        os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
        os.environ.pop("SPECIES_TWO_MODEL_ROUTE", None)
        return "single-model"
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall).replace("\\", "/")
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = route
    os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = str(int(frontier_hops))
    return f"two-model route={route}"


def _rollout_phi(
    *,
    data,
    wall_ckpt: Path,
    offwall: Path | None,
    route: str,
    frontier_hops: int,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    flow: str,
    label: str,
):
    note = _configure_two_model(offwall=offwall, route=route, frontier_hops=frontier_hops)
    print(f"[i] rollout {label} ({note})...", flush=True)
    t0 = time.perf_counter()
    bundle = load_species_gnn_rollout_bundle(wall_ckpt, device=device)
    if bundle is None:
        raise SystemExit(f"[ERR] could not load wall ckpt: {wall_ckpt}")
    static = prepare_species_gnn_rollout_static(data, device=device)
    traj = rollout_species_gnn_phi_trajectory(
        data,
        bundle,
        static,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        flow_source=flow,
    )
    print(f"[i] {label} done in {time.perf_counter() - t0:.1f}s", flush=True)
    clear_offwall_model_cache()
    return traj


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare GT | model1 | model2 with hop-colored clot nodes"
    )
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--ckpt", required=True, help="Wall / model-1 ckpt (e.g. locked WC_v7)")
    ap.add_argument(
        "--ckpt-b",
        default="",
        help="Optional alternate wall ckpt for model 2 (default: same as --ckpt)",
    )
    ap.add_argument(
        "--offwall-ckpt",
        default="",
        help="Growth specialist for model-2 compound (wall-route)",
    )
    ap.add_argument("--two-model-route", default="wall", choices=("wall", "frontier", "growth"))
    ap.add_argument("--two-model-frontier-hops", type=int, default=2)
    ap.add_argument("--label-a", default="WC_v7")
    ap.add_argument("--label-b", default="Compound S")
    ap.add_argument("--flow", default="kinematics", choices=("gt", "kinematics"))
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    wall_a = _resolve(root, args.ckpt)
    if not wall_a.is_file():
        raise SystemExit(f"[ERR] missing --ckpt: {wall_a}")
    wall_b = _resolve(root, args.ckpt_b) if args.ckpt_b.strip() else wall_a
    if not wall_b.is_file():
        raise SystemExit(f"[ERR] missing --ckpt-b: {wall_b}")

    offwall = None
    if args.offwall_ckpt.strip():
        offwall = _resolve(root, args.offwall_ckpt)
        if not offwall.is_file():
            raise SystemExit(f"[ERR] missing --offwall-ckpt: {offwall}")

    payload = torch.load(wall_a, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_compare")
    if args.mat_leg.strip():
        apply_mat_growth_leg_env(args.mat_leg.strip(), force=True)
    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(
        overrides={"T0_R4_FLOW_SOURCE": args.flow if args.flow != "kinematics" else flow_eval}
    )

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    wall_mask = _wall_mask_from_data(data, device, n_nodes)
    hop_np = compute_hop_distances(data.edge_index, wall_mask, n_nodes).detach().cpu().numpy()
    times = ladder_viz_times(int(data.y.shape[0]), max_frames=int(args.max_frames))
    mask = torch.ones(n_nodes, device=device, dtype=torch.bool)

    print(f"[i] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"[i] compare {args.anchor}: GT | {args.label_a} | {args.label_b}",
        flush=True,
    )

    phi_a = _rollout_phi(
        data=data,
        wall_ckpt=wall_a,
        offwall=None,
        route="wall",
        frontier_hops=int(args.two_model_frontier_hops),
        phys=phys,
        bio=bio,
        device=device,
        flow=args.flow,
        label=args.label_a,
    )
    phi_b = _rollout_phi(
        data=data,
        wall_ckpt=wall_b,
        offwall=offwall,
        route=str(args.two_model_route),
        frontier_hops=int(args.two_model_frontier_hops),
        phys=phys,
        bio=bio,
        device=device,
        flow=args.flow,
        label=args.label_b,
    )

    row_labels = [
        "Ground truth (GT)",
        str(args.label_a),
        str(args.label_b),
    ]
    fig, axes = plt.subplots(
        3,
        len(times),
        figsize=(2.7 * len(times), 2.6 * 3),
        squeeze=False,
    )
    fig.suptitle(
        f"Hop-colored clot compare -- {args.anchor} | {args.label_a} vs {args.label_b}",
        fontsize=11,
        y=1.01,
    )

    frames: list[dict] = []
    s = float(args.scatter_size)
    for j, t in enumerate(times):
        phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
        pa = phi_a[int(t)]
        pb = phi_b[int(t)]
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        ma = clot_trigger_viz_f1(pa, phi_gt, mask)
        mb = clot_trigger_viz_f1(pb, phi_gt, mask)

        gt_np = phi_gt.detach().cpu().numpy()
        a_np = pa.detach().cpu().numpy()
        b_np = pb.detach().cpu().numpy()

        c_gt = scatter_clot_hop_panel(
            axes[0, j], pos, gt_np, hop_np, row_labels[0] if j == 0 else "", s=s
        )
        c_a = scatter_clot_hop_panel(
            axes[1, j], pos, a_np, hop_np, row_labels[1] if j == 0 else "", s=s
        )
        c_b = scatter_clot_hop_panel(
            axes[2, j], pos, b_np, hop_np, row_labels[2] if j == 0 else "", s=s
        )
        axes[0, j].set_title(f"t={t}  tau={tau:.2f}", fontsize=9, pad=4)
        axes[1, j].set_title(
            f"F1={ma['clot_f1']:.2f}  n={c_a['n_clot']}  ge2={c_a['hop2']+c_a['hop3']+c_a['hop4']}",
            fontsize=8,
            pad=3,
        )
        axes[2, j].set_title(
            f"F1={mb['clot_f1']:.2f}  n={c_b['n_clot']}  ge2={c_b['hop2']+c_b['hop3']+c_b['hop4']}",
            fontsize=8,
            pad=3,
        )
        frames.append(
            {
                "time": int(t),
                "tau": tau,
                "f1_a": float(ma["clot_f1"]),
                "f1_b": float(mb["clot_f1"]),
                "counts_gt": c_gt,
                "counts_a": c_a,
                "counts_b": c_b,
            }
        )

    clot_hop_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.88, 0.98])

    if args.out.strip():
        out = Path(args.out)
    else:
        out = (
            root
            / "outputs/biochem/viz/mat_growth"
            / f"clot_compare_hop_{args.anchor}_{args.label_a}_vs_{args.label_b}.png"
        )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}", flush=True)

    meta_out = out.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "label_a": args.label_a,
                "label_b": args.label_b,
                "ckpt_a": str(wall_a),
                "ckpt_b": str(wall_b),
                "offwall_ckpt": "" if offwall is None else str(offwall),
                "two_model_route": args.two_model_route if offwall is not None else "",
                "mat_leg": args.mat_leg.strip(),
                "flow_source": args.flow,
                "rows": row_labels,
                "times": times,
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {meta_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
