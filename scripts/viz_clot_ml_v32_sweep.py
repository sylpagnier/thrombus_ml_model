"""V3.2 sweep viz: frozen rule budget vs leg rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_forecast import build_clot_forecast_pair_step  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_eval_device  # noqa: E402
from src.training.clot_ml_step1_residual import rollout_frozen_rule_phi, resolve_step1_rule_cfg  # noqa: E402
from src.training.clot_ml_v2_growth_gnn import load_v3_checkpoint, rollout_v3_growth_gnn  # noqa: E402
from src.training.clot_ml_v32_growth_ranker import (  # noqa: E402
    V32SweepLegConfig,
    apply_v32_env,
    load_leg_checkpoint,
    rollout_v32_ranker,
)
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="V3.2 sweep leg viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--leg", default="v32_ranker")
    ap.add_argument("--arch", default="ranker", choices=["ranker", "euler"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    full_region = np.ones(n_nodes, dtype=bool)
    n_steps = int(data.y.shape[0])
    rule_cfg = resolve_step1_rule_cfg(root / args.step0_json)
    ckpt_path = root / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)

    reset_temporal_kinematics_cache()
    phi_rule = rollout_frozen_rule_phi(
        data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=1.0
    )

    leg = V32SweepLegConfig(name=args.leg, arch=args.arch)
    if args.arch == "ranker":
        apply_v32_env()
        model, _ = load_leg_checkpoint(ckpt_path, leg, device=device)
        reset_temporal_kinematics_cache()
        phi_leg = rollout_v32_ranker(
            model, data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio
        )
        tag = "V3.2 ranker"
    else:
        v31 = "v31" in args.leg
        model, _ = load_v3_checkpoint(ckpt_path, device=device, v31=v31)
        reset_temporal_kinematics_cache()
        phi_leg = rollout_v3_growth_gnn(
            model, data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio
        )
        tag = args.leg

    frames: list[dict] = []
    for t in _pick_times(n_steps, int(args.max_frames)):
        t_in = max(0, t - 1)
        step = build_clot_forecast_pair_step(data, t_in, t, phys, bio, device)
        pr = phi_rule[int(t)]
        pl = phi_leg[int(t)]
        br = _clot_metrics(pr, step.phi_gt, step.loss_mask)
        bl = _clot_metrics(pl, step.phi_gt, step.loss_mask)
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_rule": pr.detach().cpu().numpy(),
                "phi_leg": pl.detach().cpu().numpy(),
                "f1_rule": float(br["clot_f1"]),
                "f1_leg": float(bl["clot_f1"]),
                "pred_frac_rule": float(br["pred_pos_frac"]),
                "pred_frac_leg": float(bl["pred_pos_frac"]),
            }
        )

    ncols = len(frames)
    fig, axes = plt.subplots(2, ncols, figsize=(max(2.5 * ncols, 10), 5.5), squeeze=False)
    fig.suptitle(
        f"{tag} -- {args.anchor} | row0=frozen rule budget | row1={args.leg}",
        fontsize=11,
    )
    for j, fr in enumerate(frames):
        title = (
            f"t={fr['t']} tau={fr['tau']:.2f}\n"
            f"rule F1={fr['f1_rule']:.2f} leg F1={fr['f1_leg']:.2f}"
        )
        _scatter_fullmesh_region(
            axes[0, j],
            pos,
            fr["phi_rule"],
            full_region,
            "rule" if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=float(args.scatter_size),
            layer_positive_on_top=True,
        )
        _scatter_fullmesh_region(
            axes[1, j],
            pos,
            fr["phi_leg"],
            full_region,
            tag if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=float(args.scatter_size),
            layer_positive_on_top=True,
        )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_v2/{args.leg}_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "leg": args.leg,
                "arch": args.arch,
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_rule", "phi_leg")}
                    for fr in frames
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
