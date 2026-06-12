"""V1 viz: ceiling step1 vs nucleation step1 phi timeline."""

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
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_eval_device  # noqa: E402
from src.training.clot_ml_step1_residual import (  # noqa: E402
    apply_step1_eval_env,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
)
from src.training.clot_ml_v2_step1_nucleation import rollout_step1_v1_nucleation  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="V1 ceiling vs nucleation phi viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--max-frames", type=int, default=12)
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
    n_steps = int(data.y.shape[0])

    rule_cfg = resolve_step1_rule_cfg(root / args.step0_json)
    model, meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(meta.get("alpha", 0.35))

    reset_temporal_kinematics_cache()
    apply_step1_eval_env()
    phi_ceil = rollout_step1_phi(
        data, rule_cfg, model, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
    )

    reset_temporal_kinematics_cache()
    phi_nuc = rollout_step1_v1_nucleation(
        data, rule_cfg, model, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
    )

    frames: list[dict] = []
    for t in _pick_times(n_steps, int(args.max_frames)):
        t_in = max(0, t - 1)
        step = build_clot_forecast_pair_step(data, t_in, t, phys, bio, device)
        pc = phi_ceil[int(t)]
        pn = phi_nuc[int(t)]
        elig = resolve_nucleation_eligibility(
            data, t, device, phys, bio, growth_seed="pred", phi_pred_by_time=phi_nuc
        )
        bc = _clot_metrics(pc, step.phi_gt, step.loss_mask)
        bn = _clot_metrics(pn, step.phi_gt, step.loss_mask)
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_ceiling": pc.detach().cpu().numpy(),
                "phi_nucleation": pn.detach().cpu().numpy(),
                "elig": elig.detach().cpu().numpy().astype(bool),
                "f1_ceil": float(bc["clot_f1"]),
                "f1_nuc": float(bn["clot_f1"]),
                "pred_frac_ceil": float(bc["pred_pos_frac"]),
                "pred_frac_nuc": float(bn["pred_pos_frac"]),
            }
        )

    ncols = len(frames)
    fig, axes = plt.subplots(2, ncols, figsize=(max(2.6 * ncols, 10), 5.5), squeeze=False)
    fig.suptitle(
        f"V1 step1 -- {args.anchor} | row0=ceiling mask | row1=nucleation E_seed",
        fontsize=11,
    )

    for j, fr in enumerate(frames):
        title = (
            f"t={fr['t']} tau={fr['tau']:.2f}\n"
            f"ceil F1={fr['f1_ceil']:.2f} nuc F1={fr['f1_nuc']:.2f}"
        )
        _scatter_fullmesh_region(
            axes[0, j], pos, fr["phi_ceiling"], fr["elig"], "ceiling" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=float(args.scatter_size), layer_positive_on_top=True,
        )
        _scatter_fullmesh_region(
            axes[1, j], pos, fr["phi_nucleation"], fr["elig"], "nucleation" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=float(args.scatter_size), layer_positive_on_top=True,
        )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_default = root / f"outputs/biochem/viz/clot_v2/v1_step1_{args.anchor}_ceiling_vs_nuc.png"
    out_path = Path(args.out) if args.out.strip() else out_default
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
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_ceiling", "phi_nucleation", "elig")}
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
