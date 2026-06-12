"""V2 viz: V1 nucleation (in-window) vs V2 continuous tau (in-window + extrap)."""

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
from src.core_physics.clot_continuous_time import (  # noqa: E402
    comsol_final_index,
    macro_tau_at_index,
    rollout_time_indices,
)
from src.core_physics.clot_forecast import build_clot_forecast_pair_step  # noqa: E402
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_eval_device  # noqa: E402
from src.training.clot_ml_step1_residual import load_step1_checkpoint, resolve_step1_rule_cfg  # noqa: E402
from src.training.clot_ml_v2_continuous_tau import rollout_step1_v2_continuous_tau  # noqa: E402
from src.training.clot_ml_v2_step1_nucleation import rollout_step1_v1_nucleation  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(t_indices: list[int], max_frames: int) -> list[int]:
    if not t_indices:
        return []
    if max_frames <= 0 or len(t_indices) <= max_frames:
        return list(t_indices)
    idx = np.linspace(0, len(t_indices) - 1, num=max_frames, dtype=int)
    return sorted({int(t_indices[i]) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 continuous tau timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--step1-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--sim-end-scale", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=14)
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
    t_comsol = comsol_final_index(data)

    rule_cfg = resolve_step1_rule_cfg(root / args.step0_json)
    model, meta = load_step1_checkpoint(root / args.step1_ckpt, device=device)
    alpha = float(meta.get("alpha", 0.35))
    scale = float(args.sim_end_scale)

    reset_temporal_kinematics_cache()
    phi_v1 = rollout_step1_v1_nucleation(
        data, rule_cfg, model, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
    )

    reset_temporal_kinematics_cache()
    phi_v2 = rollout_step1_v2_continuous_tau(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
        alpha=alpha,
        sim_end_scale=scale,
    )

    t_all = rollout_time_indices(data, sim_end_scale=scale)
    times = _pick_times(t_all, int(args.max_frames))

    frames: list[dict] = []
    for t in times:
        t_in = max(0, t - 1)
        in_window = int(t) <= t_comsol
        step = None
        if in_window:
            step = build_clot_forecast_pair_step(data, t_in, t, phys, bio, device)
        pv1 = phi_v1.get(int(t))
        pv2 = phi_v2[int(t)]
        if pv1 is None:
            pv1 = pv2
        elig = resolve_nucleation_eligibility(
            data, t, device, phys, bio, growth_seed="pred", phi_pred_by_time=phi_v2
        )
        row: dict = {
            "t": int(t),
            "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
            "in_window": bool(in_window),
            "phi_v1": pv1.detach().cpu().numpy(),
            "phi_v2": pv2.detach().cpu().numpy(),
            "elig": elig.detach().cpu().numpy().astype(bool),
        }
        if step is not None:
            bc = _clot_metrics(pv1, step.phi_gt, step.loss_mask)
            bn = _clot_metrics(pv2, step.phi_gt, step.loss_mask)
            row["f1_v1"] = float(bc["clot_f1"])
            row["f1_v2"] = float(bn["clot_f1"])
            row["pred_frac_v1"] = float(bc["pred_pos_frac"])
            row["pred_frac_v2"] = float(bn["pred_pos_frac"])
        else:
            row["pred_frac_v2"] = float((pv2.reshape(-1) >= 0.5).float().mean().item())
            row["pred_frac_v1"] = float((pv1.reshape(-1) >= 0.5).float().mean().item())
        frames.append(row)

    ncols = len(frames)
    fig, axes = plt.subplots(3, ncols, figsize=(max(2.4 * ncols, 12), 7.5), squeeze=False)
    fig.suptitle(
        f"V2 continuous tau -- {args.anchor} | scale={scale:.1f} | "
        f"row0=V1 in-window | row1=V2 in-window | row2=V2 extrap",
        fontsize=10,
    )

    for j, fr in enumerate(frames):
        extrap_tag = " EXTRAP" if not fr["in_window"] else ""
        if fr["in_window"]:
            title = (
                f"t={fr['t']} tau={fr['tau']:.2f}{extrap_tag}\n"
                f"v1 F1={fr.get('f1_v1', float('nan')):.2f} "
                f"v2 F1={fr.get('f1_v2', float('nan')):.2f}"
            )
        else:
            title = (
                f"t={fr['t']} tau={fr['tau']:.2f}{extrap_tag}\n"
                f"pred+ v1={fr['pred_frac_v1']:.3f} v2={fr['pred_frac_v2']:.3f}"
            )
        _scatter_fullmesh_region(
            axes[0, j], pos, fr["phi_v1"], fr["elig"], "V1" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=float(args.scatter_size), layer_positive_on_top=True,
        )
        _scatter_fullmesh_region(
            axes[1, j], pos, fr["phi_v2"], fr["elig"], "V2 in" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=float(args.scatter_size), layer_positive_on_top=True,
        )
        _scatter_fullmesh_region(
            axes[2, j], pos, fr["phi_v2"], fr["elig"], "V2 extrap" if j == 0 else "",
            cmap="Oranges", vmin=0, vmax=1, s=float(args.scatter_size), layer_positive_on_top=True,
        )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_default = root / f"outputs/biochem/viz/clot_v2/v2_tau_{args.anchor}_s{int(scale)}.png"
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
                "sim_end_scale": scale,
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_v1", "phi_v2", "elig")}
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
