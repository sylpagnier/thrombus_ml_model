"""Zoomed lower-wall panel: GT vs rule phi (highlights recess localization)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_forecast import build_clot_forecast_pair_step  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    rollout_temporal_phi,
    temporal_rule_config_from_env,
)
from src.core_physics.clot_t0_pattern_probe import _wall_mask  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.channel_schema import infer_missing_schema  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _lower_wall_mask(data, device: torch.device) -> torch.Tensor:
    n = int(data.num_nodes)
    wall = _wall_mask(data, device, n)
    pos = data.x[:, :2].to(device)
    ym = pos[wall, 1].median()
    return wall & (pos[:, 1] <= ym)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lower-wall zoom clot viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--t-out", type=int, default=37)
    ap.add_argument("--pad", type=float, default=0.15, help="bbox pad fraction")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    cfg = temporal_rule_config_from_env()

    path = root / f"data/processed/graphs_biochem_anchors/{args.anchor}.pt"
    data = torch.load(path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")

    lower = _lower_wall_mask(data, device)
    pos = data.x[:, :2].detach().cpu().numpy()
    lw = pos[lower.cpu().numpy()]
    xmin, ymin = lw.min(axis=0)
    xmax, ymax = lw.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = dx * args.pad, dy * args.pad

    phi_by_t = rollout_temporal_phi(data, cfg, device=device, phys_cfg=phys, bio_cfg=bio)
    t_out = max(0, min(args.t_out, int(data.y.shape[0]) - 1))
    phi = phi_by_t[t_out].detach().cpu().numpy()
    step = build_clot_forecast_pair_step(data, 0, t_out, phys, bio, device)
    phi_gt = step.phi_gt.detach().cpu().numpy()
    ceiling = resolve_ceiling_mask(data, device, bio).detach().cpu().numpy().astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"{args.anchor} lower wall zoom @ t_out={t_out} | {cfg.describe()}", fontsize=11)
    _scatter_fullmesh_region(
        axes[0], pos, phi_gt, step.loss_mask.cpu().numpy(), "GT phi (band)", s=12, mask_outside_region=True
    )
    _scatter_fullmesh_region(
        axes[1], pos, phi, ceiling, "Rule phi (ceiling)", s=12, mask_outside_region=True
    )
    for ax in axes:
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal")

    out = Path(args.out) if args.out else root / f"outputs/biochem/viz/clot_deploy/lower_wall_zoom_{args.anchor}_t{t_out}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
