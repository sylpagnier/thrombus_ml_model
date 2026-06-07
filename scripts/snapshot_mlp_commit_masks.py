"""Headless PNG: oracle gt_clot vs deploy neighbor MLP commit masks @ t_final.

Usage:
  python scripts/snapshot_mlp_commit_masks.py --anchor patient007
  python scripts/snapshot_mlp_commit_masks.py --anchor patient007 --leg B_deploy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run_mlp_clot_inject_probe import (  # noqa: E402
    _configure_leg,
    _load_teacher,
    _normalize_probe_leg,
    _rollout,
)
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_mu_inject import (
    compute_mlp_commit_gates_at_rollout_frame,
    mlp_mu_map_mask_mode,
)
from src.evaluation.clot_phi_checkpoint_env import (  # noqa: E402
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.evaluation.visualize_pipeline import _nearest_time_indices  # noqa: E402
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.core_physics.clot_phi_simple import build_clot_phi_model
from src.utils.paths import get_project_root


def _load_clot_model(ckpt: Path, device: torch.device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    model = build_clot_phi_model(
        in_dim=int(cfg.get("in_dim", 6)), hidden=int(cfg.get("hidden", 32))
    ).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model


def _mask_png(
    pos: np.ndarray,
    gt_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    anchor: str,
    leg: str,
    active_mode: str,
    t_si: float,
    out_path: Path,
) -> None:
    _C_COMMIT = "#c0392b"
    _C_BULK = "#bdc3c7"
    overlap = int((gt_mask & active_mask).sum())
    gt_n = int(gt_mask.sum())
    act_n = int(active_mask.sum())
    dice = (2.0 * overlap / (gt_n + act_n)) if (gt_n + act_n) > 0 else 0.0

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mask, title in (
        (axs[0], gt_mask, f"Oracle commit (gt_clot) n={gt_n}"),
        (axs[1], active_mask, f"Active commit ({active_mode}) n={act_n}"),
    ):
        colors = np.where(mask.reshape(-1), _C_COMMIT, _C_BULK)
        ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        f"MLP commit masks leg={leg} anchor={anchor} t~{t_si:.0f}s "
        f"overlap={overlap} dice={dice:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {out_path}", flush=True)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot MLP commit masks (gt_clot vs active)")
    ap.add_argument("--teacher-checkpoint", default="outputs/biochem/clot_baseline/teacher_best_high_mu.pth")
    ap.add_argument("--clot-phi-checkpoint", default="outputs/biochem/clot_baseline/clot_phi_best.pth")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--leg", default="B_deploy", help="B or B_deploy (sets active env mask)")
    ap.add_argument("--time-stride", type=int, default=5)
    ap.add_argument("--mu-ratio-max", type=float, default=20.0)
    ap.add_argument("--out-dir", default="outputs/biochem/viz/commit_masks")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_ROLLOUT_PROGRESS"] = "0"

    leg = _normalize_probe_leg(args.leg)
    if leg not in ("B", "B_deploy"):
        print("[ERR] --leg must be B or B_deploy", file=sys.stderr)
        return 1

    ckpt = root / args.teacher_checkpoint.replace("/", os.sep)
    clot_ckpt = root / args.clot_phi_checkpoint.replace("/", os.sep)
    graph_path = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not ckpt.is_file() or not clot_ckpt.is_file() or not graph_path.is_file():
        print("[ERR] missing ckpt or anchor graph", file=sys.stderr)
        return 1

    mu_ratio = args.mu_ratio_max
    teacher, phys, bio = _load_teacher(ckpt, device, mu_ratio, fast=False)
    _configure_leg(teacher, device, leg, clot_ckpt=clot_ckpt)
    attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))

    data = torch.load(graph_path, map_location=device, weights_only=False)
    pred = _rollout(teacher, data, bio, device, time_stride=max(1, int(args.time_stride)), fast=False)
    clot_model = _load_clot_model(clot_ckpt, device)

    t_si = bio.resolve_biochem_times(data, device)
    ti = int(t_si.shape[0]) - 1
    pred_ti = min(ti, int(pred.shape[0]) - 1)
    t_val = float(t_si[ti].item())

    pred_t = pred[pred_ti]
    prev_t = pred[pred_ti - 1] if pred_ti > 0 else None
    prev_mu = None
    if prev_t is not None:
        prev_mu = phys.viscosity_nd_to_si(prev_t[:, STATE_CHANNEL_MU_EFF_ND])

    gate_gt, gate_active = compute_mlp_commit_gates_at_rollout_frame(
        clot_model,
        data,
        ti,
        u_nd=pred_t[:, 0],
        v_nd=pred_t[:, 1],
        species_log=pred_t[:, 4:16],
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        prev_mu_eff_si=prev_mu,
    )
    pos = data.x[:, :2].detach().cpu().numpy()
    active_mode = mlp_mu_map_mask_mode()
    out_dir = root / args.out_dir.replace("/", os.sep)
    out_path = out_dir / f"leg_{leg}_commit_masks_{args.anchor}_t{int(t_val)}.png"
    _mask_png(
        pos,
        gate_gt.detach().cpu().numpy().astype(bool),
        gate_active.detach().cpu().numpy().astype(bool),
        anchor=args.anchor,
        leg=leg,
        active_mode=active_mode,
        t_si=t_val,
        out_path=out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
