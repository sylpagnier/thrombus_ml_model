"""Headless clot-band PNG for biochem teacher vs COMSOL (same layout as viz_clot_phi_simple).

Compares GT phi / capped mu_eff vs teacher rollout mu_eff in the supervision band
(neighbor + dgamma slice when env matches clot-phi training).

Usage:
  python scripts/snapshot_biochem_teacher_clotband.py --checkpoint outputs/biochem/biochem_teacher_last.pth
  python scripts/snapshot_biochem_teacher_clotband.py --checkpoint ... --anchor-dir outputs/biochem/passive_species_clotband_focus/anchors_clotband_72 --time-index 4
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

from scripts.dump_teacher_species_to_anchors import _build_teacher
from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_simple import (
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_mask_mode,
    clot_phi_mu_cap_si,
    phi_gt_binary,
    phi_gt_soft,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _apply_gt_flow_env() -> None:
    os.environ.setdefault("BIOCHEM_GT_KINE_VEL", "1")
    os.environ.setdefault("BIOCHEM_GT_KINE_SKIP_DEQ", "1")
    os.environ.setdefault("BIOCHEM_TEACHER_MU_RATIO_MAX", "1.0")
    os.environ.setdefault("BIOCHEM_ADJOINT_RK4_SUBSTEPS", "1")
    os.environ.setdefault("BIOCHEM_TBPTT_MAX_WINDOW", "6")


def _ensure_clot_phi_mask_env() -> None:
    """Match scripts/_clot_phi_shared_env.ps1 / viz_clot_phi_simple defaults."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_SOFT_LABELS": "1",
        "CLOT_PHI_DGAMMA_FEATURE_TIME": "current",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def _scatter_panel(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    title: str,
    *,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=vals,
        s=14,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def _phi_from_mu_cap(mu_cap: torch.Tensor, mu_c: torch.Tensor, region: torch.Tensor, phys_cfg: PhysicsConfig) -> torch.Tensor:
    use_soft = (os.environ.get("CLOT_PHI_SOFT_LABELS") or "0").strip().lower() in ("1", "true", "yes", "on")
    if use_soft:
        return phi_gt_soft(mu_cap, mu_c, region)
    return phi_gt_binary(mu_cap, region, phys_cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Teacher clot-band snapshot (phi / mu_eff vs GT).")
    ap.add_argument("--checkpoint", default="outputs/biochem/biochem_teacher_last.pth")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument(
        "--anchor-dir",
        default="",
        help="Optional dumped-anchor directory (default: graphs_biochem_anchors).",
    )
    ap.add_argument("--time-index", type=int, default=-1, help="Time index (-1 = final).")
    ap.add_argument("--out", default="", help="Output PNG path.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _apply_gt_flow_env()
    _ensure_clot_phi_mask_env()

    root = get_project_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = root / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    if args.anchor_dir:
        anchor_dir = Path(args.anchor_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchor_path = anchor_dir / f"{args.anchor}.pt"
    if not anchor_path.is_file():
        print(f"[ERR] anchor not found: {anchor_path}", file=sys.stderr)
        return 2

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher = _build_teacher(ckpt, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device)

    data = torch.load(anchor_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    n_t = int(data.y.shape[0])
    ti = int(args.time_index)
    if ti < 0:
        ti = n_t + ti
    ti = max(0, min(ti, n_t - 1))

    t_si = bio_cfg.resolve_biochem_times(data, device)
    eval_t = to_t_nd(t_si, float(getattr(bio_cfg, "t_final", 30000.0)))

    with torch.no_grad():
        pred = teacher(
            data,
            eval_t,
            y_true_trajectory=data.y,
            teacher_forcing_ratio=1.0,
            start_idx=0,
            initial_species=None,
            detach_macro_state=True,
        )
    if isinstance(pred, tuple):
        pred = pred[0]

    step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, device, rollout_state=None)
    pred_nd = pred[ti, :, STATE_CHANNEL_MU_EFF_ND].to(device=device, dtype=torch.float32)
    mu_pred_si = phys_cfg.viscosity_nd_to_si(pred_nd)
    mu_pred_cap = cap_mu_eff_si(mu_pred_si)
    phi_pred = _phi_from_mu_cap(mu_pred_cap, step.mu_c_si, step.region, phys_cfg)

    pos = data.x[:, :2].detach().cpu().numpy()
    m = step.region.detach().cpu().numpy().astype(bool)
    idx = np.where(m)[0]
    phi_gt = step.phi_gt.detach().cpu().numpy()
    phi_pr = phi_pred.detach().cpu().numpy()
    mu_gt = step.mu_gt_cap.detach().cpu().numpy()
    mu_pr = mu_pred_cap.detach().cpu().numpy()

    gt_pos = int((phi_gt[m] > 0.05).sum())
    pr_pos = int((phi_pr[m] >= 0.5).sum())
    print(
        f"[i]  t={ti} region_n={int(m.sum())} gt_pos_n={gt_pos} "
        f"mean_pred_phi={float(phi_pr[m].mean()):.3f} mean_gt_phi={float(phi_gt[m].mean()):.3f} "
        f"frac_pred_phi>=0.5={float((phi_pr[m] >= 0.5).mean()):.3f} "
        f"mean_pred_mu={float(mu_pr[m].mean()):.4f} mean_gt_mu={float(mu_gt[m].mean()):.4f}",
        flush=True,
    )

    cap = float(clot_phi_mu_cap_si())
    band = clot_phi_mask_mode()
    fig = plt.figure(figsize=(12, 10))
    panels = [
        (phi_gt, f"GT phi (t={ti}) {band} band", 0.0, 1.0, "Reds"),
        (phi_pr, "Teacher pred phi (from mu_eff)", 0.0, 1.0, "Reds"),
        (mu_gt, f"GT mu cap {cap:.2f} Pa*s", 0.0, cap, "bwr"),
        (mu_pr, "Teacher pred mu_eff cap", 0.0, cap, "bwr"),
    ]
    for plot_i, (vals, title, vmin, vmax, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, plot_i)
        _scatter_panel(ax, pos[idx], vals[idx], title, cmap=cmap, vmin=vmin, vmax=vmax)

    src = anchor_dir.name if args.anchor_dir else "biochem_anchors"
    fig.suptitle(
        f"Teacher clot-band — {args.anchor} ({ckpt_path.name}, {src}, t={ti})",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "outputs" / "biochem" / "viz" / f"teacher_clotband_{args.anchor}_t{ti}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  Wrote {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
