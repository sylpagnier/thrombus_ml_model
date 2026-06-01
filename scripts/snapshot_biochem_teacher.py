"""Headless PNG snapshot of biochem teacher rollout vs COMSOL (GT flow when env set).

Usage:
  python scripts/snapshot_biochem_teacher.py --checkpoint outputs/biochem/biochem_teacher_last.pth
  python scripts/snapshot_biochem_teacher.py --checkpoint ... --anchor patient007 --out outputs/biochem/viz/gnode91_p007.png
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
from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.utils.channel_schema import assert_graph_schema, infer_missing_schema
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _apply_gt_flow_env() -> None:
    os.environ.setdefault("BIOCHEM_GT_KINE_VEL", "1")
    os.environ.setdefault("BIOCHEM_GT_KINE_SKIP_DEQ", "1")
    os.environ.setdefault("BIOCHEM_TEACHER_MU_RATIO_MAX", "1.0")
    os.environ.setdefault("BIOCHEM_ADJOINT_RK4_SUBSTEPS", "1")
    os.environ.setdefault("BIOCHEM_TBPTT_MAX_WINDOW", "6")
    os.environ.setdefault("VIZ_FAST", "1")


def _resolve_y_index(names: list[str], key: str, fallback: int) -> int:
    if key in names:
        return int(names.index(key))
    return fallback


def _scatter_panel(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=vals,
        s=10,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> int:
    ap = argparse.ArgumentParser(description="Save teacher rollout vs GT species/flow PNG.")
    ap.add_argument("--checkpoint", default="outputs/biochem/biochem_teacher_last.pth")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument(
        "--out",
        default="",
        help="Output PNG (default outputs/biochem/viz/teacher_snapshot_<anchor>.png)",
    )
    ap.add_argument(
        "--time-indices",
        default="0,-1",
        help="Comma-separated time indices into rolled-out series (default 0,-1).",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    _apply_gt_flow_env()
    root = get_project_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = root / args.checkpoint
    if not ckpt_path.is_file():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchor_path = anchor_dir / f"{args.anchor}.pt"
    if not anchor_path.is_file():
        print(f"[ERR] anchor not found: {anchor_path}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    teacher = _build_teacher(ckpt, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device)

    data = torch.load(anchor_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data)

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

    y_names = list(getattr(data, "y_channel_names", []) or [])
    if not y_names:
        y_names = [
            "u_nd",
            "v_nd",
            "p_nd",
            "mu_eff_nd",
            "RP_log1p_nd",
            "AP_log1p_nd",
            "APR_log1p_nd",
            "APS_log1p_nd",
            "PT_log1p_nd",
            "T_log1p_nd",
            "AT_log1p_nd",
            "FG_log1p_nd",
            "FI_log1p_nd",
            "M_log1p_nd",
            "Mas_log1p_nd",
            "Mat_log1p_nd",
        ]
    i_fi = _resolve_y_index(y_names, "FI_log1p_nd", 12)
    i_mat = _resolve_y_index(y_names, "Mat_log1p_nd", 15)

    pos = data.x[:, :2].detach().cpu().numpy()
    y_gt = data.y.detach().cpu().numpy()
    y_pr = pred.detach().cpu().numpy()
    n_t = int(y_gt.shape[0])

    t_indices: list[int] = []
    for raw in (args.time_indices or "0,-1").split(","):
        raw = raw.strip()
        if not raw:
            continue
        ti = int(raw)
        if ti < 0:
            ti = n_t + ti
        ti = max(0, min(ti, n_t - 1))
        if ti not in t_indices:
            t_indices.append(ti)
    if not t_indices:
        t_indices = [0, n_t - 1]

    n_rows = len(t_indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.6 * n_rows), squeeze=False)
    for row, ti in enumerate(t_indices):
        gt = y_gt[ti]
        pr = y_pr[ti]
        speed_gt = np.sqrt(np.maximum(gt[:, 0] ** 2 + gt[:, 1] ** 2, 0.0))
        speed_pr = np.sqrt(np.maximum(pr[:, 0] ** 2 + pr[:, 1] ** 2, 0.0))
        fi_gt, fi_pr = gt[:, i_fi], pr[:, i_fi]
        mat_gt, mat_pr = gt[:, i_mat], pr[:, i_mat]
        vmax_fi = float(np.nanpercentile(np.concatenate([fi_gt, fi_pr]), 99))
        vmax_mat = float(np.nanpercentile(np.concatenate([mat_gt, mat_pr]), 99))
        vmax_sp = float(np.nanpercentile(np.concatenate([speed_gt, speed_pr]), 99))
        if row == n_rows - 1:
            panels = [
                (speed_gt, f"t={ti} GT |u|", "plasma", 0.0, max(vmax_sp, 1e-6)),
                (speed_pr, f"t={ti} Pred |u|", "plasma", 0.0, max(vmax_sp, 1e-6)),
                (mat_gt, f"t={ti} GT Mat log1p", "cividis", 0.0, max(vmax_mat, 1e-6)),
                (mat_pr, f"t={ti} Pred Mat log1p", "cividis", 0.0, max(vmax_mat, 1e-6)),
            ]
        else:
            panels = [
                (speed_gt, f"t={ti} GT |u|", "plasma", 0.0, max(vmax_sp, 1e-6)),
                (speed_pr, f"t={ti} Pred |u|", "plasma", 0.0, max(vmax_sp, 1e-6)),
                (fi_gt, f"t={ti} GT FI log1p", "magma", 0.0, max(vmax_fi, 1e-6)),
                (fi_pr, f"t={ti} Pred FI log1p", "magma", 0.0, max(vmax_fi, 1e-6)),
            ]
        for col, (vals, title, cmap, vmin, vmax) in enumerate(panels):
            _scatter_panel(axes[row, col], pos, vals, title, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.suptitle(
        f"Teacher snapshot — {args.anchor} ({ckpt_path.name})",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "outputs" / "biochem" / "viz" / f"teacher_snapshot_{args.anchor}.png"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  Wrote {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
