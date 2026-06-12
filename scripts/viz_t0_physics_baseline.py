"""T0 baseline physics viz: mu, gamma, gelation, phi (GT vs deploy physics).

Usage::

    python scripts/viz_t0_physics_baseline.py --anchor patient007
"""

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
from src.core_physics.clot_phi_simple import (  # noqa: E402
    build_clot_phi_step,
    cap_mu_eff_si,
    physics_mu_eff_si,
    resolve_gamma_dot_nd_for_carreau,
    species_log1p_nd_to_si,
    mu1_comsol_from_mat_si,
    mu2_comsol_from_fi_si,
)
from src.core_physics.clot_trigger_rollout import rollout_clot_trigger_physics  # noqa: E402
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1, scatter_clot_vessel  # noqa: E402
from src.training.clot_trigger_stack import apply_clot_trigger_honest_env  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 baseline physics field viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--scatter-size", type=float, default=3.0)
    ap.add_argument("--ratio-max", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    ratio_max = float(args.ratio_max)

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])

    traj = rollout_clot_trigger_physics(data, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1)
    step0 = build_clot_phi_step(data, 0, phys, bio, device)
    mu_anchor = cap_mu_eff_si(
        physics_mu_eff_si(
            step0.mu_c_si,
            step0.species_log_gt,
            bio,
            device=device,
            data=data,
            u_nd=step0.u_flow_nd,
            v_nd=step0.v_flow_nd,
            phys_cfg=phys,
            time_index=0,
        )
    ).reshape(-1)
    mu_gt_t0 = phys.viscosity_nd_to_si(data.y[0, :, 3])

    row_labels = [
        "mu_GT (spf.mu)",
        "mu_physics (comsol_carreau)",
        "|mu_phys - mu_GT|",
        "log10 gamma_nd (resolved)",
        "phi_GT",
        "phi_deploy",
        "dmu_phys (above t0 anchor)",
        "gel M (Mat+FI hard)",
    ]
    nrows = len(row_labels)
    times = _pick_times(n_steps, int(args.max_frames))
    ncols = len(times)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.2 * nrows), squeeze=False)
    fig.suptitle(
        f"T0 baseline physics -- {args.anchor} | gamma=max(graph,poi,|u|/width) | M_max={ratio_max}",
        fontsize=11,
    )

    mu_phys_all: list[torch.Tensor] = []
    mu_gt_all: list[torch.Tensor] = []

    for j, t in enumerate(times):
        y = data.y[int(t)]
        u_nd, v_nd = y[:, 0], y[:, 1]
        mu_gt = phys.viscosity_nd_to_si(y[:, 3])
        mu_gt_all.append(mu_gt)
        sp = species_log1p_nd_to_si(y[:, 4:16], bio)
        gel = (
            mu1_comsol_from_mat_si(sp[:, 11], bio, ratio_max)
            + mu2_comsol_from_fi_si(sp[:, 8], bio, ratio_max)
        )
        step = build_clot_phi_step(data, int(t), phys, bio, device)
        mu_phys = cap_mu_eff_si(
            physics_mu_eff_si(
                step.mu_c_si,
                step.species_log_gt,
                bio,
                device=device,
                data=data,
                u_nd=step.u_flow_nd,
                v_nd=step.v_flow_nd,
                phys_cfg=phys,
                time_index=int(t),
            )
        ).reshape(-1)
        mu_phys_all.append(mu_phys)
        g_res = resolve_gamma_dot_nd_for_carreau(data, u_nd, v_nd, device=device)
        log_g = torch.log10(g_res.clamp(min=1e-12))
        err = (mu_phys - mu_gt).abs()
        dmu = (mu_phys - mu_anchor).clamp(min=0.0)
        phi_gt = step.phi_gt.reshape(-1)
        phi_dep = traj[int(t)]["phi"].reshape(-1)
        tau = float(macro_tau_at_index(data, int(t), bio_cfg=bio))
        m_f1 = clot_trigger_viz_f1(phi_dep, phi_gt, step.loss_mask.reshape(-1).bool())
        col_title = f"t={t} tau={tau:.2f}\nF1={m_f1['clot_f1']:.2f}"

        panels = [
            (mu_gt.cpu().numpy(), "viridis", 0.0, 0.10),
            (mu_phys.cpu().numpy(), "viridis", 0.0, 0.10),
            (err.cpu().numpy(), "magma", 0.0, None),
            (log_g.cpu().numpy(), "plasma", None, None),
            (phi_gt.cpu().numpy(), "bwr", 0.0, 1.0),
            (phi_dep.cpu().numpy(), "bwr", 0.0, 1.0),
            (dmu.cpu().numpy(), "inferno", 0.0, 0.05),
            (gel.cpu().numpy(), "cividis", 1.0, float(ratio_max)),
        ]
        for i, (vals, cmap, vmin, vmax) in enumerate(panels):
            title = row_labels[i] if j == 0 else ""
            if i >= 4:
                scatter_clot_vessel(
                    axes[i, j],
                    pos,
                    vals,
                    title,
                    scatter_size=float(args.scatter_size),
                )
            else:
                _scatter_fullmesh(
                    axes[i, j],
                    pos,
                    vals,
                    title,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=float(args.scatter_size),
                )
            if i == 0:
                axes[i, j].set_title(col_title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_physics_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    bulk = mu_gt_t0 < 0.012
    summary = {
        "anchor": args.anchor,
        "times": times,
        "t0_bulk_gt_over_phys_median": float(
            (mu_gt_t0[bulk] / mu_phys_all[0][bulk].clamp(min=1e-8)).median().item()
        )
        if bulk.any()
        else None,
        "gamma_mode": "max",
        "mu_base": "comsol_carreau",
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
