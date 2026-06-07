"""Visualize simple clot-phi model vs capped GT on one anchor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState, clot_phi_rollout_enabled
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_hybrid_enabled,
    clot_phi_mask_mode,
    clot_phi_minimal_features_enabled,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
)
from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_config_from_checkpoint
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _ensure_training_mask_env() -> None:
    """Match scripts/_clot_phi_shared_env.ps1 so viz uses the same region as train."""
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_REF_TIME": "0",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_MINIMAL_FEATURES": "1",
        "CLOT_PHI_HYBRID": "1",
        "CLOT_PHI_SOFT_LABELS": "1",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Viz clot_phi_simple model")
    parser.add_argument("--anchor", default="patient007", help="Anchor stem (default patient007)")
    parser.add_argument(
        "--checkpoint",
        default="outputs/biochem/clot_phi_best.pth",
        help="Path to clot_phi_best.pth",
    )
    parser.add_argument("--time-index", type=int, default=-1, help="Time index (-1 = final)")
    parser.add_argument(
        "--plot-mode",
        choices=("tri", "scatter"),
        default="tri",
        help="tri=gouraud tripcolor (can look like full-wall rim); scatter=points in region only",
    )
    parser.add_argument("--out", default="", help="Output PNG path (default auto)")
    args = parser.parse_args()
    _ensure_training_mask_env()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchor_dir = anchor_dir.resolve()
    graph_path = anchor_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = root / args.checkpoint
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw.get("config", {})
    hidden = int(cfg.get("hidden", 64))
    in_dim = int(cfg.get("in_dim", 6))
    oracle_mu = bool(cfg.get("oracle_mu", False))
    os.environ["CLOT_PHI_ORACLE_MU"] = "1" if oracle_mu else "0"
    if "species_features" in cfg:
        os.environ["CLOT_PHI_SPECIES_FEATURES"] = "1" if bool(cfg.get("species_features")) else "0"
    if "joint_bio" in cfg:
        os.environ["CLOT_PHI_JOINT_BIO"] = "1" if bool(cfg.get("joint_bio")) else "0"
    if "use_prior_features" in cfg:
        use_prior = bool(cfg.get("use_prior_features"))
        prior_n = int(cfg.get("prior_n", 2))
    else:
        # Backward-compatible inference for older checkpoints:
        # base=6, +1 if oracle_mu, remainder are prior columns.
        prior_n = max(0, in_dim - 6 - (1 if oracle_mu else 0))
        use_prior = prior_n > 0
    os.environ["CLOT_PHI_USE_PRIOR_FEATURES"] = "1" if use_prior else "0"
    os.environ["CLOT_PHI_PRIOR_N"] = str(max(0, prior_n))
    if "minimal_features" in cfg:
        os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1" if bool(cfg.get("minimal_features")) else "0"
    if "hybrid" in cfg:
        os.environ["CLOT_PHI_HYBRID"] = "1" if bool(cfg.get("hybrid")) else "0"
    if "mlp_depth" in cfg:
        os.environ["CLOT_PHI_MLP_DEPTH"] = str(int(cfg.get("mlp_depth") or 1))
    if "dropout" in cfg:
        os.environ["CLOT_PHI_DROPOUT"] = str(float(cfg.get("dropout") or 0.0))
    if "model_kind" in cfg:
        os.environ["CLOT_PHI_MODEL"] = str(cfg.get("model_kind") or "mlp")
    apply_clot_phi_config_from_checkpoint(cfg)
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    data = torch.load(graph_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    rollout_state = ClotPhiRolloutState() if clot_phi_rollout_enabled() else None
    with torch.no_grad():
        step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, device, rollout_state=rollout_state)
        if rollout_state is not None and ti > 0:
            rollout_state = ClotPhiRolloutState()
            for t_run in range(0, ti):
                step_run = build_clot_phi_step(
                    data, t_run, phys_cfg, bio_cfg, device, rollout_state=rollout_state
                )
                phi_r = model(step_run.features)
                if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
                    mu_r = mu_eff_from_delta_log_si(
                        step_run.mu_c_si, model.forward_delta_log_mu(step_run.features)
                    )
                else:
                    mu_r = log_blend_mu_eff_si(step_run.mu_c_si, phi_r)
                rollout_state.update_from_pred(phi_r, mu_r, detach=True)
            step = build_clot_phi_step(
                data, ti, phys_cfg, bio_cfg, device, rollout_state=rollout_state
            )
        phi_pred = model(step.features)
        if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
            mu_pred = mu_eff_from_delta_log_si(step.mu_c_si, model.forward_delta_log_mu(step.features))
        else:
            mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)

    pos = data.x[:, :2].detach().cpu().numpy()
    m = step.region.detach().cpu().numpy().astype(bool)
    phi_gt = step.phi_gt.detach().cpu().numpy()
    phi_pr = phi_pred.detach().cpu().numpy()
    mu_gt = step.mu_gt_cap.detach().cpu().numpy()
    mu_pr = mu_pred.detach().cpu().numpy()

    idx = np.where(m)[0]
    gt_pos = int((phi_gt[m] > 0.05).sum())
    y_sl = data.y[ti].to(device=device, dtype=torch.float32)
    pred_state = y_sl.clone()
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(
        torch.tensor(mu_pr, device=device, dtype=torch.float32)
    )
    shape_m = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys_cfg,
    )
    print(
        f"[i]  t={ti} region_n={int(m.sum())} gt_pos_n={gt_pos} "
        f"mean_pred_phi={float(phi_pr[m].mean()):.3f} mean_gt_phi={float(phi_gt[m].mean()):.3f} "
        f"frac_pred_phi>=0.5={float((phi_pr[m] >= 0.5).mean()):.3f} | "
        f"clot_shape={shape_m['clot_shape']:.3f} (full mesh mu>={shape_m['clot_mu_thresh_si']:.3f}) "
        f"rec={shape_m['clot_recall']:.3f} pred_frac={shape_m['clot_pred_frac']:.3f} gt_frac={shape_m['clot_gt_frac']:.3f}",
        flush=True,
    )

    def _plot_panel(vals, title, vmin=None, vmax=None, cmap="hot"):
        ax = fig.add_subplot(2, 2, plot_i[0])
        plot_i[0] += 1
        if args.plot_mode == "scatter":
            sc = ax.scatter(
                pos[idx, 0],
                pos[idx, 1],
                c=vals[idx],
                s=14,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, fraction=0.046)
        else:
            v = np.where(m, vals, np.nan)
            triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
            tri_pts = pos[triang.triangles]
            d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
            d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
            d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
            max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)
            triang.set_mask(max_edge_sq > (np.median(max_edge_sq) * 10.0))
            tc = ax.tripcolor(triang, v, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
            fig.colorbar(tc, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    fig = plt.figure(figsize=(12, 10))
    plot_i = [1]
    band = clot_phi_mask_mode()
    _plot_panel(phi_gt, f"GT phi (t={ti}) {band} band", 0.0, 1.0, "Reds")
    _plot_panel(phi_pr, f"Pred phi", 0.0, 1.0, "Reds")
    cap = float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))
    _plot_panel(mu_gt, f"GT mu cap {cap:.2f} Pa*s", 0.0, cap, "bwr")
    _plot_panel(mu_pr, f"Pred mu eff", 0.0, cap, "bwr")
    fig.suptitle(f"clot_phi_simple — {args.anchor} (ckpt {ckpt_path.name})", fontsize=14)
    fig.tight_layout()

    out = Path(args.out) if args.out else root / "outputs" / "biochem" / f"clot_phi_viz_{args.anchor}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[OK]  Wrote {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
