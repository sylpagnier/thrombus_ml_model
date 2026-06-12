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
from src.core_physics.clot_forecast import (
    build_clot_forecast_pair_step,
    clot_forecast_one_step_enabled,
    clot_forecast_pair_stride,
    clot_forecast_pair_schedule,
)
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState, clot_phi_rollout_enabled
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_hard_support_projection_enabled,
    clot_phi_hybrid_enabled,
    clot_phi_mask_mode,
    clot_support_band_mode,
    clot_phi_thresh_si,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    resolve_clot_support_band_for_step,
)
from src.core_physics.clot_growth_masks import (
    clot_ceiling_hops,
    growth_seed_mode,
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
    resolve_t0_dgamma_wall_mask,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
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


def _clot_binary(mu_si: np.ndarray, thresh: float) -> np.ndarray:
    return (mu_si.reshape(-1) >= float(thresh)).astype(np.float32)


def _resolve_forecast_t_in(t_out: int, t_steps: int) -> int:
    schedule = clot_forecast_pair_schedule()
    if schedule in ("static_final", "from_t0"):
        return 0
    stride = clot_forecast_pair_stride()
    return max(0, int(t_out) - int(stride))


def _project_mu_if_enabled(
    data,
    step,
    mu_pred: torch.Tensor,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    forecast_one_step: bool,
    time_index: int,
) -> torch.Tensor:
    from src.core_physics.clot_phi_simple import project_deploy_mu_with_support

    bulk_t = int(time_index) if forecast_one_step else None
    return project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu_pred,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=forecast_one_step,
        time_index=int(time_index),
        bulk_time_index=bulk_t,
    )


def _predict_mu_phi(
    model,
    data,
    ti: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Return (step, phi_pred, mu_pred projected) for label time index ti."""
    forecast_one_step = clot_forecast_one_step_enabled()
    rollout_state = ClotPhiRolloutState() if clot_phi_rollout_enabled() else None

    with torch.no_grad():
        if forecast_one_step:
            t_out = int(ti)
            t_in = _resolve_forecast_t_in(t_out, int(data.y.shape[0]))
            forecast_state = None
            step = build_clot_forecast_pair_step(
                data,
                t_in,
                t_out,
                phys_cfg,
                bio_cfg,
                device,
                forecast_state=forecast_state,
                train_epoch=None,
            )
            phi_pred = model(step.features)
            if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
                mu_pred = mu_eff_from_delta_log_si(
                    step.mu_c_si, model.forward_delta_log_mu(step.features)
                )
            else:
                mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
            mu_pred = _project_mu_if_enabled(
                data,
                step,
                mu_pred,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                forecast_one_step=True,
                time_index=t_out,
            )
            return step, phi_pred, mu_pred

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
        mu_pred = _project_mu_if_enabled(
            data,
            step,
            mu_pred,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            forecast_one_step=False,
            time_index=int(ti),
        )
        return step, phi_pred, mu_pred


def _scatter_fullmesh(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    title: str,
    *,
    cmap: str = "bwr",
    vmin=None,
    vmax=None,
    s: float = 8,
) -> None:
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=vals.reshape(-1),
        s=s,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def _scatter_fullmesh_region(
    ax,
    pos: np.ndarray,
    vals: np.ndarray,
    region: np.ndarray,
    title: str,
    *,
    cmap: str = "bwr",
    vmin=None,
    vmax=None,
    s: float = 8,
    off_color: str = "#cccccc",
    layer_positive_on_top: bool = False,
    positive_thresh: float = 0.5,
    mask_outside_region: bool = False,
) -> None:
    """Scatter clot phi on the mesh. Default: full vessel (no grey band masking).

    Set ``mask_outside_region=True`` to grey out nodes outside ``region`` (legacy band viz).

    When ``layer_positive_on_top``, draw phi<thresh (blue bulk) under phi>=thresh (red flags)
    so dense ceiling negatives do not occlude sparse positives in scatter order.
    """
    reg = region.reshape(-1).astype(bool)
    v = vals.reshape(-1)
    if not mask_outside_region:
        reg = np.ones(v.shape[0], dtype=bool)
    off = ~reg
    if off.any():
        ax.scatter(
            pos[off, 0],
            pos[off, 1],
            c=off_color,
            s=max(s * 0.65, 1.0),
            linewidths=0,
            alpha=0.45,
            zorder=1,
        )
    if reg.any() and layer_positive_on_top:
        low = reg & (v < float(positive_thresh))
        high = reg & (v >= float(positive_thresh))
        if low.any():
            ax.scatter(
                pos[low, 0],
                pos[low, 1],
                c="#6baed6",
                s=s,
                linewidths=0,
                alpha=0.45,
                zorder=2,
            )
        if high.any():
            ax.scatter(
                pos[high, 0],
                pos[high, 1],
                c="#d62728",
                s=max(s * 1.35, s + 2.0),
                linewidths=0,
                alpha=0.98,
                zorder=4,
            )
        import matplotlib.cm as cm

        norm = plt.Normalize(vmin=0.0 if vmin is None else vmin, vmax=1.0 if vmax is None else vmax)
        sc = cm.ScalarMappable(cmap=cmap, norm=norm)
        sc.set_array([])
    else:
        sc = ax.scatter(
            pos[reg, 0],
            pos[reg, 1],
            c=v[reg],
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.95,
            zorder=2,
        )
    plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def _resolve_support_ceiling_masks(
    data,
    ti: int,
    step,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t0_mask, support_at_t, ceiling_mask) as bool numpy arrays."""
    forecast_one_step = clot_forecast_one_step_enabled()
    if clot_support_band_mode() == "ceiling_growth":
        t0 = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
        ceiling = resolve_ceiling_mask(data, device, bio_cfg)
        support = resolve_growth_support_at_time(data, ti, device, phys_cfg, bio_cfg)
    else:
        t0 = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
        ceiling = resolve_ceiling_mask(data, device, bio_cfg)
        support = resolve_clot_support_band_for_step(
            data,
            device,
            step,
            phys_cfg,
            bio_cfg,
            forecast_one_step=forecast_one_step,
            time_index=int(ti),
        )
    return (
        t0.detach().cpu().numpy().astype(bool),
        support.detach().cpu().numpy().astype(bool),
        ceiling.detach().cpu().numpy().astype(bool),
    )


def _write_mask_overlay_figure(
    *,
    pos: np.ndarray,
    t0_mask: np.ndarray,
    support: np.ndarray,
    ceiling: np.ndarray,
    pred_clot: np.ndarray,
    anchor: str,
    ti: int,
    out: Path,
    scatter_size: float = 6.0,
) -> None:
    """Separate figure: t0 / support / ceiling panels + combined zone overlay."""
    from matplotlib.colors import ListedColormap

    n_t0 = int(t0_mask.sum())
    n_sup = int(support.sum())
    n_ceil = int(ceiling.sum())
    hops = clot_ceiling_hops()
    seed = growth_seed_mode()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, mask, title in (
        (axes[0, 0], t0_mask, f"t0 dgamma wall (n={n_t0})"),
        (axes[0, 1], support, f"support(t={ti}) seed={seed} (n={n_sup})"),
        (axes[1, 0], ceiling, f"ceiling wall+{hops} hops (n={n_ceil})"),
    ):
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=mask.astype(np.float32),
            s=scatter_size,
            cmap="bwr",
            vmin=0,
            vmax=1,
            linewidths=0,
            alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    # Combined: 0=off ceiling, 1=ceiling-only bulk, 2=support no clot, 3=support pred clot
    zone = np.zeros(int(pos.shape[0]), dtype=np.float32)
    zone[ceiling & ~support] = 1.0
    zone[support & ~pred_clot.astype(bool)] = 2.0
    zone[support & pred_clot.astype(bool)] = 3.0
    cmap = ListedColormap(["#404040", "#4a90d9", "#f4a261", "#d62828"])
    ax = axes[1, 1]
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=zone,
        s=scatter_size,
        cmap=cmap,
        vmin=0,
        vmax=3,
        linewidths=0,
        alpha=0.95,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(["off ceiling", "ceiling bulk", "support bulk", "support clot"])
    n_pred_in_sup = int(np.logical_and(support, pred_clot).sum())
    ax.set_title(f"zones + pred clot in support (n={n_pred_in_sup})")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        f"clot masks — {anchor} t={ti} | band={clot_support_band_mode()} ceiling_hops={hops}",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_band_panels(fig, plot_i, args, pos, m, idx, phi_gt, phi_pr, mu_gt, mu_pr, ti, band, cap):
    """Legacy 2x2 band-only phi/mu panels."""

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

    _plot_panel(phi_gt, f"GT phi (t={ti}) {band} band", 0.0, 1.0, "Reds")
    _plot_panel(phi_pr, "Pred phi", 0.0, 1.0, "Reds")
    _plot_panel(mu_gt, f"GT mu cap {cap:.2f} Pa*s", 0.0, cap, "bwr")
    _plot_panel(mu_pr, "Pred mu eff", 0.0, cap, "bwr")


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
        "--layout",
        choices=("fullmesh", "band"),
        default="fullmesh",
        help="fullmesh=GT/pred clot on full mesh + mu row (default); band=legacy loss-band phi",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("tri", "scatter"),
        default="scatter",
        help="scatter (default) or tri; band layout only for tri, fullmesh always scatter",
    )
    parser.add_argument(
        "--scatter-size",
        type=float,
        default=6.0,
        help="Scatter marker size for fullmesh + mask overlay (default 6)",
    )
    parser.add_argument("--out", default="", help="Output PNG path (default auto)")
    parser.add_argument(
        "--mask-overlay-out",
        default="",
        help="Optional second PNG: t0/support/ceiling mask overlay (omit to skip)",
    )
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
    apply_clot_phi_eval_defaults()
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    data = torch.load(graph_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = args.time_index if args.time_index >= 0 else int(data.y.shape[0]) - 1
    step, phi_pred, mu_pred = _predict_mu_phi(
        model, data, ti, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device
    )

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
    thresh = clot_phi_thresh_si(phys_cfg)
    proj_tag = " proj=1" if clot_phi_hard_support_projection_enabled() else ""
    print(
        f"[i]  t={ti} region_n={int(m.sum())} gt_pos_n={gt_pos} "
        f"mean_pred_phi={float(phi_pr[m].mean()) if m.any() else 0.0:.3f} "
        f"mean_gt_phi={float(phi_gt[m].mean()) if m.any() else 0.0:.3f} | "
        f"clot_shape={shape_m['clot_shape']:.3f} (full mesh mu>={shape_m['clot_mu_thresh_si']:.3f}{proj_tag}) "
        f"rec={shape_m['clot_recall']:.3f} pred_frac={shape_m['clot_pred_frac']:.3f} "
        f"gt_frac={shape_m['clot_gt_frac']:.3f}",
        flush=True,
    )

    cap = float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))
    band = clot_phi_mask_mode()
    fig = plt.figure(figsize=(12, 10))

    if args.layout == "band":
        plot_i = [1]
        _plot_band_panels(fig, plot_i, args, pos, m, idx, phi_gt, phi_pr, mu_gt, mu_pr, ti, band, cap)
        fig.suptitle(f"clot_phi_simple — {args.anchor} (ckpt {ckpt_path.name})", fontsize=14)
    else:
        gt_bin = _clot_binary(mu_gt, thresh)
        pr_bin = _clot_binary(mu_pr, thresh)
        dot = float(args.scatter_size)
        dot_mu = max(dot, dot * 1.25)
        axes = fig.subplots(2, 2)
        _scatter_fullmesh(
            axes[0, 0],
            pos,
            gt_bin,
            f"GT clot (t={ti}) mu>={thresh:.3f} Pa*s",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
        )
        _scatter_fullmesh(
            axes[0, 1],
            pos,
            pr_bin,
            f"Pred clot (projected mu) shape={shape_m['clot_shape']:.3f}",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
        )
        _scatter_fullmesh(
            axes[1, 0],
            pos,
            mu_gt,
            f"GT mu cap {cap:.2f} Pa*s",
            cmap="bwr",
            vmin=0,
            vmax=cap,
            s=dot_mu,
        )
        _scatter_fullmesh(
            axes[1, 1],
            pos,
            mu_pr,
            f"Pred mu eff (projected)",
            cmap="bwr",
            vmin=0,
            vmax=cap,
            s=dot_mu,
        )
        schedule = clot_forecast_pair_schedule() if clot_forecast_one_step_enabled() else "legacy"
        fig.suptitle(
            f"clot_phi — {args.anchor} t={ti} | full mesh | schedule={schedule}{proj_tag} | "
            f"pred_frac={shape_m['clot_pred_frac']:.3f} gt_frac={shape_m['clot_gt_frac']:.3f}",
            fontsize=12,
        )

    fig.tight_layout()

    out = Path(args.out) if args.out else root / "outputs" / "biochem" / f"clot_phi_viz_{args.anchor}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[OK]  Wrote {out.resolve()}", flush=True)

    if args.mask_overlay_out:
        t0_m, support_m, ceiling_m = _resolve_support_ceiling_masks(
            data, ti, step, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device
        )
        pr_bin = _clot_binary(mu_pr, thresh)
        overlay_out = Path(args.mask_overlay_out)
        if not overlay_out.is_absolute():
            overlay_out = root / overlay_out
        _write_mask_overlay_figure(
            pos=pos,
            t0_mask=t0_m,
            support=support_m,
            ceiling=ceiling_m,
            pred_clot=pr_bin,
            anchor=args.anchor,
            ti=ti,
            out=overlay_out,
            scatter_size=float(args.scatter_size),
        )
        print(f"[OK]  Wrote mask overlay {overlay_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
