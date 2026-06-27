"""Timeline viz: one-step clot forecast predictions across macro times."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_forecast import (
    build_clot_forecast_pair_step,
    clot_forecast_mu_carry_enabled,
    clot_forecast_one_step_enabled,
    clot_forecast_pair_stride,
)
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_fixed_mu_from_phi_enabled,
    clot_phi_hard_support_projection_enabled,
    clot_phi_hybrid_enabled,
    clot_phi_shape_use_t_out_mu,
    clot_phi_thresh_si,
    gt_mu_anchor_cap_si,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    project_deploy_mu_with_support,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _resolve_anchor_path(root: Path, anchor: str) -> Path:
    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    return anchor_dir.resolve() / f"{anchor}.pt"


def _load_model(ckpt_path: Path, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(raw.get("config") or {})
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    hidden = int(cfg.get("hidden", 32))
    in_dim = int(cfg.get("in_dim", 4))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


def _predict_forecast_step(
    model,
    data,
    t_in: int,
    t_out: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    forecast_state: ClotPhiRolloutState | None = None,
    train_epoch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (phi_pred, mu_pred_si, phi_gt) at t_out for one-step forecast pair."""
    step = build_clot_forecast_pair_step(
        data,
        t_in,
        t_out,
        phys_cfg,
        bio_cfg,
        device,
        forecast_state=forecast_state,
        train_epoch=train_epoch,
    )
    with torch.no_grad():
        phi_pred = model(step.features).reshape(-1)
        step_out = None
        if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
            mu_pred = mu_eff_from_delta_log_si(step.mu_c_si, model.forward_delta_log_mu(step.features))
            mu_c_proj = step.mu_c_si
        elif clot_phi_fixed_mu_from_phi_enabled() and clot_phi_shape_use_t_out_mu():
            step_out = build_clot_phi_step(data, t_out, phys_cfg, bio_cfg, device)
            mu_pred = log_blend_mu_eff_si(step_out.mu_c_si, phi_pred)
            mu_c_proj = step_out.mu_c_si
        else:
            mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
            mu_c_proj = step.mu_c_si
        if clot_phi_hard_support_projection_enabled():
            mu_pred = project_deploy_mu_with_support(
                data=data,
                step=step,
                mu_pred=mu_pred,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                forecast_one_step=True,
                time_index=int(t_out),
                bulk_time_index=int(t_out),
            )
    return phi_pred, mu_pred.reshape(-1), step.phi_gt.reshape(-1)


def _clot_binary(mu_si: np.ndarray, anchor_si: np.ndarray, thresh: float) -> np.ndarray:
    growth = np.maximum(mu_si.reshape(-1) - anchor_si.reshape(-1), 0.0)
    return (growth >= thresh).astype(np.float32)


def _pick_t_out_indices(t_steps: int, pair_stride: int, keyframes: int) -> list[int]:
    t_out_min = pair_stride
    t_out_max = t_steps - 1
    if t_out_max <= t_out_min:
        return [t_out_max]
    if keyframes <= 1:
        return [t_out_max]
    outs = np.linspace(t_out_min, t_out_max, num=keyframes, dtype=int)
    return sorted(set(int(x) for x in outs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Clot forecast timeline (one-step t_in -> t_out)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--keyframes", type=int, default=8, help="Number of t_out snapshots")
    ap.add_argument("--out", default="")
    ap.add_argument("--summary-json", default="", help="Optional JSONL path for per-time metrics")
    ap.add_argument(
        "--no-carry",
        action="store_true",
        help="Disable mu carry (GT mu @ t_in each step; oracle ablation)",
    )
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = root / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    model, cfg = _load_model(ckpt_path, device)
    if not clot_forecast_one_step_enabled():
        raise RuntimeError("checkpoint is not one-step forecast; use viz_clot_phi_simple instead")

    use_carry = clot_forecast_mu_carry_enabled() and not args.no_carry
    forecast_state = ClotPhiRolloutState() if use_carry else None
    carry_tag = "carry" if use_carry else "gt_mu_in"
    graph_path = _resolve_anchor_path(root, args.anchor)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    pair_stride = clot_forecast_pair_stride()
    t_steps = int(data.y.shape[0])
    t_out_list = _pick_t_out_indices(t_steps, pair_stride, max(1, args.keyframes))
    thresh = clot_phi_thresh_si(phys_cfg)
    mu_anchor_np = gt_mu_anchor_cap_si(data, phys_cfg, device).detach().cpu().numpy()
    pos = data.x[:, :2].detach().cpu().numpy()

    rows: list[dict] = []
    gt_panels: list[np.ndarray] = []
    pr_panels: list[np.ndarray] = []
    titles: list[str] = []

    for t_out in t_out_list:
        t_in = max(0, t_out - pair_stride)
        phi_pred, mu_pred, _phi_gt = _predict_forecast_step(
            model,
            data,
            t_in,
            t_out,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            forecast_state=forecast_state,
            train_epoch=None,
        )
        if forecast_state is not None:
            forecast_state.update_from_pred(phi_pred, mu_pred, detach=True)
        y_out = data.y[t_out].to(device=device, dtype=torch.float32)
        mu_gt = cap_mu_eff_si(phys_cfg.viscosity_nd_to_si(y_out[:, STATE_CHANNEL_MU_EFF_ND]))
        pred_state = y_out.clone()
        pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(mu_pred)
        sm = compute_clot_shape_metrics(
            pred_state=pred_state,
            gt_state=y_out,
            edge_index=data.edge_index.to(device),
            phys_cfg=phys_cfg,
            gt_anchor_state=data.y[0].to(device=device, dtype=torch.float32),
        )
        mu_gt_np = mu_gt.detach().cpu().numpy()
        mu_pr_np = mu_pred.detach().cpu().numpy()
        gt_panels.append(_clot_binary(mu_gt_np, mu_anchor_np, thresh))
        pr_panels.append(_clot_binary(mu_pr_np, mu_anchor_np, thresh))
        titles.append(f"t_out={t_out}\nshape={sm['clot_shape']:.3f}")
        rows.append(
            {
                "anchor": args.anchor,
                "t_in": t_in,
                "t_out": t_out,
                "mu_in_mode": carry_tag,
                "clot_shape": float(sm["clot_shape"]),
                "clot_recall": float(sm["clot_recall"]),
                "clot_pred_frac": float(sm["clot_pred_frac"]),
                "clot_gt_frac": float(sm["clot_gt_frac"]),
                "mean_pred_phi": float(phi_pred.mean().item()),
            }
        )
        print(
            f"[i]  t_in={t_in} t_out={t_out} shape={sm['clot_shape']:.3f} "
            f"rec={sm['clot_recall']:.3f} pred_frac={sm['clot_pred_frac']:.3f} "
            f"gt_frac={sm['clot_gt_frac']:.3f}",
            flush=True,
        )

    ncols = len(t_out_list)
    fig, axes = plt.subplots(2, ncols, figsize=(max(3.2 * ncols, 10), 6.5), squeeze=False)
    mask_mode = (cfg.get("forecast_mask") or os.environ.get("CLOT_FORECAST_MASK") or "target")
    anchor_tag = Path(args.anchor).stem
    fig.suptitle(
        f"Clot forecast timeline — {anchor_tag} | mask={mask_mode} | mu_in={carry_tag} | "
        f"stride={pair_stride} | mu>={thresh:.3f} Pa*s",
        fontsize=13,
    )

    for j, (gt_c, pr_c, title) in enumerate(zip(gt_panels, pr_panels, titles)):
        for row_i, panel, label in ((0, gt_c, "GT clot"), (1, pr_c, "Pred clot")):
            ax = axes[row_i, j]
            ax.scatter(pos[:, 0], pos[:, 1], c=panel, s=6, cmap="Reds", vmin=0, vmax=1, linewidths=0, alpha=0.85)
            ax.set_aspect("equal")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(label, fontsize=11)
            if row_i == 0:
                ax.set_title(title, fontsize=9)

    fig.tight_layout()
    out = Path(args.out) if args.out else (
        root / "outputs/biochem/viz" / f"clot_forecast_timeline_{anchor_tag}_{ckpt_path.parent.name}.png"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[OK]  Wrote {out.resolve()}", flush=True)

    # 4-panel phi/mu detail at final t_out
    t_last = t_out_list[-1]
    t_in_last = max(0, t_last - pair_stride)
    phi_pred, mu_pred, phi_gt = _predict_forecast_step(
        model,
        data,
        t_in_last,
        t_last,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_state=forecast_state,
        train_epoch=None,
    )
    step_last = build_clot_forecast_pair_step(
        data,
        t_in_last,
        t_last,
        phys_cfg,
        bio_cfg,
        device,
        forecast_state=forecast_state,
        train_epoch=None,
    )
    m = step_last.loss_mask.detach().cpu().numpy().astype(bool)
    idx = np.where(m)[0]
    mu_gt_last = cap_mu_eff_si(
        phys_cfg.viscosity_nd_to_si(data.y[t_last][:, STATE_CHANNEL_MU_EFF_ND].to(device))
    ).detach().cpu().numpy()
    phi_gt_np = phi_gt.detach().cpu().numpy()
    phi_pr_np = phi_pred.detach().cpu().numpy()
    mu_pr_np = mu_pred.detach().cpu().numpy()

    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9))
    panels = (
        (phi_gt_np, "GT phi @ t_out", "Reds", 0, 1),
        (phi_pr_np, "Pred phi @ t_out", "Reds", 0, 1),
        (mu_gt_last, f"GT mu cap", "bwr", 0, float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))),
        (mu_pr_np, "Pred mu eff", "bwr", 0, float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))),
    )
    for ax, (vals, title, cmap, vmin, vmax) in zip(axes2.flat, panels):
        ax.scatter(
            pos[idx, 0], pos[idx, 1], c=vals[idx], s=16, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig2.suptitle(f"{anchor_tag} t_out={t_last} (loss band n={len(idx)})", fontsize=12)
    fig2.tight_layout()
    detail_out = out.with_name(out.stem + "_tfinal_band.png")
    fig2.savefig(detail_out, dpi=140)
    plt.close(fig2)
    print(f"[OK]  Wrote {detail_out.resolve()}", flush=True)

    # Full-mesh deploy mu (projected) vs GT binary clot — matches clot_shape eval.
    mu_cap_v = float(os.environ.get("CLOT_PHI_MU_CAP_SI", "0.10"))
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))
    gt_bin = _clot_binary(mu_gt_last, mu_anchor_np, thresh)
    pr_bin = _clot_binary(mu_pr_np, mu_anchor_np, thresh)
    support_np = m.astype(float)
    for ax, vals, title, cmap, vmin, vmax in (
        (axes3[0], gt_bin, f"GT clot (>={thresh:.3f})", "Reds", 0, 1),
        (axes3[1], pr_bin, f"Pred clot (projected mu)", "Reds", 0, 1),
        (axes3[2], support_np, f"Support B_t (loss band n={len(idx)})", "Greens", 0, 1),
    ):
        ax.scatter(pos[:, 0], pos[:, 1], c=vals, s=8, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    pf = float(pr_bin.mean())
    gf = float(gt_bin.mean())
    fig3.suptitle(
        f"{anchor_tag} t_out={t_last} full mesh | pred_frac={pf:.3f} gt_frac={gf:.3f}",
        fontsize=11,
    )
    fig3.tight_layout()
    full_out = out.with_name(out.stem + "_tfinal_fullmesh.png")
    fig3.savefig(full_out, dpi=140)
    plt.close(fig3)
    print(f"[OK]  Wrote {full_out.resolve()}", flush=True)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        if not summary_path.is_absolute():
            summary_path = root / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[OK]  summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
