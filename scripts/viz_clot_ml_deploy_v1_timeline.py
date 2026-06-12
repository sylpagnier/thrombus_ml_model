"""Deploy v1 timeline: phi, mu_eff, flow (frozen vs coupled), extrap horizon."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index, rollout_time_indices  # noqa: E402
from src.core_physics.clot_forecast import build_clot_forecast_pair_step  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.inference.clot_ml_deploy_v1 import (  # noqa: E402
    load_deploy_v1_recipe,
    rollout_deploy_v1_mu,
    rollout_deploy_v1_phi,
)
from src.training.clot_ml_device import resolve_clot_ml_eval_device  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.metrics import rel_l2_uvp  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


@torch.no_grad()
def _predict_uv_with_mu_si(model, batch, mu_si: torch.Tensor, phys_cfg: PhysicsConfig):
    """One GINO-DEQ solve with ``MU_PRIOR`` from predicted mu (single model, no second load)."""
    mu_nd = phys_cfg.viscosity_si_to_nd(mu_si.reshape(-1, 1))
    kin_in = batch.x.clone()
    kin_in[:, NodeFeat.MU_PRIOR] = mu_nd.to(device=kin_in.device, dtype=kin_in.dtype)
    batch_k = batch.clone()
    batch_k.x = kin_in
    pred = predict_kinematics(model, batch_k)
    del batch_k, kin_in
    return pred[:, 0], pred[:, 1]


def _maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pick_times(t_indices: list[int], max_frames: int) -> list[int]:
    if not t_indices:
        return []
    if max_frames <= 0 or len(t_indices) <= max_frames:
        return list(t_indices)
    idx = np.linspace(0, len(t_indices) - 1, num=max_frames, dtype=int)
    return sorted({int(t_indices[i]) for i in idx.tolist()})


@torch.no_grad()
def collect_deploy_v1_frames(
    data,
    *,
    recipe,
    device: torch.device,
    sim_end_scale: float,
    coupled: bool,
    show_coupled_flow: bool,
    phi_only: bool,
    time_stride: int,
    max_frames: int,
    kine_max_iters: int,
) -> tuple[list[dict], dict]:
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_comsol = int(data.y.shape[0])
    t_comsol_final = n_comsol - 1

    phi_by_t = rollout_deploy_v1_phi(
        data,
        recipe,
        device=device,
        sim_end_scale=sim_end_scale,
        coupled=coupled,
    )
    mu_by_t: dict[int, torch.Tensor] = {}
    if not phi_only:
        mu_by_t = rollout_deploy_v1_mu(
            data, phi_by_t, device=device, phys_cfg=phys, bio_cfg=bio
        )

    t_all = rollout_time_indices(data, time_stride=time_stride, sim_end_scale=sim_end_scale)
    t_show = [t for t in _pick_times(t_all, max_frames) if t in phi_by_t]

    kin_phys = PhysicsConfig(phase="kinematics")
    kine_model = None
    batch = data.to(device)
    u_f = v_f = None
    speed_frozen = None
    if show_coupled_flow:
        ckpt = recipe.kine_ckpt.strip() or str(resolve_kinematics_checkpoint())
        kine_model = load_kinematics_predictor(
            ckpt, device, phys_cfg=kin_phys, max_iters=max(3, int(kine_max_iters))
        )
        pred_frozen = predict_kinematics(kine_model, batch)
        u_f = pred_frozen[:, 0]
        v_f = pred_frozen[:, 1]
        speed_frozen = torch.sqrt(u_f * u_f + v_f * v_f)
        _maybe_empty_cuda_cache(device)

    frames: list[dict] = []
    for t_out in t_show:
        phi = phi_by_t[int(t_out)]
        mu = mu_by_t.get(int(t_out))
        if show_coupled_flow and kine_model is not None and speed_frozen is not None and mu is not None:
            u_c, v_c = _predict_uv_with_mu_si(kine_model, batch, mu, kin_phys)
            speed_coupled = torch.sqrt(u_c * u_c + v_c * v_c)
            speed_delta = (speed_coupled - speed_frozen).abs()
            _maybe_empty_cuda_cache(device)
        else:
            speed_coupled = speed_delta = None

        band = None
        phi_gt_np = None
        loss_m = None
        if not phi_only and int(t_out) <= t_comsol_final:
            t_in = max(0, int(t_out) - 1)
            step = build_clot_forecast_pair_step(data, t_in, min(int(t_out), t_comsol_final), phys, bio, device)
            band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
            phi_gt_np = step.phi_gt.detach().cpu().numpy()
            loss_m = step.loss_mask.detach().cpu().numpy().astype(bool)

        tau = macro_tau_at_index(data, int(t_out), bio_cfg=bio)
        frames.append(
            {
                "t_out": int(t_out),
                "tau": float(tau),
                "extrap": int(t_out) > t_comsol_final,
                "phi": phi.detach().cpu().numpy(),
                "mu_si": mu.detach().cpu().numpy() if mu is not None else None,
                "speed_frozen": speed_frozen.detach().cpu().numpy() if speed_frozen is not None else None,
                "speed_coupled": speed_coupled.detach().cpu().numpy() if speed_coupled is not None else None,
                "speed_delta": speed_delta.detach().cpu().numpy() if speed_delta is not None else None,
                "phi_gt": phi_gt_np,
                "loss_mask": loss_m,
                "band": band,
                "n_flag": int((phi > 0.5).sum().item()),
                "pred_frac": float((phi > 0.5).float().mean().item()),
            }
        )

    meta = {
        "n_comsol_steps": n_comsol,
        "sim_end_scale": sim_end_scale,
        "coupled": coupled,
        "show_coupled_flow": show_coupled_flow,
        "phi_only": phi_only,
        "time_stride": time_stride,
        "n_frames": len(frames),
        "vel_source": "coupled" if coupled else recipe.vel_source,
    }
    if show_coupled_flow and frames and kine_model is not None and u_f is not None and v_f is not None:
        t_last = min(int(frames[-1]["t_out"]), t_comsol_final)
        gt = data.y[int(t_last)].to(device)
        mu_last = mu_by_t.get(int(t_last), mu_by_t[max(mu_by_t.keys())])
        u_c, v_c = _predict_uv_with_mu_si(kine_model, batch, mu_last, kin_phys)
        meta["rel_l2_uvp_coupled_vs_frozen"] = float(
            rel_l2_uvp(
                torch.stack([u_c, v_c, gt[:, 2]], dim=1),
                torch.stack([u_f, v_f, gt[:, 2]], dim=1),
            )
        )
    return frames, meta


def render_phi_only_png(
    frames: list[dict],
    *,
    anchor: str,
    meta: dict,
    pos: np.ndarray,
    ceiling: np.ndarray,
    out_path: Path,
    scatter_size: float,
) -> None:
    if not frames:
        raise RuntimeError("no frames to render")
    ncols = len(frames)
    fig, axes = plt.subplots(1, ncols, figsize=(max(2.4 * ncols, 10), 3.2), squeeze=False)

    fig.suptitle(
        f"clot pred phi -- {anchor} | H={meta['sim_end_scale']:.1f}x | "
        f"stride={meta['time_stride']} | flags>=0.5",
        fontsize=11,
    )

    for j, fr in enumerate(frames):
        extrap_tag = " [extrap]" if fr["extrap"] else ""
        title = (
            f"t={fr['t_out']} tau={fr['tau']:.2f}{extrap_tag}\n"
            f"flag={fr['n_flag']} pred+={fr['pred_frac']:.3f}"
        )
        _scatter_fullmesh_region(
            axes[0, j], pos, fr["phi"], ceiling, "pred phi" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=scatter_size, layer_positive_on_top=True,
        )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_timeline_png(
    frames: list[dict],
    *,
    anchor: str,
    meta: dict,
    pos: np.ndarray,
    ceiling: np.ndarray,
    out_path: Path,
    scatter_size: float,
    show_flow: bool,
) -> None:
    if not frames:
        raise RuntimeError("no frames to render")
    ncols = len(frames)
    nrows = 5 if show_flow else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.8 * ncols, 10), 2.2 * nrows + 1.5), squeeze=False)

    mode = "coupled" if meta.get("coupled") else "frozen-kine"
    fig.suptitle(
        f"deploy v1 -- {anchor} | H={meta['sim_end_scale']:.2f}x | {mode} | "
        f"stride={meta['time_stride']} | n_comsol={meta['n_comsol_steps']}",
        fontsize=11,
    )

    for j, fr in enumerate(frames):
        extrap_tag = " [extrap]" if fr["extrap"] else ""
        tau = fr["tau"]
        title = f"t={fr['t_out']} tau={tau:.2f}{extrap_tag}\nflag={fr['n_flag']} pred+={fr['pred_frac']:.3f}"
        if fr["band"] is not None:
            title += f" F1={fr['band']['clot_f1']:.2f}"

        if fr["phi_gt"] is not None and fr["loss_mask"] is not None:
            _scatter_fullmesh_region(
                axes[0, j], pos, fr["phi_gt"], fr["loss_mask"], "GT band" if j == 0 else "",
                cmap="bwr", vmin=0, vmax=1, s=scatter_size, layer_positive_on_top=True,
            )
        else:
            axes[0, j].text(0.5, 0.5, "no GT\n(extrap)", ha="center", va="center", transform=axes[0, j].transAxes)
            axes[0, j].axis("off")

        _scatter_fullmesh_region(
            axes[1, j], pos, fr["phi"], ceiling, "pred phi" if j == 0 else "",
            cmap="bwr", vmin=0, vmax=1, s=scatter_size, layer_positive_on_top=True,
        )
        if fr["mu_si"] is not None:
            mu_log = np.log10(np.clip(fr["mu_si"], 1e-4, None))
            _scatter_fullmesh_region(
                axes[2, j], pos, mu_log, ceiling, "log10 mu" if j == 0 else "",
                cmap="viridis", s=scatter_size,
            )

        if show_flow:
            _scatter_fullmesh_region(
                axes[3, j], pos, fr["speed_frozen"], ceiling, "|u| frozen" if j == 0 else "",
                cmap="plasma", s=scatter_size,
            )
            _scatter_fullmesh_region(
                axes[4, j], pos, fr["speed_delta"], ceiling, "|u_c-u_f|" if j == 0 else "",
                cmap="hot", s=scatter_size,
            )

        axes[1, j].set_title(title, fontsize=7)
        if j == 0:
            axes[0, j].set_ylabel("GT", fontsize=9)
            axes[1, j].set_ylabel("phi", fontsize=9)
            axes[2, j].set_ylabel("mu", fontsize=9)
            if show_flow:
                axes[3, j].set_ylabel("flow", fontsize=9)
                axes[4, j].set_ylabel("d flow", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_growth_curve_png(
    frames: list[dict],
    *,
    anchor: str,
    meta: dict,
    data,
    bio: BiochemConfig,
    out_path: Path,
) -> None:
    taus = [fr["tau"] for fr in frames]
    pred_frac = [fr["pred_frac"] for fr in frames]
    flags = [fr["n_flag"] for fr in frames]
    n_comsol = int(meta["n_comsol_steps"])
    t_comsol_tau = macro_tau_at_index(data, n_comsol - 1, bio_cfg=bio)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(taus, pred_frac, "b-o", label="pred commit frac", markersize=4)
    ax2 = ax1.twinx()
    ax2.plot(taus, flags, "r--", alpha=0.6, label="flag count")
    ax2.set_ylabel("flags", color="r", fontsize=8)
    if meta["sim_end_scale"] > 1.0:
        ax1.axvline(t_comsol_tau, color="gray", linestyle=":", label="COMSOL end")
    ax1.set_xlabel("macro tau")
    ax1.set_ylabel("commit fraction")
    ax1.set_title(f"deploy v1 growth curve -- {anchor} H={meta['sim_end_scale']:.2f}x")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy v1 clot+flow timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--recipe", default="data/reference/clot_ml_deploy_v1.json")
    ap.add_argument("--sim-end-scale", type=float, default=1.5)
    ap.add_argument("--time-stride", type=int, default=1, help="macro step stride (1=every step)")
    ap.add_argument("--max-frames", type=int, default=12, help="max columns in PNG")
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--phi-only", action="store_true", help="pred clot phi row only (no GT/mu/flow)")
    ap.add_argument("--coupled", action="store_true", help="coupled phi rollout + mu->DEQ panels")
    ap.add_argument("--no-flow", action="store_true", help="skip mu->DEQ flow panels even with --coupled")
    ap.add_argument("--kine-max-iters", type=int, default=12, help="DEQ iters per flow frame (lower saves VRAM)")
    ap.add_argument("--out", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--growth-curve", action="store_true", help="also write pred_frac vs tau PNG")
    args = ap.parse_args()

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    reset_temporal_kinematics_cache()

    recipe = load_deploy_v1_recipe(root / args.recipe)
    if args.coupled:
        from dataclasses import replace

        recipe = replace(recipe, coupled=True)
    recipe.apply_env()
    os.environ["CLOT_ML_SIM_END_SCALE"] = str(float(args.sim_end_scale))

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    ceiling = resolve_ceiling_mask(data, device, BiochemConfig(phase="biochem")).detach().cpu().numpy().astype(bool)

    phi_only = bool(args.phi_only)
    show_coupled_flow = bool(args.coupled) and not args.no_flow and not phi_only
    frames, meta = collect_deploy_v1_frames(
        data,
        recipe=recipe,
        device=device,
        sim_end_scale=float(args.sim_end_scale),
        coupled=bool(args.coupled),
        show_coupled_flow=show_coupled_flow,
        phi_only=phi_only,
        time_stride=max(int(args.time_stride), 1),
        max_frames=int(args.max_frames),
        kine_max_iters=int(args.kine_max_iters),
    )
    meta["anchor"] = args.anchor
    meta["n_nodes"] = int(data.num_nodes)

    suffix = f"_H{args.sim_end_scale:.1f}".replace(".", "p")
    if phi_only:
        suffix = f"_phi{suffix}"
    elif args.coupled:
        suffix += "_coupled"
    stem = f"deploy_v1_{args.anchor}_timeline{suffix}" if not phi_only else f"deploy_v1_{args.anchor}{suffix}"
    out_default = root / f"outputs/biochem/viz/clot_deploy/{stem}.png"
    out_path = Path(args.out) if args.out.strip() else out_default
    if not out_path.is_absolute():
        out_path = root / out_path

    if phi_only:
        render_phi_only_png(
            frames,
            anchor=args.anchor,
            meta=meta,
            pos=pos,
            ceiling=ceiling,
            out_path=out_path,
            scatter_size=float(args.scatter_size),
        )
    else:
        render_timeline_png(
            frames,
            anchor=args.anchor,
            meta=meta,
            pos=pos,
            ceiling=ceiling,
            out_path=out_path,
            scatter_size=float(args.scatter_size),
            show_flow=show_coupled_flow,
        )
    print(f"[save] {out_path}")

    if args.growth_curve:
        curve_path = out_path.with_name(out_path.stem + "_growth.png")
        render_growth_curve_png(
            frames,
            anchor=args.anchor,
            meta=meta,
            data=data,
            bio=BiochemConfig(phase="biochem"),
            out_path=curve_path,
        )
        print(f"[save] {curve_path}")

    summary_path = Path(args.summary_json) if args.summary_json.strip() else out_path.with_suffix(".json")
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "frames": [{k: v for k, v in fr.items() if k not in ("phi", "mu_si", "phi_gt", "loss_mask", "speed_frozen", "speed_coupled", "speed_delta")} for fr in frames]}
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {summary_path}")

    for fr in frames:
        b = fr.get("band") or {}
        print(
            f"[i] t={fr['t_out']} tau={fr['tau']:.2f} extrap={fr['extrap']} "
            f"flag={fr['n_flag']} pred+={fr['pred_frac']:.3f} "
            f"F1={float(b.get('clot_f1', float('nan'))):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
