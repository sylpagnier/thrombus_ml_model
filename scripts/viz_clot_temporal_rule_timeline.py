"""Timeline viz: temporal growing rule phi vs GT over macro times."""

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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    deploy_score_from_eval_row,
    eval_temporal_rule_on_anchor,
    reset_temporal_kinematics_cache,
    rollout_temporal_phi,
    temporal_rule_config_from_env,
    temporal_vel_source,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.metrics import rel_l2_uvp  # noqa: E402
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
from src.training.clot_ml_step0_coef import load_step0_coef_json  # noqa: E402
from src.training.clot_ml_step1_residual import (  # noqa: E402
    eval_step1_on_anchor,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
)
from src.training.clot_ml_step2_band_gnn import (  # noqa: E402
    eval_step2_on_anchor,
    load_step2_checkpoint,
    resolve_step2_rule_cfg,
    rollout_step2_phi,
)
from src.training.clot_ml_step3_temporal_gate import (  # noqa: E402
    eval_step3_on_anchor,
    load_step3_checkpoint,
    resolve_step3_rule_cfg,
    rollout_step3_phi,
)
from src.training.clot_ml_step7_band_phi import (  # noqa: E402
    eval_step7_on_anchor,
    load_step7_checkpoint,
    resolve_step7_rule_cfg,
    rollout_step7_phi,
)
from src.training.clot_ml_step7b_hybrid import (  # noqa: E402
    eval_step7b_on_anchor,
    load_step7b_checkpoint,
    resolve_step7b_rule_cfg,
    rollout_step7b_phi,
)
from src.training.clot_ml_pivot_common import load_pivot_checkpoint  # noqa: E402
from src.training.clot_ml_pivot_data_driven import (  # noqa: E402
    build_data_driven_model,
    eval_data_driven_on_anchor,
    rollout_data_driven_phi,
)
from src.training.clot_ml_pivot_rule_mixture import (  # noqa: E402
    build_rule_mixture_model,
    eval_rule_mixture_on_anchor,
    resolve_mixture_rule_cfg,
    rollout_rule_mixture_phi,
)
from src.training.clot_ml_pivot_soft_commit import (  # noqa: E402
    build_soft_commit_model,
    eval_soft_commit_on_anchor,
    resolve_soft_rule_cfg,
    rollout_soft_commit_phi,
)
from src.utils.paths import get_project_root


def _apply_viz_env() -> None:
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "from_t0")
    os.environ.setdefault("CLOT_FORECAST_PAIR_STRIDE", "1")


def _pick_keyframe_times(n_times: int, keyframes: int) -> list[int]:
    if n_times <= 0:
        return []
    if keyframes <= 0 or n_times <= keyframes:
        return list(range(n_times))
    idx = np.linspace(0, n_times - 1, num=keyframes, dtype=int)
    return sorted(set(int(i) for i in idx.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal growing rule clot timeline")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--scatter-size", type=float, default=5.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--title", default="", help="Override figure suptitle")
    ap.add_argument(
        "--vel-source",
        choices=("gt", "kinematics"),
        default="",
        help="gt=COMSOL flow; kinematics=steady GINO-DEQ (deploy)",
    )
    ap.add_argument(
        "--kine-ckpt",
        default="",
        help="GINO-DEQ checkpoint (default: outputs/kinematics/kinematics_best.pth)",
    )
    ap.add_argument("--compare-gt", action="store_true", help="Print GT vs pred-kine deploy metrics")
    ap.add_argument(
        "--step0-json",
        default="",
        help="Step0 best_coef.json (overrides env rule; forces pred kine)",
    )
    ap.add_argument(
        "--step1-ckpt",
        default="",
        help="Step1 residual checkpoint (.pth); uses frozen Step0 rule + MLP",
    )
    ap.add_argument(
        "--step2-ckpt",
        default="",
        help="Step2 band GNN checkpoint (.pth); frozen Step0 shell + learned risk",
    )
    ap.add_argument(
        "--step3-ckpt",
        default="",
        help="Step3 temporal gate checkpoint (.pth); learned per-vessel onset",
    )
    ap.add_argument(
        "--step7-ckpt",
        default="",
        help="Step7 band GNN phi checkpoint (.pth); ceiling + Step0 onset gate",
    )
    ap.add_argument(
        "--step7b-ckpt",
        default="",
        help="Step7b hybrid checkpoint (.pth); frozen rule_mixture + residual MLP",
    )
    ap.add_argument(
        "--pivot-ckpt",
        default="",
        help="Side pivot checkpoint (.pth): soft_commit | rule_mixture | data_driven",
    )
    args = ap.parse_args()

    _apply_viz_env()
    if (
        args.step1_ckpt.strip()
        or args.step2_ckpt.strip()
        or args.step3_ckpt.strip()
        or args.step7_ckpt.strip()
        or args.step7b_ckpt.strip()
        or args.pivot_ckpt.strip()
    ):
        os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    root = get_project_root()
    step0_arg = args.step0_json.strip()
    step0_path: Path | None = None
    if step0_arg:
        step0_path = Path(step0_arg)
        if not step0_path.is_absolute():
            step0_path = root / step0_path
        for key, val in load_step0_coef_json(step0_path).to_env().items():
            os.environ[key] = val
    if args.vel_source:
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = args.vel_source
    if args.kine_ckpt.strip():
        os.environ["CLOT_PHI_KINE_CKPT"] = args.kine_ckpt.strip()
    reset_temporal_kinematics_cache()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    step1_path: Path | None = None
    step1_model = None
    step1_meta: dict = {}
    step2_path: Path | None = None
    step2_model = None
    step2_meta: dict = {}
    step3_path: Path | None = None
    step3_model = None
    step3_meta: dict = {}
    step7_path: Path | None = None
    step7_model = None
    step7_meta: dict = {}
    step7b_path: Path | None = None
    step7b_mixture = None
    step7b_residual = None
    step7b_meta: dict = {}
    pivot_path: Path | None = None
    pivot_model = None
    pivot_meta: dict = {}
    pivot_type = ""
    pivot_arg = args.pivot_ckpt.strip()
    step3_arg = args.step3_ckpt.strip()
    step7_arg = args.step7_ckpt.strip()
    step7b_arg = args.step7b_ckpt.strip()
    step2_arg = args.step2_ckpt.strip()
    step1_arg = args.step1_ckpt.strip()
    if pivot_arg:
        pivot_path = Path(pivot_arg)
        if not pivot_path.is_absolute():
            pivot_path = root / pivot_path
        raw = torch.load(pivot_path, map_location=device, weights_only=False)
        pivot_meta = dict(raw.get("meta") or {})
        pivot_type = str(pivot_meta.get("pivot") or "")
        if pivot_type == "soft_commit":
            pivot_model = build_soft_commit_model(pivot_meta).to(device)
        elif pivot_type == "rule_mixture":
            pivot_model = build_rule_mixture_model(pivot_meta).to(device)
        elif pivot_type == "data_driven":
            pivot_model = build_data_driven_model(pivot_meta).to(device)
        else:
            raise ValueError(f"unknown pivot in checkpoint meta: {pivot_type!r}")
        pivot_model.load_state_dict(raw["model"], strict=True)
        pivot_model.eval()
        if pivot_type in ("soft_commit", "rule_mixture"):
            step0_from_meta = str(pivot_meta.get("step0_json") or args.step0_json or "").strip()
            if step0_from_meta:
                step0_path = Path(step0_from_meta)
                if not step0_path.is_absolute():
                    step0_path = root / step0_path
            if pivot_type == "soft_commit":
                cfg = resolve_soft_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
            else:
                cfg = resolve_mixture_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
        else:
            cfg = temporal_rule_config_from_env()
    elif step7b_arg:
        step7b_path = Path(step7b_arg)
        if not step7b_path.is_absolute():
            step7b_path = root / step7b_path
        step7b_mixture, step7b_residual, step7b_meta = load_step7b_checkpoint(step7b_path, device=device)
        step0_from_meta = str(step7b_meta.get("step0_json") or args.step0_json or "").strip()
        if step0_from_meta:
            step0_path = Path(step0_from_meta)
            if not step0_path.is_absolute():
                step0_path = root / step0_path
        cfg = resolve_step7b_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
    elif step7_arg:
        step7_path = Path(step7_arg)
        if not step7_path.is_absolute():
            step7_path = root / step7_path
        step7_model, step7_meta = load_step7_checkpoint(step7_path, device=device)
        step0_from_meta = str(step7_meta.get("step0_json") or args.step0_json or "").strip()
        if step0_from_meta:
            step0_path = Path(step0_from_meta)
            if not step0_path.is_absolute():
                step0_path = root / step0_path
        cfg = resolve_step7_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
    elif step3_arg:
        step3_path = Path(step3_arg)
        if not step3_path.is_absolute():
            step3_path = root / step3_path
        step3_model, step3_meta = load_step3_checkpoint(step3_path, device=device)
        step0_from_meta = str(step3_meta.get("step0_json") or args.step0_json or "").strip()
        if step0_from_meta:
            step0_path = Path(step0_from_meta)
            if not step0_path.is_absolute():
                step0_path = root / step0_path
        cfg = resolve_step3_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
    elif step2_arg:
        step2_path = Path(step2_arg)
        if not step2_path.is_absolute():
            step2_path = root / step2_path
        step2_model, step2_meta = load_step2_checkpoint(step2_path, device=device)
        step0_from_meta = str(step2_meta.get("step0_json") or args.step0_json or "").strip()
        if step0_from_meta:
            step0_path = Path(step0_from_meta)
            if not step0_path.is_absolute():
                step0_path = root / step0_path
        cfg = resolve_step2_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
    elif step1_arg:
        step1_path = Path(step1_arg)
        if not step1_path.is_absolute():
            step1_path = root / step1_path
        step1_model, step1_meta = load_step1_checkpoint(step1_path, device=device)
        step0_from_meta = str(step1_meta.get("step0_json") or args.step0_json or "").strip()
        if step0_from_meta:
            step0_path = Path(step0_from_meta)
            if not step0_path.is_absolute():
                step0_path = root / step0_path
        cfg = resolve_step1_rule_cfg(step0_path) if step0_path else temporal_rule_config_from_env()
    elif step0_path is not None:
        cfg = load_step0_coef_json(step0_path).to_rule_config()
    else:
        cfg = temporal_rule_config_from_env()

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    graph_path = anchor_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    if pivot_model is not None and pivot_type == "soft_commit":
        phi_by_t = rollout_soft_commit_phi(
            data,
            cfg,
            pivot_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
    elif pivot_model is not None and pivot_type == "rule_mixture":
        phi_by_t = rollout_rule_mixture_phi(
            data,
            cfg,
            pivot_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
    elif pivot_model is not None and pivot_type == "data_driven":
        phi_by_t = rollout_data_driven_phi(
            data,
            pivot_model,
            device=device,
            bio_cfg=bio,
        )
    elif step7b_mixture is not None and step7b_residual is not None:
        phi_by_t = rollout_step7b_phi(
            data,
            cfg,
            step7b_mixture,
            step7b_residual,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=float(step7b_meta.get("alpha", 0.35)),
            flow_time=cfg.risk_flow_time,
        )
    elif step7_model is not None:
        phi_by_t = rollout_step7_phi(
            data,
            cfg,
            step7_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
    elif step3_model is not None:
        phi_by_t = rollout_step3_phi(
            data,
            cfg,
            step3_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
    elif step2_model is not None:
        phi_by_t = rollout_step2_phi(
            data,
            cfg,
            step2_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            delta_scale=float(step2_meta.get("delta_scale", 0.30)),
        )
    elif step1_model is not None:
        alpha = float(step1_meta.get("alpha", 0.35))
        phi_by_t = rollout_step1_phi(
            data,
            cfg,
            step1_model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=alpha,
        )
    else:
        phi_by_t = rollout_temporal_phi(data, cfg, device=device, phys_cfg=phys, bio_cfg=bio)
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    t_outs = _pick_keyframe_times(int(data.y.shape[0]), args.keyframes)
    pair_by_out = {t_out: t_in for t_in, t_out in pairs if t_out in t_outs}
    for t_out in t_outs:
        pair_by_out.setdefault(t_out, 0)

    pos = data.x[:, :2].detach().cpu().numpy()
    ceiling = resolve_ceiling_mask(data, device, bio).detach().cpu().numpy().astype(bool)
    ncols = len(t_outs)
    dot = float(args.scatter_size)
    fig, axes = plt.subplots(2, ncols, figsize=(max(3.0 * ncols, 9), 6.5), squeeze=False)
    vel_tag = temporal_vel_source()
    rel_kine_t0 = float("nan")
    if vel_tag == "kinematics":
        ckpt = (os.environ.get("CLOT_PHI_KINE_CKPT") or "").strip() or str(
            resolve_kinematics_checkpoint()
        )
        kine_model = load_kinematics_predictor(
            ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics")
        )
        with torch.no_grad():
            pred_k = predict_kinematics(kine_model, data.to(device))
        gt_t0 = data.y[0, :, :3].to(device)
        pred_t0 = pred_k[:, :3]
        if float(torch.norm(gt_t0[:, :3]).item()) > 1e-6:
            rel_kine_t0 = rel_l2_uvp(pred_t0, gt_t0)

    if args.title.strip():
        fig.suptitle(args.title.strip(), fontsize=11)
    else:
        kine_note = f" | GINO-DEQ rel_L2@t0={rel_kine_t0:.3f}" if vel_tag == "kinematics" else ""
        if pivot_model is not None:
            rule_line = f"pivot_{pivot_type}"
        elif step7b_mixture is not None:
            rule_line = (
                f"ml_step7b_hybrid (alpha={float(step7b_meta.get('alpha', 0.35)):.2f}, "
                f"frozen rule_mixture)"
            )
        elif step7_model is not None:
            rule_line = (
                f"ml_step7_band_phi (onset={float(step7_meta.get('onset_frac', cfg.global_onset_frac)):.2f})"
            )
        elif step3_model is not None:
            rule_line = "ml_step3_temporal_gate (learned onset)"
        elif step2_model is not None:
            rule_line = (
                f"ml_step2_band_gnn (delta={float(step2_meta.get('delta_scale', 0.30)):.2f})"
            )
        elif step1_model is not None:
            rule_line = f"ml_step1_residual (alpha={float(step1_meta.get('alpha', 0.35)):.2f})"
        else:
            rule_line = f"{cfg.name}: {cfg.describe()}"
        fig.suptitle(
            f"temporal rule timeline -- {args.anchor} | vel={vel_tag}{kine_note}\n"
            f"{rule_line}",
            fontsize=11,
        )

    rows_out: list[dict] = []
    for j, t_out in enumerate(t_outs):
        t_in = int(pair_by_out[t_out])
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys, bio, device)
        band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
        phi_gt = step.phi_gt.detach().cpu().numpy()
        phi_pred = phi.detach().cpu().numpy()
        loss_m = step.loss_mask.detach().cpu().numpy().astype(bool)
        n_flag = int((phi > 0.5).sum().item())

        title = f"t_out={t_out}\nF1={band['clot_f1']:.2f} flag={n_flag} pred+={band['pred_pos_frac']:.2f}"
        _scatter_fullmesh_region(
            axes[0, j],
            pos,
            phi_gt,
            loss_m,
            "GT" if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
            layer_positive_on_top=True,
        )
        _scatter_fullmesh_region(
            axes[1, j],
            pos,
            phi_pred,
            ceiling,
            "Temporal rule" if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
            layer_positive_on_top=True,
        )
        axes[0, j].set_title(title, fontsize=8)
        if j == 0:
            axes[0, j].set_ylabel("GT (band)", fontsize=10)
            axes[1, j].set_ylabel("Pred rule (ceiling)", fontsize=10)

        row = {
            "anchor": args.anchor,
            "t_in": t_in,
            "t_out": t_out,
            "band_f1": float(band["clot_f1"]),
            "band_prec": float(band["clot_prec"]),
            "band_rec": float(band["clot_rec"]),
            "band_pred_frac": float(band["pred_pos_frac"]),
            "n_flag": n_flag,
            "rule": cfg.name,
            "rule_desc": cfg.describe(),
        }
        rows_out.append(row)
        print(
            f"[i] t_out={t_out} band F1={band['clot_f1']:.3f} pred+={band['pred_pos_frac']:.3f} flag={n_flag}",
            flush=True,
        )

    fig.tight_layout()
    if pivot_path is not None:
        suffix = f"_pivot_{pivot_type}" if pivot_type else "_pivot"
    elif step7b_path is not None:
        suffix = "_step7b"
    elif step7_path is not None:
        suffix = "_step7"
    elif step3_path is not None:
        suffix = "_step3"
    elif step2_path is not None:
        suffix = "_step2"
    elif step1_path is not None:
        suffix = "_step1"
    elif step0_path is not None:
        suffix = "_step0"
    else:
        suffix = "_predkine" if vel_tag == "kinematics" else ""
    out_default = (
        root / f"outputs/biochem/viz/clot_deploy/temporal_rule_{args.anchor}_timeline{suffix}.png"
    )
    out_path = Path(args.out) if args.out else out_default
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".jsonl")
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    print(f"[save] {summary_path}")

    if pivot_model is not None and pivot_type == "soft_commit":
        metrics = eval_soft_commit_on_anchor(
            pivot_model,
            cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
        print(
            f"[i] temperature={metrics.get('temperature', float('nan')):.4f}",
            flush=True,
        )
    elif pivot_model is not None and pivot_type == "rule_mixture":
        metrics = eval_rule_mixture_on_anchor(
            pivot_model,
            cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
        ew = metrics.get("expert_weights") or {}
        if ew:
            top = sorted(ew.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(f"[i] expert_top3={top}", flush=True)
    elif pivot_model is not None and pivot_type == "data_driven":
        metrics = eval_data_driven_on_anchor(
            pivot_model,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
    elif step7b_mixture is not None and step7b_residual is not None:
        metrics = eval_step7b_on_anchor(
            step7b_residual,
            step7b_mixture,
            cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=float(step7b_meta.get("alpha", 0.35)),
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
    elif step7_model is not None:
        metrics = eval_step7_on_anchor(
            step7_model,
            cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
    elif step3_model is not None:
        metrics = eval_step3_on_anchor(
            step3_model,
            rule_cfg=cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
        print(
            f"[i] onset_pred={metrics.get('onset_pred', float('nan')):.3f} "
            f"onset_gt={metrics.get('onset_gt', float('nan')):.3f}",
            flush=True,
        )
    elif step2_model is not None:
        metrics = eval_step2_on_anchor(
            step2_model,
            rule_cfg=cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            delta_scale=float(step2_meta.get("delta_scale", 0.30)),
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
    elif step1_model is not None:
        metrics = eval_step1_on_anchor(
            step1_model,
            cfg,
            graph_path=graph_path,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=float(step1_meta.get("alpha", 0.35)),
        )
        deploy = float(metrics.get("deploy_score", float("nan")))
    else:
        metrics = eval_temporal_rule_on_anchor(
            data, cfg, stem=args.anchor, device=device, phys_cfg=phys, bio_cfg=bio
        )
        if "deploy_score" not in metrics:
            metrics["deploy_score"] = deploy_score_from_eval_row(metrics)
        deploy = float(metrics.get("deploy_score", float("nan")))
    print(
        f"[i] vel={vel_tag} deploy={deploy:.3f} tfinal_shape={metrics.get('tfinal_clot_shape', float('nan')):.3f} "
        f"tfinal_band_F1={metrics.get('tfinal_band_f1', float('nan')):.3f} "
        f"pred+={metrics.get('tfinal_band_pred_frac', float('nan')):.3f} "
        f"gt+={metrics.get('tfinal_gt_pos_frac', float('nan')):.3f} "
        f"early_pred+={metrics.get('early_mean_pred_frac', float('nan')):.3f}",
        flush=True,
    )
    if vel_tag == "kinematics" and rel_kine_t0 == rel_kine_t0:
        print(f"[i] GINO-DEQ vs COMSOL GT @ t=0: rel_L2(uvp)={rel_kine_t0:.4f}", flush=True)

    if args.compare_gt:
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "gt"
        reset_temporal_kinematics_cache()
        gt_metrics = eval_temporal_rule_on_anchor(
            data, cfg, stem=args.anchor, device=device, phys_cfg=phys, bio_cfg=bio
        )
        gt_deploy = deploy_score_from_eval_row(gt_metrics)
        print(
            f"[i] GT-flow baseline deploy={gt_deploy:.3f} tfinal_shape={gt_metrics.get('tfinal_clot_shape', float('nan')):.3f} "
            f"tfinal_band_F1={gt_metrics.get('tfinal_band_f1', float('nan')):.3f}",
            flush=True,
        )
        if vel_tag == "kinematics":
            os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"


if __name__ == "__main__":
    main()
