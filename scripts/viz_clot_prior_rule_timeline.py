"""Timeline viz: prior rule clot predictions across macro times (S0/S1/G1/G2)."""

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
from src.core_physics.clot_forecast import (  # noqa: E402
    build_clot_forecast_pair_step,
    clot_forecast_pair_stride,
    iter_forecast_pairs,
)
from src.core_physics.clot_growth_masks import (  # noqa: E402
    growth_seed_mode,
    resolve_ceiling_mask,
)
from src.core_physics.clot_phi_simple import (  # noqa: E402
    log_blend_mu_eff_si,
    predict_phi_prior_rule,
    prior_rule_config_from_env,
    project_deploy_mu_with_support,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
from src.utils.paths import get_project_root


def apply_rule_ladder_env(stage: str) -> None:
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HYBRID", "0")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.80")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_T0_STRIP", "0")
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_FORECAST_PAIR_STRIDE", "1")
    if stage == "s0":
        os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = "static_final"
        os.environ["CLOT_PHI_GROWTH_SEED"] = "gt"
    elif stage == "s1":
        os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = "from_t0"
        os.environ["CLOT_PHI_GROWTH_SEED"] = "gt"
    elif stage == "g1":
        os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = "rolling"
        os.environ["CLOT_PHI_GROWTH_SEED"] = "gt"
    elif stage == "g2":
        os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = "rolling"
        os.environ["CLOT_PHI_GROWTH_SEED"] = "pred"
    else:
        raise ValueError(f"unknown stage {stage}")


def _pick_keyframe_pairs(pairs: list[tuple[int, int]], keyframes: int) -> list[tuple[int, int]]:
    if not pairs:
        return []
    if keyframes <= 0 or len(pairs) <= keyframes:
        return pairs
    idx = np.linspace(0, len(pairs) - 1, num=keyframes, dtype=int)
    return [pairs[int(i)] for i in sorted(set(idx.tolist()))]


def _eval_rule_frame(
    data,
    t_in: int,
    t_out: int,
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    rule,
    phi_hist: dict[int, torch.Tensor] | None,
) -> dict:
    step = build_clot_forecast_pair_step(
        data, t_in, t_out, phys, bio, device, phi_pred_by_time=phi_hist
    )
    phi, meta = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=t_in)
    mu = log_blend_mu_eff_si(step.mu_c_si, phi)
    mu = project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        forecast_one_step=True,
        time_index=t_out,
        bulk_time_index=t_out,
        phi_pred_by_time=phi_hist,
    )
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    ceiling = resolve_ceiling_mask(data, device, bio).detach().cpu().numpy().astype(bool)
    return {
        "t_in": int(t_in),
        "t_out": int(t_out),
        "phi_gt": step.phi_gt.detach().cpu().numpy(),
        "phi_pred": phi.detach().cpu().numpy(),
        "loss_mask": step.loss_mask.detach().cpu().numpy().astype(bool),
        "ceiling": ceiling,
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "band_pred_frac": band["pred_pos_frac"],
        "band_gt_frac": band["gt_pos_frac"],
        "rule": str(meta.get("rule", rule.describe())),
        "n_flag": int((phi > 0.5).sum().item()),
        "mu_pred": mu.detach().cpu().numpy(),
    }


def collect_timeline_frames(
    data,
    stage: str,
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    rule,
    keyframes: int,
) -> list[dict]:
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    pairs = _pick_keyframe_pairs(pairs, keyframes)
    phi_hist: dict[int, torch.Tensor] | None = {} if stage == "g2" else None
    frames: list[dict] = []
    for t_in, t_out in pairs:
        if phi_hist is not None and t_in not in phi_hist:
            phi_in, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=t_in)
            phi_hist[int(t_in)] = phi_in.detach()
        frame = _eval_rule_frame(
            data, t_in, t_out, phys=phys, bio=bio, device=device, rule=rule, phi_hist=phi_hist
        )
        frames.append(frame)
        if phi_hist is not None:
            phi_out, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=t_in)
            phi_hist[int(t_out)] = phi_out.detach()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Prior rule clot timeline (phi panels over macro times)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument(
        "--stage",
        default="s1",
        choices=("s0", "s1", "g1", "g2"),
        help="S1=from_t0 (default), G1=rolling flow@t_in, G2=pred growth seed",
    )
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--scatter-size", type=float, default=5.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    apply_rule_ladder_env(args.stage)
    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule = prior_rule_config_from_env()

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    graph_path = anchor_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    pos = data.x[:, :2].detach().cpu().numpy()
    frames = collect_timeline_frames(
        data, args.stage, phys=phys, bio=bio, device=device, rule=rule, keyframes=args.keyframes
    )
    if not frames:
        raise RuntimeError("no timeline frames")

    seed = growth_seed_mode()
    stride = clot_forecast_pair_stride()
    rule_label = frames[0]["rule"]
    ncols = len(frames)
    dot = float(args.scatter_size)
    fig, axes = plt.subplots(2, ncols, figsize=(max(3.0 * ncols, 9), 6.5), squeeze=False)
    fig.suptitle(
        f"prior rule timeline -- {args.anchor} stage={args.stage.upper()} | {rule_label} | "
        f"seed={seed} stride={stride} ceiling_hops={os.environ.get('CLOT_PHI_CEILING_HOPS', '2')}",
        fontsize=12,
    )

    rows_out: list[dict] = []
    for j, frame in enumerate(frames):
        t_in = frame["t_in"]
        t_out = frame["t_out"]
        loss_m = frame["loss_mask"]
        ceiling_m = frame["ceiling"]
        title = (
            f"t_out={t_out}\n"
            f"F1={frame['band_f1']:.2f} flag={frame['n_flag']}"
        )
        if args.stage in ("g1", "g2"):
            title = f"t_in={t_in} " + title

        _scatter_fullmesh_region(
            axes[0, j],
            pos,
            frame["phi_gt"],
            loss_m,
            "GT phi" if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
        )
        _scatter_fullmesh_region(
            axes[1, j],
            pos,
            frame["phi_pred"],
            ceiling_m,
            "Rule phi" if j == 0 else "",
            cmap="bwr",
            vmin=0,
            vmax=1,
            s=dot,
        )
        axes[0, j].set_title(title, fontsize=8)
        if j == 0:
            axes[0, j].set_ylabel("GT phi (band)", fontsize=10)
            axes[1, j].set_ylabel("Rule phi (ceiling)", fontsize=10)

        row = {k: v for k, v in frame.items() if k not in ("phi_gt", "phi_pred", "loss_mask", "ceiling", "mu_pred")}
        row["anchor"] = args.anchor
        row["stage"] = args.stage
        rows_out.append(row)
        print(
            f"[i]  stage={args.stage} t_in={t_in} t_out={t_out} "
            f"band F1={frame['band_f1']:.3f} prec={frame['band_prec']:.3f} "
            f"rec={frame['band_rec']:.3f} flag={frame['n_flag']}",
            flush=True,
        )

    fig.tight_layout()
    out_default = (
        root / f"outputs/biochem/viz/clot_deploy/prior_rule_{args.anchor}_{args.stage}_timeline.png"
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


if __name__ == "__main__":
    main()
