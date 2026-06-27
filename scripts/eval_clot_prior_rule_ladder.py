"""Evaluate sweep-winner prior rule on CAVO deploy ladder stages (S0/S1/G1/G2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_forecast import iter_forecast_pairs  # noqa: E402
from src.core_physics.clot_growth_masks import growth_seed_mode  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    predict_phi_prior_rule,
    prior_rule_config_from_env,
    project_deploy_mu_with_support,
    log_blend_mu_eff_si,
)
from src.core_physics.clot_forecast import build_clot_forecast_pair_step  # noqa: E402
from src.evaluation.clot_shape_score import compute_clot_shape_metrics  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics, _list_anchor_paths  # noqa: E402
from src.utils.paths import get_project_root


def _apply_deploy_env(stage: str) -> None:
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


def _eval_pair(
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
    phi, _meta = predict_phi_prior_rule(
        data, device, bio, rule=rule, t_in=t_in
    )
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
    y_sl = data.y[t_out].to(device=device, dtype=torch.float32)
    pred_state = y_sl.clone()
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys.viscosity_si_to_nd(mu.reshape(-1))
    shape = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys,
        gt_anchor_state=data.y[0].to(device=device, dtype=torch.float32),
    )
    return {
        "t_in": t_in,
        "t_out": t_out,
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "band_pred_frac": band["pred_pos_frac"],
        "band_gt_frac": band["gt_pos_frac"],
        "clot_shape": shape["clot_shape"],
        "clot_shape_rec": shape["clot_recall"],
        "n_flag": int((phi > 0.5).sum().item()),
    }


def eval_anchor_stage(
    path: Path,
    stage: str,
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    rule,
    late_frac: float = 0.5,
) -> dict:
    data = torch.load(path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    pairs = iter_forecast_pairs(n_steps, time_stride=1)
    if not pairs:
        raise ValueError(f"{path.stem}: no forecast pairs")

    phi_hist: dict[int, torch.Tensor] | None = {} if stage == "g2" else None
    rows: list[dict] = []
    for t_in, t_out in pairs:
        if phi_hist is not None and t_in not in phi_hist:
            phi_in, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=t_in)
            phi_hist[int(t_in)] = phi_in.detach()
        row = _eval_pair(
            data, t_in, t_out, phys=phys, bio=bio, device=device, rule=rule, phi_hist=phi_hist
        )
        rows.append(row)
        if phi_hist is not None:
            phi_out, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=t_in)
            phi_hist[int(t_out)] = phi_out.detach()

    late_cut = int(n_steps * late_frac)
    late = [r for r in rows if int(r["t_out"]) >= late_cut]
    use = late if late else rows

    def _mean(key: str) -> float:
        return float(sum(r[key] for r in use) / max(len(use), 1))

    tfinal = rows[-1]
    return {
        "anchor": path.stem,
        "stage": stage,
        "growth_seed": growth_seed_mode(),
        "n_pairs": len(rows),
        "mean_band_f1": _mean("band_f1"),
        "mean_band_prec": _mean("band_prec"),
        "mean_band_rec": _mean("band_rec"),
        "mean_band_pred_frac": _mean("band_pred_frac"),
        "mean_clot_shape": _mean("clot_shape"),
        "tfinal_band_f1": float(tfinal["band_f1"]),
        "tfinal_band_pred_frac": float(tfinal["band_pred_frac"]),
        "tfinal_clot_shape": float(tfinal["clot_shape"]),
        "pairs": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval prior rule on deploy ladder stage")
    ap.add_argument("--stage", required=True, choices=("s0", "s1", "g1", "g2"))
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchor", default="", help="Single anchor stem (default all)")
    ap.add_argument(
        "--out-json",
        default="",
        help="Default outputs/biochem/diagnostics/clot_prior_rule_{stage}.json",
    )
    ap.add_argument("--late-frac", type=float, default=0.5)
    args = ap.parse_args()
    _apply_deploy_env(args.stage)

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir

    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule = prior_rule_config_from_env()

    if args.anchor.strip():
        paths = [anchor_dir / f"{args.anchor.strip()}.pt"]
    else:
        paths = [Path(p) for p in _list_anchor_paths(anchor_dir.resolve())]

    summaries = []
    for path in paths:
        if not path.is_file():
            print(f"[WARN] skip missing {path}")
            continue
        s = eval_anchor_stage(
            path, args.stage, phys=phys, bio=bio, device=device, rule=rule, late_frac=args.late_frac
        )
        summaries.append({k: v for k, v in s.items() if k != "pairs"})
        print(
            f"{s['anchor']:<12} pairs={s['n_pairs']:>2} "
            f"mean F1={s['mean_band_f1']:.3f} tfinal F1={s['tfinal_band_f1']:.3f} "
            f"shape={s['mean_clot_shape']:.3f} seed={s['growth_seed']}"
        )

    if not summaries:
        print("[ERR] no anchors")
        sys.exit(1)

    out_path = Path(args.out_json) if args.out_json else (
        root / f"outputs/biochem/diagnostics/clot_prior_rule_{args.stage}.json"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": args.stage,
        "rule": rule.describe(),
        "anchors": summaries,
        "mean_band_f1": float(sum(s["mean_band_f1"] for s in summaries) / len(summaries)),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
