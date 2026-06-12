"""F5: Step 3b spike -- index vs macro tau; extrap pair count; optional rollout scale."""

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

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import (
    extrapolated_t_out_max,
    macro_tau_at_index,
    rollout_time_indices,
)
from src.core_physics.clot_forecast import iter_forecast_pairs
from src.core_physics.clot_temporal_growth_rules import (
    reset_temporal_kinematics_cache,
    rollout_temporal_phi,
    temporal_rule_config_from_env,
    temporal_vel_source,
    _time_frac_at_index,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_device import resolve_clot_ml_eval_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3b continuous-time spike")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--sim-end-scale", type=float, default=1.5)
    ap.add_argument("--compare-rollout", action="store_true")
    ap.add_argument("--out", default="outputs/biochem/clot_ml_ladder/step3b_spike/spike.json")
    args = ap.parse_args()

    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ["CLOT_ML_SIM_END_SCALE"] = str(args.sim_end_scale)

    device = resolve_clot_ml_eval_device()
    reset_temporal_kinematics_cache()
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")

    graph_path = REPO / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n = int(data.y.shape[0])

    os.environ.pop("CLOT_ML_USE_MACRO_TAU", None)
    index_fracs = [_time_frac_at_index(data, i) for i in range(n)]

    os.environ["CLOT_ML_USE_MACRO_TAU"] = "1"
    tau_fracs = [macro_tau_at_index(data, i, bio_cfg=bio) for i in range(n)]

    t_out_max = extrapolated_t_out_max(data, sim_end_scale=args.sim_end_scale)
    pairs_extrap = iter_forecast_pairs(n, t_out_max=t_out_max)
    t_indices = rollout_time_indices(data, sim_end_scale=args.sim_end_scale)

    payload: dict = {
        "anchor": args.anchor,
        "n_comsol_steps": n,
        "sim_end_scale": args.sim_end_scale,
        "tau_ref_s": macro_tau_at_index(data, n - 1, bio_cfg=bio),
        "vel_source": temporal_vel_source(),
        "index_time_frac_tail": index_fracs[-5:],
        "macro_tau_tail": tau_fracs[-5:],
        "extrap_t_out_max": t_out_max,
        "extrap_pair_count": len(pairs_extrap),
        "rollout_index_count": len(t_indices),
    }

    if args.compare_rollout:
        rule_cfg = load_step0_coef_json(REPO / args.step0_json).to_rule_config(name="step3b_spike")
        phi_in = rollout_temporal_phi(
            data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=1.0
        )
        os.environ["CLOT_ML_USE_MACRO_TAU"] = "1"
        phi_tau = rollout_temporal_phi(
            data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=args.sim_end_scale
        )
        tf = n - 1
        payload["phi_commit_frac_tfinal_index"] = float((phi_in[tf] > 0.5).float().mean().item())
        payload["phi_commit_frac_tfinal_tau_extrap"] = float(
            (phi_tau.get(t_out_max, phi_tau[tf]) > 0.5).float().mean().item()
        )
        payload["extrap_phi_commit_frac"] = float(
            (phi_tau.get(t_out_max, phi_tau[tf]) > 0.5).float().mean().item()
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[i] anchor={args.anchor} n={n} extrap_t_out_max={t_out_max} pairs={len(pairs_extrap)}")
    print(f"[i] index t_frac @ end: {index_fracs[-1]:.3f}  macro tau @ end: {tau_fracs[-1]:.3f}")
    if args.compare_rollout:
        print(
            f"[i] phi_commit in-window={payload.get('phi_commit_frac_tfinal_index', 0):.3f} "
            f"extrap={payload.get('extrap_phi_commit_frac', 0):.3f}"
        )
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
