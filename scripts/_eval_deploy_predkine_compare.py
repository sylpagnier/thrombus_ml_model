"""Compare GT-flow vs GINO-DEQ deploy metrics for promoted temporal rule."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.promote_clot_architecture_winner import _parse_rule_name  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    deploy_score_from_eval_row,
    eval_temporal_rule_on_anchor,
    reset_temporal_kinematics_cache,
    temporal_rule_config_from_env,
)


def main() -> int:
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    for key, val in _parse_rule_name("loc_prog_both_t20_s0_ndx25_inc40").items():
        os.environ[key] = val

    anchor_dir = REPO / "data/processed/graphs_biochem_anchors"
    cfg = temporal_rule_config_from_env()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    stems = sorted(p.stem for p in anchor_dir.glob("patient*.pt"))

    print("anchor          mode         deploy  tfinal_sh  band_F1  early_pred+")
    print("-" * 72)
    means: dict[str, list[float]] = {"gt": [], "kinematics": []}
    for mode in ("gt", "kinematics"):
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = mode
        reset_temporal_kinematics_cache()
        for stem in stems:
            data = torch.load(anchor_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
            m = eval_temporal_rule_on_anchor(
                data, cfg, stem=stem, device=device, phys_cfg=phys, bio_cfg=bio
            )
            deploy = deploy_score_from_eval_row(m)
            means[mode].append(deploy)
            print(
                f"{stem:14} {mode:12} {deploy:6.3f} "
                f"{m.get('tfinal_clot_shape', float('nan')):9.3f} "
                f"{m.get('tfinal_band_f1', float('nan')):8.3f} "
                f"{m.get('early_mean_pred_frac', float('nan')):11.3f}"
            )
    print("-" * 72)
    for mode in ("gt", "kinematics"):
        avg = sum(means[mode]) / len(means[mode]) if means[mode] else float("nan")
        print(f"mean deploy ({mode}): {avg:.3f} over {len(means[mode])} anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
