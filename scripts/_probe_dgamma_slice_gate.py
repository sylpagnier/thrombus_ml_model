"""Compare rule phi F1: ceiling rank vs dgamma-slice rank pool (refined winner)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import ClotPriorRuleConfig, predict_prior_rule_deploy  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics, _list_anchor_paths  # noqa: E402

BASE = ClotPriorRuleConfig(
    name="refined_ceiling",
    prior_p=0.80,
    flux_stag_top_frac=0.20,
    rank_tie_break=True,
    use_t0_strip=False,
    rank_dgamma_slice=False,
)
DGAMMA = ClotPriorRuleConfig(
    name="refined_dgamma_rank",
    prior_p=0.80,
    flux_stag_top_frac=0.20,
    rank_tie_break=True,
    use_t0_strip=False,
    rank_dgamma_slice=True,
)


def eval_anchor(path: Path, rule: ClotPriorRuleConfig) -> dict:
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(path, map_location=device, weights_only=False)
    t_out = int(data.y.shape[0]) - 1
    step, phi, _mu, meta = predict_prior_rule_deploy(
        data, t_out, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=rule
    )
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    return {
        "anchor": path.stem,
        "rule": rule.name,
        "n_ceiling": meta["n_ceiling"],
        "n_rank_mask": meta.get("n_rank_mask", meta["n_ceiling"]),
        "n_flag": meta["n_flag"],
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "band_pred_frac": band["pred_pos_frac"],
        "band_gt_frac": band["gt_pos_frac"],
    }


def main() -> None:
    anchor_dir = REPO / "data/processed/graphs_biochem_anchors"
    paths = [Path(p) for p in _list_anchor_paths(anchor_dir.resolve()) if Path(p).is_file()]
    for rule in (BASE, DGAMMA):
        rows = [eval_anchor(p, rule) for p in paths]
        mf1 = sum(r["band_f1"] for r in rows) / len(rows)
        print(f"\n=== {rule.describe()} mean_F1={mf1:.3f}")
        for r in rows:
            print(
                f"  {r['anchor']}: rank_n={r['n_rank_mask']} flag={r['n_flag']} "
                f"F1={r['band_f1']:.3f} prec={r['band_prec']:.3f} rec={r['band_rec']:.3f} "
                f"pred+={r['band_pred_frac']:.3f}"
            )


if __name__ == "__main__":
    main()
