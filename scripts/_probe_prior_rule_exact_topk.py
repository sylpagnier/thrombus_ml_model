"""Quick probe of exact top-k rule configs on patient007 tfinal."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import ClotPriorRuleConfig, predict_prior_rule_deploy
from src.training.train_clot_phi_simple import _clot_metrics


def eval_cfg(data, t: int, cfg: ClotPriorRuleConfig) -> None:
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    step, phi, _mu, meta = predict_prior_rule_deploy(
        data, t, phys_cfg=phys, bio_cfg=bio, device=device, rule=cfg
    )
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    stag = int(meta.get("n_stag_leg", 0))
    print(
        f"{cfg.describe():42s} flag={meta['n_flag']:3d} "
        f"prior={meta['n_prior_leg']:3d} stag={stag:3d} "
        f"F1={band['clot_f1']:.3f} rec={band['clot_rec']:.3f} pred+={band['pred_pos_frac']:.3f}"
    )


def main() -> None:
    data = torch.load(
        REPO / "data/processed/graphs_biochem_anchors/patient007.pt",
        map_location="cpu",
        weights_only=False,
    )
    t = int(data.y.shape[0]) - 1
    configs = [
        ClotPriorRuleConfig(name="p80", prior_p=0.80, use_t0_strip=False),
        ClotPriorRuleConfig(name="p85|st10", prior_p=0.85, flux_stag_top_frac=0.10, use_t0_strip=False),
        ClotPriorRuleConfig(name="p90|st15", prior_p=0.90, flux_stag_top_frac=0.15, use_t0_strip=False),
        ClotPriorRuleConfig(name="p95|st20", prior_p=0.95, flux_stag_top_frac=0.20, use_t0_strip=False),
        ClotPriorRuleConfig(name="st20", prior_p=None, flux_stag_top_frac=0.20, use_t0_strip=False),
        ClotPriorRuleConfig(name="p80|st15", prior_p=0.80, flux_stag_top_frac=0.15, use_t0_strip=False),
        ClotPriorRuleConfig(name="p95|st10 old winner", prior_p=0.95, flux_stag_top_frac=0.10, use_t0_strip=False),
    ]
    for cfg in configs:
        eval_cfg(data, t, cfg)


if __name__ == "__main__":
    main()
