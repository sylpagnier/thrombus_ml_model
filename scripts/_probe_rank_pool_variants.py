"""Compare rank pools: ceiling, ceiling+dgamma, neighbor+dgamma (deploy vs oracle)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("CLOT_PHI_MASK_MODE", "neighbor")
os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
os.environ.setdefault("CLOT_PHI_DGAMMA_REF_TIME", "0")
os.environ.setdefault("CLOT_PHI_DGAMMA_WALL_MIN_SI", "100")
os.environ.setdefault("CLOT_PHI_DGAMMA_OFFWALL_PCT", "80")
os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    ClotPriorRuleConfig,
    cap_mu_eff_si,
    clot_phi_thresh_si,
    dgamma_dx_slice_mask,
    neighbor_supervision_mask,
    predict_prior_rule_deploy,
)
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402

RULE = ClotPriorRuleConfig(
    prior_p=0.80,
    flux_stag_top_frac=0.20,
    rank_tie_break=True,
    use_t0_strip=False,
)


def rank_masks(data, device, bio, phys, ti: int):
    ceiling = resolve_ceiling_mask(data, device, bio)
    mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[ti, :, STATE_CHANNEL_MU_EFF_ND]))
    clot_gt = mu_cap.reshape(-1) >= clot_phi_thresh_si(phys)
    neighbor_oracle = neighbor_supervision_mask(data, device, clot_gt)
    neighbor_deploy = neighbor_supervision_mask(
        data, device, torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
    )
    empty = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
    return {
        "ceiling": ceiling,
        "ceiling_dgamma": dgamma_dx_slice_mask(data, device, ceiling, empty, bio),
        "neighbor_oracle_dgamma": dgamma_dx_slice_mask(
            data, device, neighbor_oracle, clot_gt, bio
        ),
        "neighbor_deploy_dgamma": dgamma_dx_slice_mask(
            data, device, neighbor_deploy, empty, bio
        ),
    }


def eval_with_rank(data, device, phys, bio, t_out: int, rank_mask: torch.Tensor) -> dict:
    rule = ClotPriorRuleConfig(
        name="tmp",
        prior_p=RULE.prior_p,
        flux_stag_top_frac=RULE.flux_stag_top_frac,
        rank_tie_break=RULE.rank_tie_break,
        use_t0_strip=False,
        rank_dgamma_slice=False,
    )
    # Inject rank by temporarily using rank_dgamma_slice path with pre-built mask:
    # reuse predict with custom config field - add rank_mask override via env hack?
    # Call predict_phi_prior_rule with rank_dgamma_slice and swap - cleaner to add rank_mask_override
    from src.core_physics.clot_phi_simple import predict_phi_prior_rule

    step, _, _, _ = predict_prior_rule_deploy(
        data, t_out, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=rule
    )
    phi_base, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=0)
    phi_rank, _ = predict_phi_prior_rule(
        data,
        device,
        bio,
        rule=ClotPriorRuleConfig(
            name="ranked",
            prior_p=RULE.prior_p,
            flux_stag_top_frac=RULE.flux_stag_top_frac,
            rank_tie_break=RULE.rank_tie_break,
            use_t0_strip=False,
            rank_dgamma_slice=True,
        ),
        t_in=0,
    )
    del phi_rank
    # Direct re-rank: use rank_mask as pool
    from src.core_physics.clot_phi_simple import (
        _anchor_flow_props,
        _top_frac_mask,
        clot_prior_score_flat,
        compute_clot_kinematics_fields,
    )

    y0 = data.y[0]
    u, v = y0[:, 0], y0[:, 1]
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio, props)
    prior = clot_prior_score_flat(data, u, v, bio, props)
    leg_p = _top_frac_mask(prior, rank_mask, 0.20)
    leg_s = _top_frac_mask(fields.flux_stag.reshape(-1), rank_mask, 0.20)
    phi = torch.clamp(leg_p.float() + leg_s.float(), 0, 1)
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    return {
        "n_rank": int(rank_mask.sum()),
        "n_flag": int((phi >= 0.5).sum()),
        "band_f1": band["clot_f1"],
        "band_prec": band["clot_prec"],
        "band_rec": band["clot_rec"],
        "pred_pos_frac": band["pred_pos_frac"],
    }


def main() -> None:
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    path = REPO / "data/processed/graphs_biochem_anchors/patient007.pt"
    data = torch.load(path, map_location=device, weights_only=False)
    t_out = int(data.y.shape[0]) - 1
    masks = rank_masks(data, device, bio, phys, t_out)

    print("patient007 rank pool sizes:")
    for k, m in masks.items():
        print(f"  {k}: n={int(m.sum())}")

    print("\nRule OR (p80|st20|tie) ranked inside each pool:")
    for name, m in masks.items():
        r = eval_with_rank(data, device, phys, bio, t_out, m)
        print(
            f"  {name:26s} rank_n={r['n_rank']:4d} flag={r['n_flag']:4d} "
            f"F1={r['band_f1']:.3f} prec={r['band_prec']:.3f} rec={r['band_rec']:.3f} "
            f"pred+={r['pred_pos_frac']:.3f}"
        )

    # ceiling baseline via full deploy path
    step, phi, _, meta = predict_prior_rule_deploy(
        data, t_out, phys_cfg=phys, bio_cfg=bio, device=device, t_in=0, rule=RULE
    )
    band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
    print(
        f"\n  {'ceiling_full_deploy':26s} rank_n={meta['n_ceiling']:4d} flag={meta['n_flag']:4d} "
        f"F1={band['clot_f1']:.3f} prec={band['clot_prec']:.3f} rec={band['clot_rec']:.3f} "
        f"pred+={band['pred_pos_frac']:.3f}"
    )


if __name__ == "__main__":
    main()
