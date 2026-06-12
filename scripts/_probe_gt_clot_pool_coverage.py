"""Where do GT clot nodes live vs ceiling / dgamma pools (patient007 tfinal)?"""

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
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    cap_mu_eff_si,
    clot_phi_thresh_si,
    dgamma_dx_slice_mask,
    gt_neg_dgamma_dx_phys,
    neighbor_supervision_mask,
    predict_phi_prior_rule,
    ClotPriorRuleConfig,
)
from src.training.train_clot_phi_simple import build_clot_phi_step  # noqa: E402

device = torch.device("cpu")
phys = PhysicsConfig(phase="biochem")
bio = BiochemConfig(phase="biochem")
data = torch.load(
    REPO / "data/processed/graphs_biochem_anchors/patient007.pt",
    map_location=device,
    weights_only=False,
)
t_out = int(data.y.shape[0]) - 1
step = build_clot_phi_step(data, t_out, phys, bio, device)

ceiling = resolve_ceiling_mask(data, device, bio)
mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[t_out, :, STATE_CHANNEL_MU_EFF_ND]))
clot_gt = mu_cap.reshape(-1) >= clot_phi_thresh_si(phys)
empty = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
neighbor_oracle = neighbor_supervision_mask(data, device, clot_gt)
ceiling_dgamma = dgamma_dx_slice_mask(data, device, ceiling, empty, bio)
train_pool = dgamma_dx_slice_mask(data, device, neighbor_oracle, clot_gt, bio)

phi_gt = step.phi_gt.reshape(-1).bool()
loss_m = step.loss_mask.reshape(-1).bool()
gt_in_loss = phi_gt & loss_m

neg_dx = gt_neg_dgamma_dx_phys(data, 0, bio, device)
wall = data.mask_wall.view(-1).bool().to(device)

rule = ClotPriorRuleConfig(
    prior_p=0.80, flux_stag_top_frac=0.20, rank_tie_break=True, use_t0_strip=False
)
phi_ceiling, _ = predict_phi_prior_rule(data, device, bio, rule=rule, t_in=0)
phi_dg, _ = predict_phi_prior_rule(
    data,
    device,
    bio,
    rule=ClotPriorRuleConfig(
        prior_p=0.80,
        flux_stag_top_frac=0.20,
        rank_tie_break=True,
        use_t0_strip=False,
        rank_dgamma_slice=True,
    ),
    t_in=0,
)

pred_ceiling = (phi_ceiling >= 0.5) & loss_m
pred_dg = (phi_dg >= 0.5) & loss_m


def report(label: str, mask: torch.Tensor) -> None:
    m = mask & gt_in_loss
    n_gt = int(gt_in_loss.sum())
    n_hit = int(m.sum())
    print(f"{label:36s} covers GT clots: {n_hit}/{n_gt} ({100*n_hit/max(n_gt,1):.1f}%)")


print(f"GT clot nodes in loss band: {int(gt_in_loss.sum())}")
print(f"Pool sizes: ceiling={int(ceiling.sum())} ceiling_dgamma={int(ceiling_dgamma.sum())} train_pool={int(train_pool.sum())}")
print()

report("ceiling pool", ceiling)
report("ceiling_dgamma pool", ceiling_dgamma)
report("train loss pool (231)", train_pool)
report("ceiling rule flags", pred_ceiling)
report("dgamma-rank rule flags", pred_dg)

# GT clots in ceiling but NOT in ceiling_dgamma
lost = gt_in_loss & ceiling & ~ceiling_dgamma
kept = gt_in_loss & ceiling & ceiling_dgamma
print()
print(f"GT clots inside ceiling but OUTSIDE ceiling_dgamma: {int(lost.sum())}")
print(f"GT clots inside ceiling AND ceiling_dgamma:       {int(kept.sum())}")

if int(lost.sum()) > 0:
    on_w = lost & wall
    off_w = lost & ~wall
    print(f"  lost on-wall: {int(on_w.sum())}  off-wall: {int(off_w.sum())}")
    if int(on_w.sum()) > 0:
        print(f"  lost on-wall neg_dx @t0: min={neg_dx[on_w].min():.1f} max={neg_dx[on_w].max():.1f} (need>={100})")
    if int(off_w.sum()) > 0:
        thr = torch.quantile(neg_dx[ceiling & ~wall], 0.80)
        print(f"  lost off-wall neg_dx @t0: min={neg_dx[off_w].min():.1f} max={neg_dx[off_w].max():.1f} (need>={thr:.1f})")

# GT clots in train pool but not ceiling_dgamma
extra_train = gt_in_loss & train_pool & ~ceiling_dgamma
print()
print(f"GT clots in train pool but NOT ceiling_dgamma: {int(extra_train.sum())} (GT seed bypass)")
