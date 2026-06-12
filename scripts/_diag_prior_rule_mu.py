"""One-off: why fullmesh pred clot is red everywhere."""
import os

import torch

os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.85")
os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_growth_masks import (
    resolve_bulk_carreau_mu_si,
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
)
from src.core_physics.clot_phi_simple import (
    clot_phi_thresh_si,
    predict_phi_prior_rule_baseline,
    predict_prior_rule_deploy,
)

data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt", weights_only=False)
device = torch.device("cpu")
phys = PhysicsConfig(phase="biochem")
bio = BiochemConfig(phase="biochem")
ti = int(data.y.shape[0]) - 1
thr = clot_phi_thresh_si(phys)
ceil = resolve_ceiling_mask(data, device, bio)
sup = resolve_growth_support_at_time(data, ti, device, phys, bio)
phi, meta = predict_phi_prior_rule_baseline(data, device, bio, t_in=0)
step, phi2, mu, _ = predict_prior_rule_deploy(data, ti, phys_cfg=phys, bio_cfg=bio, device=device)
mu_gt = phys.viscosity_nd_to_si(data.y[ti][:, STATE_CHANNEL_MU_EFF_ND])
carreau_t53 = resolve_bulk_carreau_mu_si(data, ti, phys, device)

n = int(data.num_nodes)
pred_clot = mu >= thr
gt_clot = mu_gt >= thr
print(f"n_nodes={n} thr={thr:.4f}")
print(f"phi>0.5={int((phi > 0.5).sum())} rule_flag={meta['n_flag']} prior_thr={meta['prior_thr']:.4f}")
print(f"ceiling={int(ceil.sum())} support@53={int(sup.sum())}")
print(f"GT clot mesh={int(gt_clot.sum())} ({100*gt_clot.float().mean():.2f}%)")
print(f"pred clot mesh={int(pred_clot.sum())} ({100*pred_clot.float().mean():.2f}%)")
print(f"pred clot OFF support={int((pred_clot & ~sup).sum())} OFF ceiling={int((pred_clot & ~ceil).sum())}")
print(f"carreau@t53>=thr={int((carreau_t53 >= thr).sum())} ({100*(carreau_t53 >= thr).float().mean():.2f}%)")
print(
    f"carreau@t53 mean={float(carreau_t53.mean()):.5f} "
    f"p50={float(carreau_t53.median()):.5f} max={float(carreau_t53.max()):.5f}"
)
print(
    f"mu_pred mean={float(mu.mean()):.5f} min={float(mu.min()):.5f} max={float(mu.max()):.5f}"
)
print(f"mu_pred off_support mean={float(mu[~sup].mean()):.5f} n={int((~sup).sum())}")
if (sup & (phi <= 0.5)).any():
    m = sup & (phi <= 0.5)
    print(f"mu_pred on_support phi=0 mean={float(mu[m].mean()):.5f} >=thr={int((mu[m]>=thr).sum())}")
if (sup & (phi > 0.5)).any():
    m = sup & (phi > 0.5)
    print(f"mu_pred on_support phi=1 mean={float(mu[m].mean()):.5f} n={int(m.sum())}")
