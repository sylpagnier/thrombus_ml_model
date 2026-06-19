"""Quick probe: T0 mu/phi under COMSOL-oracle vs deploy-proxy settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.core_physics.clot_phi_simple import build_clot_phi_step, cap_mu_eff_si, physics_mu_eff_si
from src.core_physics.clot_trigger_rollout import rollout_clot_trigger_physics

data = torch.load(
    REPO / "data/processed/graphs_biochem_anchors/patient007.pt",
    map_location="cpu",
    weights_only=False,
)
phys = PhysicsConfig(phase="biochem")
bio = BiochemConfig(phase="biochem")
os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "comsol_carreau"
os.environ["CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR"] = "patient007"

for gamma in ("max", "comsol_sr"):
    for rm in (4, 80):
        os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = gamma
        os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = str(rm)
        traj = rollout_clot_trigger_physics(
            data, phys_cfg=phys, bio_cfg=bio, device=torch.device("cpu")
        )
        t = 53
        step = build_clot_phi_step(data, t, phys, bio, torch.device("cpu"))
        mu_gt = phys.viscosity_nd_to_si(data.y[t, :, 3])
        mu_phys = cap_mu_eff_si(
            physics_mu_eff_si(
                step.mu_c_si,
                step.species_log_gt,
                bio,
                device=torch.device("cpu"),
                data=data,
                u_nd=step.u_flow_nd,
                v_nd=step.v_flow_nd,
                phys_cfg=phys,
                time_index=t,
            )
        )
        growth = gt_growth_commit_mask_at_time(data, t, phys, torch.device("cpu"))
        gr = (
            pearsonr(mu_phys[growth].numpy(), mu_gt[growth].numpy())[0]
            if growth.sum() > 10
            else float("nan")
        )
        bulk = mu_gt < 0.012
        bulk_ratio = float((mu_gt[bulk] / mu_phys[bulk].clamp(1e-8)).median())
        print(
            f"gamma={gamma} rm={rm} bulk_mu_ratio={bulk_ratio:.3f} "
            f"mu_r={pearsonr(mu_phys.numpy(), mu_gt.numpy())[0]:.3f} growth_mu_r={gr:.3f} "
            f"phi>0.5={int((traj[t]['phi'] > 0.5).sum())}"
        )
