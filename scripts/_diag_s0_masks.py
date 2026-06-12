"""One-off: mask vs GT/pred stats for S0 ceiling_growth."""
from __future__ import annotations

import os

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step
from src.core_physics.clot_growth_masks import (
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    clot_phi_thresh_si,
    log_blend_mu_eff_si,
    project_deploy_mu_with_support,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.utils.paths import get_project_root

os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "static_final")
os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "5")
os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
os.environ.setdefault("CLOT_PHI_HYBRID", "0")


def main() -> None:
    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(root / "data/processed/graphs_biochem_anchors/patient007.pt", weights_only=False)
    ckpt_path = root / "outputs/biochem/clot_deploy/s0_static_final/clot_phi_best.pth"
    if not ckpt_path.is_file():
        print(f"[ERR] missing {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    apply_clot_phi_config_from_checkpoint(ckpt.get("config", {}))
    apply_clot_phi_eval_defaults()
    cfg = ckpt["config"]
    model = build_clot_phi_model(in_dim=int(cfg["in_dim"]), hidden=int(cfg["hidden"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ti = 53
    t0 = resolve_t0_dgamma_wall_mask(data, device, bio)
    ceil = resolve_ceiling_mask(data, device, bio)
    sup = resolve_growth_support_at_time(data, ti, device, phys, bio)
    thr = clot_phi_thresh_si(phys)
    y = data.y[ti]
    mu_gt = phys.viscosity_nd_to_si(y[:, 3])
    gt_clot = (mu_gt >= thr).cpu()
    step = build_clot_forecast_pair_step(data, 0, ti, phys, bio, device)
    with torch.no_grad():
        phi = model(step.features)
        mu = log_blend_mu_eff_si(step.mu_c_si, phi)
        mu = project_deploy_mu_with_support(
            data=data,
            step=step,
            mu_pred=mu,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            forecast_one_step=True,
            time_index=ti,
            bulk_time_index=ti,
        )
    pred_clot = (mu >= thr).cpu()
    phi_clot = (phi > 0.5).cpu()
    n = int(data.num_nodes)
    gt_n = int(gt_clot.sum())
    print(f"n_nodes={n}")
    print(f"t0={int(t0.sum())} ceiling={int(ceil.sum())} support@53={int(sup.sum())}")
    print(f"GT clot mesh={gt_n} ({100*gt_n/n:.2f}%)")
    print(f"GT in support={int((gt_clot & sup.cpu()).sum())} in ceiling={int((gt_clot & ceil.cpu()).sum())}")
    print(f"GT outside ceiling={int((gt_clot & ~ceil.cpu()).sum())}")
    print(f"pred clot mesh={int(pred_clot.sum())} ({100*int(pred_clot.sum())/n:.2f}%)")
    print(f"pred in support={int((pred_clot & sup.cpu()).sum())} phi>0.5 in support={int((phi_clot & sup.cpu()).sum())}")
    if gt_n:
        print(f"support recall on GT={float((gt_clot & sup.cpu()).sum())/gt_n:.3f}")
    print(f"mean phi support={float(phi[sup].mean()):.3f} mean phi_gt support={float(step.phi_gt[sup].mean()):.3f}")


if __name__ == "__main__":
    main()
