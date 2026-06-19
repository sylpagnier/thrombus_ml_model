"""Sanity: does the kine model slow flow when MU_PRIOR is raised? (uniform + clot-plug)"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si
from src.core_physics.clot_phi_simple import log_blend_mu_eff_si, clot_phi_mu_solid_si
from src.core_physics.clot_phi_rollout import KinematicsUvProvider
from src.config import NodeFeat
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")


def main():
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    provider = KinematicsUvProvider(dev)
    d = torch.load(s1b.ANCHOR_DIR / "patient007.pt", map_location=dev, weights_only=False)
    T = d.y.shape[0]
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    base_mu_nd = d.x[:, NodeFeat.MU_PRIOR].clone()
    print(f"[i] baseline MU_PRIOR(nd): min={float(base_mu_nd.min()):.3g} med={float(base_mu_nd.median()):.3g} "
          f"max={float(base_mu_nd.max()):.3g}  mu_solid_si={clot_phi_mu_solid_si():.3g}")
    print(f"[i] viscosity_si_to_nd(mu_solid)={float(phys.viscosity_si_to_nd(torch.tensor([[clot_phi_mu_solid_si()]]))):.3g}")

    with torch.no_grad():
        u0, v0 = (lambda p: (p[:, 0], p[:, 1]))(predict_kinematics(model, d.to(dev)))
    sp0 = torch.sqrt(u0 ** 2 + v0 ** 2)

    # uniform mu x10, x100 via provider (inject SI = nd->si of base*factor)
    base_si = phys.viscosity_nd_to_si(base_mu_nd.reshape(-1, 1)).reshape(-1)
    for fac in (10.0, 100.0):
        with torch.no_grad():
            u, v = provider.uv_nd_from_mu_si(d.to(dev), base_si * fac)
        sp = torch.sqrt(u ** 2 + v ** 2)
        print(f"[uniform x{fac:>5.0f}] mean speed {float(sp0.mean()):.4g} -> {float(sp.mean()):.4g}  "
              f"in-clot {float(sp0[gt].mean()):.4g} -> {float(sp[gt].mean()):.4g}")

    # clot plug
    mu_c = resolve_bulk_carreau_mu_si(d, T - 1, phys, dev, u_nd=u0, v_nd=v0)
    mu_eff = log_blend_mu_eff_si(mu_c, gt.float())
    inj_nd = phys.viscosity_si_to_nd(mu_eff.reshape(-1, 1)).reshape(-1)
    print(f"[plug] injected MU_PRIOR(nd) in-clot med={float(inj_nd[gt].median()):.3g}  "
          f"off-clot med={float(inj_nd[~gt].median()):.3g}  (baseline med {float(base_mu_nd.median()):.3g})")


if __name__ == "__main__":
    main()
