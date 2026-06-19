"""Why did the coupling loop not move the footprint? Decompose on p007.

Best-case one-shot: build the strongest clot plug (phi from final frozen Mat), re-solve kine,
and measure (a) how much the flow changes, (b) how much the WLS/wallfunc low-shear SET changes,
(c) the resulting gate F1. Tells us if the limiter is the kine diversion or the shear operator.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si
from src.core_physics.clot_phi_simple import log_blend_mu_eff_si
from src.core_physics.clot_phi_rollout import KinematicsUvProvider
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s2_kine_flow_test as kft

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
lss = float(cfg.lss); crit = float(cfg.viscosity_mat_crit)


def setf1(pred_set, gt, wall):
    p = pred_set & wall
    tp = float((p & gt).sum()); pr = tp / max(float(p.sum()), 1); rc = tp / max(float(gt.sum()), 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


def main():
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    provider = KinematicsUvProvider(dev)
    d = torch.load(s1b.ANCHOR_DIR / "patient007.pt", map_location=dev, weights_only=False)
    T = d.y.shape[0]
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()

    with torch.no_grad():
        pred = predict_kinematics(model, d.to(dev))
    u0, v0 = pred[:, 0].detach(), pred[:, 1].detach()

    # strongest plug: gel the TRUE clot footprint (oracle phi) -> upper bound on diversion
    phi = gt.float()
    mu_c = resolve_bulk_carreau_mu_si(d, T - 1, phys, dev, u_nd=u0, v_nd=v0)
    mu_eff = log_blend_mu_eff_si(mu_c, phi)
    with torch.no_grad():
        u1, v1 = provider.uv_nd_from_mu_si(d.to(dev), mu_eff)
    u1, v1 = u1.detach(), v1.detach()

    sp0 = torch.sqrt(u0 ** 2 + v0 ** 2); sp1 = torch.sqrt(u1 ** 2 + v1 ** 2)
    rel = float((torch.norm(torch.stack([u1 - u0, v1 - v0])) / max(torch.norm(torch.stack([u0, v0])), 1e-9)))
    print(f"[flow change] relL2(uv1-uv0)={rel:.3f}  speed in-clot: frozen={float(sp0[gt].mean()):.4g} "
          f"-> coupled={float(sp1[gt].mean()):.4g}  speed off-clot wall: "
          f"{float(sp0[wall & ~gt].mean()):.4g}->{float(sp1[wall & ~gt].mean()):.4g}")

    for name, shear_uv in [("wls", kft.wls_shear_uv), ("wallfunc", kft.wallfunc_shear_uv)]:
        s0 = shear_uv(d, u0, v0, dev); s1v = shear_uv(d, u1, v1, dev)
        g0 = (s0 < lss); g1 = (s1v < lss)
        flips = int((g0 ^ g1).sum())
        print(f"[{name:<8}] low-set frozen={int((g0 & wall).sum())}  coupled={int((g1 & wall).sum())}  "
              f"flips={flips}  gateF1 frozen={setf1(g0, gt, wall):.3f}  coupled={setf1(g1, gt, wall):.3f}")

    # reference: exact spf.sr gate (coupled COMSOL flow) on this graph
    sr_ex = s1b._exact_shear_p007(d, dev)
    ge = (sr_ex[-1] < lss)
    print(f"[exact   ] exact spf.sr (final) gateF1={setf1(ge, gt, wall):.3f}  "
          f"low-set={int((ge & wall).sum())}")
    print(f"[i] in-clot exact spf.sr mean={float(sr_ex[-1][gt].mean()):.3g}  "
          f"wls(coupled) in-clot mean={float(shear_uv(d,u1,v1,dev)[gt].mean()):.3g}")


if __name__ == "__main__":
    main()
