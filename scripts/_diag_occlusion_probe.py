"""Does encoding the clot as GEOMETRY (wall/occlusion) make the kine model divert?

Contrast with viscosity injection (_diag_kine_mu_response: in-clot speed went UP). Here we
re-express the oracle clot as a solid: recompute SDF = dist to (wall U clot), shrink the
hydraulic WIDTH accordingly, zero the velocity prior inside the clot, and re-solve. If the kine
model respects this (it was trained on many channel widths), in-clot speed should DROP and the
open lumen should speed up = diversion. That would make a deployable geometry-coupling loop viable.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.spatial import cKDTree
from src.config import BiochemConfig, PhysicsConfig, NodeFeat
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.data_gen.lib.graph_velocity_priors import (
    smooth_width_nd_on_edges, width_nd_to_radius_nd, mass_conserving_umax_nd)
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s2_kine_flow_test as kft

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
lss = float(cfg.lss)


def setf1(pred_set, gt, wall):
    p = pred_set & wall
    tp = float((p & gt).sum()); pr = tp / max(float(p.sum()), 1); rc = tp / max(float(gt.sum()), 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


def run_model(model, d, x):
    b = d.clone(); b.x = x
    with torch.no_grad():
        pred = predict_kinematics(model, b.to(dev))
    return pred[:, 0].detach(), pred[:, 1].detach()


def main():
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    d = torch.load(s1b.ANCHOR_DIR / "patient007.pt", map_location=dev, weights_only=False)
    T = d.y.shape[0]
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    pos = d.x[:, NodeFeat.XY].cpu().numpy()

    u0, v0 = run_model(model, d, d.x)
    sp0 = torch.sqrt(u0 ** 2 + v0 ** 2)

    # --- occlusion geometry: SDF = dist to (wall U clot) ---
    solid = (wall | gt).cpu().numpy()
    tree = cKDTree(pos[solid])
    dist, _ = tree.query(pos)                      # nd distance to nearest solid
    new_sdf = torch.tensor(dist, dtype=torch.float32).clamp_min(1e-6)
    old_sdf = d.x[:, NodeFeat.SDF].reshape(-1)
    width_new = smooth_width_nd_on_edges((2.0 * new_sdf).view(-1, 1), d.edge_index, d.num_nodes).reshape(-1)

    x_occ = d.x.clone()
    x_occ[:, NodeFeat.SDF] = new_sdf.view(-1, 1)
    if d.x.shape[1] > NodeFeat.WIDTH_ND.start:
        x_occ[:, NodeFeat.WIDTH_ND] = width_new.view(-1, 1)
    # rescale velocity prior by mass conservation (~1/R), zero inside clot
    R0 = width_nd_to_radius_nd(2.0 * old_sdf); R1 = width_nd_to_radius_nd(width_new)
    scale = (mass_conserving_umax_nd(R1) / mass_conserving_umax_nd(R0)).clamp(0.2, 5.0)
    x_occ[:, NodeFeat.UV_PRIOR] = d.x[:, NodeFeat.UV_PRIOR] * scale.view(-1, 1)
    solid_t = torch.tensor(solid)
    x_occ[solid_t, NodeFeat.UV_PRIOR] = 0.0

    u1, v1 = run_model(model, d, x_occ)
    sp1 = torch.sqrt(u1 ** 2 + v1 ** 2)
    rel = float(torch.norm(torch.stack([u1 - u0, v1 - v0])) / max(torch.norm(torch.stack([u0, v0])), 1e-9))
    print(f"[geometry occlusion]  relL2(uv1-uv0)={rel:.3f}")
    print(f"  in-clot speed   : frozen={float(sp0[gt].mean()):.4g} -> occluded={float(sp1[gt].mean()):.4g}  "
          f"({'DOWN ok' if sp1[gt].mean() < sp0[gt].mean() else 'UP (no divert)'})")
    print(f"  open-lumen speed: frozen={float(sp0[~wall & ~gt].mean()):.4g} -> occluded={float(sp1[~wall & ~gt].mean()):.4g}")
    print(f"  SDF in-clot     : {float(old_sdf[gt].mean()):.4g} -> {float(new_sdf[gt].mean()):.4g}")

    for name, shear_uv in [("wls", kft.wls_shear_uv), ("wallfunc", kft.wallfunc_shear_uv)]:
        g0 = (shear_uv(d, u0, v0, dev) < lss); g1 = (shear_uv(d, u1, v1, dev) < lss)
        print(f"  [{name:<8}] gateF1 frozen={setf1(g0, gt, wall):.3f} -> occluded={setf1(g1, gt, wall):.3f}")


if __name__ == "__main__":
    main()
