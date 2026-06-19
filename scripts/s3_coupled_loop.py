"""S3: wire the flow<->clot COUPLING loop into the validated S-ladder.

S0-S2 ran on FROZEN initial flow and capped at swept-F1 ~0.50 even with exact shear
(scripts/_diag_ml_direction.py). The ceiling decomposition showed the +0.29 lever is the
flow time-evolution: as the clot gels it raises mu_eff and reroutes flow, sharpening the
stagnation. This script closes that loop using only deployable pieces + the already-trained
kine model -- no GT flow, no per-anchor spf.sr export.

Per macro step j:
  1. gate g_low = [shear(current uv) < lss]          (wallfunc/wls shear on the live flow)
  2. integrate the surface deposition ODE over [t_{j-1}, t_j]   (S2 closed loop, autocat)
  3. phi = [Mat >= crit];  mu_eff = log-blend(Carreau_bulk(uv), phi)   (mu1(Mat) gelation)
  4. uv = kine.uv_nd_from_mu_si(mu_eff)               (GINO-DEQ re-solve with the clot plug)
flow is refreshed every --refresh-stride frames (kine solve is the cost); deposition every frame.

Compares FROZEN (no coupling) vs COUPLED swept-F1, vs the p007 exact-shear coupled ceiling 0.77.
Run: python scripts/s3_coupled_loop.py [patient007 ...] [--refresh-stride 20] [--all]
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scipy.spatial import cKDTree  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig, NodeFeat  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.comsol_surface_deposition import DepositionConstants  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si  # noqa: E402
from src.core_physics.clot_phi_simple import log_blend_mu_eff_si  # noqa: E402
from src.core_physics.clot_phi_rollout import KinematicsUvProvider  # noqa: E402
from src.data_gen.lib.graph_velocity_priors import (  # noqa: E402
    smooth_width_nd_on_edges, width_nd_to_radius_nd, mass_conserving_umax_nd)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b  # noqa: E402
import scripts.s1c_soft_eval as s1c  # noqa: E402
import scripts.s2_deploy_forward as s2  # noqa: E402
import scripts.s2_kine_flow_test as kft  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s3_coupled_loop.json"
N_SUB = 4
COMPLETE_FRAMES = 201
P007_EXACT_COUPLED_CEIL = 0.77   # _diag_ml_direction.py (time-varying exact spf.sr gate)


def _initial_uv(model, d, dev):
    with torch.no_grad():
        pred = predict_kinematics(model, d.to(dev))
    return pred[:, 0].detach(), pred[:, 1].detach()


def _occluded_solve(model, d, clot_mask, pos, wall_np, old_sdf, dev):
    """Re-express the clot as a solid wall (geometry occlusion) and re-solve the kine model.

    The kine model ignores clot-scale MU_PRIOR (out-of-distribution, wrong sign) but DOES
    respect channel geometry. So we recompute SDF=dist-to-(wall U clot), shrink WIDTH, rescale
    the velocity prior by mass conservation (~1/R), and zero it inside the clot. (_diag_occlusion_probe)
    """
    solid = wall_np | clot_mask.cpu().numpy()
    dist, _ = cKDTree(pos[solid]).query(pos)
    new_sdf = torch.tensor(dist, dtype=torch.float32, device=dev).clamp_min(1e-6)
    width_new = smooth_width_nd_on_edges((2.0 * new_sdf).view(-1, 1), d.edge_index, d.num_nodes).reshape(-1)
    x = d.x.clone()
    x[:, NodeFeat.SDF] = new_sdf.view(-1, 1)
    if x.shape[1] > NodeFeat.WIDTH_ND.start:
        x[:, NodeFeat.WIDTH_ND] = width_new.view(-1, 1)
    R0 = width_nd_to_radius_nd(2.0 * old_sdf); R1 = width_nd_to_radius_nd(width_new)
    scale = (mass_conserving_umax_nd(R1) / mass_conserving_umax_nd(R0)).clamp(0.2, 5.0)
    x[:, NodeFeat.UV_PRIOR] = d.x[:, NodeFeat.UV_PRIOR] * scale.view(-1, 1)
    x[torch.tensor(solid, device=dev), NodeFeat.UV_PRIOR] = 0.0
    return run_model_x(model, d, x, dev)


def run_model_x(model, d, x, dev):
    b = d.clone(); b.x = x
    with torch.no_grad():
        pred = predict_kinematics(model, b.to(dev))
    return pred[:, 0].detach(), pred[:, 1].detach()


def run(d, cfg, phys, dev, model, provider, *, gate="wallfunc", couple=True, mode="geom",
        refresh_stride=20, n_sub=N_SUB):
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    rp, ap = s2._resting_bulk(d, cfg, dev)
    k = DepositionConstants.si(cfg)
    Da, Minf = float(cfg.surface_damkohler), float(cfg.Minf)
    crit, lss = float(cfg.viscosity_mat_crit), float(cfg.lss)
    t_s, step2t = sp["t_s"], sp["step2t"]
    dep_rate_const = k.k_rs * rp + k.k_as * ap
    shear_uv = kft.wallfunc_shear_uv if gate == "wallfunc" else kft.wls_shear_uv

    N = int(d.num_nodes)
    u, v = _initial_uv(model, d, dev)
    pos = d.x[:, NodeFeat.XY].cpu().numpy()
    wall_np = d.mask_wall.reshape(-1).bool().cpu().numpy()
    old_sdf = d.x[:, NodeFeat.SDF].reshape(-1).to(dev)
    M = torch.zeros(N, device=dev); Mas = torch.zeros(N, device=dev); Mat = torch.zeros(N, device=dev)
    n_solves = 0
    for j in range(1, T):
        gl = (shear_uv(d, u, v, dev) < lss).float()
        dt = float(t_s[j] - t_s[j - 1]) / n_sub; s2t = float(step2t[j])
        for _ in range(n_sub):
            avail = (1.0 - (M + Mas + Mat) / Minf).clamp(0.0, 1.0)
            cdep = avail * gl * dep_rate_const
            cauto = gl * (Mas / Minf) * k.k_aa * ap
            dMas = Da * s2t * cdep; dMat = Da * s2t * (cdep + cauto)
            M = M + dt * dMas; Mas = Mas + dt * dMas; Mat = Mat + dt * dMat
        if couple and (j % refresh_stride == 0 or j == T - 1):
            clot = (Mat >= crit)
            if int(clot.sum()) > 0:
                if mode == "geom":
                    u, v = _occluded_solve(model, d, clot, pos, wall_np, old_sdf, dev)
                else:   # legacy viscosity-injection (dead end, kept for comparison)
                    mu_c = resolve_bulk_carreau_mu_si(d, j, phys, dev, u_nd=u, v_nd=v)
                    mu_eff = log_blend_mu_eff_si(mu_c, clot.float())
                    with torch.no_grad():
                        u, v = provider.uv_nd_from_mu_si(d.to(dev), mu_eff)
                    u, v = u.detach(), v.detach()
                n_solves += 1
    return Mat, n_solves


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = 20
    for a in sys.argv[1:]:
        if a.startswith("--refresh-stride"):
            refresh = int(a.split("=")[-1]) if "=" in a else 20
    run_all = "--all" in sys.argv
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); s1c.cfg = cfg
    crit = float(cfg.viscosity_mat_crit)
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev,
                                      phys_cfg=PhysicsConfig(phase="kinematics"))
    provider = KinematicsUvProvider(dev)

    if run_all or not args:
        anchors = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    else:
        anchors = args
    if not run_all and not args:
        anchors = ["patient007"]   # quick default
    print(f"[i] anchors={anchors}  refresh_stride={refresh}  crit={crit:.3g}  lss={cfg.lss}\n")
    print(f"{'patient':<12}{'frozen':>9}{'coupled':>9}{'delta':>8}{'solves':>8}")
    rep = {"refresh_stride": refresh, "per_patient": {}}
    fr_all, co_all = [], []
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        wall = d.mask_wall.reshape(-1).bool()
        mat_fr, _ = run(d, cfg, phys, dev, model, provider, couple=False, refresh_stride=refresh)
        mat_co, ns = run(d, cfg, phys, dev, model, provider, couple=True, refresh_stride=refresh)
        f_fr = s1c._scores(d, mat_fr, gt, wall, crit)["swept_best_f1"]
        f_co = s1c._scores(d, mat_co, gt, wall, crit)["swept_best_f1"]
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep["per_patient"][a] = {"cohort": cohort, "frozen": f_fr, "coupled": f_co, "solves": ns}
        if cohort == "complete":
            fr_all.append(f_fr); co_all.append(f_co)
        print(f"{a:<12}{f_fr:>9.3f}{f_co:>9.3f}{f_co - f_fr:>+8.3f}{ns:>8d}")

    if fr_all:
        rep["complete_mean"] = {"frozen": float(np.mean(fr_all)), "coupled": float(np.mean(co_all))}
        print(f"\n[complete-cohort mean] frozen={np.mean(fr_all):.3f}  coupled={np.mean(co_all):.3f}  "
              f"delta={np.mean(co_all) - np.mean(fr_all):+.3f}")
    print(f"[i] reference: p007 exact-shear coupled ceiling ~{P007_EXACT_COUPLED_CEIL}; "
          f"S1c oracle-Mas headline 0.863")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
