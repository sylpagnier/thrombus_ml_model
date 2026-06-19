"""Is S2's deployable shortfall magnitude (calibration) or footprint (gate)?

Wide threshold sweep + magnitude + gate coverage on complete patients, carreau gate.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b
import scripts.s2_deploy_forward as s2

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
crit = float(cfg.viscosity_mat_crit)


def f1(pred, gt):
    tp = float((pred & gt).sum()); p = tp / max(float(pred.sum()), 1); r = tp / max(float(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


for a in ["patient007", "patient006", "patient010", "patient005", "patient001"]:
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    rp, ap = s2._resting_bulk(d, cfg, dev)
    gates = s2._gate_sources(d, dev, cfg)
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    ei = d.edge_index
    for s in ["carreau", "wls"]:
        gl = gates[s]
        mat = s2._integrate_closed_loop(d, cfg, dev, gl, ap, rp, sp["step2t"], sp["t_s"])
        gtmat = sp["mat"][-1]                       # oracle GT Mat final
        # wide sweep on absolute Mat threshold
        best, bthr = 0.0, 0.0
        for thr in np.geomspace(mat.max().item() * 1e-4 + 1e-30, mat.max().item() + 1e-30, 40):
            pp = s1b._dilate((mat >= thr) & wall, ei, 1)
            fv = f1(pp, gt)
            if fv > best: best, bthr = fv, thr
        gl_clot = float(gl[-1][gt].mean())          # frac clot nodes with gate on (final)
        gl_wall = float(gl[-1][wall].mean())
        print(f"{a:<11}{s:<8} matMedClot={float(mat[gt].median()):.2e} crit={crit:.1e} "
              f"GTmatMedClot={float(gtmat[gt].median()):.2e} | wideSweptF1={best:.3f}@{bthr:.1e} "
              f"gate_on(clot)={gl_clot:.2f} gate_on(wall)={gl_wall:.2f}")
