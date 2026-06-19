"""Does the EXACT COMSOL gate (physics, not learned) reproduce the clot once fed
accurate spf.sr? p007 closed-loop: WLS shear vs exact spf.sr, low-shear vs +separation.

Answers: is the gap a shear-COMPUTATION problem (physics gate is fine, WLS is bad) or a
gate-FORMULATION problem (the physics gate itself is insufficient)?
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b
import scripts.s1c_soft_eval as s1c
import scripts.s2_deploy_forward as s2

cfg = BiochemConfig(phase="biochem"); s1c.cfg = cfg
phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
crit = float(cfg.viscosity_mat_crit); lss = float(cfg.lss); sgt = float(cfg.sgt)
L, gm = cfg.L_char, cfg.gamma_m

d = torch.load(s1b.ANCHOR_DIR / "patient007.pt", map_location=dev, weights_only=False)
sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
rp, ap = s2._resting_bulk(d, cfg, dev)
gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
wall = d.mask_wall.reshape(-1).bool()

# shear sources
sr_wls, dsr_wls = s1b._wls_shear(d, dev)                  # [T,N]
sr_exact = s1b._exact_shear_p007(d, dev)                  # [T,N] exact COMSOL spf.sr
# exact dshear/ds along x (approx) via graph grad of exact shear
Gx = d.G_x.to(dev); d_bar = float(d.d_bar.view(-1)[0])
dsr_exact = torch.stack([torch.sparse.mm(Gx, sr_exact[t].reshape(-1, 1)).squeeze(1) / d_bar
                         for t in range(T)], 0)


def gate_low(sr):
    return (sr < lss).float()


def gate_full(sr, dsr):
    g_low = (sr < lss).float()
    g_sep = (dsr < sgt).float() * (L / gm) * dsr.abs()
    return torch.clamp(g_low + g_sep, 0.0, 1.0)


def run(g):
    mat = s2._integrate_closed_loop(d, cfg, dev, g, ap, rp, sp["step2t"], sp["t_s"])
    return s1c._scores(d, mat, gt, wall, crit)


for name, g in [("wls  low",   gate_low(sr_wls)),
                ("exact low",  gate_low(sr_exact)),
                ("wls  full",  gate_full(sr_wls, dsr_wls)),
                ("exact full", gate_full(sr_exact, dsr_exact))]:
    sc = run(g)
    print(f"{name:<12} hardF1={sc['hard_f1']:.3f}  sweptF1={sc['swept_best_f1']:.3f}  "
          f"softDice={sc['soft_dice']:.3f}")
print(f"\n[i] gate_on(clot) wls={float(gate_low(sr_wls)[-1][gt].mean()):.2f} "
      f"exact={float(gate_low(sr_exact)[-1][gt].mean()):.2f}  (frac clot nodes the gate fires on)")
