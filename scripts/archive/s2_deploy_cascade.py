"""S2 step 1: deployable cascade reactor -> does produced `ap` track GT `ap`?

The full deployable target is ap/Mas from IC + cascade + kine flow (no oracle). A from-
scratch ADR march is CFL-unstable (advective time d_bar/u_ref ~0.16 s vs frame dt 150 s;
COMSOL/the DEQ solve implicitly). So we approximate bulk transport as a per-node
residence-time reactor: dC/dt = R(C, shear) - k_wash (C - C_in), k_wash = speed/d_bar,
seeded from the resting IC. Washout is treated semi-implicitly (unconditionally stable);
the exact COMSOL reaction kernel (BiochemKinetics.compute_species_reactions) is reused so
no reaction units are re-derived.

This step validates the cascade in isolation (GT flow) by comparing produced ap/th to GT.

Run: python scripts/s2_deploy_cascade.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels  # noqa: E402
from src.core_physics.t0_rung4_ladder import resting_species_log_nd  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
import scripts.s1_kaa_closure_generalization as s1  # noqa: E402

ANCHOR_DIR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
BULK = ["RP", "AP", "APR", "APS", "PT", "T", "AT", "FG", "FI"]   # y ch 4..12
N_SUB = 20


def run_cascade(d, cfg, dev, n_sub=N_SUB, vel="gt"):
    """Integrate the 9 bulk species per-node; return [T,N,9] linear working units."""
    kin = BiochemPhysicsKernels(cfg, None).kinetics
    scales = cfg.get_species_scales(device=dev)[:9]
    t_s = d.t.reshape(-1).float().to(dev)
    T = t_s.numel()
    d_bar = float(d.d_bar.view(-1)[0]); u_ref = float(d.u_ref.view(-1)[0])
    rest = resting_species_log_nd(d, dev)[:, :9]            # log1p nd resting IC
    nd_in = torch.expm1(rest.clamp(-10, 8))                 # nd resting
    sr_all, _ = s1._shear_series(d, dev)                    # [T,N] 1/s (GT flow)
    y = d.y.to(dev)
    speed_nd = torch.sqrt(y[:, :, 0] ** 2 + y[:, :, 1] ** 2)   # nd speed [T,N]
    out = torch.zeros(T, nd_in.shape[0], 9, device=dev)
    nd = nd_in.clone()
    out[0] = nd * scales
    for k in range(1, T):
        dt = float(t_s[k] - t_s[k - 1]) / n_sub
        sr = sr_all[k]
        kwash = (speed_nd[k] * u_ref / d_bar).reshape(-1, 1)    # 1/s
        for _ in range(n_sub):
            lin = nd * scales
            sd = {BULK[i]: lin[:, i] for i in range(9)}
            R = kin.compute_species_reactions(sd, sr)           # working/s
            R_nd = torch.stack([R[BULK[i]] / float(scales[i]) for i in range(9)], dim=1)
            nd = (nd + dt * (R_nd + kwash * nd_in)) / (1.0 + dt * kwash)
            nd = nd.clamp(min=0.0)
        out[k] = nd * scales
    return out


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu")
    scales = cfg.get_species_scales(device=dev)
    anchors = ["patient007", "patient001", "patient010"]
    for a in anchors:
        d = torch.load(ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = d.y.shape[0]; tl = T - 1
        casc = run_cascade(d, cfg, dev)                        # [T,N,9] working units
        # GT linear working units
        gt_ap = torch.expm1(d.y[:, :, 5].clamp(-10, 8)) * float(scales[1])
        gt_rp = torch.expm1(d.y[:, :, 4].clamp(-10, 8)) * float(scales[0])
        gt_th = torch.expm1(d.y[:, :, 9].clamp(-10, 8)) * float(scales[5])
        pr_ap, pr_rp, pr_th = casc[:, :, 1], casc[:, :, 0], casc[:, :, 5]
        clot = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()

        def cmp(name, pr, gt):
            x = pr[tl].numpy(); y = gt[tl].numpy()
            m = np.isfinite(x) & np.isfinite(y)
            corr = np.corrcoef(x[m], y[m])[0, 1] if x[m].std() > 0 and y[m].std() > 0 else float("nan")
            cc = np.corrcoef(x[clot.numpy()], y[clot.numpy()])[0, 1] if int(clot.sum()) > 2 else float("nan")
            rg = float(np.median(x[clot.numpy()]) / (np.median(y[clot.numpy()]) + 1e-30)) if int(clot.sum()) else 0
            print(f"   {name}: corr(all)={corr:.3f} corr(clot)={cc:.3f} "
                  f"pr/gt median(clot)={rg:.2g}  GTmed={np.median(y[clot.numpy()]):.2g} PRmed={np.median(x[clot.numpy()]):.2g}")
        print(f"\n=== {a}  T={T} clot={int(clot.sum())} ===")
        cmp("ap", pr_ap, gt_ap); cmp("rp", pr_rp, gt_rp); cmp("th", pr_th, gt_th)


if __name__ == "__main__":
    main()
