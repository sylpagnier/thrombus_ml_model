"""Is p011/p008 no-gate failure an undershoot (numerics) or wrong shape (physics)?"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, numpy as np
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu"); crit = float(cfg.viscosity_mat_crit)


def check(a):
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    tl = T - 1
    gt = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()
    ones = torch.ones(T, sp["mat"].shape[1])
    # replicate _integrate but expose internals
    Da, krs, kas = cfg.surface_damkohler, cfg.k_rs, cfg.k_as
    step2t = sp["step2t"]
    dep = Da * ones * (sp["sat"] * (krs * sp["rp"] + kas * sp["ap"])) * step2t
    auto = ones * (sp["mas"] / sp["Minf"]) * sp["ap"] * step2t
    s2 = step2t.expand_as(sp["mat"]); msk = (s2 > 0.5) & torch.isfinite(sp["dmat"])
    A = torch.stack([dep[msk], auto[msk]], 1).cpu().numpy(); b = sp["dmat"][msk].cpu().numpy()
    ok = np.all(np.isfinite(A), 1) & np.isfinite(b)
    coef, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    c_dep, k_aa = float(coef[0]), float(coef[1])
    rate = torch.nan_to_num(c_dep * dep + k_aa * auto)
    t = sp["t_s"]; dt = (t[1:] - t[:-1]).reshape(-1, 1)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    mat_rec = sp["mat"][:1] + torch.cat([torch.zeros(1, rate.shape[1]), torch.cumsum(incr, 0)], 0)
    mr = mat_rec[tl]; mg = sp["mat"][tl]
    cn = gt
    print(f"\n=== {a}  clot={int(cn.sum())} ===")
    print(f" fitted c_dep={c_dep:.3g}  k_aa={k_aa:.3g}")
    print(f" crit={crit:.2g}")
    print(f" GT  Mat on clot: median={float(mg[cn].median()):.3g} max={float(mg[cn].max()):.3g}  >=crit:{int((mg[cn]>=crit).sum())}/{int(cn.sum())}")
    print(f" REC Mat on clot: median={float(mr[cn].median()):.3g} max={float(mr[cn].max()):.3g}  >=crit:{int((mr[cn]>=crit).sum())}/{int(cn.sum())}")
    # shape correlation on clot nodes
    x = mr[cn].numpy(); yv = mg[cn].numpy()
    if x.std() > 0 and yv.std() > 0:
        print(f" corr(REC,GT) on clot nodes: {np.corrcoef(x, yv)[0,1]:.3f}")
    # if we scaled REC to match GT median, how many gel?
    sc = float(mg[cn].median() / (mr[cn].median() + 1e-30))
    print(f" if REC scaled x{sc:.2g} -> gel {int(((mr[cn]*sc)>=crit).sum())}/{int(cn.sum())}")


for a in ("patient007", "patient011", "patient008"):
    check(a)
