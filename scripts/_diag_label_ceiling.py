"""Decompose the ~0.85 label ceiling: is it MESH resolution, or the wall+1-hop reconstruction?

For each complete anchor, with PERFECT GT Mat:
  - what fraction of the clot label is wall / within k hops of wall-clot (dilation reachability)
  - swept-best-F1 of the deployable footprint (wall-seed + dilate-{1,2,3})
  - swept-best-F1 of an UNrestricted footprint (gtMat>=thr anywhere, no wall seed)
If relaxing the reconstruction (more hops / no wall seed) -> ~1.0, the ceiling is the RECON method,
not the mesh. If it stays <1, the residual is label-threshold ambiguity / genuine mesh limit.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
crit = float(cfg.viscosity_mat_crit); COMPLETE = 201


def f1(pred, gt):
    tp = float((pred & gt).sum()); p = tp / max(float(pred.sum()), 1); r = tp / max(float(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def swept(seed_fn, gt, ei, hops):
    best = 0.0
    for thr in np.geomspace(0.3, 3.0, 24):
        pp = s1b._dilate(seed_fn(thr), ei, hops) if hops else seed_fn(thr)
        best = max(best, f1(pp, gt))
    return best


def main():
    anchors = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    print(f"{'patient':<11}{'clot':>6}{'%wall':>7}{'%w+1h':>7}{'%w+2h':>7}"
          f"{'wall+1':>8}{'wall+2':>8}{'wall+3':>8}{'free':>8}")
    rows = []
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = d.y.shape[0]
        if T < COMPLETE:
            continue
        sp = s1b._species(d, cfg, dev); ei = d.edge_index
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        wall = d.mask_wall.reshape(-1).bool()
        mat = sp["mat"][-1]
        wall_clot = (mat >= crit) & wall
        cov = {k: float((s1b._dilate(wall_clot, ei, k) & gt).sum()) / max(float(gt.sum()), 1) for k in (0, 1, 2)}
        c1 = swept(lambda t: ((mat / crit >= t) & wall), gt, ei, 1)
        c2 = swept(lambda t: ((mat / crit >= t) & wall), gt, ei, 2)
        c3 = swept(lambda t: ((mat / crit >= t) & wall), gt, ei, 3)
        cf = swept(lambda t: (mat / crit >= t), gt, ei, 0)   # unrestricted, no wall seed
        rows.append((c1, c2, c3, cf))
        print(f"{a:<11}{int(gt.sum()):>6}{cov[0]*100:>7.0f}{cov[1]*100:>7.0f}{cov[2]*100:>7.0f}"
              f"{c1:>8.3f}{c2:>8.3f}{c3:>8.3f}{cf:>8.3f}")
    r = np.array(rows)
    print(f"\n{'MEAN':<11}{'':>6}{'':>7}{'':>7}{'':>7}"
          f"{r[:,0].mean():>8.3f}{r[:,1].mean():>8.3f}{r[:,2].mean():>8.3f}{r[:,3].mean():>8.3f}")
    print("[i] %wall/%w+kh = fraction of the clot LABEL reachable from wall-clot within k hops")
    print("[i] wall+k = label ceiling with k-hop dilation; free = best footprint with NO wall seed")


if __name__ == "__main__":
    main()
