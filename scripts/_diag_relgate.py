"""Does a RELATIVE low-shear gate (percentile of wall shear at t0) recover the clot?

S2 used absolute lss=25 (marks ~50% of all wall nodes). The clot-wall vs else-wall shear
contrast is ~10x, so a percentile gate on the *initial* flow should localize far better.
Footprint = wall & (sr[0] < pth), dilate-1, F1 vs GT clot (same scoring as S2).
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


def f1(pred, gt):
    tp = float((pred & gt).sum()); p = tp / max(float(pred.sum()), 1); r = tp / max(float(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


pcts = [2, 5, 10, 15, 20, 30]
print(f"{'patient':<11}" + "".join(f"{'p'+str(p):>8}" for p in pcts) + f"{'best':>8}{'absLss':>8}")
res = []
for a in ["patient007", "patient005", "patient006", "patient010", "patient001"]:
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    T = d.y.shape[0]; wall = d.mask_wall.reshape(-1).bool(); ei = d.edge_index
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    sr0 = s1b._wls_shear(d, dev)[0][0]                    # t0 GT-flow shear
    sw = sr0[wall]
    row = []
    for p in pcts:
        th = float(torch.quantile(sw, p / 100.0))
        pred = s1b._dilate((sr0 < th) & wall, ei, 1)
        row.append(f1(pred, gt)[0])
    abs_pred = s1b._dilate((sr0 < float(cfg.lss)) & wall, ei, 1)
    abs_f1 = f1(abs_pred, gt)[0]
    best = max(row); res.append(best)
    print(f"{a:<11}" + "".join(f"{v:>8.3f}" for v in row) + f"{best:>8.3f}{abs_f1:>8.3f}")
print(f"\n[i] relative-gate best mean F1 = {np.mean(res):.3f}  (S2 carreau/combo ~0.45-0.47)")
