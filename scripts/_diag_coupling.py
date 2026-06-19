"""Does the clot CARVE its own low-shear region? (flow<->clot two-way coupling test)

If GT clot nodes are fast/high-shear at t0 but slow/low-shear at t_final, then the
stagnation that gates deposition is *created by* the growing clot (coupling), and a
static/initial-flow gate (S2) is fundamentally limited. Uses GT flow only.
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
lss = float(cfg.lss)
# Among WALL nodes only (all ~0 speed via no-slip): is clot-wall lower-shear than non-clot
# wall, and does that contrast already exist at t0 (geometry) or only at t_final (coupling)?
print(f"{'patient':<11}{'srMed_clotW_t0':>15}{'srMed_elseW_t0':>15}{'srMed_clotW_tF':>15}"
      f"{'low%clotW_t0':>14}{'low%elseW_t0':>14}{'low%clotW_tF':>14}")
for a in ["patient007", "patient005", "patient006", "patient010", "patient001"]:
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    y = d.y.to(dev); T = y.shape[0]
    wall = d.mask_wall.reshape(-1).bool()
    clot = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    cw = clot & wall; ew = (~clot) & wall                # clot-wall, else-wall
    sr = s1b._wls_shear(d, dev)[0]                        # [T,N] GT-flow shear 1/s
    def med(m, t): return float(sr[t][m].median())
    def low(m, t): return float((sr[t][m] < lss).float().mean())
    print(f"{a:<11}{med(cw,0):>15.1f}{med(ew,0):>15.1f}{med(cw,-1):>15.1f}"
          f"{low(cw,0):>14.2f}{low(ew,0):>14.2f}{low(cw,-1):>14.2f}")
