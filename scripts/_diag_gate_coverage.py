"""On GT-clot nodes, is the low-shear gate ON and is autocat reachable? p007 vs p011 vs p008."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, numpy as np
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b
import scripts.s1_kaa_closure_generalization as s1

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu"); Minf = cfg.Minf; lss = float(cfg.lss)


def analyze(a):
    d = torch.load(f"data/processed/graphs_biochem_anchors/{a}.pt", map_location=dev, weights_only=False)
    tl = d.y.shape[0] - 1
    gt = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    cnode = gt  # all clot nodes (wall% ~1 for these)
    sr_wls = s1b._wls_shear(d, dev)[0]
    sr_car = s1b._carreau_shear(d, dev)
    sr_wf = s1b._wallfunc_shear(d, dev)
    sp = s1b._species(d, cfg, dev)
    mas = sp["mas"]; ap = sp["ap"]
    # gate ever-on over time on clot nodes
    def ever_on(sr):
        g = (sr < lss)
        if sr.dim() == 1:
            return float(g[cnode].float().mean())
        return float(g.any(0)[cnode].float().mean())
    masN = (mas[tl] > mas[tl].max() * 0.01)   # meaningful Mas present
    apN = (ap[tl] > ap[tl].max() * 0.01)
    # median shear on clot vs non-clot (WLS final, wallfunc final)
    def med(sr):
        s = sr[tl] if sr.dim() > 1 else sr
        return float(s[cnode].median()), float(s[~cnode & wall].median())
    print(f"\n=== {a}  clot={int(cnode.sum())} nodes (t_final={float(d.t.reshape(-1)[-1]):.0f}s) ===")
    print(f" WLS low-shear gate ON on clot nodes : {ever_on(sr_wls):.2%}")
    print(f" carreau gate ON on clot nodes       : {ever_on(sr_car):.2%}")
    print(f" wallfunc gate ON on clot nodes      : {ever_on(sr_wf):.2%}")
    print(f" Mas present on clot nodes (autocat) : {float(masN[cnode].float().mean()):.2%}")
    print(f" ap  present on clot nodes           : {float(apN[cnode].float().mean()):.2%}")
    mc, mnc = med(sr_wls); print(f" median WLS shear  clot={mc:.1f}  non-clot-wall={mnc:.1f}  (lss={lss})")
    mc, mnc = med(sr_wf);  print(f" median wallf shear clot={mc:.1f}  non-clot-wall={mnc:.1f}")
    mc, mnc = med(sr_car); print(f" median carreau shr clot={mc:.1f}  non-clot-wall={mnc:.1f}")


for a in ("patient007", "patient011", "patient008"):
    analyze(a)
