"""With oracle Mas, is the shear gate net-harmful? Compare gate vs no-gate (gate=1)."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, numpy as np
from src.config import BiochemConfig, PhysicsConfig
import scripts.s1b_gate_variants as s1b

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu")
ANCH = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
print(f"{'patient':<11}{'WLS gate':>10}{'no-gate':>10}{'clot':>7}")
ng, wl = [], []
for a in ANCH:
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    sr = s1b._wls_shear(d, dev)[0]
    mf, _ = s1b._integrate(sp, s1b._gate(sr, cfg), cfg); rw = s1b._score(d, mf, cfg, phys, dev)
    ones = torch.ones(T, sp["mat"].shape[1])
    mf2, _ = s1b._integrate(sp, ones, cfg); rn = s1b._score(d, mf2, cfg, phys, dev)
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    nclot = int(gt_clot_phi_at_time(d, T - 1, phys, device=dev).sum())
    wl.append(rw["f1"]); ng.append(rn["f1"])
    print(f"{a:<11}{rw['f1']:>10.3f}{rn['f1']:>10.3f}{nclot:>7}   "
          f"(no-gate P={rn['precision']:.2f} R={rn['recall']:.2f})")
print(f"\n{'MEAN':<11}{np.mean(wl):>10.3f}{np.mean(ng):>10.3f}")
