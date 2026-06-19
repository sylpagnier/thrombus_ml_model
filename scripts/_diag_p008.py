"""Is p008 an incomplete-sim data artifact? Compare growth curves & frame freezing."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu"); crit = float(cfg.viscosity_mat_crit); Minf = cfg.Minf


def analyze(a):
    d = torch.load(f"data/processed/graphs_biochem_anchors/{a}.pt", map_location=dev, weights_only=False)
    y = d.y; T = y.shape[0]; t = d.t.reshape(-1)
    # frame-to-frame change of whole y (detect freezing / padding)
    dchg = (y[1:] - y[:-1]).abs().flatten(1).mean(1)
    matgt = torch.expm1(y[:, :, 15].clamp(-10, 8)) * Minf
    gel = (matgt >= crit).sum(1)
    clot = torch.tensor([int(gt_clot_phi_at_time(d, k, phys, device=dev).sum()) for k in range(T)])
    # last frame where y actually changes
    moving = (dchg > 1e-6).nonzero().reshape(-1)
    last_move = int(moving[-1]) + 1 if len(moving) else 0
    print(f"\n=== {a}  (T={T}, t_final={float(t[-1]):.0f}s) ===")
    print(f" last frame with y-change: {last_move} (t={float(t[last_move]):.0f}s)")
    print(f" gelled nodes:  t0={int(gel[0])}  mid={int(gel[T//2])}  final={int(gel[-1])}  max={int(gel.max())}")
    print(f" GT clot nodes: t0={int(clot[0])}  mid={int(clot[T//2])}  final={int(clot[-1])}  max={int(clot.max())}")
    # show growth at a few frames
    idxs = [0, 20, 40, 47, 60, 100, 200]
    idxs = [i for i in idxs if i < T]
    print("  frame:  " + "".join(f"{i:>6}" for i in idxs))
    print("  t(s) :  " + "".join(f"{float(t[i]):>6.0f}" for i in idxs))
    print("  gel  :  " + "".join(f"{int(gel[i]):>6}" for i in idxs))
    print("  clot :  " + "".join(f"{int(clot[i]):>6}" for i in idxs))
    return last_move, T


for a in ("patient008", "patient007", "patient011"):
    analyze(a)
