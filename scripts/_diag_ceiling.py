"""How much of the F1 gap is recall vs label-ceiling? Per-patient breakdown."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu")
J = json.load(open("outputs/reports/comsol_validation/s1b_gate_variants.json"))
pp = J["per_patient"]
crit = float(cfg.viscosity_mat_crit); Minf = cfg.Minf


def f1(p, g):
    tp = float((p & g).sum()); P = tp / max(float(p.sum()), 1); R = tp / max(float(g.sum()), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


print(f"{'patient':<11}{'lrnP':>7}{'lrnR':>7}{'lrnF1':>7}   "
      f"{'lblP':>7}{'lblR':>7}{'lblF1':>7}   {'gt':>5}{'wall%':>7}")
labelF1s = []
for a in sorted(pp):
    d = torch.load(f"data/processed/graphs_biochem_anchors/{a}.pt", map_location=dev, weights_only=False)
    tl = d.y.shape[0] - 1
    gt = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    matgt = torch.expm1(d.y[:, :, 15].clamp(-10, 8)) * Minf
    lP, lR, lF = f1(matgt[tl] >= crit, gt)
    labelF1s.append(lF)
    lv = pp[a].get("learned") or {}
    wpct = float((gt & wall).sum()) / max(float(gt.sum()), 1)
    print(f"{a:<11}{lv.get('precision',0):>7.3f}{lv.get('recall',0):>7.3f}{lv.get('f1',0):>7.3f}   "
          f"{lP:>7.3f}{lR:>7.3f}{lF:>7.3f}   {int(gt.sum()):>5}{wpct:>7.2f}")
import numpy as np
print(f"\nmean label-ceiling F1 (perfect Mat reconstruction): {np.mean(labelF1s):.3f}")
