"""Survey anchor sim maturity to pick a headline-metric cutoff."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
dev = torch.device("cpu"); crit = float(cfg.viscosity_mat_crit); Minf = cfg.Minf
AD = Path("data/processed/graphs_biochem_anchors")
anchors = sorted(p.stem for p in AD.glob("patient*.pt") if "_metadata" not in p.stem)
print(f"{'patient':<11}{'frames':>7}{'t_final':>9}{'clot':>6}{'GTmat/crit(med)':>17}{'>=crit':>8}")
for a in anchors:
    d = torch.load(AD / f"{a}.pt", map_location=dev, weights_only=False)
    tl = d.y.shape[0] - 1
    gt = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()
    matgt = torch.expm1(d.y[:, :, 15].clamp(-10, 8)) * Minf
    med = float((matgt[tl][gt] / crit).median()) if int(gt.sum()) else 0.0
    nge = int((matgt[tl][gt] >= crit).sum())
    print(f"{a:<11}{d.y.shape[0]:>7}{float(d.t.reshape(-1)[-1]):>9.0f}{int(gt.sum()):>6}{med:>17.2f}{nge:>8}")
