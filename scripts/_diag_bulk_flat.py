"""Are GT bulk ap/rp ~ resting IC everywhere & all time? If so, deployable ap=rest."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_rung4_ladder import resting_species_log_nd
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
scales = cfg.get_species_scales(device=dev)
CH = {"RP": 4, "AP": 5, "APR": 6, "APS": 7, "PT": 8, "T": 9, "AT": 10, "FG": 11, "FI": 12}
SC = {"RP": 0, "AP": 1, "APR": 2, "APS": 3, "PT": 4, "T": 5, "AT": 6, "FG": 7, "FI": 8}
for a in ["patient007", "patient001", "patient010", "patient005", "patient006"]:
    d = torch.load(f"data/processed/graphs_biochem_anchors/{a}.pt", map_location=dev, weights_only=False)
    rest = torch.expm1(resting_species_log_nd(d, dev).clamp(-10, 8))  # nd
    y = d.y; tl = y.shape[0] - 1
    clot = gt_clot_phi_at_time(d, tl, phys, device=dev).reshape(-1).bool()
    print(f"\n=== {a} clot={int(clot.sum())} ===")
    for k in ["RP", "AP", "T", "APR", "APS"]:
        gt = torch.expm1(y[:, :, CH[k]].clamp(-10, 8))      # nd, [T,N]
        r = float(rest[:, SC[k]].median())
        # how far does GT move from resting? ratio of final-frame clot to resting
        fin = gt[tl]
        rc = float(fin[clot].median()) / (r + 1e-30) if int(clot.sum()) else 0
        rall = float(gt[tl].median()) / (r + 1e-30)
        # max relative excursion across all time/nodes
        mx = float(gt.max()) / (r + 1e-30) if r > 0 else float(gt.max())
        print(f"   {k}: rest={r:.3g}  finalclot/rest={rc:.3f}  finalall/rest={rall:.3f}  max/rest={mx:.2f}")
