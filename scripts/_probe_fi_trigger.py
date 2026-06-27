import glob
import os

import torch

from src.utils.paths import get_project_root

root = get_project_root() / "data/processed/graphs_biochem_anchors"
FI_Y = 4 + 8   # species block idx 8 (FI) -> y channel 12
MAT_Y = 4 + 11  # species block idx 11 (Mat) -> y channel 15
TRIG = 0.6      # viscosity_fi_crit [uM]

hdr = f"{'anchor':12} {'FI_uM_max':>10} {'FI_uM_p99.9':>12} {'>=0.6 frac':>11} {'>=0.6 anynode_t%':>16}"
print(hdr)
for p in sorted(glob.glob(str(root / "patient*.pt"))):
    d = torch.load(p, map_location="cpu", weights_only=False)
    fi = d.y[:, :, FI_Y].clamp(min=-10, max=8).float()
    fi_uM = 7.0 * torch.expm1(fi)            # FI_uM = c_Fg0_uM(=7) * expm1(log1p_nd)
    frac = (fi_uM >= TRIG).float().mean().item()
    # fraction of timesteps where ANY node crosses the trigger
    any_t = (fi_uM >= TRIG).any(dim=1).float().mean().item()
    q = torch.quantile(fi_uM.flatten(), 0.999).item()
    name = os.path.basename(p)[:-3]
    print(f"{name:12} {fi_uM.max().item():10.4f} {q:12.4f} {frac:11.5f} {any_t*100:16.1f}")
