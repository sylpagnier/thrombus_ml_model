"""Build the coupled-shear GNN dataset: per anchor, sample frames, occlude with the GT clot,
re-solve the kine model, and cache (full-graph features, target log1p(spf.sr)) pairs.

Features [N,11] (PRIOR_COL=0 is the wallfunc-shear residual anchor):
  0 log1p(wallfunc shear)  1 log1p(wls shear)  2 u  3 v  4 log1p(speed)
  5 sdf_occ  6 width_occ  7 x_norm  8 y_norm  9 wall  10 clot
Cache -> data/processed/coupled_shear_ds/<anchor>.pt   Run: python scripts/build_coupled_shear_dataset.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig, NodeFeat
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.utils.kinematics_inference import (
    load_kinematics_predictor, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s2_kine_flow_test as kft
import scripts.s3_corrector_loop as cl
import scripts.spfsr_lib as spfsr

DS = ROOT / "data" / "processed" / "coupled_shear_ds"
N_FRAMES = 8
FEAT_DIM = 11
PRIOR_COL = 0


def shear_features(d, u, v, sdf, width, clot_mask, dev):
    speed = torch.sqrt(u ** 2 + v ** 2)
    wf = torch.log1p(kft.wallfunc_shear_uv(d, u, v, dev).clamp(min=0))
    wls = torch.log1p(kft.wls_shear_uv(d, u, v, dev).clamp(min=0))
    pos = d.x[:, NodeFeat.XY].to(dev)
    posn = (pos - pos.mean(0)) / pos.std(0).clamp(min=1e-6)
    wall = d.mask_wall.reshape(-1).float().to(dev)
    clot = (clot_mask.float().to(dev) if clot_mask is not None else torch.zeros_like(wall))
    return torch.stack([wf, wls, u, v, torch.log1p(speed.clamp(min=0)),
                        sdf.to(dev), width.to(dev), posn[:, 0], posn[:, 1], wall, clot], dim=1)


def build_anchor(a, cfg, phys, dev, model):
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    T = d.y.shape[0]
    pos = d.x[:, NodeFeat.XY].cpu().numpy()
    wall_np = d.mask_wall.reshape(-1).bool().cpu().numpy()
    old_sdf = d.x[:, NodeFeat.SDF].reshape(-1).to(dev)
    sr = spfsr.aligned(d, dev, a)["sr"]
    frames = np.unique(np.linspace(0, T - 1, N_FRAMES).astype(int))
    out = {"edge_index": d.edge_index.cpu(), "frames": []}
    for f in frames:
        clot = (None if f == 0 else
                gt_clot_phi_at_time(d, int(f), phys, device=dev).reshape(-1).bool())
        x, sdf, width = cl._occlude(d, clot, pos, wall_np, old_sdf, dev)
        u, v = cl._predict_uv(model, d, x, dev)
        X = shear_features(d, u, v, sdf, width, clot, dev)
        y = torch.log1p(sr[int(f)].clamp(min=0))
        out["frames"].append({"X": X.cpu(), "y": y.cpu(),
                              "wall": d.mask_wall.reshape(-1).bool().cpu(),
                              "clot": (clot.cpu() if clot is not None else torch.zeros(X.shape[0], dtype=torch.bool)),
                              "frame": int(f)})
    DS.mkdir(parents=True, exist_ok=True)
    torch.save(out, DS / f"{a}.pt")
    print(f"[{a}] frames={list(frames)}  N={X.shape[0]}  -> {DS / f'{a}.pt'}")


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu")
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    anchors = [a for a in sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
               if spfsr.has_cache(a)]
    for a in anchors:
        build_anchor(a, cfg, phys, dev, model)


if __name__ == "__main__":
    main()
