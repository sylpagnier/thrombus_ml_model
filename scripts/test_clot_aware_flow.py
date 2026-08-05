"""Quantitative and viz test for ClotAwareFlow using SDF wall-ification."""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_device import require_cuda_device
from src.config import PhysicsConfig
from src.inference.corrector_coupling import ClotAwareFlow
from src.utils.paths import data_root

def main():
    device = require_cuda_device()
    
    # 1. Load graph
    graph_path = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n = data.num_nodes
    
    # 2. Init ClotAwareFlow
    phys_cfg = PhysicsConfig(phase="kinematics")
    # Force a low threshold for kine resolve so we trigger it easily
    os.environ["BIOCHEM_KINE_RESOLVE_ON_CLOT"] = "1"
    os.environ["BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES"] = "100"
    
    flow = ClotAwareFlow(device, phys_cfg=phys_cfg)
    
    # 3. Baseline step (no clot)
    print("\n--- STEP 1: Baseline (Frozen) ---")
    mu_eff_base = torch.full((n,), float(phys_cfg.mu_inf), device=device)
    state1 = flow.update(data, mu_eff_base)
    print(f"Mode: {state1.mode}")
    assert state1.mode == "frozen"
    speed1 = torch.sqrt(state1.u**2 + state1.v**2)
    
    # 4. Small clot step (triggers corrector)
    print("\n--- STEP 2: Small Clot (Corrector) ---")
    # Create small clot (50 nodes)
    pos = data.x[:, 0:2].to(device)
    dist_to_center = torch.sqrt((pos[:, 0] - pos[:, 0].mean())**2 + (pos[:, 1] - pos[:, 1].mean())**2)
    small_clot_idx = torch.argsort(dist_to_center)[:50]
    
    mu_eff_small = mu_eff_base.clone()
    mu_eff_small[small_clot_idx] = 2.0  # high viscosity
    state2 = flow.update(data, mu_eff_small)
    print(f"Mode: {state2.mode}")
    assert state2.mode == "corrector"
    
    # 5. Large clot step (triggers macro-resolve)
    print("\n--- STEP 3: Large Clot (Macro-Resolve) ---")
    large_clot_idx = torch.argsort(dist_to_center)[:500] # > 100 threshold
    mu_eff_large = mu_eff_base.clone()
    mu_eff_large[large_clot_idx] = 2.0
    state3 = flow.update(data, mu_eff_large)
    print(f"Mode: {state3.mode}")
    assert state3.mode == "resolved"
    assert state3.z_kin is not None
    speed3 = torch.sqrt(state3.u**2 + state3.v**2)
    
    print(f"\nQuantitative check on macro-resolve:")
    clot_speed = speed3[large_clot_idx]
    print(f"Clot node speed max: {clot_speed.max().item():.6f} (Should be ~0)")
    
    # Viz
    out_dir = Path("outputs/viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = axes[0].scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), c=speed1.cpu(), cmap="viridis", s=1)
    axes[0].set_title("Baseline Speed")
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), c=speed3.cpu(), cmap="viridis", s=1)
    axes[1].scatter(pos[large_clot_idx, 0].cpu(), pos[large_clot_idx, 1].cpu(), color="red", s=2, alpha=0.5, label="Clot")
    axes[1].set_title("Macro-Resolve Speed (SDF Wallified)")
    axes[1].legend()
    plt.colorbar(scatter2, ax=axes[1])
    
    out_path = out_dir / "clot_aware_flow_test.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved visualization to {out_path}")
    
if __name__ == "__main__":
    sys.exit(main())
