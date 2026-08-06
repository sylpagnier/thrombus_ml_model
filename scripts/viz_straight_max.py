import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data_gen.lib.vessel_generator import VesselGenerator
import json

def plot_vessels():
    # Use a temp directory
    out_dir = REPO / "outputs/temp_vessels"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vg = VesselGenerator(phase="kinematics", output_dir=out_dir)
    # Generate 3 straight_max vessels
    vg.run_pipeline(n=3, level=0, pathology_mode="straight_max", start_idx=0)
    
    # Plot them
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        json_path = out_dir / f"vessel_{i}.json"
        if not json_path.exists():
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        ax = axes[i]
        
        top_pts = data.get("top_wall_pts", [])
        if top_pts:
            xs = [p[0] for p in top_pts]
            ys = [p[1] for p in top_pts]
            ax.plot(xs, ys, 'b-', linewidth=2)
            
        bot_pts = data.get("bot_wall_pts", [])
        if bot_pts:
            xs = [p[0] for p in bot_pts]
            ys = [p[1] for p in bot_pts]
            ax.plot(xs, ys, 'b-', linewidth=2)
            
        # Draw inlet (first points of top and bot)
        if top_pts and bot_pts:
            ax.plot([top_pts[0][0], bot_pts[0][0]], [top_pts[0][1], bot_pts[0][1]], 'g-', linewidth=2)
            # Draw outlet (last points)
            ax.plot([top_pts[-1][0], bot_pts[-1][0]], [top_pts[-1][1], bot_pts[-1][1]], 'r-', linewidth=2)
            
        # Also plot the pathology descriptor if present
        v_type = data.get("type", "")
        max_sten = data.get("max_stenosis", "")
        max_aneu = data.get("max_aneurysm", "")
        
        title = f"patient{i:03d}\n{v_type}"
        ax.set_title(title)
        ax.axis('equal')
        
    out_png = REPO / "outputs/temp_vessels/straight_max_preview.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Saved preview to {out_png}")

if __name__ == "__main__":
    plot_vessels()
