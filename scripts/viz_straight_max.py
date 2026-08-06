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
        json_path = out_dir / f"patient{i:03d}.json"
        if not json_path.exists():
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # The JSON has boundaries which we can plot
        ax = axes[i]
        
        walls = data.get("wall_segments", [])
        for seg in walls:
            pts = seg.get("points", [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, 'b-', linewidth=2)
                
        inlet = data.get("inlet_segment", {}).get("points", [])
        if inlet:
            ax.plot([p[0] for p in inlet], [p[1] for p in inlet], 'g-', linewidth=2)
            
        outlet = data.get("outlet_segment", {}).get("points", [])
        if outlet:
            ax.plot([p[0] for p in outlet], [p[1] for p in outlet], 'r-', linewidth=2)
            
        # Also plot the pathology descriptor if present
        v_type = data.get("v_type", "")
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
