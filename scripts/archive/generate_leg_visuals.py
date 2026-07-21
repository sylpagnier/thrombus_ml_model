import json
import subprocess
from pathlib import Path

legs = ["WC_v7_fresh_canonical", "WC_v7_clot_phi_mse", "WC_v7_high_precision"]
anchors = ["patient007", "patient001", "patient004"]
root = Path("outputs/biochem/biochem_gnn")

# Create temporary manifest for each leg
for leg in legs:
    manifest_data = {
        "species_gnn_ckpt": f"outputs/biochem/biochem_gnn/{leg}/species/best.pth",
        "viscosity_beta": "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth",
        "kinematics_ckpt": "outputs/kinematics/kinematics_best.pth",
        "ckpt_overrides": {},
        "beta_overrides": {}
    }
    
    manifest_path = Path("outputs/biochem/biochem_gnn") / leg / "manifest_temp.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    
    # Run the viz script for each of the 3 anchors
    for anchor in anchors:
        out_png = f"outputs/biochem/viz/species_gnn_deploy/deploy_{leg}_{anchor}_kinematics.png"
        print(f"Generating viz for leg={leg}, anchor={anchor} -> {out_png}")
        
        args = [
            "python", "scripts/viz_species_gnn_deploy.py",
            "--anchor", anchor,
            "--manifest", str(manifest_path),
            "--flow", "kinematics",
            "--out", out_png
        ]
        
        # Execute
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[ERR] Failed: {res.stderr}")
        else:
            print(f"[OK] Saved {out_png}")
