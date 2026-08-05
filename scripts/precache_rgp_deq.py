import argparse
import os
import sys
from pathlib import Path
import glob

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.kinematics_inference import load_kinematics_predictor, predict_kinematics_and_latent
from src.inference.corrector_coupling import resolve_kinematics_checkpoint
from src.biochem_gnn.config import apply_deploy_env

def main():
    parser = argparse.ArgumentParser(description="Precache RGP-DEQ kinematics directly on PyG Data objects.")
    parser.add_argument("--graph-dir", type=str, default="data/processed/graphs_biochem_anchors", help="Directory containing the graph .pt files.")
    
    args = parser.parse_args()
    
    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        print(f"[WARN] Directory {graph_dir} does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = resolve_kinematics_checkpoint()
    print(f"Loading kinematics predictor from {ckpt_path}")
    kine = load_kinematics_predictor(ckpt_path, device)
    kine.eval()
    
    apply_deploy_env()

    pt_files = sorted(glob.glob(str(graph_dir / "*.pt")))
    if not pt_files:
        print(f"[WARN] No .pt files found in {graph_dir}")
        return

    for file_path_str in pt_files:
        file_path = Path(file_path_str)
        anchor = file_path.stem
        
        print(f"[{anchor}] Loading data...")
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        
        # If already cached, we can optionally skip, but let's just recompute to be safe
        # since we want to guarantee u0_pred, v0_pred, and z_kin_pred are all in sync.
        
        data_cuda = data.to(device)
        print(f"[{anchor}] Running RGP-DEQ...")
        with torch.no_grad():
            pred, z_kin = predict_kinematics_and_latent(kine, data_cuda)
            
        # Extract u0, v0
        u0 = pred[:, 0].contiguous()
        v0 = pred[:, 1].contiguous()
        
        # Save back to CPU
        data.u0_pred = u0.unsqueeze(1).cpu()
        data.v0_pred = v0.unsqueeze(1).cpu()
        data.z_kin_pred = z_kin.cpu()
        
        print(f"[{anchor}] Saving attached predictions to {file_path.name}...")
        torch.save(data, file_path)
        print(f"[OK] {anchor} complete.")

if __name__ == '__main__':
    main()
