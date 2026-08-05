import torch
from src.utils.kinematics_inference import predict_kinematics_and_latent
from src.biochem_gnn.config import apply_deploy_env
from src.inference.corrector_coupling import resolve_kinematics_checkpoint
from src.utils.kinematics_inference import load_kinematics_predictor
import os

def main():
    anchors = [
        "patient005", "patient006", "patient010", 
        "patient023", "patient002", "patient020", "patient034"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = resolve_kinematics_checkpoint()
    print(f"Loading kinematics predictor from {ckpt_path}")
    kine = load_kinematics_predictor(ckpt_path, device)
    kine.eval()
    
    apply_deploy_env()
    
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    
    for anchor in anchors:
        print(f"Precomputing {anchor}...")
        graph_path = f"data/processed/graphs_biochem_anchors/{anchor}.pt"
        data = torch.load(graph_path, weights_only=False).to(device)
        
        from pathlib import Path
        pred, z_kin = predict_kinematics_and_latent(
            kine, 
            data, 
            disk_cache_dir=Path(".cache/kinematics_t0"),
            disk_cache_key=f"{stem}/{anchor}"
        )
        print(f"Done {anchor}. Cache populated.")

if __name__ == "__main__":
    main()
