"""
Sweep script for the 10 zero-touch temporal ML architectures.
Runs a smoke test (forward and backward pass) to verify that gradients
flow through the BPTT and the models don't trivially collapse into NaN.
"""
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from src.differentiable_wall_model.temporal_models import (
    TemporalDifferentiableWallModel,
    get_temporal_model_by_name
)

from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
from src.config import PhysicsConfig

def run_sweep():
    print("[i] Loading patient020.pt for temporal sweep test...")
    data_path = Path("data/processed/graphs_biochem_anchors/patient020.pt")
    if not data_path.exists():
        print(f"[!] Cannot find {data_path}")
        return
        
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # Pre-calculate target for real BCE loss
    phys_cfg = PhysicsConfig(phase="biochem")
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    target = gt_clot_phi_at_time(data, t_eval, phys_cfg, device=device).reshape(-1)
    target = target * data.mask_wall.reshape(-1).float().to(device)
    
    architectures = [
        "PredictorCorrectorGNN",
        "TGCN",
        "UNetODEPeriodic",
        "GraphormerLite",
        "PseudoSDE",
        "FNOProxy",
        "PseudoCGNODE",
        "PseudoCDE",
        "TrajectoryMatchingMPC",
        "ThresholdGNN",
    ]
    
    import numpy as np
    from src.differentiable_wall_model.evaluation import evaluate_vessel
    import json
    results_file = Path("sweep_results_score.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Only test the best 2 architectures + baseline
    architectures = [
        "Baseline",
        "PredictorCorrectorGNN",
        "PseudoCGNODE",
    ]
    print(f"\n[OK] Starting sweep over {len(architectures)} temporal architectures on {device}")
    
    for arch_name in architectures:
        if arch_name in results:
            print(f"\n--- Leg: {arch_name} (Already completed, median score: {results[arch_name]:.4f}) ---")
            continue
            
        print(f"\n--- Leg: {arch_name} ---")
        try:
            if arch_name == "Baseline":
                corrector = None
            else:
                corrector_cls = get_temporal_model_by_name(arch_name)
                corrector = corrector_cls().to(device)
            
            model = TemporalDifferentiableWallModel(temporal_corrector=corrector).to(device)
            
            if corrector is not None:
                optimizer = torch.optim.Adam(corrector.parameters(), lr=1e-4)
            else:
                optimizer = None
                
            scores = []
            
            for epoch in range(25): # increase to 25 to let them stretch their legs
                model.train()
                t0 = time.time()
                out = model(data, flow_source="gt", device=device)
                prob_clot = out["prob_clot"]
                
                # Tie loss function directly to the evaluation metric (F1 Score) using Dice Loss + BCE
                mask = data.mask_wall.reshape(-1).float()
                p_c = prob_clot[mask > 0]
                t_c = target[mask > 0].clamp(0,1)
                
                bce_loss = F.binary_cross_entropy(p_c, t_c)
                
                # Soft Dice Loss (differentiable surrogate for F1)
                intersection = (p_c * t_c).sum()
                dice = (2. * intersection + 1e-6) / (p_c.sum() + t_c.sum() + 1e-6)
                dice_loss = 1.0 - dice
                
                # Combined Loss
                loss = bce_loss + 2.0 * dice_loss
                
                if corrector is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    has_nan = False
                    for p in corrector.parameters():
                        if p.grad is not None and torch.isnan(p.grad).any():
                            has_nan = True
                            break
                            
                    if has_nan or torch.isnan(loss):
                        print(f"[WARN] Epoch {epoch}: NaN encountered. Stopping early.")
                        break
                        
                    optimizer.step()
                
                t1 = time.time()
                
                # Evaluate real score
                res = evaluate_vessel(model, data, flow_source="gt", device=device)
                score = res["deploy_clot_score"]
                scores.append(score)
                
                print(f"  Epoch {epoch+1}/25 - Score: {score:.4f} (BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f}) ({t1-t0:.2f}s)")
                
            if len(scores) > 0:
                median_score = float(np.median(scores))
                results[arch_name] = median_score
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                
        except Exception as e:
            import traceback
            print(f"[ERROR] {arch_name} failed: {e}")
            traceback.print_exc()

    print("\n--- Final Results (Median Clot Score) ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_sweep()
