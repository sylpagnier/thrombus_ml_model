import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path

from src.differentiable_wall_model.temporal_models import (
    TemporalDifferentiableWallModel,
    PseudoCGNODE
)
from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
from src.config import PhysicsConfig
from src.differentiable_wall_model.evaluation import evaluate_vessel

def train_combined():
    print("[i] Loading patient020.pt for combined architecture training...")
    data_path = Path("data/processed/graphs_biochem_anchors/patient020.pt")
    if not data_path.exists():
        print(f"[!] Cannot find {data_path}")
        return
        
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    # Pre-calculate target for real BCE + Dice loss
    phys_cfg = PhysicsConfig(phase="biochem")
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    target = gt_clot_phi_at_time(data, t_eval, phys_cfg, device=device).reshape(-1)
    target = target * data.mask_wall.reshape(-1).float().to(device)

    print("\n[OK] Assembling combined architecture (MeshGraphNet + PseudoCGNODE) on", device)
    
    # 1. Temporal Corrector
    temporal_corrector = PseudoCGNODE().to(device)
    
    # 2. Base Physics Wrapped with Temporal Corrector
    temporal_base_model = TemporalDifferentiableWallModel(temporal_corrector=temporal_corrector).to(device)
    
    # 3. Outer Spatial Corrector wrapping the Temporal Base Model
    # MeshGraphNetCorrector extracts in_channels from data.x (usually 14 for flow + pos features)
    in_channels = data.x.size(1)
    model = MeshGraphNetCorrector(
        base_model=temporal_base_model, 
        in_channels=in_channels, 
        hidden_channels=32, 
        num_layers=2
    ).to(device)
    
    # Zero-init the spatial head so we start identically to the baseline (0.7472) instead of noise!
    nn.init.zeros_(model.node_decoder.weight)
    nn.init.zeros_(model.node_decoder.bias)
    
    # Unified Optimizer for both correctors
    # The base physics parameters are frozen inside MeshGraphNetCorrector's init.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    
    epochs = 40
    best_score = 0.0
    best_ckpt_path = Path("outputs/biochem/best_combined.pth")
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n[i] Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        
        # Forward Pass
        out = model(data, flow_source="gt", device=device)
        prob_clot = out["prob_clot"]
        
        # Tie loss directly to F1 evaluation metric
        mask = data.mask_wall.reshape(-1).float()
        p_c = prob_clot[mask > 0]
        t_c = target[mask > 0].clamp(0, 1)
        
        bce_loss = F.binary_cross_entropy(p_c, t_c)
        
        # Soft Dice Loss
        intersection = (p_c * t_c).sum()
        dice = (2. * intersection + 1e-6) / (p_c.sum() + t_c.sum() + 1e-6)
        dice_loss = 1.0 - dice
        
        loss = bce_loss + 2.0 * dice_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Detect NaNs before stepping
        has_nan = False
        for p in trainable_params:
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan = True
                break
                
        if has_nan or torch.isnan(loss):
            print(f"[WARN] Epoch {epoch}: NaN encountered in gradients. Skipping step or stopping.")
            break
            
        # Gradient Clipping (Prevents the collapse seen in isolated temporal sweep)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        optimizer.step()
        
        t1 = time.time()
        
        # Evaluate 
        res = evaluate_vessel(model, data, flow_source="gt", device=device)
        score = res["deploy_clot_score"]
        
        print(f"  Epoch {epoch+1:02d}/{epochs} - Score: {score:.4f} (BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f}) ({t1-t0:.2f}s)")
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    [*] New Best Score! Saved to {best_ckpt_path}")

    print(f"\n[OK] Training Complete. Best Score: {best_score:.4f}")

if __name__ == "__main__":
    train_combined()
