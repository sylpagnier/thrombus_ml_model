import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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

# Custom Focal Loss
def focal_loss_bce(p_c, t_c, gamma=2.0):
    bce = F.binary_cross_entropy(p_c, t_c, reduction='none')
    p_t = p_c * t_c + (1 - p_c) * (1 - t_c)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()

def run_stage(
    model, 
    data, 
    target, 
    trainable_params, 
    device, 
    stage_name, 
    epochs=100, 
    patience=10,
    lr=1e-4, 
    max_norm=1.0, 
    best_score_threshold=0.0,
    dice_weight=2.0,
    focal_weight=0.0,
    save_path=None
):
    print(f"\n{'='*50}")
    print(f"=== {stage_name} ===")
    print(f"{'='*50}")
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    best_score = best_score_threshold
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        
        out = model(data, flow_source="gt", device=device)
        prob_clot = out["prob_clot"]
        
        mask = data.mask_wall.reshape(-1).float()
        p_c = prob_clot[mask > 0]
        t_c = target[mask > 0].clamp(0, 1)
        
        bce = F.binary_cross_entropy(p_c, t_c)
        
        intersection = (p_c * t_c).sum()
        dice = (2. * intersection + 1e-6) / (p_c.sum() + t_c.sum() + 1e-6)
        dice_loss = 1.0 - dice
        
        fl = focal_loss_bce(p_c, t_c)
        
        loss = bce + (dice_weight * dice_loss) + (focal_weight * fl)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)
        optimizer.step()
        
        t1 = time.time()
        
        res = evaluate_vessel(model, data, flow_source="gt", device=device)
        score = res["deploy_clot_score"]
        
        print(f"  Epoch {epoch+1:02d}/{epochs} - Score: {score:.4f} (BCE: {bce.item():.4f}, Dice: {dice_loss.item():.4f}, Focal: {fl.item():.4f}) ({t1-t0:.2f}s)")
        
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [*] New Best Score for {stage_name}: {best_score:.4f}")
            if save_path:
                torch.save(best_state, save_path)
                print(f"    [*] Saved weights to {save_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"[i] Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs).")
            break

    print(f"[OK] {stage_name} Complete. Best Score: {best_score:.4f}")
    model.load_state_dict(best_state)
    return best_score

def train_best_model():
    # Winning Hyperparameters
    hidden_channels = 32
    num_layers = 3
    dice_weight = 2.7967487855778583
    focal_weight = 0.6058746123069086
    spatial_lr = 0.0006355376701714655
    temporal_lr = 2.5267018503130577e-05
    joint_lr = 7.535708484799129e-06
    
    save_path = Path("outputs/biochem/best_optuna_model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n[i] Retraining best Optuna model to save weights...")
    
    data_path = Path("data/processed/graphs_biochem_anchors/patient020.pt")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    phys_cfg = PhysicsConfig(phase="biochem")
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    target = gt_clot_phi_at_time(data, t_eval, phys_cfg, device=device).reshape(-1)
    target = target * data.mask_wall.reshape(-1).float().to(device)

    temporal_corrector = PseudoCGNODE().to(device)
    temporal_base_model = TemporalDifferentiableWallModel(temporal_corrector=temporal_corrector).to(device)
    
    class CustomMeshGraphNetCorrector(MeshGraphNetCorrector):
        def forward(self, data, *, flow_source="pred", device=None):
            device = device or (data.x.device if hasattr(data, "x") else torch.device("cpu"))
            wall_mask = data.mask_wall.reshape(-1).float().to(device)
            
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]
            
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

            x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)

            v = self.node_encoder(x_aug)
            e = self.edge_encoder(edge_attr)
            
            row, col = edge_index
            for i in range(self.processor_steps):
                e_in = torch.cat([e, v[row], v[col]], dim=-1)
                e_out = e + self.edge_mlps[i](e_in)
                e = e_out
                
                from torch_geometric.utils import scatter
                agg_e = scatter(e, row, dim=0, dim_size=v.size(0), reduce="sum")
                v_in = torch.cat([v, agg_e], dim=-1)
                v_out = v + self.node_mlps[i](v_in)
                v = v_out

            delta_logits = self.node_decoder(v).squeeze(-1)

            p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
            base_logits = torch.log(p_clamp / (1.0 - p_clamp))
            final_logits = base_logits + delta_logits
            final_prob = torch.sigmoid(final_logits) * wall_mask

            base_out["prob_clot"] = final_prob
            return base_out

    in_channels = data.x.size(1)
    model = CustomMeshGraphNetCorrector(
        base_model=temporal_base_model, 
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers
    ).to(device)
    
    nn.init.zeros_(model.node_decoder.weight)
    nn.init.zeros_(model.node_decoder.bias)
    
    spatial_params = []
    for name, param in model.named_parameters():
        if "base_model" not in name and param.requires_grad:
            spatial_params.append(param)
            
    temporal_params = []
    for name, param in model.named_parameters():
        if "temporal_corrector" in name:
            param.requires_grad = True
            temporal_params.append(param)
            
    joint_params = spatial_params + temporal_params

    # Single pass curriculum
    current_best = run_stage(
        model, data, target, spatial_params, device, 
        stage_name="Stage 1: Spatial Pre-Training", 
        epochs=100, patience=10, lr=spatial_lr, max_norm=1.0, 
        best_score_threshold=0.0,
        dice_weight=dice_weight, focal_weight=focal_weight,
        save_path=save_path
    )
    
    current_best = run_stage(
        model, data, target, temporal_params, device, 
        stage_name="Stage 2: Temporal Pre-Training", 
        epochs=100, patience=10, lr=temporal_lr, max_norm=1.0, 
        best_score_threshold=current_best,
        dice_weight=dice_weight, focal_weight=focal_weight,
        save_path=save_path
    )
    
    final_score = run_stage(
        model, data, target, joint_params, device, 
        stage_name="Stage 3: Joint Fine-Tuning", 
        epochs=100, patience=10, lr=joint_lr, max_norm=0.5, 
        best_score_threshold=current_best,
        dice_weight=dice_weight, focal_weight=focal_weight,
        save_path=save_path
    )
    
    print(f"\n[OK] Retraining Complete. Final Score: {final_score:.4f}. Weights saved to {save_path}")

if __name__ == "__main__":
    train_best_model()
