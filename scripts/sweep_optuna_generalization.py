import optuna
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
    focal_weight=0.0
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
        
        # Forward Pass
        out = model(data, flow_source="gt", device=device)
        prob_clot = out["prob_clot"]
        
        mask = data.mask_wall.reshape(-1).float()
        p_c = prob_clot[mask > 0]
        t_c = target[mask > 0].clamp(0, 1)
        
        # Losses
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
        
        # Fast Eval on Patient020
        res = evaluate_vessel(model, data, flow_source="gt", device=device)
        score = res["deploy_clot_score"]
        
        print(f"  Epoch {epoch+1:02d}/{epochs} - Proxy Score (P020): {score:.4f} (BCE: {bce.item():.4f}, Dice: {dice_loss.item():.4f}, Focal: {fl.item():.4f}) ({t1-t0:.2f}s)")
        
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [*] New Best Proxy Score for {stage_name}: {best_score:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"[i] Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs).")
            break

    print(f"[OK] {stage_name} Complete. Best Proxy Score: {best_score:.4f}")
    # Restore best state
    model.load_state_dict(best_state)
    return best_score

def objective(trial):
    # 1. Hyperparameters (Expanded Search Space)
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
    dice_weight = trial.suggest_float("dice_weight", 0.0, 10.0)
    focal_weight = trial.suggest_float("focal_weight", 0.0, 10.0)
    num_loops = trial.suggest_categorical("num_loops", [1, 2])
    
    spatial_lr = trial.suggest_float("spatial_lr", 1e-5, 1e-3, log=True)
    temporal_lr = trial.suggest_float("temporal_lr", 1e-5, 1e-3, log=True)
    joint_lr = trial.suggest_float("joint_lr", 1e-6, 1e-4, log=True)
    
    print(f"\n\n{'#'*80}")
    print(f"### TRIAL {trial.number}")
    print(f"### Params: {trial.params}")
    print(f"{'#'*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")

    # 2. Proxy Data for Fast Training (patient020)
    data_path = Path("data/processed/graphs_biochem_anchors/patient020.pt")
    data = torch.load(data_path, map_location="cpu", weights_only=False).to(device)
    
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    target = gt_clot_phi_at_time(data, t_eval, phys_cfg, device=device).reshape(-1)
    target = target * data.mask_wall.reshape(-1).float().to(device)

    # 3. Validation Cohort Data for Final Median Reward
    cohort_names = ["patient012", "patient015", "patient016", "patient017", "patient020"]
    cohort_data = []
    for name in cohort_names:
        p = Path(f"data/processed/graphs_biochem_anchors/{name}.pt")
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False).to(device)
            cohort_data.append((name, d))

    # 4. Models
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

    # 5. Iterative Curriculum
    current_best = 0.0
    for loop in range(num_loops):
        current_best = run_stage(
            model, data, target, spatial_params, device, 
            stage_name=f"Loop {loop+1}: Spatial Pre-Training", 
            epochs=100, patience=10, lr=spatial_lr, max_norm=1.0, 
            best_score_threshold=current_best,
            dice_weight=dice_weight, focal_weight=focal_weight
        )
        
        current_best = run_stage(
            model, data, target, temporal_params, device, 
            stage_name=f"Loop {loop+1}: Temporal Pre-Training", 
            epochs=100, patience=10, lr=temporal_lr, max_norm=1.0, 
            best_score_threshold=current_best,
            dice_weight=dice_weight, focal_weight=focal_weight
        )
        
    # 6. Joint Fine-Tuning
    run_stage(
        model, data, target, joint_params, device, 
        stage_name="Stage 3: Joint Fine-Tuning", 
        epochs=100, patience=10, lr=joint_lr, max_norm=0.5, 
        best_score_threshold=current_best,
        dice_weight=dice_weight, focal_weight=focal_weight
    )
    
    # 7. Final Generalization Median Evaluation
    print("\n[i] Performing Final Cohort Evaluation...")
    cohort_scores = []
    model.eval()
    for name, v_data in cohort_data:
        try:
            res = evaluate_vessel(model, v_data, flow_source="gt", device=device)
            cohort_scores.append(res["deploy_clot_score"])
            print(f"  {name}: {res['deploy_clot_score']:.4f}")
        except Exception as e:
            print(f"  {name}: ERROR ({e})")
            
    median_score = float(np.median(cohort_scores))
    print(f"\n[OK] Final Cohort Median Score: {median_score:.4f}")
    
    return median_score

if __name__ == "__main__":
    study_name = "generalization_optimization"
    storage_path = Path("outputs/biochem/optuna_generalization.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=f"sqlite:///{storage_path}", 
        load_if_exists=True
    )
    
    print("\n[i] Starting 2-Hour Generalization Optuna Study...")
    
    # 2-hour timeout limit
    study.optimize(objective, timeout=7200)
    
    print("\n[OK] Optuna Study Complete.")
    print("Best Trial:")
    print(f"  Median Cohort Score: {study.best_trial.value:.4f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
