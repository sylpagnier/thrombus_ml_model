import sys
from pathlib import Path
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.differentiable_wall_model.temporal_models import (
    TemporalDifferentiableWallModel,
    PseudoCGNODE
)
from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.config import PhysicsConfig
from src.evaluation.clot_relaxed_metrics import (
    compute_clot_relaxed_metrics,
    metrics_to_deploy_prefix,
    clot_score_from_deploy_dict,
)

def compute_median_score():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_vessels = ["patient020", "patient034", "patient015", "patient016", "patient043"]
    
    hidden_channels = 64
    num_layers = 2
    
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

    data_path = Path(f"data/processed/graphs_biochem_anchors/{val_vessels[0]}.pt")
    dummy_data = torch.load(data_path, map_location="cpu", weights_only=False).to(device)
    in_channels = dummy_data.x.size(1)
    
    model = CustomMeshGraphNetCorrector(
        base_model=temporal_base_model, 
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers
    ).to(device)
    
    ckpt_path = Path("outputs/biochem/best_generalization_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    phys_cfg = PhysicsConfig(phase="biochem")
    all_scores = []
    
    for anchor in val_vessels:
        data_path = Path(f"data/processed/graphs_biochem_anchors/{anchor}.pt")
        if not data_path.exists():
            continue
            
        data = torch.load(data_path, map_location="cpu", weights_only=False).to(device)
        t_max = int(data.y.shape[0])
        wall_mask = data.mask_wall.reshape(-1).float().to(device)
        
        with torch.no_grad():
            out = model(data, flow_source="gt", device=device)
            mat_traj = out["mat_traj"]  # Shape: (t_max, num_nodes)
            gate_init = out["gate_init"]
            
            for t_idx in range(t_max):
                mat_t = mat_traj[t_idx]
                mat_crit = getattr(model.base_model, 'mat_crit', 1.0)
                p_seed_t = torch.sigmoid((mat_t / mat_crit - 1.0) / 0.1) * wall_mask
                
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
                x_aug = torch.cat([x, p_seed_t.unsqueeze(-1), gate_init.unsqueeze(-1)], dim=-1)

                v = model.node_encoder(x_aug)
                e = model.edge_encoder(edge_attr)
                
                row, col = edge_index
                for i in range(model.processor_steps):
                    e_in = torch.cat([e, v[row], v[col]], dim=-1)
                    e_out = e + model.edge_mlps[i](e_in)
                    e = e_out
                    
                    from torch_geometric.utils import scatter
                    agg_e = scatter(e, row, dim=0, dim_size=v.size(0), reduce="sum")
                    v_in = torch.cat([v, agg_e], dim=-1)
                    v_out = v + model.node_mlps[i](v_in)
                    v = v_out

                delta_logits = model.node_decoder(v).squeeze(-1)
                p_clamp = torch.clamp(p_seed_t, min=1e-5, max=1.0 - 1e-5)
                base_logits = torch.log(p_clamp / (1.0 - p_clamp))
                final_logits = base_logits + delta_logits
                final_prob_t = torch.sigmoid(final_logits) * wall_mask
                
                binary_pred = (final_prob_t >= 0.5).float()
                
                target_t = gt_clot_phi_at_time(data, t_idx, phys_cfg, device=device).reshape(-1)
                target_t = target_t * wall_mask
                
                m = compute_clot_relaxed_metrics(
                    binary_pred,
                    target_t,
                    edge_index,
                    wall_mask=wall_mask.bool(),
                )
                d = metrics_to_deploy_prefix(m)
                deploy_score = clot_score_from_deploy_dict(d)
                all_scores.append(deploy_score)
                
    median_score = np.median(all_scores)
    mean_score = np.mean(all_scores)
    print(f"Total time steps evaluated across {len(val_vessels)} vessels: {len(all_scores)}")
    print(f"Median Deploy Clot Score across all times: {median_score:.4f}")
    print(f"Mean Deploy Clot Score across all times: {mean_score:.4f}")

if __name__ == "__main__":
    compute_median_score()
