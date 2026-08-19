import sys
from pathlib import Path
import json
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch.nn as nn
from src.differentiable_wall_model.temporal_models import (
    TemporalDifferentiableWallModel,
    PseudoCGNODE
)
from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.config import PhysicsConfig
from src.core_physics.physics_wall_model import node_positions

def gen_viz_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_vessels = ["patient020", "patient034", "patient015", "patient016", "patient043"]
    N_FRAMES = 13
    MAX_BG_POINTS = 1800
    
    # Build Model with Hyperparameters from train_best_generalization_model.py
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

    # Initialize model on first data piece to get channels
    first_data_path = Path(f"data/processed/graphs_biochem_anchors/{val_vessels[0]}.pt")
    dummy_data = torch.load(first_data_path, map_location="cpu", weights_only=False).to(device)
    in_channels = dummy_data.x.size(1)
    
    model = CustomMeshGraphNetCorrector(
        base_model=temporal_base_model, 
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers
    ).to(device)
    
    ckpt_path = Path("outputs/biochem/best_generalization_model.pth")
    print(f"[i] Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    phys_cfg = PhysicsConfig(phase="biochem")
    out_payload = {}
    
    for anchor in val_vessels:
        data_path = Path(f"data/processed/graphs_biochem_anchors/{anchor}.pt")
        if not data_path.exists():
            print(f"Skipping {anchor}, data not found.")
            continue
            
        data = torch.load(data_path, map_location="cpu", weights_only=False).to(device)
        pos = node_positions(data)
        wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
        n = len(wall)
        interior = np.where(~wall)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]
        
        t_max = int(data.y.shape[0])
        print(f"\n[i] Evaluating {anchor} over {t_max} time steps...")
        
        wall_mask = data.mask_wall.reshape(-1).float().to(device)
        
        # We need to simulate t from 0 to 30000 by 150 increments.
        # But data.y.shape[0] usually is 201
        t_array = np.linspace(0, 30000, t_max)
        
        model_hot = np.zeros((t_max, n), dtype=bool)
        gt_hot = np.zeros((t_max, n), dtype=bool)
        
        with torch.no_grad():
            out = model(data, flow_source="gt", device=device)
            mat_traj = out["mat_traj"]  # Shape: (t_max, num_nodes)
            gate_init = out["gate_init"]
            
            for t_idx in range(t_max):
                mat_t = mat_traj[t_idx]
                mat_crit = getattr(model.base_model, 'mat_crit', 1.0)
                p_seed_t = torch.sigmoid((mat_t / mat_crit - 1.0) / 0.1) * wall_mask
                
                # Apply MeshGraphNetCorrector
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
                
                binary_pred = (final_prob_t >= 0.5).cpu().numpy()
                model_hot[t_idx] = binary_pred & wall
                
                target_t = gt_clot_phi_at_time(data, t_idx, phys_cfg, device=torch.device("cpu")).reshape(-1).numpy()
                gt_hot[t_idx] = (target_t > 0.5) & wall
        
        frame_idx = np.linspace(0, t_max - 1, N_FRAMES).round().astype(int)
        wall_idx = np.where(wall)[0]

        out_payload[anchor] = {
            "t_final": float(t_array[-1]),
            "n_wall": int(wall.sum()),
            "bg": [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in bg],
            "wall_pos": [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in wall_idx],
            "frame_t": [round(float(t_array[i]), 1) for i in frame_idx],
            "frame_gt": [[bool(x) for x in gt_hot[i][wall_idx]] for i in frame_idx],
            "frame_model": [[bool(x) for x in model_hot[i][wall_idx]] for i in frame_idx],
            "count_t": [round(float(x), 1) for x in t_array],
            "count_gt": [int(gt_hot[i].sum()) for i in range(t_max)],
            "count_model": [int(model_hot[i].sum()) for i in range(t_max)],
        }
        print(f"{anchor}: wall={wall.sum()} bg={len(bg)} frames={N_FRAMES} "
              f"gt_final={out_payload[anchor]['count_gt'][-1]} model_final={out_payload[anchor]['count_model'][-1]}")

    out_path = Path("outputs/temporal_viz_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload), encoding="utf-8")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    gen_viz_data()
