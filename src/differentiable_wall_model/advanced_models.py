"""Advanced Machine Learning models for biochem sweep."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, NNConv, CGConv, TransformerConv

from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.architecture.ginodeq import AttentionGlobalMixingBlock

# =============================================================================
# 1. GATv2ResidualCorrector
# =============================================================================
class GATv2ResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList()
        # +2 for base_prob and base_gate
        self.convs.append(GATv2Conv(in_channels + 2, hidden_channels // 4, heads=4))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels // 4, heads=4))
            
        if num_layers > 1:
            self.final_conv = GATv2Conv(hidden_channels, 1, heads=1)
        else:
            self.final_conv = GATv2Conv(in_channels + 2, 1, heads=1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)

        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        delta_logits = self.final_conv(h, edge_index).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        return base_out


# =============================================================================
# 2. SpatiotemporalResidualCorrector
# =============================================================================
class SpatiotemporalResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # We process mat_traj which has shape [T, N] -> [N, 1, T]. We use a 1D conv to summarize it.
        self.traj_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.traj_linear = nn.Linear(16, 16) # we will pool over time

        self.convs = nn.ModuleList()
        # +2 for base_prob, base_gate, +16 for trajectory summary
        self.convs.append(SAGEConv(in_channels + 2 + 16, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        if num_layers > 1:
            self.final_conv = SAGEConv(hidden_channels, 1)
        else:
            self.final_conv = SAGEConv(in_channels + 2 + 16, 1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]
            mat_traj = base_out["mat_traj"] # [T, N]

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process trajectory
        # mat_traj is [T, N]. We want [N, 1, T] for Conv1D
        traj = mat_traj.t().unsqueeze(1) # [N, 1, T]
        traj_feat = F.relu(self.traj_conv(traj)) # [N, 16, T']
        traj_feat = torch.mean(traj_feat, dim=2) # [N, 16] - global average pooling over time
        traj_feat = F.relu(self.traj_linear(traj_feat))

        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1), traj_feat], dim=-1)

        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        delta_logits = self.final_conv(h, edge_index).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        return base_out


# =============================================================================
# 3. EdgeConditionedResidualCorrector
# =============================================================================
class EdgeConditionedResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        edge_dim = 3 # dx, dy, length
        
        self.convs = nn.ModuleList()
        # NNConv takes an MLP that maps edge_dim -> in_channels * out_channels
        nn1 = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(), nn.Linear(16, (in_channels + 2) * hidden_channels))
        self.convs.append(NNConv(in_channels + 2, hidden_channels, nn1, aggr='mean'))
        
        for _ in range(num_layers - 2):
            nn_k = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(), nn.Linear(16, hidden_channels * hidden_channels))
            self.convs.append(NNConv(hidden_channels, hidden_channels, nn_k, aggr='mean'))
            
        if num_layers > 1:
            nn_final = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(), nn.Linear(16, hidden_channels * 1))
            self.final_conv = NNConv(hidden_channels, 1, nn_final, aggr='mean')
        else:
            nn_final = nn.Sequential(nn.Linear(edge_dim, 16), nn.ReLU(), nn.Linear(16, (in_channels + 2) * 1))
            self.final_conv = NNConv(in_channels + 2, 1, nn_final, aggr='mean')

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)

        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)

        delta_logits = self.final_conv(h, edge_index, edge_attr).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        return base_out


# =============================================================================
# 4. StateInjectedResidualCorrector
# =============================================================================
class StateInjectedResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList()
        # +2 for base_prob, base_gate, +1 for mat_final, +2 for sr, dsrx
        self.convs.append(SAGEConv(in_channels + 5, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        if num_layers > 1:
            self.final_conv = SAGEConv(hidden_channels, 1)
        else:
            self.final_conv = SAGEConv(in_channels + 5, 1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]
            mat_final = base_out["mat_final"]
            sr, dsrx = self.base_model.compute_flow_fields(data, device, flow_source=flow_source)

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        sr_norm = torch.log1p(torch.abs(sr)) * torch.sign(sr)
        dsrx_norm = torch.log1p(torch.abs(dsrx)) * torch.sign(dsrx)

        x_aug = torch.cat([
            x, 
            base_prob.unsqueeze(-1), 
            base_gate.unsqueeze(-1),
            mat_final.unsqueeze(-1),
            sr_norm.unsqueeze(-1),
            dsrx_norm.unsqueeze(-1)
        ], dim=-1)

        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        delta_logits = self.final_conv(h, edge_index).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        return base_out


# =============================================================================
# 5. MultiTaskResidualCorrector
# =============================================================================
class MultiTaskResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels + 2, hidden_channels))
        for _ in range(num_layers - 1): # note -1 here to keep hidden state for multi-head
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.delta_head = nn.Linear(hidden_channels, 1)
        self.wss_head = nn.Linear(hidden_channels, 1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)

        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        delta_logits = self.delta_head(h).squeeze(-1)
        wss_pred = self.wss_head(h).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        base_out["wss_pred"] = wss_pred
        return base_out


# =============================================================================
# 6. MeshGraphNetCorrector
# =============================================================================
class MeshGraphNetCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        edge_dim = 3
        
        self.node_encoder = nn.Sequential(nn.Linear(in_channels + 2, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        
        self.processor_steps = num_layers
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels * 3, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            for _ in range(self.processor_steps)
        ])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            for _ in range(self.processor_steps)
        ])
        
        self.node_decoder = nn.Linear(hidden_channels, 1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
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
            # Edge update
            e_in = torch.cat([e, v[row], v[col]], dim=-1)
            e_out = e + self.edge_mlps[i](e_in)
            e = e_out
            
            # Node update
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


# =============================================================================
# 7. RGPResidualCorrector (GAT + Transformer Hybrid)
# =============================================================================
class RGPResidualCorrector(nn.Module):
    def __init__(self, base_model: DifferentiableWallModel, in_channels: int, hidden_channels: int = 32, num_layers: int = 2):
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # We simulate the RGP-DEQ style: local attention + global perceiver mixing.
        self.encoder = nn.Linear(in_channels + 2, hidden_channels)
        
        self.local_convs = nn.ModuleList()
        self.global_mixers = nn.ModuleList()
        
        for _ in range(num_layers):
            # PyG TransformerConv is an attention layer similar to standard transformer MHA
            self.local_convs.append(TransformerConv(hidden_channels, hidden_channels // 4, heads=4, edge_dim=3))
            self.global_mixers.append(AttentionGlobalMixingBlock(hidden_channels, num_global_tokens=8))
            
        self.decoder = nn.Linear(hidden_channels, 1)

    def forward(self, data, *, flow_source: str = "pred", device: torch.device | None = None) -> dict[str, torch.Tensor]:
        from src.utils.batching import get_batch_tensor
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        batch = get_batch_tensor(data, x.size(0), device)

        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)
        
        h = self.encoder(x_aug)
        
        for conv, mixer in zip(self.local_convs, self.global_mixers):
            h_local = F.relu(conv(h, edge_index, edge_attr))
            h_global = mixer(h, batch)
            h = h + h_local + h_global # residual connection

        delta_logits = self.decoder(h).squeeze(-1)

        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        base_out["prob_clot"] = final_prob
        return base_out
