"""Physics-Gated Neural Operator (Level 2) for clot growth dynamics.

Replaces the discrete graph dilation step with a continuous Graph Neural Operator
that learns spatial spread dynamics, strictly gated by physics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GrowthGNO(nn.Module):
    """Predicts continuous clot growth based on seed probability and local features."""

    def __init__(self, in_channels: int, hidden_channels: int = 16, num_layers: int = 3):
        super().__init__()
        # Input is p_seed (1) + spatial/flow features (in_channels)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels + 1, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.final_conv = SAGEConv(hidden_channels, 1)

    def forward(self, data, p_seed: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # Ensure p_seed is [N, 1]
        if p_seed.dim() == 1:
            p_seed = p_seed.unsqueeze(1)
            
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Concatenate seed probability with features
        h = torch.cat([p_seed, x], dim=-1)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            
        raw_growth = self.final_conv(h, edge_index)
        
        # We output a positive growth multiplier/addition. 
        # It gets clamped and multiplied by p_adm in differentiable_ode.py
        return F.softplus(raw_growth).squeeze(-1)
