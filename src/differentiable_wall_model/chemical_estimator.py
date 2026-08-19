"""Deep Chemical State Estimator (Level 3).

Predicts fractional depletion of Activated (AP) and Resting (RP) platelets
based on the current clot mass and flow state at the wall.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ChemicalStateEstimator(nn.Module):
    """Predicts dynamic AP/RP depletion fractions over time."""

    def __init__(self, in_channels: int, hidden_channels: int = 16, num_layers: int = 2):
        super().__init__()
        # Input: mat (1) + sr_eff (1) + dsrx_eff (1) + data.x (in_channels)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels + 2, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.final_conv = SAGEConv(hidden_channels, 2) # AP and RP depletion

    def forward(
        self, 
        mat: torch.Tensor, 
        sr_eff: torch.Tensor, 
        data, 
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # Ensure dimensions match
        if mat.dim() == 1: mat = mat.unsqueeze(1)
        if sr_eff.dim() == 1: sr_eff = sr_eff.unsqueeze(1)
        
        # Approximate dsrx_eff simply or pass it directly (we will just use sr_eff and an empty dsrx for simplicity 
        # or we could require dsrx_eff to be passed. Let's just use mat and sr_eff and pad 1)
        # We need a dummy tensor if we specified 3 extra channels. Let's adjust to 2 extra channels (mat, sr_eff).
        # Actually, let's just require dsrx_eff to be passed in, or just pass mat, sr_eff. 
        # Wait, differentiable_ode passes: ap_frac, rp_frac = self.chem_estimator(mat, sr_eff, data, device)
        # So we only have mat and sr_eff. That's 2 extra channels.
        pass

        h = torch.cat([mat, sr_eff, x], dim=-1)
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            
        out = self.final_conv(h, edge_index)
        
        # We want to output fractional depletion [0, 1)
        ap_frac = torch.sigmoid(out[:, 0])
        rp_frac = torch.sigmoid(out[:, 1])
        
        return ap_frac, rp_frac
