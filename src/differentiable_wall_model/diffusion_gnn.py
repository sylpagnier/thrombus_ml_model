"""Level 1.4: Deep Chemical Diffusion Growth using an MPNN.

This module provides DeepChemicalDiffusion, which replaces the manual graph-hop
front growth mechanism with a learned message passing network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class DeepChemicalDiffusion(nn.Module):
    """Learned message passing to simulate chemical diffusion and front growth."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        # Input features + p_seed (the initial activation from the physics model)
        self.convs.append(SAGEConv(in_channels + 1, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        if num_layers > 1:
            self.final_conv = SAGEConv(hidden_channels, 1)
        else:
            self.final_conv = SAGEConv(in_channels + 1, 1)

    def forward(
        self,
        data,
        p_seed: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Propagate the initial activation p_seed through the graph."""
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Augment node features with the seed probability
        h = torch.cat([x, p_seed.unsqueeze(-1)], dim=-1)

        # 2. Forward pass through diffusion MPNN
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            
        # 3. Predict final bounded probability
        out = self.final_conv(h, edge_index).squeeze(-1)
        
        # We model this as a residual on top of the seed logits
        p_clamp = torch.clamp(p_seed, min=1e-5, max=1.0 - 1e-5)
        seed_logits = torch.log(p_clamp / (1.0 - p_clamp))
        final_logits = seed_logits + out
        
        return torch.sigmoid(final_logits)
