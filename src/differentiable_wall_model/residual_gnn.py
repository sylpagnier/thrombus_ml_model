"""Level 1.3: Residual Physics Corrector using a GNN.

This module provides ResidualPhysicsCorrector, which takes a frozen baseline
physics model (e.g. the 0-param model or the tuned global model), runs it to get
a strong physics prior (base probability and Mat fields), and then uses a
GNN to predict a residual correction delta_prob.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel


class ResidualPhysicsCorrector(nn.Module):
    """Predicts a residual probability correction on top of a frozen physics model."""

    def __init__(
        self,
        base_model: DifferentiableWallModel,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        # Freeze the base model
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # GNN for feature extraction and residual prediction
        # Input to GNN will be the original node features + base_prob + base_gate
        self.convs = nn.ModuleList()
        # +2 for base_prob and base_gate
        self.convs.append(SAGEConv(in_channels + 2, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        if num_layers > 1:
            self.final_conv = SAGEConv(hidden_channels, 1)
        else:
            self.final_conv = SAGEConv(in_channels + 2, 1)

    def forward(
        self,
        data,
        *,
        flow_source: str = "pred",
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        # 1. Run the frozen physics model to get the base predictions
        with torch.no_grad():
            self.base_model.eval()
            base_out = self.base_model(data, flow_source=flow_source, device=device)
            base_prob = base_out["prob_clot"]
            base_gate = base_out["gate_init"]

        # 2. Extract node features
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Concatenate physics priors
        x_aug = torch.cat([x, base_prob.unsqueeze(-1), base_gate.unsqueeze(-1)], dim=-1)

        # 4. GNN Forward pass
        h = x_aug
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        # Predict residual (delta pre-sigmoid)
        delta_logits = self.final_conv(h, edge_index).squeeze(-1)

        # 5. Combine base probability and residual
        # Base prob is in [0, 1]. We convert it to a logit, add delta, and sigmoid back.
        # Avoid log(0) by clamping
        p_clamp = torch.clamp(base_prob, min=1e-5, max=1.0 - 1e-5)
        base_logits = torch.log(p_clamp / (1.0 - p_clamp))
        
        final_logits = base_logits + delta_logits
        final_prob = torch.sigmoid(final_logits) * wall_mask

        # Return dict matching DifferentiableWallModel interface
        return {
            "prob_clot": final_prob,
            "mat_final": base_out["mat_final"],
            "mat_traj": base_out["mat_traj"],
            "gate_init": base_gate,
            "params": base_out["params"],
        }
