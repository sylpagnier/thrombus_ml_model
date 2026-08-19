"""Level 1.2: Local Parameter Prediction using a GNN.

This module provides LocalPhysicsGNN, a ParameterProvider that outputs
spatially varying physical parameters based on local geometric and flow features.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.differentiable_wall_model.parameters import ParameterMap, ParameterProvider, GlobalPhysicsParameters


class LocalPhysicsGNN(ParameterProvider):
    """Predicts physical parameters per-node using a Graph Neural Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        initial_params: Optional[Dict[str, float]] = None,
        trainable_keys: tuple[str, ...] = ("da_scale", "wake", "lss", "sgt_cgs", "tau_low", "relax"),
    ):
        super().__init__()
        self.trainable_keys = set(trainable_keys)

        # Graph Neural Network for feature extraction
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        if num_layers > 1:
            self.final_conv = SAGEConv(hidden_channels, 8)
        else:
            self.final_conv = SAGEConv(in_channels, 8)

        # Parameter order matching GlobalPhysicsParameters
        self.param_names = [
            "da_scale", "wake", "lss", "sgt_cgs", 
            "tau_low", "tau_sep", "relax", "phi_temp"
        ]
        
        # Initialize final layer to output near the 1.1 optimal values
        self._initialize_final_layer(initial_params)

    def _initialize_final_layer(self, initial_params: Optional[Dict[str, float]]):
        """Initialize final layer bias to optimal values and weights to near zero."""
        if initial_params is None:
            # Fallback defaults if none provided
            initial_params = {
                "da_scale": 40.0,
                "wake": 8.0,
                "lss": 25.0,
                "sgt_cgs": -750.0,
                "tau_low": 0.25,
                "tau_sep": 0.25,
                "relax": 2.0,
                "phi_temp": 0.05,
            }

        # Calculate inverse softplus for the biases
        biases = []
        for name in self.param_names:
            val = initial_params.get(name, 1.0)
            if name == "sgt_cgs":
                val = abs(val) # Parameterized as negative of softplus
            inv_val = GlobalPhysicsParameters._inv_softplus(val)
            biases.append(inv_val)

        # Apply initialization
        with torch.no_grad():
            self.final_conv.lin_l.weight.data.normal_(0, 1e-4) # Small weights
            self.final_conv.lin_r.weight.data.normal_(0, 1e-4)
            # Biases are added via root_weight (if bias=True which is default, but SAGEConv handles it via lin_l bias usually)
            # Actually torch_geometric SAGEConv has a specific bias parameter
            if hasattr(self.final_conv, "lin_l") and self.final_conv.lin_l.bias is not None:
                self.final_conv.lin_l.bias.data = torch.tensor(biases, dtype=torch.float32)
            else:
                raise ValueError("SAGEConv must have bias=True in lin_l for proper initialization")

    def forward(self, data, num_nodes: int, device: torch.device) -> ParameterMap:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # Replace NaN/Inf in features with 0 (sometimes SDF or curvature might have NaNs in bad meshes)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Output raw parameter values [N, 8]
        raw_params = self.final_conv(x, edge_index)

        # Map to valid ranges exactly like GlobalPhysicsParameters
        # We index according to self.param_names
        
        # da_scale > 0
        da_scale = F.softplus(raw_params[:, 0]) if "da_scale" in self.trainable_keys else F.softplus(raw_params[:, 0]).detach()
        # wake >= 0
        wake = F.softplus(raw_params[:, 1]) if "wake" in self.trainable_keys else F.softplus(raw_params[:, 1]).detach()
        # lss > 0
        lss = F.softplus(raw_params[:, 2]) if "lss" in self.trainable_keys else F.softplus(raw_params[:, 2]).detach()
        # sgt_cgs < 0
        sgt_cgs = -F.softplus(raw_params[:, 3]) if "sgt_cgs" in self.trainable_keys else -F.softplus(raw_params[:, 3]).detach()
        # tau_low > 0
        tau_low = F.softplus(raw_params[:, 4]) if "tau_low" in self.trainable_keys else F.softplus(raw_params[:, 4]).detach()
        # tau_sep > 0
        tau_sep = F.softplus(raw_params[:, 5]) if "tau_sep" in self.trainable_keys else F.softplus(raw_params[:, 5]).detach()
        # relax > 0
        relax = F.softplus(raw_params[:, 6]) if "relax" in self.trainable_keys else F.softplus(raw_params[:, 6]).detach()
        # phi_temp > 0
        phi_temp = F.softplus(raw_params[:, 7]) if "phi_temp" in self.trainable_keys else F.softplus(raw_params[:, 7]).detach()

        return ParameterMap(
            da_scale=da_scale,
            wake=wake,
            lss=lss,
            sgt_cgs=sgt_cgs,
            tau_low=tau_low,
            tau_sep=tau_sep,
            relax=relax,
            phi_temp=phi_temp,
        )
