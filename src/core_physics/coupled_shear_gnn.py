"""Coupled wall-shear surrogate GNN: (mesh geometry + clot occlusion + kine-flow prior) -> COMSOL spf.sr.

Why this exists (docs/SPECIES_LEARNING_STRATEGY.md S6.6): the exact COMSOL coupled spf.sr gate hits
~0.75 F1, but the deployable kine surrogate + geometry occlusion only supports ~0.38 read with ANY
local operator (learned readout == wallfunc). The gap is flow fidelity, not the readout: per-node
features lack the global momentum balance. This GNN adds a global mesh receptive field (message
passing over the full graph + a pooled global-context term) to map the kine-flow prior to COMSOL's
coupled spf.sr field, supervised directly on the exported spf.sr, leave-one-anchor-out.

The kine model is used where it helps: it supplies the velocity prior at t=0 and is re-solved via
geometry occlusion once the clot is large enough to reroute flow (handled by the training/eval loop).
This module only holds the nn.Module + (de)serialization; feature construction lives with the loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CoupledShearGNN(nn.Module):
    """Full-mesh GraphSAGE with a pooled global-context term -> per-node log1p(spf.sr).

    Predicts a residual on top of the wallfunc-shear prior feature (passed in ``x`` and re-added at
    the readout), so an untrained / weak model degrades gracefully to the physics baseline.
    """

    def __init__(self, in_dim: int, *, hidden: int = 128, n_layers: int = 6, prior_col: int | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.prior_col = prior_col      # column of the wallfunc-shear prior in x (residual anchor)
        self.convs = nn.ModuleList(
            [SAGEConv(self.in_dim if i == 0 else hidden, hidden) for i in range(self.n_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden + self.in_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        ctx = h.mean(dim=0, keepdim=True).expand(h.shape[0], -1)     # global context
        fused = torch.cat([h, x, ctx], dim=-1)
        out = self.readout(fused).reshape(-1)
        if self.prior_col is not None:
            out = out + x[:, int(self.prior_col)]                    # residual on the wallfunc prior
        return out                                                  # log1p(spf.sr) per node


def save_checkpoint(path: Path | str, model: CoupledShearGNN, meta: dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "in_dim": model.in_dim, "hidden": model.hidden,
                "n_layers": model.n_layers, "prior_col": model.prior_col, "meta": meta}, p)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(path: Path | str, device: torch.device | None = None) -> CoupledShearGNN:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl = torch.load(path, map_location=dev, weights_only=False)
    m = CoupledShearGNN(int(pl["in_dim"]), hidden=int(pl["hidden"]), n_layers=int(pl["n_layers"]),
                        prior_col=pl.get("prior_col"))
    m.load_state_dict(pl["model_state"]); m.to(dev); m.eval()
    return m
