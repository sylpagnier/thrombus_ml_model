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
from torch_geometric.nn import GATv2Conv, SAGEConv


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


class LocalKinematicCorrector(nn.Module):
    """Local k-hop kinematic corrector for velocity diversion around micro-clots.

    Predicts a per-node residual ``[dU, dV]`` added to the frozen GINO-DEQ base
    flow on the subgraph extracted around nucleating clot nodes. A GATv2 stack is
    used so attention can learn the anisotropic diversion (flow reroutes over and
    around a clot far more than it reverses behind it).

    Expected input features (``in_channels=6``):
        ``[dx, dy, dist_to_wall, u0, v0, delta_mu]`` where ``dx, dy`` are
        coordinates relative to the clot center of mass.
    """

    def __init__(self, in_channels: int = 6, hidden_dim: int = 64, *, heads: int = 4):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.conv1 = GATv2Conv(self.in_channels, self.hidden_dim, heads=self.heads, concat=False)
        self.conv2 = GATv2Conv(self.hidden_dim, self.hidden_dim, heads=self.heads, concat=False)
        self.conv3 = GATv2Conv(self.hidden_dim, self.hidden_dim, heads=self.heads, concat=False)
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )
        # Start as a near-identity correction so an untrained model leaves the
        # frozen base flow essentially unchanged.
        nn.init.xavier_uniform_(self.readout[-1].weight, gain=0.01)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return kinematic correction ``[N_sub, 2]`` for the subgraph."""
        h = F.gelu(self.conv1(x, edge_index))
        h = F.gelu(self.conv2(h, edge_index))
        h = F.gelu(self.conv3(h, edge_index))
        return self.readout(h)


LOCAL_CORRECTOR_IN_CHANNELS = 6
LOCAL_CORRECTOR_FEATURE_NAMES = ("dx", "dy", "dist_to_wall", "u0", "v0", "delta_mu")


def assemble_local_corrector_features(
    pos_nd: torch.Tensor,
    sdf_nd: torch.Tensor,
    u0_nd: torch.Tensor,
    v0_nd: torch.Tensor,
    delta_mu_nd: torch.Tensor,
    clot_nodes: torch.Tensor,
    subset: torch.Tensor,
) -> torch.Tensor:
    """Build the corrector input ``[dx, dy, dist_to_wall, u0, v0, delta_mu]`` for ``subset``.

    Single source of truth shared by training, the live verification tool, and the
    deploy rollout so the corrector always sees the *same* feature convention.

    All inputs MUST already be non-dimensionalized in the GINO-DEQ convention:
    positions by the geometric length scale (``d_bar`` on patient graphs, channel
    height on synthetic patches), velocity by ``u_ref``, and viscosity by
    ``PhysicsConfig.mu_viscosity_nd_scale``.

    Translation invariance: ``(dx, dy)`` are centered on the clot's center of mass,
    averaged over ``clot_nodes`` ONLY -- never the (possibly boundary-truncated)
    ``subset`` -- so the patch is dynamically re-centered the same way in train and
    deploy.
    """
    pos_nd = pos_nd.reshape(-1, 2)
    com = pos_nd[clot_nodes].mean(dim=0, keepdim=True)
    dx_dy = pos_nd[subset] - com
    feats = torch.cat(
        [
            dx_dy,
            sdf_nd.reshape(-1, 1)[subset],
            u0_nd.reshape(-1, 1)[subset],
            v0_nd.reshape(-1, 1)[subset],
            delta_mu_nd.reshape(-1, 1)[subset],
        ],
        dim=-1,
    )
    return feats.to(torch.float32)


def save_local_corrector(
    path: Path | str, model: LocalKinematicCorrector, meta: dict[str, Any] | None = None
) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": model.in_channels,
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "meta": meta or {},
        },
        p,
    )


_LOCAL_CORRECTOR_CACHE: dict[tuple[str, str], "LocalKinematicCorrector"] = {}


def load_local_corrector(
    path: Path | str, device: torch.device | None = None, *, cache: bool = True
) -> LocalKinematicCorrector:
    """Load the local kinematic corrector; session-caches by ``(path, device)`` when ``cache``."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(path).resolve()
    key = (str(ckpt), str(dev))
    if cache and key in _LOCAL_CORRECTOR_CACHE:
        return _LOCAL_CORRECTOR_CACHE[key]
    pl = torch.load(ckpt, map_location=dev, weights_only=False)
    m = LocalKinematicCorrector(
        in_channels=int(pl.get("in_channels", 6)),
        hidden_dim=int(pl.get("hidden_dim", 64)),
        heads=int(pl.get("heads", 4)),
    )
    m.load_state_dict(pl["model_state"])
    m.to(dev)
    m.eval()
    if cache:
        _LOCAL_CORRECTOR_CACHE[key] = m
    return m


def clear_local_corrector_cache() -> None:
    _LOCAL_CORRECTOR_CACHE.clear()


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
