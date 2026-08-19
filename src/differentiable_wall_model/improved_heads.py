"""Heads that stay residual: bounded, uncertainty-gated, and rate-multiplicative.

Four changes to the corrector stack, each answering a measured defect.

1. TRAJECTORY SUPERVISION (see ``scripts/sweep_ml_v2.py``).  The old loss read only
   ``gt_clot_phi_at_time(t_eval)``.  The temporal corrector fires at all ~200 Euler steps
   but received gradient from the endpoint alone, so the component whose stated job is the
   rollout was never optimised for it -- and backprop through 200 recurrent steps on a
   single endpoint signal is what produced the DEV trace oscillating 0.87 <-> 0.77.

2. SPLIT OBJECTIVES.  Both stages previously minimised the same final-mask scalar, so the
   temporal head shifted ``mat`` and the spatial head re-corrected the logits downstream of
   it -- they competed.  Here the temporal head owns the trajectory loss and the spatial
   head owns the final mask.

3. BOUNDED RESIDUALS.  ``final_logits = base_logits + delta`` with unbounded ``delta`` lets
   the head overwrite the physics outright.  With the base at 0.9093 on sealed against a
   flow-oracle ceiling of 0.9066, there is almost no legitimate headroom on the mask, so
   the correction is capped: ``delta = c * tanh(raw)``.

4. UNCERTAINTY GATING.  The win/loss pattern was diagnostic -- gains where physics was weak
   (patient031 +0.116, patient010 +0.072), the one large loss where it was strong
   (patient007 -0.114).  Scaling the correction by ``4*p*(1-p)`` lets the head move only
   nodes the physics base is already unsure about, and pins it to zero where the base is
   confident.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.differentiable_wall_model.advanced_models import MeshGraphNetCorrector


class BoundedMeshGraphNet(MeshGraphNetCorrector):
    """MeshGraphNet residual head with a capped, uncertainty-gated correction.

    ``delta_cap`` bounds the logit shift; ``uncertainty_gate`` multiplies it by
    ``4*p*(1-p)``, which is 1 at p=0.5 and 0 at p in {0,1}.  Gradients reach the physics
    base (the parent wraps it in ``no_grad``, which would strand the temporal corrector).
    """

    def __init__(self, *args, delta_cap: float = 2.0, uncertainty_gate: bool = True,
                 x_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_cap = float(delta_cap)
        self.uncertainty_gate = bool(uncertainty_gate)
        self.x_provider = x_provider          # optional: deployable prior channels

    def forward(self, data, *, flow_source="pred", device=None):
        from torch_geometric.utils import scatter

        device = device or torch.device("cpu")
        wall = data.mask_wall.reshape(-1).float().to(device)
        base_out = self.base_model(data, flow_source=flow_source, device=device)
        bp, bg = base_out["prob_clot"], base_out["gate_init"]

        x_raw = self.x_provider(data, flow_source) if self.x_provider is not None else data.x
        x = torch.nan_to_num(x_raw.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ea = torch.nan_to_num(data.edge_attr.to(device), nan=0.0, posinf=0.0, neginf=0.0)

        v = self.node_encoder(torch.cat([x, bp.unsqueeze(-1), bg.unsqueeze(-1)], -1))
        e = self.edge_encoder(ea)
        row, col = data.edge_index.to(device)
        for i in range(self.processor_steps):
            e = e + self.edge_mlps[i](torch.cat([e, v[row], v[col]], -1))
            agg = scatter(e, row, dim=0, dim_size=v.size(0), reduce="sum")
            v = v + self.node_mlps[i](torch.cat([v, agg], -1))
        raw = self.node_decoder(v).squeeze(-1)

        delta = self.delta_cap * torch.tanh(raw)
        if self.uncertainty_gate:
            delta = delta * (4.0 * bp * (1.0 - bp))

        p = torch.clamp(bp, 1e-5, 1 - 1e-5)
        base_out["prob_clot"] = torch.sigmoid(torch.log(p / (1 - p)) + delta) * wall
        base_out["delta_logit"] = delta
        return base_out


class RateMultiplierCorrector(nn.Module):
    """Temporal corrector that scales the deposition RATE instead of patching the state.

    The shipped correctors do ``mat = mat + delta * 0.01``: an additive patch on a state
    variable that grows autocatalytically.  It can drive ``mat`` negative, and -- worse --
    it can create clot where ``gate == 0``, which the law forbids (every deposition term is
    gated, so chemistry cannot ignite an ungated node).

    This applies ``d_mat -> d_mat * exp(clamp(delta, -cap, cap))`` instead.  The hook runs
    after ``mat += h*d_mat``, so the equivalent update is
    ``mat += h*d_mat*(exp(delta) - 1)``.  Positivity is preserved, ``gate == 0`` stays 0,
    and the correction is multiplicative rather than a fixed absolute step -- far better
    conditioned through a 200-step recurrence.
    """

    def __init__(self, hidden_channels: int = 16, cap: float = 1.0, init_scale: float = 1e-3):
        super().__init__()
        self.conv = SAGEConv(4, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        self.cap = float(cap)
        # NOT zero-init. A zero final layer makes ``d(delta)/d(conv output)`` identically
        # zero, so every upstream parameter receives EXACTLY zero gradient on the first
        # step -- measured: conv.lin_l/lin_r grad norms 0.000e+00 while head grad was 4.1.
        # A small random weight keeps the model within ~0.1% of the pure physics at init
        # while letting the conv learn from step one.
        nn.init.normal_(self.head.weight, std=init_scale)
        nn.init.zeros_(self.head.bias)

    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state,
                sr, dsrx, wall_mask):
        crit = 2.0e7
        feats = torch.stack([
            torch.log1p(torch.clamp(mat, min=0.0) / crit),
            torch.log1p(torch.clamp(mas, min=0.0) / crit),
            torch.log1p(torch.clamp(d_mat, min=0.0) * h / crit),
            torch.tanh(sr / 50.0),
        ], dim=-1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        z = F.relu(self.conv(feats, data.edge_index.to(feats.device)))
        delta = torch.tanh(self.head(z).squeeze(-1)) * self.cap
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0) * wall_mask
        mat = mat + h * d_mat * (torch.exp(delta) - 1.0)
        return torch.clamp(mat, min=0.0), mas, hidden_state


def trajectory_probs(mat_traj, crit, phi_temp, wall_mask, idx):
    """Soft clot probability at the sampled timesteps ``idx`` of a ``[T, N]`` trajectory."""
    sel = mat_traj[idx]                                    # [K, N]
    return torch.sigmoid((sel / crit - 1.0) / phi_temp) * wall_mask.unsqueeze(0)
