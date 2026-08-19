"""Physics-backbone onset models and the differentiable form of ``growth_l1``.

THE POINT OF THIS MODULE.  ``growth_l1 = mean_t |n_pred(t) - n_gt(t)| / N_gt`` depends only
on the COUNT curve, so replacing the hard step "node i has committed by t" with a sigmoid
makes the metric itself differentiable:

    n_soft(t) = sum_i sigmoid((t - onset_i) / tau)

There is no thresholded readout and no recurrence anywhere in the graph.  That matters
because all three earlier ML attempts died on exactly those two things (PHASE6_HANDOFF 4):
backprop through a 200-step stiff ODE, supervised through a near-step readout in which
**0.00% of wall nodes fell in the sigmoid's gradient band**.  Here every node is in the
band by construction and the loss is the metric, not a surrogate for it.

THE BACKBONE IS UNLEARNED.  Every model predicts a RESIDUAL on the physics ODE's own onset,
and every one of them reduces to the physics exactly at zero parameters:

    ``GlobalAffine``  onset = m + alpha*(ode - m) + beta          2 params, alpha=1 beta=0
    ``NodeMLP``       onset = ode + cap*tanh(mlp(x))              mlp=0 -> physics
    ``NodeGNN``       same, with k rounds of mesh message passing

``alpha`` is in the model because the measured error has exactly two features: the front
arrives too early (a shift) and the curve is too steep (a scale).  A 2-parameter model that
can express both is the right thing to beat before any network is allowed to claim capacity
-- PHASE6_HANDOFF 19.2's standing lesson is that 187k parameters once bought +0.024 over a
logistic regression.

NEVER-COMMITTING NODES.  ``onset > 1`` is allowed and read as "does not commit within the
horizon".  That is a real degree of freedom -- some vessels over-commit by 70% -- but it
moves the final mask, so callers must report ``mask_delta`` whenever it is enabled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def soft_count(onset: torch.Tensor, grid: torch.Tensor, tau: float) -> torch.Tensor:
    """``n(t)`` for every t in ``grid``; differentiable in ``onset``.  Shapes [N], [T] -> [T]."""
    return torch.sigmoid((grid[:, None] - onset[None, :]) / tau).sum(dim=1)


def growth_l1_soft(onset: torch.Tensor, gt_curve: torch.Tensor, grid: torch.Tensor,
                   n_gt: float, tau: float) -> torch.Tensor:
    """The metric itself, with the step function relaxed.  ``tau -> 0`` recovers it exactly."""
    return (soft_count(onset, grid, tau) - gt_curve).abs().mean() / max(n_gt, 1.0)


class GlobalAffine(nn.Module):
    """``onset = median + alpha*(ode - median) + beta``.  Two parameters, cohort-wide.

    The measured defects are a shift (+0.15 of the final count arrives too early at
    t/T=0.3) and a scale (the curve is too steep).  This is the smallest model that can
    express both, and it is the bar every larger model has to clear.
    """

    def __init__(self):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, ode, x=None, edges=None):
        m = ode.median()
        return m + torch.exp(self.log_alpha) * (ode - m) + self.beta


class VesselAffine(nn.Module):
    """``alpha``/``beta`` predicted from POOLED vessel descriptors, not fitted per vessel.

    Per-vessel constants would be unlearnable at deploy time; this reads only aggregates of
    the same deploy-legal node features, so it transfers.
    """

    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_feat, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ode, x, edges=None):
        d = torch.cat([x.mean(0), x.std(0)], dim=0)
        a, b = self.net(d)
        m = ode.median()
        return m + torch.exp(a.clamp(-2, 2)) * (ode - m) + 0.5 * torch.tanh(b)


class NodeMLP(nn.Module):
    """Per-node residual on the physics onset.  Zero-initialised: it starts AS the physics."""

    def __init__(self, n_feat, hidden=64, layers=2, cap=0.5):
        super().__init__()
        dims = [n_feat] + [hidden] * layers
        seq = []
        for a, b in zip(dims[:-1], dims[1:]):
            seq += [nn.Linear(a, b), nn.SiLU()]
        seq += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*seq)
        self.cap = cap
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ode, x, edges=None):
        return ode + self.cap * torch.tanh(self.net(x).squeeze(-1))


class NodeGNN(nn.Module):
    """Mesh message passing, then the same zero-initialised residual head.

    Symmetric mean aggregation on the wall subgraph.  PHASE6_RESULTS 3.4 measured that
    ISOTROPIC smoothing of a physical sink makes the fit worse, so the graph here is given
    learnable channels rather than being used as a smoother, and the ablation against
    ``NodeMLP`` (identical head, no message passing) is what says whether the graph earns
    its place at all.
    """

    def __init__(self, n_feat, hidden=64, rounds=2, cap=0.5):
        super().__init__()
        self.inp = nn.Linear(n_feat, hidden)
        self.msg = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(rounds)])
        self.out = nn.Linear(hidden, 1)
        self.cap = cap
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ode, x, edges):
        h = torch.nn.functional.silu(self.inp(x))
        src, dst = edges
        n = x.shape[0]
        deg = torch.zeros(n, device=x.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp(min=1.0)
        for layer in self.msg:
            agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg[:, None]
            h = torch.nn.functional.silu(layer(torch.cat([h, agg], dim=1))) + h
        return ode + self.cap * torch.tanh(self.out(h).squeeze(-1))


MODELS = {"global_affine": GlobalAffine, "vessel_affine": VesselAffine,
          "node_mlp": NodeMLP, "node_gnn": NodeGNN}


def build(name: str, n_feat: int, **kw) -> nn.Module:
    if name == "global_affine":
        return GlobalAffine()
    if name == "vessel_affine":
        return VesselAffine(n_feat, hidden=kw.get("hidden", 16))
    if name == "node_mlp":
        return NodeMLP(n_feat, hidden=kw.get("hidden", 64), layers=kw.get("layers", 2),
                       cap=kw.get("cap", 0.5))
    if name == "node_gnn":
        return NodeGNN(n_feat, hidden=kw.get("hidden", 64), rounds=kw.get("rounds", 2),
                       cap=kw.get("cap", 0.5))
    raise ValueError(f"unknown model {name!r}")


@torch.no_grad()
def hard_growth_l1(onset_frac, gt_onset, nt, n_gt, allow_never: bool):
    """The REAL metric on a model's output, for reporting.  Never used as a loss."""
    from src.core_physics.growth_count_metrics import count_curve, growth_error

    o = onset_frac.detach().cpu().numpy()
    idx = np.clip(np.round(o * (nt - 1)), 0, nt - 1).astype(int)
    if allow_never:
        idx = np.where(o > 1.0, -1, idx)
    g = count_curve(gt_onset, nt)
    m = count_curve(idx, nt)
    return float(np.abs(m - g).mean() / max(n_gt, 1.0)), int((idx >= 0).sum())
