"""Discrete-time survival head: predict WHEN each wall node ignites.

WHY THIS SHAPE OF MODEL.  The physics model already gets the committed SET right --
0.9093 on sealed against a flow-oracle ceiling of 0.9066, i.e. no headroom.  What it gets
wrong is timing: patient043 commits all 84 nodes in one step at t=3000 s while GT climbs
from ~7000 s to 30000 s.  Holding the set fixed and varying only the onset times is worth
+0.084 (train) / +0.036 (sealed) on the time-resolved deploy score
(``scripts/diag_time_resolved_ceiling.py``).

So the head predicts onset, not mask.  Consequences, all deliberate:

  * the committed set is supplied by the physics and never touched, so the final-time
    score CANNOT regress -- the head is strictly additive;
  * supervision goes from 19 units (one mask per vessel) to **10,343** node-subjects
    (1,769 events + 8,574 right-censored) -- 544x more;
  * it is feed-forward on t=0 features, so none of the pathologies that killed the
    in-ODE corrector apply: no 200-step recurrence, no bifurcation, and no near-step
    readout (measured: 0.00% of nodes fell in the old readout's gradient band);
  * "never commits" is right-censoring, which the survival likelihood handles natively
    rather than as a fudged negative.

Time is normalised to each vessel's own horizon before binning, because horizons differ
(23062-30000 s) and the project's recurring failure is that per-vessel LEVELS do not
transfer while orderings do.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

FEATURE_NAMES = (
    "log_sr", "sr_margin", "dsrx_scaled", "sep_margin", "gate", "gate_low", "gate_sep",
    "nbr_gate", "nbr_gate_low", "sdf", "mu_t0", "speed", "phys_onset", "phys_commits",
)


def build_features(data, bio_cfg, phys_cfg, *, flow="gt", stencil=None):
    """[N, F] deploy-legal node features at t=0, plus the physics model's own onset.

    Including ``phys_onset`` means the head starts from the physics ordering and learns a
    residual on it, rather than having to rediscover the gate structure from scratch.
    """
    import scipy.sparse as sp

    from src.core_physics.mls_gradient import node_positions
    from src.core_physics.physics_wall_model import (
        first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
    )
    from src.differentiable_wall_model.deploy_features import carreau_mu_nd

    stencil = stencil if stencil is not None else (3 if flow == "gt" else 4)
    f = t0_flow_fields(data, bio_cfg, hops=stencil, flow_source=flow)
    n = int(data.num_nodes)
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    lss = float(bio_cfg.lss)
    sgt = float(bio_cfg.sgt) / 100.0

    ei = data.edge_index.cpu().numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).reshape(-1)
    deg[deg == 0] = 1.0

    gate = f.gate
    g_low = f.gate_low
    g_sep = f.gate_sep

    # physics onset, normalised to [0,1] of this vessel's horizon; 1.0 if it never commits
    g_hard = graded_gate(f, bio_cfg, mode="hard") * wall
    traj, t = integrate_mat_trajectory(data, bio_cfg, g_hard, da_scale=40.0)
    idx = first_crossing(traj, float(bio_cfg.viscosity_mat_crit))
    T = len(t)
    phys_onset = np.where(idx >= 0, idx / max(T - 1, 1), 1.0)
    phys_commits = (idx >= 0).astype(np.float64)

    if flow == "pred":
        u = data.u0_pred.reshape(-1).detach().cpu().numpy()
        v = data.v0_pred.reshape(-1).detach().cpu().numpy()
    else:
        u = data.y[0, :, 0].detach().cpu().numpy()
        v = data.y[0, :, 1].detach().cpu().numpy()
    speed = np.hypot(u, v)

    from src.utils.channel_schema import KINE_X_SCHEMA, X_SCHEMAS
    ch = list(X_SCHEMAS[KINE_X_SCHEMA].channels)
    sdf = data.x[:, ch.index("sdf_nd")].detach().cpu().numpy()
    mu = carreau_mu_nd(f.sr, phys_cfg)          # recomputed from THIS arm's flow

    feats = np.stack([
        np.log1p(np.clip(f.sr, 0, None)) / 5.0,
        np.clip((lss - f.sr) / lss, -5.0, 1.0),
        np.tanh(f.dsrx / 1000.0),
        np.clip((sgt - f.dsrx) / abs(sgt), -5.0, 5.0),
        np.tanh(gate),
        g_low, g_sep,
        (A @ gate) / deg, (A @ g_low) / deg,
        sdf, np.log1p(np.clip(mu, 0, None)), np.tanh(speed * 5.0),
        phys_onset, phys_commits,
    ], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(feats, dtype=torch.float32), wall, phys_onset, idx


def onset_targets(gt_onset, n_times, n_bins):
    """(bin index, event flag) from a per-node GT onset index; -1 means right-censored."""
    ev = gt_onset >= 0
    frac = np.where(ev, gt_onset / max(n_times - 1, 1), 1.0)
    b = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    return torch.tensor(b, dtype=torch.long), torch.tensor(ev, dtype=torch.bool)


class SurvivalOnsetHead(nn.Module):
    """Per-node discrete-time hazard over ``n_bins`` of the vessel's normalised horizon."""

    def __init__(self, in_dim: int, hidden: int = 64, n_bins: int = 20, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_bins = int(n_bins)
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_bins)
        # Start with a low hazard everywhere so the initial cumulative incidence is gentle
        # rather than an immediate flash -- the exact failure mode being corrected.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -3.0)

    def forward(self, x, edge_index):
        h = self.enc(x)
        for c in self.convs:
            h = F.relu(h + c(h, edge_index))
        return self.head(self.drop(h))                    # hazard logits [N, K]


def survival_nll(logits, bin_idx, event, mask):
    """Discrete-time survival negative log-likelihood with right-censoring.

    event at bin k : log h_k + sum_{j<k} log(1 - h_j)
    censored       : sum_{j<K} log(1 - h_j)
    """
    lg = logits[mask]
    b = bin_idx[mask]
    e = event[mask]
    log_h = F.logsigmoid(lg)
    log_s = F.logsigmoid(-lg)                              # log(1 - h)
    cum = torch.cumsum(log_s, dim=1)
    prior = torch.where(b > 0, cum.gather(1, (b - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                        torch.zeros_like(cum[:, 0]))
    ll_event = log_h.gather(1, b.unsqueeze(1)).squeeze(1) + prior
    ll_cens = cum[:, -1]
    ll = torch.where(e, ll_event, ll_cens)
    return -ll.mean()


def conditional_pmf(logits):
    """``p_k = S(k-1) * h_k`` renormalised over eventual committers. Differentiable."""
    log_h = F.logsigmoid(logits)
    log_s = F.logsigmoid(-logits)
    cum = torch.cumsum(log_s, dim=1)
    prior = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
    pmf = torch.exp(log_h + prior)
    return pmf / pmf.sum(dim=1, keepdim=True).clamp(min=1e-8)


def expected_onset_frac(logits):
    """Conditional EXPECTED onset in [0,1] -- continuous, and differentiable.

    The median-of-bins readout this replaces was effectively degenerate: with 20 bins and
    a near-flat hazard it produced only 2-8 distinct values per vessel (std 0.03 against
    GT's 0.08-0.21), i.e. a learned constant delay with no ordering.  An expectation keeps
    per-node resolution even when hazards differ only slightly, and being differentiable it
    lets an ordering loss act directly on the quantity the metric consumes.
    """
    k = logits.shape[1]
    centers = (torch.arange(k, device=logits.device, dtype=logits.dtype) + 0.5) / k
    return (conditional_pmf(logits) * centers.unsqueeze(0)).sum(dim=1)


@torch.no_grad()
def predicted_onset_frac(logits):
    return expected_onset_frac(logits)
