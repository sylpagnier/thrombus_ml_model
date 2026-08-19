"""Flow-aware message passing on the mesh, with the physics backbone as a residual base.

Design follows the measured physics rather than a generic GNN recipe:

  * **Anisotropic messages.** PHASE6_RESULTS 3.4 measured that isotropic mesh smoothing of
    the source makes the fit *worse* -- the non-locality is advective, not diffusive.  So
    every edge carries the projection of the t=0 velocity onto it, and messages are gated
    by upstream/downstream sign.  An isotropic GNN is the wrong prior here.
  * **Physics as a base, not a competitor.** The backbone's own ``log(Mat/crit)`` enters as
    a feature *and* as an additive base for the regression head, so the network learns a
    residual and ``residual = 0`` recovers the physics.
  * **Two heads.** GT clot IS ``{Mat >= crit}``, so regressing ``log1p(Mat/crit)`` is the
    physically-faithful target and the classifier is the readout the score actually uses.
    Training both shares the representation and keeps the regression honest.
  * **Domain embedding.** Wall / first-shell / interior behave differently (a wall node
    accumulates its own flux, a shell node inherits ~0.16x its owner's), so the node type
    is an explicit input rather than something to infer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_features(pos: np.ndarray, ei: np.ndarray, u: np.ndarray, v: np.ndarray,
                  h_edge: float) -> np.ndarray:
    src, dst = ei[0], ei[1]
    d = (pos[dst] - pos[src]) / max(h_edge, 1e-12)
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    dh = d / (ln + 1e-9)
    fs = np.stack([u[src], v[src]], 1)
    fd = np.stack([u[dst], v[dst]], 1)
    ns = np.linalg.norm(fs, axis=1, keepdims=True) + 1e-9
    nd = np.linalg.norm(fd, axis=1, keepdims=True) + 1e-9
    cos_s = (dh * fs).sum(1, keepdims=True) / ns
    cos_d = (dh * fd).sum(1, keepdims=True) / nd
    spd_s = np.log1p(ns)
    return np.concatenate([dh, ln, cos_s, cos_d, cos_s * spd_s, spd_s], axis=1).astype(np.float32)


class MPLayer(nn.Module):
    """One anisotropic message-passing step: separate upstream / downstream aggregation."""

    def __init__(self, dim: int, edim: int, drop: float = 0.1):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * dim + edim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        self.upd = nn.Sequential(nn.Linear(4 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, ei, ea, w_up, w_dn):
        src, dst = ei[0], ei[1]
        m = self.msg(torch.cat([x[src], x[dst], ea], dim=1))
        n = x.shape[0]
        agg_u = torch.zeros_like(x).index_add_(0, dst, m * w_up)
        agg_d = torch.zeros_like(x).index_add_(0, dst, m * w_dn)
        cu = torch.zeros(n, 1, device=x.device).index_add_(0, dst, w_up).clamp_min(1e-6)
        cd = torch.zeros(n, 1, device=x.device).index_add_(0, dst, w_dn).clamp_min(1e-6)
        mx = torch.zeros_like(x).index_reduce_(0, dst, m, "amax", include_self=False)
        h = self.upd(torch.cat([x, agg_u / cu, agg_d / cd, mx], dim=1))
        return self.norm(x + self.drop(h))


class ClotGNN(nn.Module):
    def __init__(self, in_dim: int, edim: int, dim: int = 96, layers: int = 6,
                 drop: float = 0.1, extra_dim: int = 0):
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.enc = nn.Sequential(nn.Linear(in_dim + self.extra_dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mp = nn.ModuleList([MPLayer(dim, edim, drop) for _ in range(layers)])
        self.head_cls = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_reg = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        # residual base: start at the physics prediction exactly
        nn.init.zeros_(self.head_reg[-1].weight)
        nn.init.zeros_(self.head_reg[-1].bias)

    def forward(self, x, ei, ea, w_up, w_dn, mat_phys, extra=None):
        h = self.enc(x if extra is None else torch.cat([x, extra], dim=1))
        for layer in self.mp:
            h = layer(h, ei, ea, w_up, w_dn)
        logit = self.head_cls(h).reshape(-1)
        reg = mat_phys + self.head_reg(h).reshape(-1)
        return logit, reg


def to_device(S: dict, mu: np.ndarray, sd: np.ndarray, dev: torch.device) -> dict:
    ei = S["edge_index"]
    pos, u, v = S["pos"], S["u"], S["v"]
    h_edge = float(np.median(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)))
    ea = edge_features(pos, ei, u, v, h_edge)
    cos_s = ea[:, 4:5]
    w_up = np.clip(cos_s, 0.0, None)
    w_dn = np.clip(-cos_s, 0.0, None)
    t = lambda a, d=torch.float32: torch.tensor(np.ascontiguousarray(a), dtype=d, device=dev)
    return dict(
        x=t((S["X"] - mu) / sd), ei=t(ei, torch.long), ea=t(ea),
        w_up=t(w_up), w_dn=t(w_dn),
        mat_phys=t(np.log1p(np.maximum(S["mat_phys"], 0.0) / 2e7)),
        y=t(S["y"]), mat_gt=t(S["mat_gt"]),
        wall=t(S["wall"].astype(np.float32)), n=int(len(S["wall"])))
