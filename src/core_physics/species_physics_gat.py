"""Physics-biased GAT trunk for species / mat-growth pushforward.

Faithful port of Stage-A ``MultiHeadPhysicsGATConv`` (RGP-DEQ):
  * geometric ``edge_attr`` (dx, dy, length) feeds ``edge_proj``;
  * wall rheology / advection / curvature mods are *separate* additive logits;
  * production order: multiply-by-edge-proj then add mods.

Mesh ``WALL_NORMAL`` + ``SDF`` are preferred over estimated geometry.
Used when ``PushforwardConfig.arch == "physics_gat"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Stage-A / biochem graphs: [dx, dy, length].
GEOM_EDGE_DIM = 3
# Back-compat alias (packed prior width is no longer used for edge_proj).
PHYSICS_EDGE_DIM = GEOM_EDGE_DIM


@dataclass
class PhysicsEdgeBundle:
    """Stage-A-style edge geometry + separate wall modulators."""

    edge_attr: Tensor  # (E, 3)
    mod_rheo: Tensor  # (E, 1)
    mod_adv: Tensor  # (E, 1)
    mod_curve: Tensor  # (E, 1)


def estimate_wall_normals(
    pos: Tensor,
    wall_mask: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Fallback inward wall normals when mesh normals are unavailable.

    Prefer ``NodeFeat.WALL_NORMAL`` from kinematics-layout ``data.x`` (not
    ``BiochemNodeFeat``, which indexes ``data.x_biochem``).
    """
    device = pos.device
    dtype = pos.dtype
    n = pos.size(0)
    normals = torch.zeros((n, 2), device=device, dtype=dtype)
    wall_mask = wall_mask.to(device=device).reshape(-1).bool()
    if n == 0 or not wall_mask.any():
        return normals

    wall_idx = wall_mask.nonzero(as_tuple=False).view(-1)
    wall_pos = pos[wall_idx]
    d2 = torch.cdist(pos, wall_pos, p=2)
    nn_local = d2.argmin(dim=1)
    nearest = wall_idx[nn_local]
    diff = pos - pos[nearest]
    norms = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normals = diff / norms

    row, col = edge_index
    fluid_nbr = wall_mask[row] & (~wall_mask[col])
    if fluid_nbr.any():
        acc = torch.zeros((n, 2), device=device, dtype=dtype)
        cnt = torch.zeros((n, 1), device=device, dtype=dtype)
        vec = pos[col[fluid_nbr]] - pos[row[fluid_nbr]]
        acc.index_add_(0, row[fluid_nbr], vec)
        ones = torch.ones((int(fluid_nbr.sum()), 1), device=device, dtype=dtype)
        cnt.index_add_(0, row[fluid_nbr], ones)
        has = (cnt.squeeze(-1) > 0) & wall_mask
        if has.any():
            normals[has] = acc[has] / cnt[has].clamp_min(1e-8)
            normals[has] = F.normalize(normals[has], p=2, dim=-1, eps=1e-8)
    else:
        any_nbr = wall_mask[row]
        if any_nbr.any():
            acc = torch.zeros((n, 2), device=device, dtype=dtype)
            cnt = torch.zeros((n, 1), device=device, dtype=dtype)
            vec = pos[col[any_nbr]] - pos[row[any_nbr]]
            acc.index_add_(0, row[any_nbr], vec)
            ones = torch.ones((int(any_nbr.sum()), 1), device=device, dtype=dtype)
            cnt.index_add_(0, row[any_nbr], ones)
            has = (cnt.squeeze(-1) > 0) & wall_mask
            if has.any():
                normals[has] = F.normalize(
                    acc[has] / cnt[has].clamp_min(1e-8), p=2, dim=-1, eps=1e-8
                )

    return normals


def geometric_edge_attr(edge_index: Tensor, pos: Tensor) -> Tensor:
    """Build Stage-A ``(E, 3)`` edge geometry: ``[dx, dy, length]``."""
    row, col = edge_index
    delta = pos[row] - pos[col]
    length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat([delta, length], dim=-1)


def build_physics_edge_bundle(
    edge_index: Tensor,
    *,
    pos: Tensor | None = None,
    wall_normals: Tensor | None = None,
    sdf: Tensor | None = None,
    wall_mask: Tensor | None = None,
    edge_attr_geom: Tensor | None = None,
    rheo_log_clamp_min: float = 1e-3,
    adv_log_clamp_min: float = 1e-3,
    curve_log_clamp_min: float = 1e-3,
    edge_decay_k: float = 5.0,
) -> PhysicsEdgeBundle:
    """Stage-A edge geometry + wall mods (mesh normals/SDF when available).

    ``edge_decay_k`` defaults to 5.0 to match Stage-A ``PhysicsConfig.gino_edge_decay_k``.
    Does **not** consume velocity. Flow conditioning belongs in node features /
    corrector coupling, never as a COMSOL GT shortcut into attention.
    """
    if edge_index is None or edge_index.numel() == 0:
        z = torch.zeros((0, GEOM_EDGE_DIM), dtype=torch.float32)
        z1 = torch.zeros((0, 1), dtype=torch.float32)
        return PhysicsEdgeBundle(edge_attr=z, mod_rheo=z1, mod_adv=z1, mod_curve=z1)

    row, col = edge_index
    e = int(row.numel())
    device = edge_index.device
    dtype = torch.float32
    if pos is not None:
        dtype = pos.dtype
        device = pos.device
    elif edge_attr_geom is not None:
        dtype = edge_attr_geom.dtype
        device = edge_attr_geom.device

    if edge_attr_geom is not None and edge_attr_geom.size(0) == e:
        edge_attr = edge_attr_geom.to(device=device, dtype=dtype)
        if edge_attr.size(-1) < GEOM_EDGE_DIM:
            pad = torch.zeros(
                (e, GEOM_EDGE_DIM - edge_attr.size(-1)), device=device, dtype=dtype
            )
            edge_attr = torch.cat([edge_attr, pad], dim=-1)
        elif edge_attr.size(-1) > GEOM_EDGE_DIM:
            edge_attr = edge_attr[:, :GEOM_EDGE_DIM]
    elif pos is not None:
        edge_attr = geometric_edge_attr(edge_index, pos.to(device=device, dtype=dtype))
    else:
        edge_attr = torch.zeros((e, GEOM_EDGE_DIM), device=device, dtype=dtype)

    zeros = torch.zeros((e, 1), device=device, dtype=dtype)
    if pos is None:
        return PhysicsEdgeBundle(
            edge_attr=edge_attr, mod_rheo=zeros, mod_adv=zeros, mod_curve=zeros
        )

    pos = pos.to(device=device, dtype=dtype)
    edge_vec = edge_attr[:, :2]
    e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)

    # Mesh normals preferred; estimate only as last resort.
    if wall_normals is not None and wall_normals.shape[0] == pos.shape[0]:
        normals = wall_normals.to(device=device, dtype=dtype)
        if normals.dim() == 1:
            normals = normals.view(-1, 2)
    elif wall_mask is not None and wall_mask.shape[0] == pos.shape[0]:
        normals = estimate_wall_normals(pos, wall_mask, edge_index)
    else:
        return PhysicsEdgeBundle(
            edge_attr=edge_attr, mod_rheo=zeros, mod_adv=zeros, mod_curve=zeros
        )

    n_row = F.normalize(normals[row], p=2, dim=-1, eps=1e-8)
    n_col = F.normalize(normals[col], p=2, dim=-1, eps=1e-8)

    if sdf is not None and sdf.shape[0] == pos.shape[0]:
        sdf_nd = sdf.to(device=device, dtype=dtype).reshape(-1)
        # Match Stage-A: SDF as-is (biochem/kine graphs store non-negative SDF).
        sdf_edge = sdf_nd[row].unsqueeze(-1)
    elif wall_mask is not None and wall_mask.shape[0] == pos.shape[0]:
        # Fallback: hop-0 walls + distance proxy (avoid when mesh SDF exists).
        wall_mask_b = wall_mask.to(device=device).reshape(-1).bool()
        wall_idx = wall_mask_b.nonzero(as_tuple=False).view(-1)
        if wall_idx.numel() > 0:
            d_wall = torch.cdist(pos, pos[wall_idx], p=2).min(dim=1).values
            med = d_wall.median().clamp_min(1e-6)
            sdf_edge = (d_wall / med)[row].unsqueeze(-1)
        else:
            sdf_edge = zeros
    else:
        sdf_edge = zeros

    decay = torch.exp(-float(edge_decay_k) * sdf_edge)
    dot_prod = torch.abs((e_dir * n_row).sum(dim=-1, keepdim=True)).clamp(max=1.0)
    curve_dot = (n_row * n_col).sum(dim=-1, keepdim=True)
    mod_rheo = torch.log(torch.clamp(dot_prod, min=rheo_log_clamp_min, max=1.0)) * decay
    mod_adv = torch.log(torch.clamp(1.0 - dot_prod, min=adv_log_clamp_min, max=1.0)) * decay
    mod_curve = (
        torch.log(torch.clamp(1.0 - curve_dot, min=curve_log_clamp_min, max=1.0)) * decay
    )
    return PhysicsEdgeBundle(
        edge_attr=edge_attr,
        mod_rheo=mod_rheo,
        mod_adv=mod_adv,
        mod_curve=mod_curve,
    )


def build_physics_edge_priors(
    edge_index: Tensor,
    pos: Tensor | None,
    velocity: Tensor | None = None,
    wall_mask: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """Deprecated packed ``(E, 3)`` geometric attrs (mods discarded).

    Prefer :func:`build_physics_edge_bundle`. ``velocity`` is ignored (no GT
    upwind shortcut into attention).
    """
    del velocity  # deploy-faithful policy: GAT does not consume UV
    bundle = build_physics_edge_bundle(
        edge_index, pos=pos, wall_mask=wall_mask, **kwargs
    )
    return bundle.edge_attr


class SpeciesPhysicsGATConv(MessagePassing):
    """Single-head Stage-A PM-GAT (species trunk).

    Matches ``MultiHeadPhysicsGATConv`` message algebra with production
    multiply-then-add prior order (kinematics production default).

    Species pushforward is a single forward (not a DEQ). With Xavier-init
    ``edge_proj ≈ 0``, multiply-before-add wipes content logits and leaves
    softmax driven only by wall mods (O(1-7)). Call
    :meth:`init_edge_proj_identity` after any blanket Linear re-init, and keep
    ``prior_scale`` << 1 so mods stay soft biases.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        edge_dim: int = GEOM_EDGE_DIM,
        temperature: float = 1.5,
        priors_multiply_before_add: bool = True,
        prior_scale: float = 0.05,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        kwargs.setdefault("node_dim", 0)
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)
        self.temperature = float(temperature)
        self.priors_multiply_before_add = bool(priors_multiply_before_add)
        self.prior_scale = float(prior_scale)

        self.lin_src = nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.lin_dst = nn.Linear(self.in_channels, self.out_channels, bias=True)
        self.att = nn.Linear(self.out_channels, 1, bias=True)
        self.edge_proj = nn.Linear(self.edge_dim, self.out_channels, bias=True)
        self.init_edge_proj_identity()

    def init_edge_proj_identity(self) -> None:
        """Make ``edge_proj(edge_attr) ≈ 1`` so multiply-before-add keeps content logits."""
        nn.init.zeros_(self.edge_proj.weight)
        if self.edge_proj.bias is not None:
            nn.init.ones_(self.edge_proj.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        mod_adv: Tensor | None = None,
        mod_rheo: Tensor | None = None,
        mod_curve: Tensor | None = None,
        size: Optional[Tuple[int, int]] = None,
        *,
        # Accept PhysicsEdgeBundle via keyword for call-site convenience.
        bundle: PhysicsEdgeBundle | None = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        e = int(edge_index.size(1)) if edge_index is not None else 0
        device = x[0].device
        dtype = x[0].dtype

        if bundle is not None:
            edge_attr = bundle.edge_attr
            mod_adv = bundle.mod_adv
            mod_rheo = bundle.mod_rheo
            mod_curve = bundle.mod_curve

        if edge_attr is None:
            edge_attr = torch.zeros((e, self.edge_dim), device=device, dtype=dtype)
        else:
            ea = edge_attr.to(device=device, dtype=dtype)
            if ea.size(-1) < self.edge_dim:
                pad = torch.zeros(
                    (ea.size(0), self.edge_dim - ea.size(-1)),
                    device=device,
                    dtype=dtype,
                )
                edge_attr = torch.cat([ea, pad], dim=-1)
            else:
                edge_attr = ea[:, : self.edge_dim]

        def _mod(m: Tensor | None) -> Tensor:
            if m is None:
                return torch.zeros((e, 1), device=device, dtype=dtype)
            m = m.to(device=device, dtype=dtype)
            return m.reshape(e, -1)[:, :1]

        mod_adv_t = _mod(mod_adv)
        mod_rheo_t = _mod(mod_rheo)
        mod_curve_t = _mod(mod_curve)

        x_src = self.lin_src(x[0])
        x_dst = self.lin_dst(x[1])
        alpha_src = self.att(x_src)
        alpha_dst = self.att(x_dst)

        return self.propagate(
            edge_index,
            size=size,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_attr=edge_attr,
            mod_adv=mod_adv_t,
            mod_rheo=mod_rheo_t,
            mod_curve=mod_curve_t,
        )

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: Tensor,
        edge_attr: Tensor,
        mod_adv: Tensor,
        mod_rheo: Tensor,
        mod_curve: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        alpha = (alpha_j + alpha_i) / self.temperature
        mods = float(self.prior_scale) * (mod_adv + mod_rheo + mod_curve)
        # Production Stage-A order (kinematics): edge_proj then additive wall mods.
        if self.priors_multiply_before_add:
            alpha = alpha * self.edge_proj(edge_attr)
            alpha = alpha + mods
        else:
            alpha = alpha + mods
            alpha = alpha * self.edge_proj(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha
