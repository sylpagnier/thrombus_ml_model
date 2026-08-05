"""Time-varying clot support: t0 dgamma wall seed, hop growth, fixed ceiling.

Deploy simplification (CAVO):
- ``t0_mask``: dgamma wall adhesion @ ref time, **no** initial clot seeds (growth seed).
- ``ceiling_mask``: **full wall** + ``CLOT_PHI_CEILING_HOPS`` off-wall hops (default 2).
- ``support(t)``: grows 1 hop per step from t0 bootstrap, then from **new** clots (capped by ceiling).

Off ``support(t)`` (and always off ``ceiling_mask``) -> Carreau bulk mu at the **current**
time index (not t_in).
"""

from __future__ import annotations

import os

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    _graph_dilate,
    _wall_mask_from_data,
    cap_mu_eff_si,
    carreau_mu_si_from_uv,
    clot_phi_dgamma_slice_enabled,
    clot_phi_thresh_si,
    dgamma_dx_slice_mask,
)


def clot_ceiling_hops() -> int:
    return max(int(float(os.environ.get("CLOT_PHI_CEILING_HOPS", "3") or "3")), 0)


def growth_seed_mode() -> str:
    """``gt`` = expand support from GT new clots (train oracle); ``pred`` = from model phi."""
    raw = (os.environ.get("CLOT_PHI_GROWTH_SEED") or "gt").strip().lower()
    if raw in ("pred", "model", "deploy"):
        return "pred"
    return "gt"


def graph_dilate_hops(active: torch.Tensor, edge_index: torch.Tensor, hops: int) -> torch.Tensor:
    out = active.reshape(-1).bool().clone()
    h = max(int(hops), 0)
    for _ in range(h):
        out = _graph_dilate(out, edge_index)
    return out


def bool_mask_connected_components(
    mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> list[torch.Tensor]:
    """Undirected connected components of a boolean node mask on ``edge_index``.

    Returns one bool mask per uninterrupted region (same shape as ``mask``).
    Empty / all-False input yields ``[]``.
    """
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return []
    n = int(m.numel())
    device = m.device
    ei = edge_index.to(device=device)
    row = ei[0].long()
    col = ei[1].long()
    # Restrict adjacency to nodes inside the mask.
    keep = m[row] & m[col]
    row = row[keep]
    col = col[keep]

    # CSR-ish neighbor lists on CPU for BFS (small active sets).
    active_idx = m.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
    if not active_idx:
        return []
    from collections import defaultdict

    nbrs: dict[int, list[int]] = defaultdict(list)
    if row.numel() > 0:
        r_cpu = row.detach().cpu().tolist()
        c_cpu = col.detach().cpu().tolist()
        for a, b in zip(r_cpu, c_cpu):
            nbrs[int(a)].append(int(b))
            nbrs[int(b)].append(int(a))

    seen: set[int] = set()
    comps: list[torch.Tensor] = []
    for start in active_idx:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = [start]
        while stack:
            u = stack.pop()
            for v in nbrs.get(u, ()):
                if v not in seen and m[v].item():
                    seen.add(v)
                    stack.append(v)
                    members.append(v)
        comp = torch.zeros(n, dtype=torch.bool, device=device)
        comp[torch.tensor(members, dtype=torch.long, device=device)] = True
        comps.append(comp)
    # Largest first (stable exploratory preference).
    comps.sort(key=lambda c: int(c.sum().item()), reverse=True)
    return comps


def resolve_t0_dgamma_wall_mask(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Wall-only dgamma adhesion band @ ref time; assume **no clots** at t=0."""
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    if not clot_phi_dgamma_slice_enabled():
        return wall
    clot_seed = torch.zeros(n, device=device, dtype=torch.bool)
    return dgamma_dx_slice_mask(data, device, wall, clot_seed, bio_cfg).reshape(-1).bool()


def resolve_ceiling_mask(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    ceiling_hops: int | None = None,
) -> torch.Tensor:
    """Fixed upper bound: ``mask_wall`` + ``ceiling_hops`` lumen hops (default 2)."""
    del bio_cfg  # ceiling is wall-topology only
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hops = clot_ceiling_hops() if ceiling_hops is None else int(ceiling_hops)
    if hops <= 0:
        return wall.reshape(-1).bool()
    ei = data.edge_index.to(device=device)
    return graph_dilate_hops(wall, ei, hops)


def gt_clot_mask_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> torch.Tensor:
    """GT clot nodes: growth-only above per-node mu at macro t=0."""
    return gt_growth_commit_mask_at_time(data, time_index, phys_cfg, device)


def gt_growth_commit_mask_at_time(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> torch.Tensor:
    """Growth-only GT clot: ``relu(mu_eff(t) - mu_eff(t=0)) >= thresh``."""
    from src.core_physics.clot_phi_simple import gt_mu_anchor_cap_si

    y = data.y[int(time_index)].to(device=device)
    mu = cap_mu_eff_si(phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
    anchor = gt_mu_anchor_cap_si(data, phys_cfg, device).reshape(-1)
    growth = (mu - anchor).clamp(min=0.0)
    return (growth >= clot_phi_thresh_si(phys_cfg)).bool()


def pred_clot_mask(phi: torch.Tensor, *, thresh: float = 0.5) -> torch.Tensor:
    return phi.reshape(-1) >= float(thresh)


def resolve_growth_support_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    growth_seed: str | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
    ceiling_hops: int | None = None,
) -> torch.Tensor:
    """Support B_t at graph time ``time_index`` (capped by ``ceiling_mask``)."""
    t = max(int(time_index), 0)
    seed_mode = growth_seed if growth_seed is not None else growth_seed_mode()
    t0 = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg, ceiling_hops=ceiling_hops)
    ei = data.edge_index.to(device=device)

    try:
        from src.architecture.pushforward_config import resolve_config
        _cfg = resolve_config()
        dil_hops = int(_cfg.growth_dilation) if _cfg is not None else int(os.environ.get("SPECIES_GROWTH_DILATION", "1"))
    except Exception:
        dil_hops = int(os.environ.get("SPECIES_GROWTH_DILATION", "1"))
    if t <= 0:
        return t0 & ceiling

    support = t0 | graph_dilate_hops(t0, ei, dil_hops)
    if t == 1:
        return support & ceiling

    for k in range(2, t + 1):
        if seed_mode == "gt":
            clot_km1 = gt_growth_commit_mask_at_time(data, k - 1, phys_cfg, device)
            clot_km2 = gt_growth_commit_mask_at_time(data, k - 2, phys_cfg, device)
            new_clot = clot_km1 & ~clot_km2
        else:
            if phi_pred_by_time is None:
                # No rollout history (viz/debug): static wall bootstrap only.
                return (t0 | graph_dilate_hops(t0, ei, dil_hops)) & ceiling
            phi_m1 = phi_pred_by_time.get(k - 1)
            if phi_m1 is None:
                new_clot = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
            else:
                phi_m2 = phi_pred_by_time.get(k - 2)
                if phi_m2 is None:
                    new_clot = pred_clot_mask(phi_m1)
                else:
                    new_clot = pred_clot_mask(phi_m1) & ~pred_clot_mask(phi_m2)
        support = support | graph_dilate_hops(new_clot, ei, dil_hops)

    return support & ceiling


def resolve_bulk_carreau_mu_si(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
    *,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Carreau bulk viscosity at ``time_index`` (GT or patched u,v)."""
    if u_nd is not None and v_nd is not None:
        return carreau_mu_si_from_uv(
            data,
            u_nd.reshape(-1),
            v_nd.reshape(-1),
            phys_cfg,
        ).reshape(-1)
    y = data.y[int(time_index)].to(device=device)
    return carreau_mu_si_from_uv(data, y[:, 0], y[:, 1], phys_cfg).reshape(-1)


def snapshot_clot_growth_config() -> dict[str, object]:
    return {
        "ceiling_hops": clot_ceiling_hops(),
        "growth_seed": growth_seed_mode(),
    }
