"""V2 nucleation eligibility: wall or N-hop from prior commits (no ceiling cap).

Replaces the fixed ``ceiling_mask`` crutch for Track V2. Track V1 may still use
``clot_growth_masks.resolve_ceiling_mask``.
"""

from __future__ import annotations

import os

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import (
    graph_dilate_hops,
    gt_clot_mask_at_time,
    gt_growth_commit_mask_at_time,
    growth_seed_mode,
    pred_clot_mask,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_phi_simple import _wall_mask_from_data


def nucleation_hops_from_env() -> int:
    raw = (os.environ.get("CLOT_V2_NUCLEATION_HOPS") or "1").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 1


def catalytic_hops_from_env() -> int:
    raw = (os.environ.get("CLOT_V2_CATALYTIC_HOPS") or "1").strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return 1


def catalytic_beta_from_env() -> float:
    raw = (os.environ.get("CLOT_V2_CATALYTIC_BETA") or "1.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def resolve_wall_mask(data, device: torch.device) -> torch.Tensor:
    n = int(data.num_nodes)
    return _wall_mask_from_data(data, device, n).reshape(-1).bool()


def resolve_commits_at_time(
    data,
    time_index: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    commit_thresh: float = 0.5,
    growth_seed: str | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Committed nodes C(., tau) from GT (audit) or predicted phi history (deploy)."""
    mode = growth_seed if growth_seed is not None else growth_seed_mode()
    t = max(int(time_index), 0)
    if mode == "gt":
        return gt_growth_commit_mask_at_time(data, t, phys_cfg, device)
    if phi_pred_by_time is None:
        raise ValueError("pred growth_seed requires phi_pred_by_time")
    phi = phi_pred_by_time.get(t)
    if phi is None:
        return torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
    return pred_clot_mask(phi, thresh=commit_thresh)


def resolve_nucleation_eligibility(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    commits_prev: torch.Tensor | None = None,
    growth_seed: str | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
    commit_thresh: float = 0.5,
    nucleation_hops: int | None = None,
    use_dgamma_wall_seed: bool = True,
) -> torch.Tensor:
    """Nodes where **new** clot may nucleate at ``time_index``.

    E_seed = wall  OR  N-hop dilate(commits at t-1).

    At t=0 with no prior commits, only wall (or t0 dgamma wall band if enabled).
    """
    n = int(data.num_nodes)
    t = max(int(time_index), 0)
    hops = nucleation_hops_from_env() if nucleation_hops is None else max(int(nucleation_hops), 0)
    ei = data.edge_index.to(device=device)

    if int(t) <= 0:
        if use_dgamma_wall_seed:
            wall = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
        else:
            wall = resolve_wall_mask(data, device)
        return wall.reshape(-1).bool()

    if commits_prev is not None:
        prev = commits_prev.reshape(-1).to(device=device).bool()
    else:
        prev = resolve_commits_at_time(
            data,
            t - 1,
            device=device,
            phys_cfg=phys_cfg,
            commit_thresh=commit_thresh,
            growth_seed=growth_seed,
            phi_pred_by_time=phi_pred_by_time,
        )

    wall = resolve_wall_mask(data, device)
    if hops <= 0 or not bool(prev.any().item()):
        return wall
    neighbor = graph_dilate_hops(prev, ei, hops)
    return wall | neighbor


def resolve_catalytic_hood(
    commits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    catalytic_hops: int | None = None,
) -> torch.Tensor:
    """Soft clot-friendly neighborhood H(x): dilate existing commits (not a hard mask)."""
    hops = catalytic_hops_from_env() if catalytic_hops is None else max(int(catalytic_hops), 0)
    if hops <= 0:
        return commits.reshape(-1).bool().clone()
    return graph_dilate_hops(commits.reshape(-1).bool(), edge_index, hops)


def catalytic_rate_multiplier(hood: torch.Tensor, *, beta: float | None = None) -> torch.Tensor:
    """Rate multiplier (1 + beta * H) for V3+ growth heads."""
    b = catalytic_beta_from_env() if beta is None else max(float(beta), 0.0)
    h = hood.reshape(-1).float()
    return 1.0 + b * h


def gt_new_commit_mask(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    device: torch.device,
) -> torch.Tensor:
    """GT nodes that newly cross clot threshold at ``time_index``."""
    t = max(int(time_index), 0)
    cur = gt_growth_commit_mask_at_time(data, t, phys_cfg, device)
    if t <= 0:
        return cur
    prev = gt_growth_commit_mask_at_time(data, t - 1, phys_cfg, device)
    return cur & ~prev


def project_phi_with_nucleation(
    phi_raw: torch.Tensor,
    phi_prev: torch.Tensor | None,
    eligibility: torch.Tensor,
    *,
    commit_thresh: float = 0.5,
    hard_commit: bool = True,
) -> torch.Tensor:
    """Monotone commit projection inside nucleation eligibility (absorbing commits)."""
    n = int(phi_raw.numel())
    device = phi_raw.device
    prev = (
        phi_prev.reshape(-1).float().to(device=device)
        if phi_prev is not None
        else torch.zeros(n, device=device, dtype=phi_raw.dtype)
    )
    elig = eligibility.reshape(-1).bool().to(device=device)
    raw = phi_raw.reshape(-1).float().clamp(0.0, 1.0)
    proposed = torch.where(elig, raw, prev)
    out = torch.maximum(prev, proposed)
    if hard_commit and commit_thresh > 0:
        committed = out >= float(commit_thresh)
        out = torch.where(committed, torch.ones_like(out), out)
    return out


def snapshot_nucleation_config() -> dict[str, object]:
    return {
        "nucleation_hops": nucleation_hops_from_env(),
        "catalytic_hops": catalytic_hops_from_env(),
        "catalytic_beta": catalytic_beta_from_env(),
        "growth_seed": growth_seed_mode(),
    }
