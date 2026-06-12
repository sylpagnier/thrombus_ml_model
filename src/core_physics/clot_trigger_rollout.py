"""Temporal clot-trigger rollout with deploy-faithful nucleation projection.

Forward contract (all T0+ stars at eval/deploy):
  phi_raw  = physics gelation and/or learned head (full mesh features)
  E(tau)   = wall @ tau=0 OR 1-hop from prior **predicted** commits
  phi(tau) = project_phi_with_nucleation(phi_raw, phi_prev, E(tau))

Loss / F1 support B_t is separate (``resolve_clot_loss_mask``); it may use GT
commits during T0-T2 training diagnostics but must not drive forward E(tau).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import growth_seed_mode, pred_clot_mask
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_phi_simple import build_clot_phi_step, clot_phi_model_uses_mpnn

GrowthSeed = Literal["gt", "pred"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def clot_trigger_nucleation_enabled() -> bool:
    """Apply ``project_phi_with_nucleation`` each macro step (default on)."""
    return _env_bool("CLOT_TRIGGER_NUCLEATION", True)


def clot_trigger_forward_seed_mode() -> GrowthSeed:
    """Seed for forward envelope E(tau): ``pred`` (deploy) or ``gt`` (oracle upper bound)."""
    raw = (os.environ.get("CLOT_TRIGGER_FORWARD_SEED") or "pred").strip().lower()
    if raw in ("gt", "oracle", "ground_truth"):
        return "gt"
    return "pred"


def clot_trigger_ic_phi_zero() -> bool:
    """Force phi=0 at macro t=0 (no clots at initial condition)."""
    raw = (os.environ.get("CLOT_TRIGGER_IC_PHI_ZERO") or "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def clot_trigger_commit_thresh() -> float:
    try:
        return float(os.environ.get("CLOT_TRIGGER_COMMIT_THRESH", "0.5") or "0.5")
    except ValueError:
        return 0.5


def clot_trigger_use_dgamma_wall_seed() -> bool:
    """At tau=0, use dgamma wall band instead of geometry wall (debug only)."""
    return _env_bool("CLOT_TRIGGER_DGAMMA_WALL_SEED", False)


def clot_trigger_train_soft_commit() -> bool:
    """Training: avoid absorbing hard commits so BCE gradients reach the MLP head."""
    return _env_bool("CLOT_TRIGGER_TRAIN_SOFT_COMMIT", True)


def snapshot_clot_trigger_rollout_config() -> dict[str, object]:
    out = dict(snapshot_nucleation_config())
    out.update(
        {
            "nucleation_projection": clot_trigger_nucleation_enabled(),
            "forward_seed": clot_trigger_forward_seed_mode(),
            "loss_growth_seed": growth_seed_mode(),
            "commit_thresh": clot_trigger_commit_thresh(),
            "ic_phi_zero": clot_trigger_ic_phi_zero(),
            "dgamma_wall_seed": clot_trigger_use_dgamma_wall_seed(),
        }
    )
    return out


def _project_step_phi(
    phi_raw: torch.Tensor,
    phi_prev: torch.Tensor | None,
    data,
    time_index: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    phi_pred_by_time: dict[int, torch.Tensor],
    growth_seed: GrowthSeed,
    hard_commit: bool | None = None,
) -> torch.Tensor:
    if not clot_trigger_nucleation_enabled():
        return phi_raw.reshape(-1).float()
    if hard_commit is None:
        hc = clot_trigger_commit_thresh() > 0 and not clot_trigger_train_soft_commit()
    else:
        hc = bool(hard_commit)
    elig = resolve_nucleation_eligibility(
        data,
        int(time_index),
        device,
        phys_cfg,
        bio_cfg,
        growth_seed=growth_seed,
        phi_pred_by_time=phi_pred_by_time if growth_seed == "pred" else None,
        commit_thresh=clot_trigger_commit_thresh(),
        use_dgamma_wall_seed=clot_trigger_use_dgamma_wall_seed(),
    )
    return project_phi_with_nucleation(
        phi_raw,
        phi_prev,
        elig,
        commit_thresh=clot_trigger_commit_thresh(),
        hard_commit=hc,
    )


def _apply_ic_phi_zero(
    phi_proj: torch.Tensor,
    phi_raw: torch.Tensor,
    time_index: int,
    n_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(time_index) <= 0 and clot_trigger_ic_phi_zero():
        z = torch.zeros(n_nodes, device=device, dtype=phi_proj.dtype)
        return z, z
    return phi_proj.reshape(-1), phi_raw.reshape(-1)


@torch.no_grad()
def rollout_clot_trigger_physics(
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_stride: int = 1,
    growth_seed: GrowthSeed | None = None,
    use_soft: bool = True,
    species_log1p: torch.Tensor | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Physics-only trigger trajectory with optional nucleation projection."""
    from src.training.clot_trigger_stack import forward_physics_trigger_phi

    seed = growth_seed if growth_seed is not None else clot_trigger_forward_seed_mode()
    n_steps = int(data.y.shape[0])
    stride = max(1, int(time_stride))
    out: dict[int, dict[str, torch.Tensor]] = {}
    phi_prev: torch.Tensor | None = None
    phi_by_t: dict[int, torch.Tensor] = {}
    mu_anchor_si: torch.Tensor | None = None

    for t in range(0, n_steps, stride):
        step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        phi_raw, mu_raw = forward_physics_trigger_phi(
            step,
            data,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            species_log1p=species_log1p,
            use_soft=use_soft,
            apply_region=False,
            time_index=int(t),
            mu_anchor_si=mu_anchor_si,
        )
        from src.core_physics.clot_phi_simple import clot_phi_physics_subtract_t0_mu

        if mu_anchor_si is None and clot_phi_physics_subtract_t0_mu():
            mu_anchor_si = mu_raw.reshape(-1).clone()
            phi_raw, mu_raw = forward_physics_trigger_phi(
                step,
                data,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                species_log1p=species_log1p,
                use_soft=use_soft,
                apply_region=False,
                time_index=int(t),
                mu_anchor_si=mu_anchor_si,
            )
        phi_proj = _project_step_phi(
            phi_raw,
            phi_prev,
            data,
            t,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_pred_by_time=phi_by_t,
            growth_seed=seed,
        )
        n_nodes = int(data.num_nodes)
        phi_proj, phi_raw = _apply_ic_phi_zero(phi_proj, phi_raw, t, n_nodes, device)
        phi_by_t[int(t)] = phi_proj.reshape(-1)
        out[int(t)] = {
            "phi_raw": phi_raw.reshape(-1),
            "phi": phi_proj.reshape(-1),
            "mu": mu_raw.reshape(-1),
        }
        phi_prev = phi_proj.reshape(-1)
    return out


def clot_phi_trigger_rollout_enabled() -> bool:
    return _env_bool("CLOT_PHI_TRIGGER_ROLLOUT", False)


def clot_phi_trigger_rollout_detach_prev() -> bool:
    raw = (os.environ.get("CLOT_PHI_TRIGGER_TBPTT_DETACH") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def rollout_clot_trigger_hybrid_core(
    model: nn.Module,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_stride: int = 1,
    growth_seed: GrowthSeed | None = None,
    use_soft: bool = True,
    species_log1p: torch.Tensor | None = None,
    mode: str = "hybrid",
    detach_prev: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """Hybrid trigger trajectory with nucleation projection (trainable when not no_grad)."""
    from src.core_physics.clot_phi_simple import clot_phi_physics_subtract_t0_mu
    from src.training.clot_trigger_stack import forward_clot_trigger_hybrid

    seed = growth_seed if growth_seed is not None else clot_trigger_forward_seed_mode()
    edge_index = data.edge_index.to(device) if clot_phi_model_uses_mpnn(model) else None
    n_steps = int(data.y.shape[0])
    stride = max(1, int(time_stride))
    out: dict[int, dict[str, torch.Tensor]] = {}
    phi_prev: torch.Tensor | None = None
    phi_by_t: dict[int, torch.Tensor] = {}
    mu_anchor_si: torch.Tensor | None = None

    for t in range(0, n_steps, stride):
        step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        bundle = forward_clot_trigger_hybrid(
            model,
            step,
            data,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            edge_index=edge_index,
            species_log1p=species_log1p,
            use_soft=use_soft,
            time_index=int(t),
            mu_anchor_si=mu_anchor_si,
        )
        if mu_anchor_si is None and clot_phi_physics_subtract_t0_mu():
            mu_anchor_si = bundle["mu_phys"].reshape(-1).detach().clone()
        if mode == "physics":
            phi_raw = bundle["phi_phys"]
        elif mode == "ml":
            phi_raw = bundle["phi_ml"]
        else:
            phi_raw = bundle["phi_hybrid"]
        phi_prev_in = phi_prev.detach() if detach_prev and phi_prev is not None else phi_prev
        phi_proj = _project_step_phi(
            phi_raw,
            phi_prev_in,
            data,
            t,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_pred_by_time=phi_by_t,
            growth_seed=seed,
        )
        n_nodes = int(data.num_nodes)
        phi_proj, phi_raw = _apply_ic_phi_zero(phi_proj, phi_raw, t, n_nodes, device)
        phi_store = phi_proj.reshape(-1).detach() if detach_prev else phi_proj.reshape(-1)
        phi_by_t[int(t)] = phi_store
        out[int(t)] = {
            "phi_raw": phi_raw.reshape(-1),
            "phi": phi_proj.reshape(-1),
            "phi_phys": bundle["phi_phys"].reshape(-1),
            "phi_ml": bundle["phi_ml"].reshape(-1),
            "phi_hybrid": bundle["phi_hybrid"].reshape(-1),
            "mu_hybrid": bundle["mu_hybrid"].reshape(-1),
        }
        phi_prev = phi_store if detach_prev else phi_proj.reshape(-1)
    return out


@torch.no_grad()
def rollout_clot_trigger_hybrid(
    model: nn.Module,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_stride: int = 1,
    growth_seed: GrowthSeed | None = None,
    use_soft: bool = True,
    species_log1p: torch.Tensor | None = None,
    mode: str = "hybrid",
) -> dict[int, dict[str, torch.Tensor]]:
    """Eval-only hybrid trigger trajectory."""
    return rollout_clot_trigger_hybrid_core(
        model,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        time_stride=time_stride,
        growth_seed=growth_seed,
        use_soft=use_soft,
        species_log1p=species_log1p,
        mode=mode,
        detach_prev=True,
    )


def rollout_clot_trigger_hybrid_trainable(
    model: nn.Module,
    data,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    time_stride: int = 1,
    growth_seed: GrowthSeed | None = None,
    use_soft: bool = True,
    species_log1p: torch.Tensor | None = None,
    mode: str = "hybrid",
    detach_prev: bool | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    """Trainable hybrid rollout (TBPTT: detach phi_prev each step by default)."""
    tbptt = clot_phi_trigger_rollout_detach_prev() if detach_prev is None else bool(detach_prev)
    return rollout_clot_trigger_hybrid_core(
        model,
        data,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        time_stride=time_stride,
        growth_seed=growth_seed,
        use_soft=use_soft,
        species_log1p=species_log1p,
        mode=mode,
        detach_prev=tbptt,
    )


def lumen_false_positive_frac(
    phi: torch.Tensor,
    phi_gt: torch.Tensor,
    *,
    data,
    device: torch.device,
    thresh: float | None = None,
) -> float:
    """Fraction of eligible lumen nodes with phi>=thresh where GT is negative."""
    from src.core_physics.clot_phi_simple import lumen_eligible_mask

    commit = clot_trigger_commit_thresh() if thresh is None else float(thresh)
    elig = lumen_eligible_mask(data, device).reshape(-1).bool()
    gt = phi_gt.reshape(-1).bool()
    pred_pos = pred_clot_mask(phi.reshape(-1), thresh=commit)
    bulk_fp = elig & pred_pos & ~gt
    n_elig = int(elig.sum().item())
    if n_elig <= 0:
        return float("nan")
    return float(bulk_fp.sum().item()) / float(n_elig)


def forward_path_uses_gt_commits(growth_seed: GrowthSeed | None = None) -> bool:
    """True when forward envelope E(tau) reads GT clot history (not deploy)."""
    seed = growth_seed if growth_seed is not None else clot_trigger_forward_seed_mode()
    return seed == "gt"
