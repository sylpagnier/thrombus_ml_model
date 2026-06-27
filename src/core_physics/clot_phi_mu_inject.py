"""MLP clot-phi -> mu_eff coupling for GNODE / GINO-DEQ closed loop.

Leg B v1 (``BIOCHEM_MLP_CLOT_INJECT=1``):
  soft phi trigger -> constant clot mu blend into learned mu_eff.

Leg B v2 (``BIOCHEM_MLP_MU_MAP=1``):
  ``mu_eff = Carreau(u,v) + clot_mask * (mu_mlp - Carreau)`` — no GNODE mu head in forward.

Env:
  BIOCHEM_MLP_CLOT_INJECT=1            # v1 trigger inject (legacy)
  BIOCHEM_MLP_MU_MAP=1                 # v2 MLP mu map (preferred Leg B)
  BIOCHEM_MLP_MU_MAP_PHI_GATE=1        # v2: gate MLP overlay (default on)
  BIOCHEM_MLP_MU_MAP_MASK=gt_clot      # gt_clot|neighbor|phi|adaptive_phi|excess|ratio
  BIOCHEM_MLP_NEIGHBOR_SEED=pred_clot       # neighbor: prev mu seeds (+ optional phi)
  BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI=1   # neighbor: also require phi >= thresh inside band
  BIOCHEM_MLP_MU_MAP_BULK=cap_low_shear  # cap_low_shear|mu_inf|carreau (bulk baseline)
  BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND=0.01 # cap_low_shear: use mu_inf below this gamma_dot
  BIOCHEM_MLP_MU_MAP_GEO_CAP=0         # v2: also multiply by REGION mask (default off)
  BIOCHEM_MLP_CLOT_CKPT=outputs/biochem/clot_baseline/clot_phi_best.pth
  BIOCHEM_MLP_CLOT_MU_SI=0.10          # v1 only: constant clot level [Pa*s]
  BIOCHEM_MLP_CLOT_BLEND=1.0           # clot mix strength (v1 + v2)
  BIOCHEM_MLP_CLOT_PHI_THRESH=0.0      # v1 hard floor; v2 hard floor when PHI_GATE=1
  BIOCHEM_MLP_CLOT_REGION=supervision  # supervision|neighbor_wall|phi_seed (geo cap only)
  BIOCHEM_MLP_CLOT_USE_PRED_SPECIES=1  # patch rollout species into y slice for features
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.utils import species_channels as sc
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    carreau_mu_si_from_uv,
    clot_phi_hybrid_enabled,
    clot_phi_mu_cap_si,
    gt_mu_anchor_cap_si,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    neighbor_supervision_mask,
    phi_gt_binary,
    supervision_region_mask,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.utils.paths import get_project_root
from src.utils.rheology import compute_shear_rate


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def biochem_mlp_clot_inject_enabled() -> bool:
    return _env_bool("BIOCHEM_MLP_CLOT_INJECT", False)


def biochem_mlp_mu_map_enabled() -> bool:
    """Leg B v2: MLP owns mu_eff in neighbor_wall; Carreau bulk elsewhere."""
    return _env_bool("BIOCHEM_MLP_MU_MAP", False)


def biochem_mlp_coupling_enabled() -> bool:
    return biochem_mlp_clot_inject_enabled() or biochem_mlp_mu_map_enabled()


def resolve_mlp_clot_ckpt(user_path: str | Path | None = None) -> Path | None:
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = get_project_root() / p
        return p if p.is_file() else None
    raw = (os.environ.get("BIOCHEM_MLP_CLOT_CKPT") or "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else get_project_root() / p)
    root = get_project_root()
    candidates.extend(
        [
            root / "outputs/biochem/clot_baseline/clot_phi_best.pth",
            root / "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
            root / "outputs/biochem/clot_phi_best.pth",
        ]
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def mlp_mu_map_phi_gate_enabled() -> bool:
    """v2 default: commit MLP mu only where phi > 0 (clot map), not entire wall band."""
    if not biochem_mlp_mu_map_enabled():
        return False
    return _env_bool("BIOCHEM_MLP_MU_MAP_PHI_GATE", True)


def mlp_mu_map_geo_cap_enabled() -> bool:
    """v2 optional: also require REGION mask (supervision / neighbor_wall / phi_seed)."""
    return biochem_mlp_mu_map_enabled() and _env_bool("BIOCHEM_MLP_MU_MAP_GEO_CAP", False)


def mlp_clot_region_mode() -> str:
    default = "supervision"
    raw = (os.environ.get("BIOCHEM_MLP_CLOT_REGION") or default).strip().lower()
    if raw in ("neighbor", "neighbor_wall", "wall"):
        return "neighbor_wall"
    if raw in ("phi", "phi_seed", "clot"):
        return "phi_seed"
    return "supervision"


def mlp_clot_blend_alpha() -> float:
    return max(0.0, min(_env_float("BIOCHEM_MLP_CLOT_BLEND", 1.0), 1.0))


def mlp_clot_mu_si_default() -> float:
    raw = (os.environ.get("BIOCHEM_MLP_CLOT_MU_SI") or "").strip()
    if raw:
        return max(_env_float("BIOCHEM_MLP_CLOT_MU_SI", clot_phi_mu_cap_si()), 1e-6)
    return clot_phi_mu_cap_si()


def mlp_clot_phi_thresh() -> float:
    return max(0.0, min(_env_float("BIOCHEM_MLP_CLOT_PHI_THRESH", 0.0), 1.0))


def mlp_mu_map_phi_thresh() -> float:
    """phi|ratio mask floor when ``BIOCHEM_MLP_MU_MAP_MASK=phi``."""
    raw = (os.environ.get("BIOCHEM_MLP_MU_MAP_PHI_THRESH") or "").strip()
    if raw:
        return max(0.0, min(float(raw), 1.0))
    return 0.5


def normalize_mlp_mu_map_mask_mode(raw: str | None) -> str:
    """Normalize mask mode string (env value or override)."""
    key = (raw or "gt_clot").strip().lower()
    if key in ("gt_clot", "gt", "clot_gt"):
        return "gt_clot"
    if key in ("neighbor", "pred_neighbor", "deploy", "wall_hop"):
        return "neighbor"
    if key in ("seed_growth", "restricted_growth", "gt_seed_growth", "seed_grow"):
        return "seed_growth"
    if key in ("mlp_band", "mlp", "clot_mlp", "band_mlp", "wired"):
        return "mlp_band"
    if key in ("phi", "pred_phi"):
        return "phi"
    if key in ("adaptive_phi", "phi_adaptive", "phi_q"):
        return "adaptive_phi"
    if key in ("excess", "delta_mu", "mu_excess"):
        return "excess"
    if key in ("ratio", "mu_ratio"):
        return "ratio"
    return "gt_clot"


def mlp_mu_map_mask_mode() -> str:
    """Deploy trigger for v2 MLP mu overlay (default: GT clot seeds when labels exist)."""
    return normalize_mlp_mu_map_mask_mode(os.environ.get("BIOCHEM_MLP_MU_MAP_MASK"))


def mlp_mu_map_uses_gt_labels() -> bool:
    """True when the v2 commit mask reads COMSOL mu / clot labels every step."""
    return mlp_mu_map_mask_mode() == "gt_clot"


def mlp_seed_growth_hops() -> int:
    raw = (
        os.environ.get("BIOCHEM_MLP_DEPLOY_VISION_GROW_HOPS")
        or os.environ.get("BIOCHEM_MLP_SEED_GROWTH_HOPS")
        or "1"
    ).strip()
    return max(int(float(raw)), 0)


def mlp_deploy_vision_restrict_enabled() -> bool:
    """Neighbor deploy: cap commit vision (t=0 GT supervision band, optional growth)."""
    if not biochem_mlp_mu_map_enabled():
        return False
    mode = mlp_mu_map_mask_mode()
    if mode in ("seed_growth", "mlp_band"):
        return True
    if mode != "neighbor":
        return False
    return _env_bool("BIOCHEM_MLP_DEPLOY_VISION_RESTRICT", True)


def mlp_deploy_vision_grow_enabled() -> bool:
    if not mlp_deploy_vision_restrict_enabled():
        return False
    return _env_bool("BIOCHEM_MLP_DEPLOY_VISION_GROW", True)


def mlp_deploy_no_commit_at_t0() -> bool:
    """Deploy modes: macro step 0 = no MLP mu commit (physically no clot at t=0)."""
    if not biochem_mlp_mu_map_enabled() or mlp_mu_map_uses_gt_labels():
        return False
    mode = mlp_mu_map_mask_mode()
    if mode not in ("neighbor", "seed_growth", "mlp_band"):
        return False
    return _env_bool("BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0", True)


def deploy_vision_init_time_index(macro_label_index: int) -> int:
    raw = (
        os.environ.get("BIOCHEM_MLP_DEPLOY_VISION_INIT")
        or os.environ.get("BIOCHEM_MLP_SEED_GROWTH_INIT")
        or "comsol_t0"
    ).strip().lower()
    if raw in ("comsol_t0", "t0", "0"):
        return 0
    if raw in ("eval", "eval_step", "macro", "macro_t0", "label"):
        return int(macro_label_index)
    return 0


def init_deploy_supervision_vision_mask(
    data,
    device: torch.device,
    time_index: int,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
) -> torch.Tensor:
    """Initial deploy vision = clot-phi supervision region from GT (wall, 1-hop, dgamma, shear).

    Uses ``build_clot_phi_step`` / ``supervision_region_mask`` so ``CLOT_PHI_DGAMMA_SLICE``,
    ``CLOT_PHI_MASK_MODE``, shear knobs, etc. match clot-phi training (GT at init time only).
    """
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    ti = deploy_vision_init_time_index(int(time_index))
    step = build_clot_phi_step(data, ti, phys, bio, device)
    return step.region.reshape(-1).bool()


def init_seed_growth_allowed_mask(
    data,
    device: torch.device,
    time_index: int,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    y_reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias: seed-growth vision = GT supervision band at init (not sparse gt_clot commit)."""
    _ = y_reference  # supervision uses ``data.y[ti]``; kept for API compat.
    return init_deploy_supervision_vision_mask(
        data, device, time_index, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )


def expand_seed_growth_allowed_mask(
    allowed: torch.Tensor,
    mu_committed_si: torch.Tensor,
    data,
    device: torch.device,
    *,
    phys_cfg: PhysicsConfig | None = None,
) -> torch.Tensor:
    """Grow vision only where pred clot (committed mu) lies inside current allowed mask."""
    from src.core_physics.clot_phi_simple import _graph_dilate

    phys = phys_cfg or PhysicsConfig(phase="biochem")
    allowed_b = allowed.reshape(-1).to(device=device).bool()
    mu = mu_committed_si.reshape(-1).to(device=device)
    thr = resolve_clot_mu_commit_thresh_si(phys)
    pred_clot = allowed_b & (mu >= thr)
    if not bool(pred_clot.any().item()):
        return allowed_b
    h = mlp_seed_growth_hops()
    grown = pred_clot.clone()
    ei = data.edge_index.to(device=device)
    for _ in range(h):
        grown = _graph_dilate(grown, ei)
    return allowed_b | grown


def resolve_deploy_mlp_band_commit_mask(
    allowed: torch.Tensor,
    phi: torch.Tensor,
    mu_mlp_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    *,
    phys_cfg: PhysicsConfig | None = None,
) -> torch.Tensor:
    """Deploy: commit MLP mu inside allowed vision when phi + mu_mlp pass thresholds.

    Mirrors offline clot-map readout (hybrid MLP mu) but capped to ``allowed`` — no GT
    clot labels, no wall-nucleation flood. Intended for closed-loop Leg B wiring.
    """
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    device = phi.device
    allowed_b = allowed.reshape(-1).to(device=device).bool()
    phi_ok = phi.reshape(-1).to(device=device) >= mlp_mu_map_phi_thresh()
    mu_thr = resolve_clot_mu_commit_thresh_si(phys)
    mu_ok = mu_mlp_si.reshape(-1).to(device=device) >= mu_thr
    gate = allowed_b & phi_ok & mu_ok
    excess_min = mlp_deploy_mu_excess_si()
    if excess_min > 0.0:
        excess = mu_mlp_si.reshape(-1).to(device=device) - mu_c_si.reshape(-1).to(device=device)
        gate = gate & (excess >= excess_min)
    if _env_bool("BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS", True):
        gate = gate & mu_ok
    return gate.reshape(-1).bool()


def resolve_deploy_seed_growth_commit_mask(
    allowed: torch.Tensor,
    phi: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Commit only inside the current allowed vision mask (optional phi gate)."""
    allowed_b = allowed.reshape(-1).to(device=device).bool()
    gate = allowed_b.clone()
    if mlp_neighbor_require_phi():
        phi_flat = phi.reshape(-1).to(device=device)
        gate = gate & (phi_flat >= mlp_mu_map_phi_thresh())
    return gate.reshape(-1).bool()


def mlp_neighbor_seed_modes() -> set[str]:
    """Seed sources for ``neighbor`` deploy mask (no GT)."""
    raw = (os.environ.get("BIOCHEM_MLP_NEIGHBOR_SEED") or "pred_clot").strip().lower()
    out: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if p in ("pred_clot", "clot", "mu", "rollout"):
            out.add("pred_clot")
        elif p in ("phi", "pred_phi"):
            out.add("phi")
        elif p in ("wall",):
            out.add("wall")
    return out or {"pred_clot"}


def mlp_neighbor_require_phi() -> bool:
    return _env_bool("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", True)


def mlp_deploy_dgamma_slice_enabled() -> bool:
    if not biochem_mlp_mu_map_enabled() or mlp_mu_map_mask_mode() != "neighbor":
        return False
    return _env_bool("BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE", False)


def mlp_deploy_dgamma_wall_min_si() -> float:
    raw = (os.environ.get("BIOCHEM_MLP_DEPLOY_DGAMMA_WALL_MIN_SI") or "").strip()
    if raw:
        return max(float(raw), 0.0)
    from src.core_physics.clot_phi_simple import clot_phi_dgamma_wall_min_si

    return clot_phi_dgamma_wall_min_si()


def mlp_deploy_phi_quantile() -> float:
    return max(0.0, min(_env_float("BIOCHEM_MLP_DEPLOY_PHI_Q", 0.0), 1.0))


def mlp_neighbor_growth_only_when_clot() -> bool:
    return _env_bool("BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY", False)


def mlp_deploy_mu_excess_si() -> float:
    return max(0.0, _env_float("BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI", 0.0))


def resolve_clot_mu_commit_thresh_si(phys_cfg: PhysicsConfig) -> float:
    """Binary clot threshold for pred-clot seeds (matches scorecard default)."""
    override = (os.environ.get("CLOT_SHAPE_MU_THRESH_SI") or "").strip()
    if override:
        return max(float(override), float(phys_cfg.mu_inf))
    return max(0.055, float(phys_cfg.mu_inf) * 1.2)


def resolve_deploy_neighbor_commit_mask(
    data,
    device: torch.device,
    *,
    phi: torch.Tensor,
    prev_mu_eff_si: torch.Tensor | None = None,
    phys_cfg: PhysicsConfig | None = None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    bio_cfg: BiochemConfig | None = None,
    mu_c_si: torch.Tensor | None = None,
    mu_mlp_si: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deploy Leg B commit mask without COMSOL mu labels.

    Physics-first nucleation (pred ``u,v`` only):
      - Wall adhesion band: ``mask_wall`` with ``-d(gamma)/dx >=`` wall min (COMSOL band).
      - High-phi on that band only (optional top-``BIOCHEM_MLP_DEPLOY_PHI_Q`` quantile).
      - Clot growth: prev committed ``mu >=`` threshold + 1-hop neighbors.
      - After clots exist: optional growth-only (no new wall nucleation).
    """
    from src.core_physics.clot_phi_simple import (
        _graph_dilate,
        _lumen_supervision_eligible,
        _wall_mask_from_data,
        pred_neg_dgamma_dx_phys,
    )

    n = int(data.num_nodes)
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    phi_flat = phi.reshape(-1).to(device=device)
    phi_thr = mlp_mu_map_phi_thresh()
    wall = _wall_mask_from_data(data, device, n)
    eligible = _lumen_supervision_eligible(data, device, wall, n)
    ei = data.edge_index.to(device=device)

    clot_seed = torch.zeros(n, device=device, dtype=torch.bool)
    modes = mlp_neighbor_seed_modes()
    if "pred_clot" in modes and prev_mu_eff_si is not None:
        thr = resolve_clot_mu_commit_thresh_si(phys)
        clot_seed = clot_seed | (prev_mu_eff_si.reshape(-1).to(device=device) >= thr)
    if "phi" in modes:
        clot_seed = clot_seed | (phi_flat >= phi_thr)

    active_seed = clot_seed & eligible
    h_touch = max(int(_env_float("CLOT_PHI_CLOT_TOUCH_HOPS", 1)), 0)
    near = active_seed.clone()
    for _ in range(h_touch):
        near = _graph_dilate(near, ei)
    lumen_band = near & ~wall & ~clot_seed & eligible

    wall_adhesion = wall & eligible
    if mlp_deploy_dgamma_slice_enabled() and u_nd is not None and v_nd is not None:
        neg_dx = pred_neg_dgamma_dx_phys(data, u_nd, v_nd, bio, device)
        wall_adhesion = wall_adhesion & (neg_dx >= mlp_deploy_dgamma_wall_min_si())

    phi_q = mlp_deploy_phi_quantile()
    if phi_q > 0.0 and bool(wall_adhesion.any().item()):
        qthr = max(phi_thr, float(phi_flat[wall_adhesion].quantile(phi_q).item()))
        wall_nucleation = wall_adhesion & (phi_flat >= qthr)
    else:
        wall_nucleation = wall_adhesion & (phi_flat >= phi_thr)

    if mlp_neighbor_growth_only_when_clot() and bool(active_seed.any().item()):
        commit = active_seed | lumen_band
    else:
        commit = wall_nucleation | active_seed | lumen_band

    if mlp_neighbor_require_phi():
        commit = commit & (phi_flat >= phi_thr)

    excess_min = mlp_deploy_mu_excess_si()
    if excess_min > 0.0 and mu_c_si is not None and mu_mlp_si is not None:
        excess = mu_mlp_si.reshape(-1).to(device=device) - mu_c_si.reshape(-1).to(device=device)
        commit = commit & (excess >= excess_min)

    # Deploy: do not commit weak MLP levels that barely cross clot eval threshold.
    if mu_mlp_si is not None and _env_bool("BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS", True):
        mu_thr = resolve_clot_mu_commit_thresh_si(phys)
        commit = commit & (mu_mlp_si.reshape(-1).to(device=device) >= mu_thr)

    return commit.reshape(-1).bool()


def mlp_mu_map_phi_soft_gate() -> bool:
    """When true, use soft phi in [0,1]; else binary mask phi >= thresh."""
    return biochem_mlp_mu_map_enabled() and _env_bool("BIOCHEM_MLP_MU_MAP_PHI_SOFT", False)


def mask_only_mu_bulk_mode() -> str:
    """Shared bulk baseline for Leg B v2 and Leg C neighbor_wall (cap_low_shear default)."""
    raw = (
        os.environ.get("BIOCHEM_MLP_MU_MAP_BULK")
        or os.environ.get("BIOCHEM_MU_NEIGHBOR_WALL_BULK")
        or "cap_low_shear"
    ).strip().lower()
    if raw in ("cap_low_shear", "cap", "low_shear_cap"):
        return "cap_low_shear"
    if raw in ("mu_inf", "inf", "constant"):
        return "mu_inf"
    return "carreau"


def mlp_mu_map_bulk_mode() -> str:
    """Bulk Carreau baseline for v2 (default fixes graph low-shear mu_0 plateau)."""
    return mask_only_mu_bulk_mode()


def neighbor_wall_mu_mask_mode() -> str:
    """Leg C overlay mask: neighbor_wall | gt_clot | supervision."""
    raw = (os.environ.get("BIOCHEM_MU_NEIGHBOR_WALL_MASK") or "neighbor_wall").strip().lower()
    if raw in ("gt_clot", "gt", "clot_gt"):
        return "gt_clot"
    if raw in ("supervision", "region"):
        return "supervision"
    return "neighbor_wall"


def compose_mask_only_mu_eff(
    mu_bulk_si: torch.Tensor,
    mu_overlay_si: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``mu = mu_bulk + mask * (mu_overlay - mu_bulk)`` (Leg B/C shared commit)."""
    mb = mu_bulk_si.reshape(-1, 1).to(dtype=torch.float32)
    mo = mu_overlay_si.reshape(-1, 1).to(dtype=mb.dtype)
    gate = mask.reshape(-1, 1).to(dtype=mb.dtype).clamp(0.0, 1.0)
    return (mb + gate * (mo - mb)).clamp(min=1e-8)


def resolve_neighbor_wall_mu_mask(
    data,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    *,
    time_index: int,
    y_slice: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary mask for Leg C GNODE mu overlay (not full-domain learned mu)."""
    mode = neighbor_wall_mu_mask_mode()
    if mode == "neighbor_wall":
        from src.core_physics.clot_phi_simple import wall_supervision_mask

        return wall_supervision_mask(data, device).reshape(-1).bool()
    if y_slice is None or not hasattr(data, "y") or data.y is None:
        mode = "neighbor_wall"
        from src.core_physics.clot_phi_simple import wall_supervision_mask

        return wall_supervision_mask(data, device).reshape(-1).bool()
    y = y_slice.to(device)
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap = cap_mu_eff_si(mu_gt)
    region = supervision_region_mask(data, device, mu_cap, phys_cfg)
    if mode == "gt_clot":
        anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
        return phi_gt_binary(mu_cap, region, phys_cfg, mu_anchor_si=anchor).reshape(-1).bool()
    return region.reshape(-1).bool()


def mlp_mu_map_gamma_thresh_nd() -> float:
    return max(_env_float("BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND", 0.01), 1e-8)


def gamma_dot_nd_from_uv(data, u_nd: torch.Tensor, v_nd: torch.Tensor) -> torch.Tensor:
    u = u_nd.reshape(-1, 1).to(dtype=torch.float32)
    v = v_nd.reshape(-1, 1).to(dtype=torch.float32)
    du_dx = torch.sparse.mm(data.G_x, u)
    du_dy = torch.sparse.mm(data.G_y, u)
    dv_dx = torch.sparse.mm(data.G_x, v)
    dv_dy = torch.sparse.mm(data.G_y, v)
    return compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).reshape(-1)


def mu_map_carreau_baseline_si(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """v2 bulk baseline: Carreau with optional low-shear cap at mu_inf (COMSOL-export aligned)."""
    mu_bulk, _ = resolve_mu_map_baselines_si(data, u_nd, v_nd, phys_cfg)
    return mu_bulk


def resolve_mu_map_baselines_si(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mu_bulk_baseline, mu_mlp_anchor) for v2 commit.

    Hybrid clot MLP was trained against uncapped Carreau (~mu_0 low-shear plateau).
    Bulk display/commit uses capped baseline (mu_inf low-shear); clot overlay keeps raw anchor.
    """
    mu_raw = carreau_mu_si_from_uv(data, u_nd.reshape(-1), v_nd.reshape(-1), phys_cfg).reshape(-1)
    mode = mlp_mu_map_bulk_mode()
    if mode == "carreau":
        return mu_raw, mu_raw
    mu_inf = torch.full_like(mu_raw, float(phys_cfg.mu_inf))
    if mode == "mu_inf":
        return mu_inf, mu_raw
    gd = gamma_dot_nd_from_uv(data, u_nd, v_nd)
    low = gd < mlp_mu_map_gamma_thresh_nd()
    mu_bulk = torch.where(low, mu_inf, mu_raw)
    return mu_bulk, mu_raw


def resolve_clot_trigger_gate(
    phi: torch.Tensor,
    mu_c_si: torch.Tensor,
    mu_mlp_si: torch.Tensor,
    *,
    region: torch.Tensor | None = None,
    mu_gt_cap_si: torch.Tensor | None = None,
    phys_cfg: PhysicsConfig | None = None,
    graph_data=None,
    prev_mu_eff_si: torch.Tensor | None = None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    bio_cfg: BiochemConfig | None = None,
    mask_mode: str | None = None,
    allowed_commit_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary (or soft-phi) gate [N,1] for where MLP mu replaces Carreau baseline."""
    mode = normalize_mlp_mu_map_mask_mode(mask_mode) if mask_mode else mlp_mu_map_mask_mode()
    dtype = mu_c_si.reshape(-1, 1).dtype
    device = mu_c_si.device

    if mode == "gt_clot":
        if (
            mu_gt_cap_si is not None
            and region is not None
            and phys_cfg is not None
            and graph_data is not None
        ):
            anchor = gt_mu_anchor_cap_si(graph_data, phys_cfg, device)
            gt = phi_gt_binary(mu_gt_cap_si, region, phys_cfg, mu_anchor_si=anchor)
            return gt.reshape(-1, 1).to(dtype=dtype)
        mode = "adaptive_phi"

    if mode == "neighbor":
        if graph_data is None:
            return torch.zeros(mu_c_si.reshape(-1).numel(), 1, device=device, dtype=dtype)
        gate = resolve_deploy_neighbor_commit_mask(
            graph_data,
            device,
            phi=phi,
            prev_mu_eff_si=prev_mu_eff_si,
            phys_cfg=phys_cfg,
            u_nd=u_nd,
            v_nd=v_nd,
            bio_cfg=bio_cfg,
            mu_c_si=mu_c_si,
            mu_mlp_si=mu_mlp_si,
        )
        if allowed_commit_mask is not None:
            gate = gate & allowed_commit_mask.reshape(-1).to(device=device).bool()
        return gate.reshape(-1, 1).to(dtype=dtype)

    if mode == "seed_growth":
        if allowed_commit_mask is None:
            return torch.zeros(mu_c_si.reshape(-1).numel(), 1, device=device, dtype=dtype)
        gate = resolve_deploy_seed_growth_commit_mask(
            allowed_commit_mask, phi, device=device
        )
        return gate.reshape(-1, 1).to(dtype=dtype)

    if mode == "mlp_band":
        if allowed_commit_mask is None:
            return torch.zeros(mu_c_si.reshape(-1).numel(), 1, device=device, dtype=dtype)
        gate = resolve_deploy_mlp_band_commit_mask(
            allowed_commit_mask,
            phi,
            mu_mlp_si,
            mu_c_si,
            phys_cfg=phys_cfg,
        )
        return gate.reshape(-1, 1).to(dtype=dtype)

    if mode == "phi":
        phi_col = phi.reshape(-1, 1).clamp(0.0, 1.0)
        if mlp_mu_map_phi_soft_gate():
            gate = phi_col
            thresh = mlp_mu_map_phi_thresh()
            if thresh > 0.0:
                gate = gate * (phi_col >= thresh).to(dtype=gate.dtype)
            return gate
        return (phi_col >= mlp_mu_map_phi_thresh()).to(dtype=dtype)

    if mode == "adaptive_phi":
        phi_flat = phi.reshape(-1)
        q = max(0.0, min(_env_float("BIOCHEM_MLP_MU_MAP_PHI_Q", 0.998), 1.0))
        thr = max(mlp_mu_map_phi_thresh(), float(phi_flat.quantile(q).item()))
        return (phi_flat >= thr).reshape(-1, 1).to(dtype=dtype)

    if mode == "excess":
        excess = mu_mlp_si.reshape(-1) - mu_c_si.reshape(-1)
        reg = region.reshape(-1).bool() if region is not None else torch.ones_like(excess, dtype=torch.bool)
        if not bool(reg.any()):
            return torch.zeros(excess.numel(), 1, device=device, dtype=dtype)
        q = max(0.0, min(_env_float("BIOCHEM_MLP_MU_MAP_EXCESS_Q", 0.99), 1.0))
        thr = float(excess[reg].quantile(q).item())
        gate = (excess >= thr) & reg
        return gate.reshape(-1, 1).to(dtype=dtype)

    if mode == "ratio":
        ratio = mu_mlp_si.reshape(-1) / mu_c_si.reshape(-1).clamp(min=1e-8)
        thr = max(1.0, _env_float("BIOCHEM_MLP_MU_MAP_RATIO_THRESH", 1.5))
        gate = ratio >= thr
        if region is not None:
            gate = gate & region.reshape(-1).bool()
        return gate.reshape(-1, 1).to(dtype=dtype)

    return torch.ones(mu_c_si.reshape(-1).numel(), 1, device=device, dtype=dtype)


def assemble_committed_mu_map(
    mu_c_si: torch.Tensor,
    mu_mlp_si: torch.Tensor,
    phi: torch.Tensor,
    *,
    region: torch.Tensor | None = None,
    mu_gt_cap_si: torch.Tensor | None = None,
    phys_cfg: PhysicsConfig | None = None,
    graph_data=None,
    prev_mu_eff_si: torch.Tensor | None = None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    bio_cfg: BiochemConfig | None = None,
    allowed_commit_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Carreau baseline + MLP clot mu only on the resolved clot trigger mask."""
    mu_c = mu_c_si.reshape(-1, 1).to(dtype=torch.float32)
    mu_mlp = mu_mlp_si.reshape(-1, 1).to(dtype=mu_c.dtype)
    alpha = mlp_clot_blend_alpha()
    if alpha <= 0.0:
        return mu_c.clamp(min=1e-8)

    if mlp_mu_map_phi_gate_enabled():
        gate = resolve_clot_trigger_gate(
            phi,
            mu_c_si,
            mu_mlp_si,
            region=region,
            mu_gt_cap_si=mu_gt_cap_si,
            phys_cfg=phys_cfg,
            graph_data=graph_data,
            prev_mu_eff_si=prev_mu_eff_si,
            u_nd=u_nd,
            v_nd=v_nd,
            bio_cfg=bio_cfg,
            allowed_commit_mask=allowed_commit_mask,
        )
    else:
        gate = torch.ones_like(mu_c)

    mix = (alpha * gate).clamp(0.0, 1.0)
    if mlp_mu_map_geo_cap_enabled() and region is not None:
        mix = mix * region.reshape(-1, 1).to(dtype=mix.dtype)
    return (mu_c + mix * (mu_mlp - mu_c)).clamp(min=1e-8)


def mlp_clot_use_pred_species() -> bool:
    return _env_bool("BIOCHEM_MLP_CLOT_USE_PRED_SPECIES", True)


@torch.no_grad()
def committed_mu_mesh_from_clot_model(
    clot_model: nn.Module,
    data,
    time_index: int,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    species_log: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-mesh v2 mu: Carreau baseline + MLP clot overlay. Returns (mu_si, mu_c, phi, mu_mlp)."""
    y_slice = data.y[time_index].to(device).clone()
    y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
    y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
    if mlp_clot_use_pred_species():
        y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
    step = build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        u_nd_override=u_nd.reshape(-1),
        v_nd_override=v_nd.reshape(-1),
        y_slice_override=y_slice,
    )
    mu_c_bulk, mu_c_mlp = resolve_mu_map_baselines_si(data, u_nd, v_nd, phys_cfg)
    if clot_phi_hybrid_enabled() and hasattr(clot_model, "forward_delta_log_mu"):
        phi = torch.sigmoid(clot_model.forward_logits(step.features)).reshape(-1)
        mu_mlp = mu_eff_from_delta_log_si(
            mu_c_mlp, clot_model.forward_delta_log_mu(step.features)
        )
    else:
        phi = clot_model(step.features).reshape(-1)
        mu_mlp = log_blend_mu_eff_si(mu_c_mlp, phi)
    mu_c = mu_c_bulk.reshape(-1)
    if biochem_mlp_mu_map_enabled() and mlp_mu_map_phi_gate_enabled():
        mu_si = assemble_committed_mu_map(
            mu_c_bulk,
            mu_mlp,
            phi,
            region=step.region,
            mu_gt_cap_si=step.mu_gt_cap,
            phys_cfg=phys_cfg,
        ).reshape(-1)
    else:
        mu_si = mu_mlp.reshape(-1)
    return mu_si, mu_c, phi, mu_mlp.reshape(-1)


@torch.no_grad()
def compute_mlp_commit_gates_at_rollout_frame(
    clot_model: nn.Module,
    data,
    time_index: int,
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    species_log: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    prev_mu_eff_si: torch.Tensor | None = None,
    allowed_commit_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (oracle gt_clot gate, active-env gate) as bool [N] from rollout state."""
    y_slice = data.y[time_index].to(device).clone()
    y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
    y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
    if mlp_clot_use_pred_species():
        y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
    step = build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        u_nd_override=u_nd.reshape(-1),
        v_nd_override=v_nd.reshape(-1),
        y_slice_override=y_slice,
    )
    mu_c_bulk, mu_c_mlp = resolve_mu_map_baselines_si(data, u_nd, v_nd, phys_cfg)
    if clot_phi_hybrid_enabled() and hasattr(clot_model, "forward_delta_log_mu"):
        phi = torch.sigmoid(clot_model.forward_logits(step.features)).reshape(-1)
        mu_mlp = mu_eff_from_delta_log_si(
            mu_c_mlp, clot_model.forward_delta_log_mu(step.features)
        )
    else:
        phi = clot_model(step.features).reshape(-1)
        mu_mlp = log_blend_mu_eff_si(mu_c_mlp, phi)
    mu_c = mu_c_bulk.reshape(-1)
    common = dict(
        phi=phi,
        mu_c_si=mu_c,
        mu_mlp_si=mu_mlp.reshape(-1),
        region=step.region,
        mu_gt_cap_si=step.mu_gt_cap,
        phys_cfg=phys_cfg,
        graph_data=data,
        prev_mu_eff_si=prev_mu_eff_si,
        u_nd=u_nd.reshape(-1),
        v_nd=v_nd.reshape(-1),
        bio_cfg=bio_cfg,
    )
    gate_gt = resolve_clot_trigger_gate(**common, mask_mode="gt_clot").reshape(-1).bool()
    if mlp_mu_map_mask_mode() == "seed_growth":
        if allowed_commit_mask is None:
            allowed_commit_mask = init_seed_growth_allowed_mask(
                data,
                device,
                time_index,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
            )
        gate_active = resolve_deploy_seed_growth_commit_mask(
            allowed_commit_mask, phi, device=device
        )
    else:
        gate_active = resolve_clot_trigger_gate(
            **common,
            allowed_commit_mask=allowed_commit_mask,
        ).reshape(-1).bool()
    return gate_gt, gate_active


@dataclass
class DeployGateFrameDiagnostics:
    """Per-macro-step deploy commit gate breakdown (inside ``allowed`` vision)."""

    macro_step: int = 0
    time_index: int = 0
    t_si: float = 0.0
    mask_mode: str = ""
    n_allowed: int = 0
    n_supervision_at_t: int = 0
    n_gt_clot_in_allowed: int = 0
    n_gt_clot_supervision_t: int = 0
    frac_phi_ge_thr_in_allowed: float = 0.0
    frac_mu_mlp_ge_thr_in_allowed: float = 0.0
    frac_both_in_allowed: float = 0.0
    frac_commit_in_allowed: float = 0.0
    frac_commit_mesh: float = 0.0
    frac_rollout_mu_ge_thr_in_allowed: float = 0.0
    phi_p50_allowed: float = 0.0
    phi_p90_allowed: float = 0.0
    mu_mlp_p50_allowed: float = 0.0
    mu_mlp_p90_allowed: float = 0.0
    phi_thr: float = 0.5
    mu_thr_si: float = 0.055
    bottleneck: str = ""
    no_commit_t0: bool = False

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _frac_true(mask: torch.Tensor, pool: torch.Tensor | None = None) -> float:
    m = mask.reshape(-1).bool()
    if pool is not None:
        m = m & pool.reshape(-1).bool()
    n = int(m.sum().item())
    denom = int(pool.reshape(-1).sum().item()) if pool is not None else int(m.numel())
    if denom <= 0:
        return 0.0
    return float(n / denom)


def _quantile_in_pool(values: torch.Tensor, pool: torch.Tensor, q: float) -> float:
    v = values.reshape(-1).to(dtype=torch.float32)
    p = pool.reshape(-1).bool()
    if not bool(p.any().item()):
        return 0.0
    return float(v[p].quantile(q).item())


def _deploy_gate_bottleneck(
    *,
    n_allowed: int,
    frac_phi: float,
    frac_mu: float,
    frac_both: float,
    frac_commit: float,
    no_commit_t0: bool,
) -> str:
    if no_commit_t0:
        return "no_commit_t0"
    if n_allowed <= 0:
        return "empty_allowed"
    if frac_commit > 0.0:
        return "commit_ok"
    if frac_both > 0.0:
        return "both_ok_zero_commit"
    if frac_phi <= 0.0 and frac_mu <= 0.0:
        return "phi_and_mu_low"
    if frac_phi <= 0.0:
        return "phi_low"
    if frac_mu <= 0.0:
        return "mu_mlp_low"
    return "phi_mu_disjoint"


@torch.no_grad()
def predict_mlp_fields_at_rollout_frame(
    clot_model: nn.Module,
    data,
    time_index: int,
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    species_log: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Rollout-state MLP fields: phi, mu_mlp, mu_c, supervision region, mu_gt_cap."""
    y_slice = data.y[time_index].to(device).clone()
    y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
    y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
    if mlp_clot_use_pred_species():
        y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
    step = build_clot_phi_step(
        data,
        time_index,
        phys_cfg,
        bio_cfg,
        device,
        u_nd_override=u_nd.reshape(-1),
        v_nd_override=v_nd.reshape(-1),
        y_slice_override=y_slice,
    )
    mu_c_bulk, mu_c_mlp = resolve_mu_map_baselines_si(data, u_nd, v_nd, phys_cfg)
    if clot_phi_hybrid_enabled() and hasattr(clot_model, "forward_delta_log_mu"):
        phi = torch.sigmoid(clot_model.forward_logits(step.features)).reshape(-1)
        mu_mlp = mu_eff_from_delta_log_si(
            mu_c_mlp, clot_model.forward_delta_log_mu(step.features)
        )
    else:
        phi = clot_model(step.features).reshape(-1)
        mu_mlp = log_blend_mu_eff_si(mu_c_mlp, phi)
    return {
        "phi": phi.reshape(-1),
        "mu_mlp": mu_mlp.reshape(-1),
        "mu_c": mu_c_bulk.reshape(-1),
        "region": step.region.reshape(-1).bool(),
        "mu_gt_cap": step.mu_gt_cap.reshape(-1),
    }


@torch.no_grad()
def compute_deploy_gate_frame_diagnostics(
    clot_model: nn.Module,
    data,
    time_index: int,
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    species_log: torch.Tensor,
    mu_rollout_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    allowed_commit_mask: torch.Tensor | None = None,
    macro_step_index: int = 0,
    t_si: float = 0.0,
    prev_mu_eff_si: torch.Tensor | None = None,
) -> DeployGateFrameDiagnostics:
    """Break down why deploy commit fires (or not) inside the vision band."""
    fields = predict_mlp_fields_at_rollout_frame(
        clot_model,
        data,
        time_index,
        u_nd=u_nd,
        v_nd=v_nd,
        species_log=species_log,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
    )
    phi = fields["phi"]
    mu_mlp = fields["mu_mlp"]
    mu_c = fields["mu_c"]
    region_t = fields["region"]
    mu_gt_cap = fields["mu_gt_cap"]

    allowed = allowed_commit_mask
    if mlp_deploy_vision_restrict_enabled():
        if allowed is None and int(macro_step_index) == 0:
            allowed = init_deploy_supervision_vision_mask(
                data, device, time_index, phys_cfg=phys_cfg, bio_cfg=bio_cfg
            )
        elif allowed is None:
            allowed = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)

    phi_thr = mlp_mu_map_phi_thresh()
    mu_thr = resolve_clot_mu_commit_thresh_si(phys_cfg)
    phi_ok = phi >= phi_thr
    mu_ok = mu_mlp >= mu_thr
    both_ok = phi_ok & mu_ok

    gt_clot = mu_gt_cap >= mu_thr
    n_supervision_t = int(region_t.sum().item())
    n_gt_clot_supervision_t = int((region_t & gt_clot).sum().item())

    if allowed is None:
        allowed = region_t
    allowed_b = allowed.reshape(-1).bool()

    gate = resolve_clot_trigger_gate(
        phi,
        mu_c,
        mu_mlp,
        region=region_t,
        mu_gt_cap_si=mu_gt_cap,
        phys_cfg=phys_cfg,
        graph_data=data,
        prev_mu_eff_si=prev_mu_eff_si,
        u_nd=u_nd.reshape(-1),
        v_nd=v_nd.reshape(-1),
        bio_cfg=bio_cfg,
        allowed_commit_mask=allowed_b,
    ).reshape(-1).bool()
    if mlp_deploy_no_commit_at_t0() and int(macro_step_index) == 0:
        gate = torch.zeros_like(gate)

    mu_roll = mu_rollout_si.reshape(-1).to(device=device)
    rollout_ok = mu_roll >= mu_thr

    n_allowed = int(allowed_b.sum().item())
    frac_phi = _frac_true(phi_ok, allowed_b)
    frac_mu = _frac_true(mu_ok, allowed_b)
    frac_both = _frac_true(both_ok, allowed_b)
    frac_commit_allowed = _frac_true(gate, allowed_b)
    frac_commit_mesh = float(gate.float().mean().item())
    frac_rollout_clot = _frac_true(rollout_ok, allowed_b)
    no_commit_t0 = bool(mlp_deploy_no_commit_at_t0() and int(macro_step_index) == 0)

    return DeployGateFrameDiagnostics(
        macro_step=int(macro_step_index),
        time_index=int(time_index),
        t_si=float(t_si),
        mask_mode=mlp_mu_map_mask_mode(),
        n_allowed=n_allowed,
        n_supervision_at_t=n_supervision_t,
        n_gt_clot_in_allowed=int((allowed_b & gt_clot).sum().item()),
        n_gt_clot_supervision_t=n_gt_clot_supervision_t,
        frac_phi_ge_thr_in_allowed=frac_phi,
        frac_mu_mlp_ge_thr_in_allowed=frac_mu,
        frac_both_in_allowed=frac_both,
        frac_commit_in_allowed=frac_commit_allowed,
        frac_commit_mesh=frac_commit_mesh,
        frac_rollout_mu_ge_thr_in_allowed=frac_rollout_clot,
        phi_p50_allowed=_quantile_in_pool(phi, allowed_b, 0.5),
        phi_p90_allowed=_quantile_in_pool(phi, allowed_b, 0.9),
        mu_mlp_p50_allowed=_quantile_in_pool(mu_mlp, allowed_b, 0.5),
        mu_mlp_p90_allowed=_quantile_in_pool(mu_mlp, allowed_b, 0.9),
        phi_thr=float(phi_thr),
        mu_thr_si=float(mu_thr),
        bottleneck=_deploy_gate_bottleneck(
            n_allowed=n_allowed,
            frac_phi=frac_phi,
            frac_mu=frac_mu,
            frac_both=frac_both,
            frac_commit=frac_commit_allowed,
            no_commit_t0=no_commit_t0,
        ),
        no_commit_t0=no_commit_t0,
    )


@torch.no_grad()
def diagnose_deploy_gate_rollout_series(
    clot_model: nn.Module,
    data,
    pred_series: torch.Tensor,
    eval_times_si: torch.Tensor,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    label_time_indices: list[int] | None = None,
) -> list[DeployGateFrameDiagnostics]:
    """Run gate diagnostics on each rollout frame (updates deploy vision state)."""
    n_frames = int(pred_series.shape[0])
    if label_time_indices is None:
        label_time_indices = [
            min(i, int(data.y.shape[0]) - 1) for i in range(n_frames)
        ]
    allowed: torch.Tensor | None = None
    rows: list[DeployGateFrameDiagnostics] = []
    for idx in range(n_frames):
        ti = int(label_time_indices[idx])
        t_si = float(eval_times_si[min(idx, int(eval_times_si.numel()) - 1)].item())
        frame = pred_series[idx]
        u_nd = frame[:, 0]
        v_nd = frame[:, 1]
        species = frame[:, sc.SPECIES_BLOCK]
        prev_mu = None
        if idx > 0:
            prev_mu = phys_cfg.viscosity_nd_to_si(pred_series[idx - 1][:, STATE_CHANNEL_MU_EFF_ND])
        mu_roll = phys_cfg.viscosity_nd_to_si(frame[:, STATE_CHANNEL_MU_EFF_ND])
        if mlp_deploy_vision_restrict_enabled() and allowed is None:
            allowed = init_deploy_supervision_vision_mask(
                data,
                device,
                ti,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
            )
        row = compute_deploy_gate_frame_diagnostics(
            clot_model,
            data,
            ti,
            u_nd=u_nd,
            v_nd=v_nd,
            species_log=species,
            mu_rollout_si=mu_roll,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            allowed_commit_mask=allowed,
            macro_step_index=idx,
            t_si=t_si,
            prev_mu_eff_si=prev_mu,
        )
        rows.append(row)
        if (
            allowed is not None
            and mlp_deploy_vision_grow_enabled()
            and not row.no_commit_t0
        ):
            allowed = expand_seed_growth_allowed_mask(
                allowed,
                mu_roll,
                data,
                device,
                phys_cfg=phys_cfg,
            )
    return rows


@dataclass
class ClotPhiInjectDiagnostics:
    phi_mean: float = 0.0
    phi_frac_ge_05: float = 0.0
    n_region: int = 0
    n_triggered: int = 0
    mu_mlp_mean_region: float = 0.0
    mu_mlp_mean_high: float = 0.0
    mode: str = "off"


class ClotPhiMuInjector:
    """Lazy-loaded clot-phi head used as spatial trigger during GNODE rollout."""

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        *,
        phys_cfg: PhysicsConfig | None = None,
        bio_cfg: BiochemConfig | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.phys_cfg = phys_cfg or PhysicsConfig(phase="biochem")
        self.bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
        self._model: nn.Module | None = None
        self._cfg: dict[str, Any] = {}
        self.last_diag = ClotPhiInjectDiagnostics()
        self.next_allowed_commit_mask: torch.Tensor | None = None

    def clear_seed_growth_state(self) -> None:
        self.next_allowed_commit_mask = None

    @property
    def enabled(self) -> bool:
        return biochem_mlp_coupling_enabled() and self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        raw = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        cfg = raw.get("config") or {}
        apply_clot_phi_config_from_checkpoint(cfg)
        apply_clot_phi_eval_defaults()
        os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
        in_dim = int(cfg.get("in_dim", 6))
        hidden = int(cfg.get("hidden", 32))
        model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(self.device)
        model.load_state_dict(raw["model_state_dict"])
        model.eval()
        self._model = model
        self._cfg = cfg

    def _region_mask(
        self,
        data,
        time_index: int,
        *,
        y_slice: torch.Tensor,
        phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = int(data.num_nodes)
        device = self.device
        mode = mlp_clot_region_mode()
        if mlp_mu_map_mask_mode() == "neighbor":
            if phi is not None:
                return resolve_deploy_neighbor_commit_mask(
                    data,
                    device,
                    phi=phi,
                    prev_mu_eff_si=None,
                    phys_cfg=self.phys_cfg,
                )
            wall = torch.zeros(n, device=device, dtype=torch.bool)
            if hasattr(data, "mask_wall") and data.mask_wall is not None:
                wall = data.mask_wall.view(-1).to(device=device).bool()
            return neighbor_supervision_mask(data, device, wall)
        if mode == "neighbor_wall":
            wall = torch.zeros(n, device=device, dtype=torch.bool)
            if hasattr(data, "mask_wall") and data.mask_wall is not None:
                wall = data.mask_wall.view(-1).to(device=device).bool()
            return neighbor_supervision_mask(data, device, wall)
        if mode == "phi_seed" and phi is not None:
            seed = phi.reshape(-1).to(device=device) >= mlp_mu_map_phi_thresh()
            return neighbor_supervision_mask(data, device, seed)
        if not mlp_mu_map_uses_gt_labels():
            wall = torch.zeros(n, device=device, dtype=torch.bool)
            if hasattr(data, "mask_wall") and data.mask_wall is not None:
                wall = data.mask_wall.view(-1).to(device=device).bool()
            if mode == "phi_seed" and phi is not None:
                seed = phi.reshape(-1).to(device=device) >= mlp_mu_map_phi_thresh()
                return neighbor_supervision_mask(data, device, seed)
            return neighbor_supervision_mask(data, device, wall)
        mu_gt = self.phys_cfg.viscosity_nd_to_si(y_slice[:, STATE_CHANNEL_MU_EFF_ND])
        mu_cap = cap_mu_eff_si(mu_gt)
        return supervision_region_mask(data, device, mu_cap, self.phys_cfg)

    @torch.no_grad()
    def predict_phi(
        self,
        data,
        time_index: int,
        *,
        u_nd: torch.Tensor,
        v_nd: torch.Tensor,
        species_log: torch.Tensor,
        y_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (phi [N], region_mask [N] bool)."""
        assert self._model is not None
        device = self.device
        y_ref = data.y[time_index].to(device) if y_reference is None else y_reference.to(device)
        y_slice = y_ref.clone()
        y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
        y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
        if mlp_clot_use_pred_species():
            y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)

        step = build_clot_phi_step(
            data,
            time_index,
            self.phys_cfg,
            self.bio_cfg,
            device,
            u_nd_override=u_nd.reshape(-1),
            v_nd_override=v_nd.reshape(-1),
            y_slice_override=y_slice,
        )
        phi = self._model(step.features).reshape(-1)
        region = self._region_mask(data, time_index, y_slice=y_slice).reshape(-1).bool()
        return phi, region

    def _build_step(
        self,
        data,
        time_index: int,
        *,
        u_nd: torch.Tensor,
        v_nd: torch.Tensor,
        species_log: torch.Tensor,
        y_reference: torch.Tensor | None = None,
    ):
        device = self.device
        y_ref = data.y[time_index].to(device) if y_reference is None else y_reference.to(device)
        y_slice = y_ref.clone()
        y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
        y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
        if mlp_clot_use_pred_species():
            y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
        return build_clot_phi_step(
            data,
            time_index,
            self.phys_cfg,
            self.bio_cfg,
            device,
            u_nd_override=u_nd.reshape(-1),
            v_nd_override=v_nd.reshape(-1),
            y_slice_override=y_slice,
        )

    @torch.no_grad()
    def predict_mu_mlp_full(
        self,
        data,
        time_index: int,
        *,
        u_nd: torch.Tensor,
        v_nd: torch.Tensor,
        species_log: torch.Tensor,
        y_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu_mlp [N,1] SI, mu_c [N,1] SI, region_mask [N] bool, phi [N])."""
        assert self._model is not None
        device = self.device
        y_ref = data.y[time_index].to(device) if y_reference is None else y_reference.to(device)
        y_slice = y_ref.clone()
        y_slice[:, 0] = u_nd.reshape(-1).to(dtype=y_slice.dtype)
        y_slice[:, 1] = v_nd.reshape(-1).to(dtype=y_slice.dtype)
        if mlp_clot_use_pred_species():
            y_slice[:, sc.SPECIES_BLOCK] = species_log.to(device=device, dtype=y_slice.dtype)
        step = self._build_step(
            data,
            time_index,
            u_nd=u_nd,
            v_nd=v_nd,
            species_log=species_log,
            y_reference=y_reference,
        )
        mu_c_bulk, mu_c_mlp = resolve_mu_map_baselines_si(data, u_nd, v_nd, self.phys_cfg)
        mu_c = mu_c_bulk.reshape(-1, 1)
        model = self._model
        if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
            delta = model.forward_delta_log_mu(step.features)
            mu_mlp = mu_eff_from_delta_log_si(mu_c_mlp, delta).reshape(-1, 1)
            phi = torch.sigmoid(model.forward_logits(step.features)).reshape(-1)
        else:
            phi = model(step.features).reshape(-1)
            mu_mlp = log_blend_mu_eff_si(mu_c_mlp, phi).reshape(-1, 1)
        region = self._region_mask(data, time_index, y_slice=y_slice, phi=phi).reshape(-1).bool()
        return mu_mlp, mu_c, region, phi

    @torch.no_grad()
    def apply(
        self,
        data,
        time_index: int,
        mu_learned_si: torch.Tensor,
        *,
        u_nd: torch.Tensor,
        v_nd: torch.Tensor,
        species_log: torch.Tensor,
        y_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend learned mu_eff toward clot constant using soft phi trigger in region."""
        if not self.enabled:
            return mu_learned_si

        alpha = mlp_clot_blend_alpha()
        if alpha <= 0.0:
            return mu_learned_si

        phi, region = self.predict_phi(
            data,
            time_index,
            u_nd=u_nd,
            v_nd=v_nd,
            species_log=species_log,
            y_reference=y_reference,
        )
        mu_learned = mu_learned_si.reshape(-1, 1).to(device=self.device, dtype=torch.float32)
        mu_c = carreau_mu_si_from_uv(data, u_nd.reshape(-1), v_nd.reshape(-1), self.phys_cfg).reshape(-1, 1)
        mu_clot = torch.full_like(mu_c, mlp_clot_mu_si_default())

        phi_col = phi.reshape(-1, 1).clamp(0.0, 1.0)
        thresh = mlp_clot_phi_thresh()
        if thresh > 0.0:
            phi_col = phi_col * (phi_col >= thresh).to(dtype=phi_col.dtype)

        # Soft target: Carreau baseline -> clot cap along phi.
        mu_target = mu_c + phi_col * (mu_clot - mu_c)
        reg = region.reshape(-1, 1)
        mix = (alpha * phi_col).clamp(0.0, 1.0) * reg.to(dtype=mu_target.dtype)
        mu_out = (1.0 - mix) * mu_learned + mix * mu_target

        idx = reg.view(-1)
        n_reg = int(idx.sum().item())
        n_trig = int((idx & (phi.reshape(-1) >= max(thresh, 0.5))).sum().item()) if n_reg else 0
        self.last_diag = ClotPhiInjectDiagnostics(
            phi_mean=float(phi[idx].mean().item()) if n_reg else 0.0,
            phi_frac_ge_05=float((phi[idx] >= 0.5).float().mean().item()) if n_reg else 0.0,
            n_region=n_reg,
            n_triggered=n_trig,
            mode="inject_v1",
        )
        return mu_out.clamp(min=1e-8)

    @torch.no_grad()
    def apply_mu_map(
        self,
        data,
        time_index: int,
        *,
        u_nd: torch.Tensor,
        v_nd: torch.Tensor,
        species_log: torch.Tensor,
        y_reference: torch.Tensor | None = None,
        prev_mu_eff_si: torch.Tensor | None = None,
        allowed_commit_mask: torch.Tensor | None = None,
        macro_step_index: int = 0,
    ) -> torch.Tensor:
        """Leg B v2: ``mu_eff = Carreau(u,v) + clot_mask * (mu_mlp - Carreau)``."""
        if not biochem_mlp_mu_map_enabled() or self._model is None:
            mu_c_only = mu_map_carreau_baseline_si(
                data, u_nd.reshape(-1), v_nd.reshape(-1), self.phys_cfg
            ).reshape(-1, 1)
            return mu_c_only.clamp(min=1e-8)

        alpha = mlp_clot_blend_alpha()
        if alpha <= 0.0:
            mu_c_only = mu_map_carreau_baseline_si(
                data, u_nd.reshape(-1), v_nd.reshape(-1), self.phys_cfg
            ).reshape(-1, 1)
            return mu_c_only.clamp(min=1e-8)

        mu_mlp, mu_c, region, phi = self.predict_mu_mlp_full(
            data,
            time_index,
            u_nd=u_nd,
            v_nd=v_nd,
            species_log=species_log,
            y_reference=y_reference,
        )
        use_gt = mlp_mu_map_uses_gt_labels()
        mu_cap = None
        if use_gt:
            y_ref = data.y[time_index].to(self.device) if y_reference is None else y_reference.to(self.device)
            mu_gt = self.phys_cfg.viscosity_nd_to_si(y_ref[:, STATE_CHANNEL_MU_EFF_ND])
            mu_cap = cap_mu_eff_si(mu_gt)

        allowed = allowed_commit_mask
        if mlp_deploy_vision_restrict_enabled():
            if allowed is None and int(macro_step_index) == 0:
                allowed = init_deploy_supervision_vision_mask(
                    data,
                    self.device,
                    time_index,
                    phys_cfg=self.phys_cfg,
                    bio_cfg=self.bio_cfg,
                )
            elif allowed is None:
                allowed = torch.zeros(int(data.num_nodes), device=self.device, dtype=torch.bool)

        gate = resolve_clot_trigger_gate(
            phi,
            mu_c,
            mu_mlp,
            region=region,
            mu_gt_cap_si=mu_cap,
            phys_cfg=self.phys_cfg,
            graph_data=data,
            prev_mu_eff_si=prev_mu_eff_si,
            u_nd=u_nd.reshape(-1),
            v_nd=v_nd.reshape(-1),
            bio_cfg=self.bio_cfg,
            allowed_commit_mask=allowed,
        )
        if mlp_deploy_no_commit_at_t0() and int(macro_step_index) == 0:
            gate = torch.zeros_like(gate)
        mu_out = assemble_committed_mu_map(
            mu_c,
            mu_mlp,
            phi,
            region=region,
            mu_gt_cap_si=mu_cap,
            phys_cfg=self.phys_cfg,
            graph_data=data,
            prev_mu_eff_si=prev_mu_eff_si,
            u_nd=u_nd.reshape(-1),
            v_nd=v_nd.reshape(-1),
            bio_cfg=self.bio_cfg,
            allowed_commit_mask=allowed,
        )
        if mlp_deploy_no_commit_at_t0() and int(macro_step_index) == 0:
            mu_out = mu_c.reshape(-1, 1).clamp(min=1e-8)
        if (
            allowed is not None
            and (
                mlp_mu_map_mask_mode() in ("seed_growth", "mlp_band")
                or (mlp_mu_map_mask_mode() == "neighbor" and mlp_deploy_vision_grow_enabled())
            )
        ):
            self.next_allowed_commit_mask = expand_seed_growth_allowed_mask(
                allowed,
                mu_out.reshape(-1),
                data,
                self.device,
                phys_cfg=self.phys_cfg,
            )
        else:
            self.next_allowed_commit_mask = None

        active = gate.reshape(-1) > 0.5
        n_active = int(active.sum().item())
        thr = resolve_clot_mu_commit_thresh_si(self.phys_cfg)
        high = active & (mu_mlp.reshape(-1) >= thr)
        mask_tag = mlp_mu_map_mask_mode()
        self.last_diag = ClotPhiInjectDiagnostics(
            phi_mean=float(phi[active].mean().item()) if n_active else 0.0,
            phi_frac_ge_05=float(gate.reshape(-1).mean().item()),
            n_region=int(region.sum().item()) if region is not None else n_active,
            n_triggered=n_active,
            mu_mlp_mean_region=float(mu_mlp.reshape(-1)[active].mean().item()) if n_active else 0.0,
            mu_mlp_mean_high=float(mu_mlp.reshape(-1)[high].mean().item()) if bool(high.any()) else 0.0,
            mode=f"mu_map_v2_{mask_tag}",
        )
        return mu_out.clamp(min=1e-8)


def build_clot_phi_mu_injector(
    device: torch.device,
    ckpt_path: str | Path | None = None,
) -> ClotPhiMuInjector | None:
    if not biochem_mlp_coupling_enabled():
        return None
    path = resolve_mlp_clot_ckpt(ckpt_path)
    if path is None:
        print("[WARN]  MLP clot coupling enabled but no clot-phi ckpt found", flush=True)
        return None
    inj = ClotPhiMuInjector(path, device)
    inj.load()
    if biochem_mlp_mu_map_enabled():
        print(
            f"[i]  MLP mu map v2: ckpt={path.name} blend={mlp_clot_blend_alpha():.2f} "
            f"mask={mlp_mu_map_mask_mode()} bulk={mlp_mu_map_bulk_mode()} "
            f"phi_gate={int(mlp_mu_map_phi_gate_enabled())} "
            f"geo_cap={int(mlp_mu_map_geo_cap_enabled())} region={mlp_clot_region_mode()} "
            f"hybrid={int(clot_phi_hybrid_enabled())}",
            flush=True,
        )
    else:
        print(
            f"[i]  MLP clot inject v1: ckpt={path.name} mu_clot={mlp_clot_mu_si_default():.3f} Pa*s "
            f"blend={mlp_clot_blend_alpha():.2f} region={mlp_clot_region_mode()}",
            flush=True,
        )
    return inj


# Extend build_clot_phi_step with y_slice_override — implemented in clot_phi_simple.
