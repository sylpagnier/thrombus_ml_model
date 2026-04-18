"""Curriculum easing for Predictor (Stage A) → Corrector (Stage B) training."""

from __future__ import annotations

import math
from typing import Any, Optional


def ease01(t: float, easing: str) -> float:
    """Map ``[0, 1]`` progress with optional smoothing (matches Tier-3 training conventions)."""
    t = min(1.0, max(0.0, float(t)))
    mode = (easing or "linear").strip().lower()
    if mode == "linear":
        return t
    if mode in ("smoothstep", "smooth"):
        return t * t * (3.0 - 2.0 * t)
    if mode == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * t))
    return t


def update_physics_curriculum(
    epoch: int,
    stage_b_start: int,
    max_epochs: int,
    base_cfg: Any,
    *,
    model: Optional[Any] = None,
    hybrid: Optional[Any] = None,
    easing: str = "smooth",
    target_mu_ratio_max: Optional[float] = None,
) -> bool:
    """
    Controls transition from Predictor (Stage A) to Corrector (Stage B).

    Stage A clamps ``mu_ratio_max`` to ``1.0`` (Newtonian clot feedback off) and
    trains exclusively on synthetic data.

    Stage B ramps toward the target non-Newtonian cap using :func:`ease01`. During
    this stage, LoRA weights are unfrozen and trained on a mixed population dataset
    (synthetic + real scans) to learn generalized geometry corrections.

    Returns ``True`` when Stage B is active (epoch >= ``stage_b_start``), i.e. when LoRA should be on.

    The target viscosity ratio is captured once from ``target_mu_ratio_max``, or from
    ``base_cfg.mu_ratio_max`` on first call, and stored on ``base_cfg._pc_target_mu_ratio_max``.
    """
    key = "_pc_target_mu_ratio_max"
    if not hasattr(base_cfg, key):
        init = (
            float(target_mu_ratio_max)
            if target_mu_ratio_max is not None
            else float(getattr(base_cfg, "mu_ratio_max", 80.0))
        )
        setattr(base_cfg, key, init)

    target_mu = float(getattr(base_cfg, key))

    inner = getattr(hybrid, "inner", None) if hybrid is not None else None

    def _sync_mu(mu: float) -> None:
        base_cfg.mu_ratio_max = float(mu)
        for m in (model, inner):
            if m is not None and hasattr(m, "mu_ratio_max"):
                m.mu_ratio_max = float(mu)

    if epoch < stage_b_start:
        _sync_mu(1.0)
        if hybrid is not None:
            hybrid.set_predictor_stage()
        return False

    denom = max(1, int(max_epochs) - int(stage_b_start))
    progress = (float(epoch) - float(stage_b_start)) / float(denom)
    smooth_progress = ease01(progress, easing=easing)
    mu = 1.0 + (target_mu - 1.0) * smooth_progress
    _sync_mu(mu)
    if hybrid is not None:
        hybrid.set_corrector_stage(mu)
    return True


__all__ = ["ease01", "update_physics_curriculum"]
