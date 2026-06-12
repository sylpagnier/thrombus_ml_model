"""Rung 4 step s5: narrow 2-ch FI/Mat head on frozen s0 gate in E(t).

Deploy::

    sp_s0 = s0 rules in E(t) from GT flow + geometry + pred commits
    delta = BandGNN(feats, edge_index) * onset(tau)   # FI + Mat log-ND delta
    sp = sp_s0 + delta_scale * delta  (masked to E(t))
    phi = gelation_physics(sp, GT flow) -> nucleation commits

Unlike s4 (gate rank only), s5 supervises **species magnitude** (FN/FP FI/Mat)
and optional **commit BCE** so gelation can flip binary clot masks.

Checkpoint: ``outputs/biochem/t0_r4_s5_gnode_fimat/best.pth`` (``T0_R4_S5_CKPT``).
Recipe id: ``s5_gnode_fimat`` (override with ``T0_R4_S5_RECIPE``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_r4_sweep import (
    T0R4SweepBundle,
    load_sweep_bundle,
    rollout_sweep_species_series,
    save_sweep_checkpoint,
)
from src.utils.paths import get_project_root

DEFAULT_S5_CKPT = "outputs/biochem/t0_r4_s5_gnode_fimat/best.pth"
DEFAULT_S5_RECIPE = "s5_gnode_fimat"


def s5_ckpt_path() -> Path:
    raw = (os.environ.get("T0_R4_S5_CKPT") or DEFAULT_S5_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def s5_recipe_id() -> str:
    return (os.environ.get("T0_R4_S5_RECIPE") or DEFAULT_S5_RECIPE).strip().lower()


def load_s5_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4SweepBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else s5_ckpt_path()
    return load_sweep_bundle(path, device=device, quiet=quiet)


@torch.no_grad()
def rollout_s5_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4SweepBundle | None = None,
    *,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    b = bundle if bundle is not None else load_s5_bundle()
    if b is None:
        raise FileNotFoundError(
            "s5_gnode_fimat requires checkpoint; train with scripts/go_t0_rung4_s5.ps1"
        )
    return rollout_sweep_species_series(
        data, phys_cfg, bio_cfg, device, b, nucleation_hops=nucleation_hops
    )


__all__ = [
    "DEFAULT_S5_CKPT",
    "DEFAULT_S5_RECIPE",
    "load_s5_bundle",
    "rollout_s5_species_series",
    "s5_ckpt_path",
    "s5_recipe_id",
    "save_sweep_checkpoint",
]
