"""Attach optional clot-phi MLP injector to a loaded GNODE teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.core_physics.clot_phi_mu_inject import (
    ClotPhiMuInjector,
    biochem_mlp_coupling_enabled,
    build_clot_phi_mu_injector,
)


def attach_clot_phi_injector_to_teacher(
    teacher: Any,
    device: torch.device,
    ckpt_path: str | Path | None = None,
) -> ClotPhiMuInjector | None:
    """Load injector when MLP coupling env is on and attach to ``GNODE_Phase3``."""
    if hasattr(teacher, "clear_clot_phi_injector"):
        teacher.clear_clot_phi_injector()
    if not biochem_mlp_coupling_enabled():
        return None
    injector = build_clot_phi_mu_injector(device, ckpt_path)
    if injector is not None and hasattr(teacher, "set_clot_phi_injector"):
        teacher.set_clot_phi_injector(injector)
    return injector
