"""Species pushforward arch dispatch (GraphSAGE / GAT / physics_gat; GNODE trunk removed)."""

from __future__ import annotations

import os

import torch.nn as nn


def species_pushforward_arch() -> str:
    """Resolve trunk arch from active PushforwardConfig, else legacy env."""
    try:
        from src.architecture.pushforward_config import resolve_config

        cfg = resolve_config()
        if cfg is not None and str(cfg.arch or "").strip():
            return str(cfg.arch).strip().lower()
    except Exception:
        pass
    return (os.environ.get("SPECIES_PUSHFORWARD_ARCH") or "sage").strip().lower()


class SpeciesGnodeDualHeadContinuousGNN(nn.Module):
    """Removed GNODE-band trunk; kept for import compatibility."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "gnode pushforward arch was removed; use arch='sage' or 'physics_gat'"
        )
