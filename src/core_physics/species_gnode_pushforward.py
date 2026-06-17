"""Species pushforward arch dispatch (GraphSAGE only; GNODE trunk removed)."""

from __future__ import annotations

import os

import torch.nn as nn


def species_pushforward_arch() -> str:
    return (os.environ.get("SPECIES_PUSHFORWARD_ARCH") or "sage").strip().lower()


class SpeciesGnodeDualHeadContinuousGNN(nn.Module):
    """Removed GNODE-band trunk; kept for import compatibility."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "gnode pushforward arch was removed; use SPECIES_PUSHFORWARD_ARCH=sage"
        )
