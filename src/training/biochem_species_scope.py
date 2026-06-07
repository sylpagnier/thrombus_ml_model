"""Species channel scope for ``L_Data_Bio`` (trigger-relevant subset vs full bulk)."""

from __future__ import annotations

import os
from typing import Sequence

import torch

# Indices within ``y[..., 4:16]`` (12 bulk log1p channels).
FI_CHANNEL = 8
MAT_CHANNEL = 11


def data_bio_species_scope() -> str:
    raw = (os.environ.get("BIOCHEM_DATA_BIO_SPECIES_SCOPE") or "all").strip().lower()
    aliases = {
        "fi+mat": "fi_mat",
        "trigger": "fi_mat",
        "clot_trigger": "fi_mat",
    }
    return aliases.get(raw, raw)


def data_bio_species_channel_indices(*, n_channels: int = 12) -> list[int]:
    """Active bulk species channels for ``L_Data_Bio`` (subset of ``4:16`` slice)."""
    scope = data_bio_species_scope()
    if scope in ("all", "bulk", "full", ""):
        return list(range(min(n_channels, 12)))
    if scope == "fi_mat":
        return [FI_CHANNEL, MAT_CHANNEL]
    if scope == "fi":
        return [FI_CHANNEL]
    if scope == "mat":
        return [MAT_CHANNEL]
    raise ValueError(
        f"Unknown BIOCHEM_DATA_BIO_SPECIES_SCOPE={scope!r}; "
        "use all|fi_mat|fi|mat"
    )


def slice_bio_species_channels(
    tensor: torch.Tensor,
    *,
    n_channels: int = 12,
) -> torch.Tensor:
    """Select supervised species channels; ``tensor`` shape ``[..., 12]``."""
    idx = data_bio_species_channel_indices(n_channels=n_channels)
    return tensor[..., idx]


def data_bio_species_scope_label() -> str:
    return data_bio_species_scope()
