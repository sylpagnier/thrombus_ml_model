"""Species channel scope for biochem supervision and GraphSAGE pushforward state."""

from __future__ import annotations

import os
from typing import Sequence

import torch

# Indices within ``y[..., 4:16]`` (12 bulk log1p channels).
THROMBIN_CHANNEL = 5
PROTHROMBIN_CHANNEL = 4
FIBRINOGEN_CHANNEL = 7
FI_CHANNEL = 8
MAT_CHANNEL = 11


def _normalize_scope(raw: str) -> str:
    aliases = {
        "fi+mat": "fi_mat",
        "trigger": "fi_mat",
        "clot_trigger": "fi_mat",
        "fi_mat_th": "fi_mat_thrombin",
        "trigger_thrombin": "fi_mat_thrombin",
        "fi_mat+th": "fi_mat_thrombin",
        "bulk": "bulk9",
    }
    return aliases.get(raw, raw)


def data_bio_species_scope() -> str:
    raw = (os.environ.get("BIOCHEM_DATA_BIO_SPECIES_SCOPE") or "all").strip().lower()
    return _normalize_scope(raw)


def pushforward_species_scope() -> str:
    """Active species channels for GraphSAGE pushforward (default FI+Mat)."""
    raw = (
        os.environ.get("BIOCHEM_PUSHFORWARD_SPECIES_SCOPE")
        or os.environ.get("BIOCHEM_DATA_BIO_SPECIES_SCOPE")
        or "fi_mat"
    ).strip().lower()
    return _normalize_scope(raw)


def _scope_channel_indices(scope: str, *, n_channels: int = 12) -> list[int]:
    if scope in ("all", "bulk_full", "full", ""):
        return list(range(min(n_channels, 12)))
    if scope == "fi_mat":
        return [FI_CHANNEL, MAT_CHANNEL]
    if scope in ("fi_mat_thrombin", "trigger_thrombin"):
        return [FI_CHANNEL, MAT_CHANNEL, THROMBIN_CHANNEL]
    if scope == "bulk9":
        return list(range(9))
    if scope == "fi":
        return [FI_CHANNEL]
    if scope == "mat":
        return [MAT_CHANNEL]
    if scope == "thrombin":
        return [THROMBIN_CHANNEL]
    raise ValueError(
        f"Unknown species scope={scope!r}; "
        "use fi_mat|fi_mat_thrombin|bulk9|all|fi|mat|thrombin"
    )


def data_bio_species_channel_indices(*, n_channels: int = 12) -> list[int]:
    """Active bulk species channels for ``L_Data_Bio`` (subset of ``4:16`` slice)."""
    return _scope_channel_indices(data_bio_species_scope(), n_channels=n_channels)


def pushforward_state_bulk_indices(*, n_channels: int = 12) -> list[int]:
    """Bulk channel indices modeled by GraphSAGE pushforward state."""
    return _scope_channel_indices(pushforward_species_scope(), n_channels=n_channels)


def pushforward_state_dim(*, n_channels: int = 12) -> int:
    return len(pushforward_state_bulk_indices(n_channels=n_channels))


def pushforward_local_index(which: str) -> int:
    """Local state index for a named bulk channel (fi|mat|thrombin)."""
    bulk = pushforward_state_bulk_indices()
    key = which.strip().lower()
    ch_map = {
        "fi": FI_CHANNEL,
        "mat": MAT_CHANNEL,
        "thrombin": THROMBIN_CHANNEL,
        "th": THROMBIN_CHANNEL,
        "pt": PROTHROMBIN_CHANNEL,
        "fg": FIBRINOGEN_CHANNEL,
    }
    if key not in ch_map:
        raise KeyError(f"unknown pushforward channel name {which!r}")
    bulk_ch = ch_map[key]
    return bulk.index(bulk_ch)


def slice_bio_species_channels(
    tensor: torch.Tensor,
    *,
    n_channels: int = 12,
) -> torch.Tensor:
    """Select supervised species channels; ``tensor`` shape ``[..., 12]``."""
    idx = data_bio_species_channel_indices(n_channels=n_channels)
    return tensor[..., idx]


def scatter_log_state_to_species_block(
    species: torch.Tensor,
    log_state: torch.Tensor,
    node_idx: torch.Tensor,
    *,
    bulk_channels: Sequence[int] | None = None,
) -> torch.Tensor:
    """Write pushforward log-state onto a 12-ch species block."""
    bulk = list(bulk_channels or pushforward_state_bulk_indices())
    out = species.clone()
    idx = node_idx.reshape(-1)
    st = log_state.reshape(-1, len(bulk)).to(device=out.device, dtype=out.dtype)
    for local_i, bulk_ch in enumerate(bulk):
        out[idx, int(bulk_ch)] = st[:, local_i]
    return out.clamp(min=0.0)


def data_bio_species_scope_label() -> str:
    return data_bio_species_scope()


def pushforward_species_scope_label() -> str:
    return pushforward_species_scope()
