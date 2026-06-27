"""Species channel scope for biochem supervision and GraphSAGE pushforward state."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from src.utils import species_channels as sc

# Indices within the 12-ch species block ``y[..., 4:16]`` (NOT full-y columns;
# add 4 for the full-y column). Single-sourced from ``src.utils.species_channels``
# so they cannot drift from the enums / channel schema.
THROMBIN_CHANNEL = sc.block_index("T")        # 5
PROTHROMBIN_CHANNEL = sc.block_index("PT")    # 4
FIBRINOGEN_CHANNEL = sc.block_index("FG")     # 7
FI_CHANNEL = sc.block_index("FI")             # 8
MAT_CHANNEL = sc.block_index("Mat")           # 11

# Species-block index -> short name (``y[:, 4:16]`` layout). Indices 9-11 are
# wall species (M, Mas, Mat) despite the historical "bulk" label.
BULK_CHANNEL_NAMES: dict[int, str] = {
    i: sc.name_at_block_index(i) for i in range(sc.SPECIES_BLOCK_WIDTH)
}

FI_MAT_BASE_CHANNELS: tuple[int, ...] = (FI_CHANNEL, MAT_CHANNEL)

# Single-channel add-on candidates for fi_mat + X screen (exclude FI/Mat).
SCREEN_ADDON_CHANNEL_INDICES: tuple[int, ...] = tuple(
    ch for ch in range(12) if ch not in FI_MAT_BASE_CHANNELS
)


def bulk_channel_name(ch: int) -> str:
    return BULK_CHANNEL_NAMES.get(int(ch), f"ch{int(ch)}")


def format_channel_list(channels: Sequence[int]) -> str:
    return ",".join(str(int(c)) for c in channels)


def canonical_pushforward_channel_order(channels: Sequence[int]) -> list[int]:
    """FI then Mat then extras (stable); matches ``fi_mat_thrombin`` = ``[8,11,5]`` not ``[5,8,11]``."""
    seen: set[int] = set()
    extras: list[int] = []
    for c in channels:
        ch = int(c)
        if ch in seen:
            continue
        seen.add(ch)
        if ch not in FI_MAT_BASE_CHANNELS:
            extras.append(ch)
    ordered_base = [c for c in FI_MAT_BASE_CHANNELS if c in seen]
    return ordered_base + sorted(extras)


def parse_channel_list(spec: str) -> list[int]:
    """Parse comma-separated bulk indices or short names (e.g. ``8,11,5`` or ``FI,Mat,T``)."""
    out: list[int] = []
    name_to_ch = {v.lower(): k for k, v in BULK_CHANNEL_NAMES.items()}
    name_to_ch.update(
        {
            "fi": FI_CHANNEL,
            "mat": MAT_CHANNEL,
            "thrombin": THROMBIN_CHANNEL,
            "th": THROMBIN_CHANNEL,
            "pt": PROTHROMBIN_CHANNEL,
            "fg": FIBRINOGEN_CHANNEL,
        }
    )
    for part in (spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            out.append(int(token))
            continue
        key = token.lower()
        if key not in name_to_ch:
            raise ValueError(f"unknown species channel name {token!r}")
        out.append(int(name_to_ch[key]))
    return canonical_pushforward_channel_order(out)


def pushforward_channels_from_env() -> list[int] | None:
    raw = (os.environ.get("BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS") or "").strip()
    if not raw:
        return None
    return parse_channel_list(raw)


def fi_mat_plus_channels(extra: Sequence[int]) -> list[int]:
    return canonical_pushforward_channel_order([*FI_MAT_BASE_CHANNELS, *(int(c) for c in extra)])


def scope_label_for_channels(channels: Sequence[int]) -> str:
    ch = sorted({int(c) for c in channels})
    base = sorted(FI_MAT_BASE_CHANNELS)
    if ch == base:
        return "fi_mat"
    if ch == [MAT_CHANNEL]:
        return "mat"
    extras = [c for c in ch if c not in base]
    if len(extras) == 1 and extras[0] == THROMBIN_CHANNEL:
        return "fi_mat_thrombin"
    extra_names = "+".join(bulk_channel_name(c) for c in extras)
    return f"fi_mat+{extra_names}"


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
    explicit = pushforward_channels_from_env()
    if explicit is not None:
        return explicit
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
    explicit = pushforward_channels_from_env()
    if explicit is not None:
        return scope_label_for_channels(explicit)
    return pushforward_species_scope()
