"""Env-driven ADR residual formulation (passive / M3 narrowing experiments)."""

from __future__ import annotations

import os

_VALID_RESIDUAL_MODES = frozenset(
    {
        "convective_nd",
        "nd",
        "log",
        "relative_nd",
        "rel_nd",
        "transport_only",
        "transport",
        "reaction_only",
        "reaction",
    }
)
_VALID_SPECIES_SCOPES = frozenset({"all", "fi", "fi_mat", "fast", "slow"})


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower()


def adr_residual_mode() -> str:
    """How bulk ADR residual is formed (default: convective_nd)."""
    mode = _env_str("BIOCHEM_ADR_RESIDUAL_MODE", "convective_nd")
    aliases = {
        "nd": "convective_nd",
        "rel_nd": "relative_nd",
        "transport": "transport_only",
        "reaction": "reaction_only",
    }
    mode = aliases.get(mode, mode)
    if mode not in _VALID_RESIDUAL_MODES:
        raise ValueError(
            f"Unknown BIOCHEM_ADR_RESIDUAL_MODE={mode!r}; "
            f"use {sorted(_VALID_RESIDUAL_MODES)}"
        )
    return mode


def adr_species_scope() -> str:
    """Subset of bulk species included in ADR sum (fi_mat -> FI only in bulk)."""
    scope = _env_str("BIOCHEM_ADR_SPECIES_SCOPE", "all")
    if scope == "fi_mat":
        return "fi"
    if scope not in _VALID_SPECIES_SCOPES:
        raise ValueError(
            f"Unknown BIOCHEM_ADR_SPECIES_SCOPE={scope!r}; use {sorted(_VALID_SPECIES_SCOPES)}"
        )
    return scope


def species_in_adr_scope(species_name: str, *, scope: str, is_fast: bool) -> bool:
    if scope in ("all", ""):
        return True
    if scope == "fi":
        return species_name == "FI"
    if scope == "fast":
        return bool(is_fast)
    if scope == "slow":
        return not bool(is_fast)
    return True
