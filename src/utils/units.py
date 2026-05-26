"""Centralized physical unit conversion helpers and mesh-unit invariants.

Two responsibilities:

1. CGS<->SI numeric multipliers used when ingesting COMSOL exports
   (``CGS_to_SI``).
2. Mesh-unit invariants for the data-generation pipelines:
   synthetic / kinematics meshes are written in **meters** while patient /
   COMSOL meshes are written in **centimeters**. Each ``vessel_*.msh`` ships
   a JSON sidecar that records its unit choice, and the graph builders /
   anchor extractor call :func:`assert_mesh_unit` so a mismatched mesh is
   rejected with a clear error rather than silently producing graphs whose
   ``d_bar`` / ``u_ref`` are off by 100x.

All ``Data`` graphs stored downstream therefore carry SI-scale ``d_bar`` and
``u_ref`` regardless of whether they originated in the synthetic or anchor
track, which is the assumption baked into ``PhysicsConfig.get_u_ref`` and
the training-time physics kernels.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Optional


class CGS_to_SI:
    """Strict conversion multipliers from CGS to SI."""

    LENGTH = 1e-2  # cm -> m
    VELOCITY = 1e-2  # cm/s -> m/s
    PRESSURE = 1e-1  # barye (dyn/cm^2) -> Pa
    WSS = 1e-1  # dyn/cm^2 -> Pa
    VISCOSITY = 1e-1  # Poise -> Pa*s
    KINEMATIC_VISC = 1e-4  # Stokes (cm^2/s) -> m^2/s
    DIFFUSION = 1e-4  # cm^2/s -> m^2/s
    CONCENTRATION = 1e6  # mol/cm^3 -> mol/m^3

    # Common COMSOL export conveniences used in this project.
    UM_TO_MOL_PER_M3 = 1e-3  # micro-molar (uM) -> mol/m^3
    PLT_PER_ML_TO_PER_M3 = 1e6  # platelets/ml -> platelets/m^3


# --- Mesh-unit invariants -------------------------------------------------

MESH_UNIT_M: str = "m"
MESH_UNIT_CM: str = "cm"
SUPPORTED_MESH_UNITS: tuple[str, ...] = (MESH_UNIT_M, MESH_UNIT_CM)


class MeshUnitMismatchError(ValueError):
    """Raised when a vessel mesh declares a length unit different from what the caller expects."""


def assert_mesh_unit(
    meta: Optional[Mapping[str, object]],
    expected: str,
    *,
    stem: str,
    builder: str,
) -> str:
    """Validate the length unit declared in a vessel mesh sidecar JSON.

    Parameters
    ----------
    meta : optional mapping from ``vessel_<idx>.json`` (already parsed). ``None``
        means no sidecar JSON was found at all -- this helper returns
        ``expected`` silently in that case so each caller can decide whether a
        missing sidecar is fatal.
    expected : ``"m"`` or ``"cm"`` -- the unit the calling pipeline assumes.
    stem : mesh stem (e.g. ``"vessel_0"``) for error / warning messages.
    builder : human-readable name of the calling component, used in messages.

    Returns
    -------
    The unit string actually present in ``meta``. When ``meta`` is missing or
    has no ``unit`` field, this is ``expected`` (with a warning in the latter
    case so legacy meshes without a unit declaration aren't a hard break).

    Raises
    ------
    MeshUnitMismatchError
        when ``meta['unit']`` is present and disagrees with ``expected``,
        or when it declares an unsupported value entirely.
    """
    if expected not in SUPPORTED_MESH_UNITS:
        raise ValueError(
            f"assert_mesh_unit: unsupported expected={expected!r}; "
            f"supported: {SUPPORTED_MESH_UNITS}."
        )

    if meta is None:
        return expected

    if "unit" not in meta:
        warnings.warn(
            f"{builder}: {stem} sidecar JSON has no 'unit' field; assuming {expected!r}. "
            "Regenerate with the current vessel_generator to attach a unit declaration.",
            stacklevel=2,
        )
        return expected

    actual = str(meta["unit"]).lower()
    if actual not in SUPPORTED_MESH_UNITS:
        raise MeshUnitMismatchError(
            f"{builder}: {stem} sidecar JSON declares unsupported unit={actual!r}; "
            f"supported: {SUPPORTED_MESH_UNITS}."
        )
    if actual != expected:
        raise MeshUnitMismatchError(
            f"{builder}: {stem} sidecar JSON declares unit={actual!r} but {builder} "
            f"requires unit={expected!r}. Pick the matching pipeline track or "
            "regenerate the mesh with the right unit."
        )
    return actual


def read_mesh_length_unit(
    meta: Optional[Mapping[str, object]],
    *,
    stem: str,
    builder: str,
    default: str = MESH_UNIT_M,
) -> str:
    """Return the mesh length unit declared in a sidecar (``m`` or ``cm``).

    Unlike :func:`assert_mesh_unit`, this does not require a specific track unit;
    use it when the caller will convert lengths to SI (e.g. COMSOL ``D_eff``).
    """
    if default not in SUPPORTED_MESH_UNITS:
        raise ValueError(
            f"read_mesh_length_unit: unsupported default={default!r}; "
            f"supported: {SUPPORTED_MESH_UNITS}."
        )
    if meta is None:
        return default
    if "unit" not in meta:
        warnings.warn(
            f"{builder}: {stem} sidecar JSON has no 'unit' field; assuming {default!r}. "
            "Regenerate with the current vessel_generator to attach a unit declaration.",
            stacklevel=2,
        )
        return default
    actual = str(meta["unit"]).lower()
    if actual not in SUPPORTED_MESH_UNITS:
        raise MeshUnitMismatchError(
            f"{builder}: {stem} sidecar JSON declares unsupported unit={actual!r}; "
            f"supported: {SUPPORTED_MESH_UNITS}."
        )
    return actual


def length_in_meters(value: float, unit: str) -> float:
    """Convert a scalar length from mesh units to SI meters."""
    u = str(unit).lower()
    v = float(value)
    if u == MESH_UNIT_M:
        return v
    if u == MESH_UNIT_CM:
        return v * CGS_to_SI.LENGTH
    raise ValueError(
        f"length_in_meters: unsupported unit={unit!r}; supported: {SUPPORTED_MESH_UNITS}."
    )


def d_bar_si_from_sidecar(
    meta: Mapping[str, object],
    *,
    stem: str,
    builder: str,
) -> tuple[float, str]:
    """Return ``(d_bar [m], mesh_unit)`` from a vessel sidecar JSON mapping."""
    if "d_bar" not in meta:
        raise KeyError(f"{builder}: {stem} sidecar JSON missing required 'd_bar'.")
    unit = read_mesh_length_unit(meta, stem=stem, builder=builder)
    d_si = length_in_meters(float(meta["d_bar"]), unit)
    if d_si <= 0.0:
        raise ValueError(f"{builder}: {stem} sidecar d_bar={d_si} m is not positive.")
    return d_si, unit
