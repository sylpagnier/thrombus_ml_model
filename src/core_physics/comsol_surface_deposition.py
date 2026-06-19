"""Canonical COMSOL phase-2 surface platelet-deposition law (unit-consistent).

Single source of truth for the wall surface reaction rates that COMSOL exports as
``J0_M`` / ``J0_Mat`` / ``J0_th``. Validated against the patient007 calibration
exports to machine precision: with constants from ``BiochemConfig`` converted to
**COMSOL-native CGS units**, the recovered Damkohler number equals
``surface_damkohler = 1e-4`` with CV ~ 1e-16 across all gate branches, and the
thrombin-source constant ``beta*phi_at`` is recovered exactly.

See ``docs/COMSOL_PHYSICS_VALIDATION.md``. Validator: ``scripts/validate_comsol_calibration.py``.

UNIT SYSTEM (COMSOL phase-2 is CGS / micromolar, NOT SI):
  - length             : cm           (BiochemConfig stores m  -> x100)
  - bulk platelets rp,ap: plt/cm^3    (BiochemConfig c_RP0 plt/m^3 -> x1e-6)
  - surface M/Mas/Mat  : plt/cm^2     (BiochemConfig Minf plt/m^2  -> x1e-4)
  - solutes (PT, ...)  : micromolar   (BiochemConfig mol/m^3 -> x1e3)
  - shear rate spf.sr  : 1/s          (same)
  - shear gradient dsrx: 1/(s*cm)     (BiochemConfig sgt 1/(s*m) -> /100)
  - adhesion rates k_* : cm/s         (BiochemConfig m/s -> x100)

This module operates entirely in CGS; callers in SI must convert at the boundary
via the ``*_M_TO_CM`` / ``*_uM`` helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config import BiochemConfig

# SI -> CGS conversion factors (multiply SI value by these to get CGS).
M_TO_CM = 100.0                 # length  m   -> cm
PER_M3_TO_PER_CM3 = 1.0e-6      # density m^-3 -> cm^-3 (bulk platelets)
PER_M2_TO_PER_CM2 = 1.0e-4      # density m^-2 -> cm^-2 (surface platelets)
MOLM3_TO_UM = 1.0e3             # mol/m^3 -> micromolar (1 mol/m^3 = 1e3 uM)
MS_TO_CMS = 100.0               # velocity m/s -> cm/s
PER_MS_TO_PER_CMS = 1.0e-2      # 1/(s*m) -> 1/(s*cm)


@dataclass(frozen=True)
class DepositionConstants:
    """COMSOL deposition-law constants in a single explicit unit system.

    The Damkohler prefactor ``Da`` is **dimensionless** and identical in CGS and SI;
    the deposition law is dimensionally consistent in either system, so the recovered
    ``Da`` equals ``surface_damkohler`` regardless of which unit system is used (as long
    as inputs and constants share that system). ``cgs()`` matches the COMSOL exports;
    ``si()`` matches the in-repo ``biochem_wall_residual`` kernel (SI throughout).
    """

    Da: float            # Damkohler prefactor (dimensionless) = surface_damkohler
    k_rs: float          # resting-platelet adhesion rate [cm/s | m/s]
    k_as: float          # activated-platelet adhesion rate [cm/s | m/s]
    k_aa: float          # autocatalytic (platelet-platelet) rate [cm/s | m/s]
    L: float             # characteristic length [cm | m]
    gamma_m: float       # reference shear rate [1/s]
    Minf: float          # surface saturation [plt/cm^2 | plt/m^2]
    lss: float           # low-shear (stagnation) threshold [1/s]
    sgt: float           # shear-gradient (separation) threshold [1/(s*cm) | 1/(s*m)]
    beta_phi_at: float   # thrombin-source constant (recovers exported J0_th; CGS units)

    @classmethod
    def cgs(cls, cfg: BiochemConfig) -> "DepositionConstants":
        return cls(
            Da=float(cfg.surface_damkohler),
            k_rs=float(cfg.k_rs) * MS_TO_CMS,
            k_as=float(cfg.k_as) * MS_TO_CMS,
            k_aa=float(cfg.k_aa) * MS_TO_CMS,
            L=float(cfg.L_char) * M_TO_CM,
            gamma_m=float(cfg.gamma_m),
            Minf=float(cfg.Minf) * PER_M2_TO_PER_CM2,
            lss=float(cfg.lss),
            # cfg.sgt is stored in SI [1/(s*m)] (negative); CGS is /100.
            sgt=float(cfg.sgt) * PER_MS_TO_PER_CMS,
            # Exported J0_th = (beta*phi_at) * Mat[plt/cm^2] * PT[uM] * step2t.
            # The numeric value equals SI beta*phi_at scaled by the unit factor
            # (validated: ratio exactly 1e6 vs cfg.beta*cfg.phi_at).
            beta_phi_at=float(cfg.beta) * float(cfg.phi_at) * 1.0e6,
        )

    @classmethod
    def si(cls, cfg: BiochemConfig) -> "DepositionConstants":
        """SI constants exactly as stored in ``BiochemConfig`` (kernel convention)."""
        return cls(
            Da=float(cfg.surface_damkohler),
            k_rs=float(cfg.k_rs),
            k_as=float(cfg.k_as),
            k_aa=float(cfg.k_aa),
            L=float(cfg.L_char),
            gamma_m=float(cfg.gamma_m),
            Minf=float(cfg.Minf),
            lss=float(cfg.lss),
            sgt=float(cfg.sgt),
            beta_phi_at=float(cfg.beta) * float(cfg.phi_at) * 1.0e6,
        )


# Backwards-compatible alias.
DepositionConstantsCGS = DepositionConstants


def _hard_or_soft_gate(
    value: torch.Tensor,
    threshold: float,
    *,
    reverse: bool,
    temperature: float | None,
) -> torch.Tensor:
    """Indicator ``value < threshold`` (reverse=True) or ``>`` (False).

    ``temperature is None`` -> hard boolean gate (matches COMSOL exported gate
    columns). Otherwise a sigmoid for differentiable use.
    """
    if temperature is None:
        gate = (value < threshold) if reverse else (value > threshold)
        return gate.to(dtype=value.dtype)
    z = (threshold - value) / temperature if reverse else (value - threshold) / temperature
    return torch.sigmoid(z.clamp(min=-50.0, max=50.0))


def deposition_common_term(
    sat_m: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    mas: torch.Tensor,
    k: DepositionConstants,
) -> torch.Tensor:
    """``common`` = Sat*(k_rs*rp + k_as*ap) + (Mas/Minf)*k_aa*ap   [plt/(area s)] / Da."""
    return sat_m * (k.k_rs * rp + k.k_as * ap) + (mas / k.Minf) * k.k_aa * ap


def j0_mat_from_constants(
    k: DepositionConstants,
    *,
    sat_m: torch.Tensor,
    shear_sr: torch.Tensor,
    dsrx: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    mas: torch.Tensor,
    step2t: torch.Tensor,
    gate_low_temp: float | None = None,
    gate_sep_temp: float | None = None,
) -> torch.Tensor:
    """Unit-agnostic COMSOL ``J0_Mat`` deposition source (inputs match ``k``'s unit system).

    ``J0_Mat = Da * ( [dsrx<sgt]*(L/gamma_m)*|dsrx|*common + [sr<lss]*common ) * step2t``
    """
    common = deposition_common_term(sat_m, rp, ap, mas, k)
    gate_sep = _hard_or_soft_gate(dsrx, k.sgt, reverse=True, temperature=gate_sep_temp)
    gate_low = _hard_or_soft_gate(shear_sr, k.lss, reverse=True, temperature=gate_low_temp)
    bracket = gate_sep * (k.L / k.gamma_m) * dsrx.abs() * common + gate_low * common
    return k.Da * bracket * step2t


def j0_mat_cgs(
    *,
    sat_m: torch.Tensor,
    shear_sr: torch.Tensor,
    dsrx: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    mas: torch.Tensor,
    step2t: torch.Tensor,
    cfg: BiochemConfig,
    gate_low_temp: float | None = None,
    gate_sep_temp: float | None = None,
) -> torch.Tensor:
    """Exported COMSOL ``J0_Mat`` deposition source [plt/(cm^2 s)] (all inputs CGS).

    With ``gate_*_temp=None`` (default) the gates are the exact COMSOL booleans, so
    the recovered Da == ``surface_damkohler`` to machine precision. Pass finite
    temperatures for a differentiable surrogate.
    """
    return j0_mat_from_constants(
        DepositionConstants.cgs(cfg),
        sat_m=sat_m, shear_sr=shear_sr, dsrx=dsrx, rp=rp, ap=ap, mas=mas,
        step2t=step2t, gate_low_temp=gate_low_temp, gate_sep_temp=gate_sep_temp,
    )


def j0_mat_si(
    *,
    sat_m: torch.Tensor,
    shear_sr: torch.Tensor,
    dsrx: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    mas: torch.Tensor,
    step2t: torch.Tensor,
    cfg: BiochemConfig,
    gate_low_temp: float | None = None,
    gate_sep_temp: float | None = None,
) -> torch.Tensor:
    """COMSOL ``J0_Mat`` deposition source [plt/(m^2 s)] with all inputs in SI.

    This is the convention used by ``biochem_wall_residual`` (rp/ap plt/m^3, dsrx
    1/(s*m), Mas plt/m^2). Equals ``j0_mat_cgs`` * 1e4 on the same physical state.
    """
    return j0_mat_from_constants(
        DepositionConstants.si(cfg),
        sat_m=sat_m, shear_sr=shear_sr, dsrx=dsrx, rp=rp, ap=ap, mas=mas,
        step2t=step2t, gate_low_temp=gate_low_temp, gate_sep_temp=gate_sep_temp,
    )


def j0_thrombin_cgs(
    *,
    mat: torch.Tensor,
    pt: torch.Tensor,
    step2t: torch.Tensor,
    cfg: BiochemConfig,
) -> torch.Tensor:
    """Exported COMSOL ``J0_th = beta*phi_at*Mat*PT*step2t`` (Mat plt/cm^2, PT uM)."""
    k = DepositionConstants.cgs(cfg)
    return k.beta_phi_at * mat * pt * step2t


def recover_damkohler_cgs(
    *,
    j0_mat_exported: torch.Tensor,
    sat_m: torch.Tensor,
    shear_sr: torch.Tensor,
    dsrx: torch.Tensor,
    rp: torch.Tensor,
    ap: torch.Tensor,
    mas: torch.Tensor,
    step2t: torch.Tensor,
    gate_low: torch.Tensor,
    gate_sep: torch.Tensor,
    cfg: BiochemConfig,
) -> torch.Tensor:
    """Recover per-point Da = J0_Mat / bracket using the COMSOL exported gate columns.

    Returns Da values on points where the bracket is non-zero (should be constant
    == ``surface_damkohler``).
    """
    k = DepositionConstants.cgs(cfg)
    common = deposition_common_term(sat_m, rp, ap, mas, k)
    bracket = (
        gate_sep * (k.L / k.gamma_m) * dsrx.abs() * common + gate_low * common
    ) * step2t
    nz = bracket.abs() > 0
    return j0_mat_exported[nz] / bracket[nz]
