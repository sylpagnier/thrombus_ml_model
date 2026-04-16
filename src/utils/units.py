"""Centralized physical unit conversion helpers."""


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
