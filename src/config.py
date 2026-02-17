from dataclasses import dataclass, field
from typing import Dict

@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation."""
    # Output
    output_dir: str = "data/raw/synthetic_v2"

    # Mesh Settings
    mesh_size_factor: float = 0.25
    mesh_lc: float = 0.0002

    # Vessel Dimensions
    base_length: float = 0.015
    width_min: float = 0.0012
    width_max: float = 0.0018
    curvature_amplitude: float = 0.0035

    # Bifurcation Specifics
    bifurcation_angle_min: float = 20.0
    bifurcation_angle_max: float = 45.0
    bifurcation_l1: float = 0.006
    bifurcation_l2: float = 0.009

    # Pathology Constraints
    stenosis_factor_min: float = 0.30
    stenosis_factor_max: float = 0.66
    aneurysm_factor_min: float = 0.40
    aneurysm_factor_max: float = 0.80

    # Control Points
    num_ctrl_pts: int = 7

    # Physical Group Tags (The Single Source of Truth)
    TAGS: Dict[str, int] = field(default_factory=lambda: {
        "Inlet": 101,
        "Outlet_1": 102,
        "Outlet_2": 103,
        "Walls": 104,
        "Fluid_Domain": 201
    })

@dataclass
class PhysicsConfig:
    rho: float = 1050.0  # kg/m^3
    mu_newtonian: float = 0.0035  # Pa*s
    re_target: float = 150.0  # Reynolds number [-]