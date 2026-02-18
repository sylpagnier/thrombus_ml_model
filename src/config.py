from dataclasses import dataclass, field
from typing import Dict, Optional
from src.utils.paths import get_project_root


@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation."""
    # --- Paths ---
    # Where the COMSOL template lives
    project_root = get_project_root()
    template_path: str = project_root / "comsol_models/phase1_template.mph"

    # 1. RAW DATA: Where meshes (.msh/.nas) and metadata (.json) are saved
    mesh_input_dir: str = project_root / "data/raw/synthetic"

    # 2. PROCESSED CFD: Where COMSOL results (.npz) are saved
    output_dir: str = project_root / "data/processed/cfd_results"

    # 3. FINAL GRAPHS: Where PyTorch Geometric graphs (.pt) are saved
    graph_output_dir: str = project_root / "data/processed/graphs"

    # Mesh Settings
    mesh_size_factor: float = .5
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

    # Physical Group Tags (Must match what is used in Gmsh and Graph Gen)
    TAGS: Dict[str, int] = field(default_factory=lambda: {
        "Inlet": 101,
        "Outlet_1": 102,
        "Outlet_2": 103,
        "Walls": 104,
        "Fluid_Domain": 201
    })


@dataclass
class PhysicsConfig:
    # Fluid Properties
    rho: float = 1050.0  # kg/m^3
    re_target: float = 150.0  # Reynolds number [-]

    # Newtonian Reference (Tier 1)
    mu_newtonian: float = 0.0035  # Pa*s (3.5 cP)

    # Carreau-Yasuda Rheology (Tier 2 - Cho & Kensey 1991 Human Blood)
    # mu_eff = mu_inf + (mu_0 - mu_inf) * (1 + (lambda * gamma_dot)^a)^((n-1)/a)
    mu_inf: float = 0.0035  # Pa*s
    mu_0: float = 0.056  # Pa*s
    lam: float = 3.313  # Relaxation time (s)
    n: float = 0.3568  # Power law index
    a: float = 2.0  # Yasuda parameter