from dataclasses import dataclass, field
from typing import Dict, Optional
from src.utils.paths import get_project_root
from pathlib import Path

@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation."""

    # 1. Define the active tier/experiment here
    tier: str = "tier1"  # e.g., "tier1", "tier2", "tier3_patients"

    # --- Paths ---
    project_root: Path = field(default_factory=get_project_root)
    template_path: Path = field(init=False)
    mesh_input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    graph_output_dir: Path = field(init=False)

    def __post_init__(self):
        """Dynamically resolve paths based on the selected tier."""
        self.template_path = self.project_root / "comsol_models/phase1_template.mph"

        # Append the tier name to isolate data generation and processing
        self.mesh_input_dir = self.project_root / f"data/raw/{self.tier}"
        self.output_dir = self.project_root / f"data/processed/cfd_results_{self.tier}"
        self.graph_output_dir = self.project_root / f"data/processed/graphs_{self.tier}"

    # Mesh Settings
    mesh_size_factor: float = 0.75
    mesh_lc: float = 0.0001

    # Vessel Dimensions
    base_length: float = 0.015
    width_min: float = 0.0012
    width_max: float = 0.0018
    curvature_amplitude: float = 0.0035

    # Pathology Constraints
    stenosis_factor_min: float = 0.30
    stenosis_factor_max: float = 0.66
    aneurysm_factor_min: float = 0.40
    aneurysm_factor_max: float = 0.80

    # Control Points
    num_ctrl_pts: int = 7

    # Physical Group Tags
    TAGS: Dict[str, int] = field(default_factory=lambda: {
        "Inlet": 101,
        "Outlet_1": 102,
        "Walls": 103,
        "Fluid_Domain": 201
    })


@dataclass
class PhysicsConfig:
    tier: str = "tier1"

    # Fluid Properties
    rho: float = 1050.0  # kg/m^3
    re_target: float = 150.0  # Reynolds number [-]

    viscosity_model: str = field(init=False)
    mu_ref: float = field(init=False)

    # Newtonian Reference (tier 1)
    mu_newtonian: float = 0.0035  # Pa*s (3.5 cP)

    # Carreau-Yasuda Rheology (tier 2 - Cho & Kensey 1991 Human Blood)
    mu_inf: float = 0.0035  # Pa*s
    mu_0: float = 0.056  # Pa*s
    lam: float = 3.313  # Relaxation time (s)
    n: float = 0.3568  # Power law index
    a: float = 2.0  # Yasuda parameter

    def __post_init__(self):
        """Automatically set the correct physics based on the project tier."""
        if self.tier == "tier1":
            self.viscosity_model = "newtonian"
            self.mu_ref = self.mu_newtonian
        elif self.tier in ["tier2", "tier3", "tier3_patients"]:
            self.viscosity_model = "carreau"
            self.mu_ref = self.mu_inf
        else:
            raise ValueError(f"Unknown tier: {self.tier}")

    def get_u_ref(self, d_bar) -> float:
        """Calculate the reference velocity for a given effective diameter."""
        return (self.re_target * self.mu_ref) / (self.rho * d_bar)

    def get_p_ref(self, u_ref) -> float:
        """Calculate the reference pressure scaling."""
        return self.rho * (u_ref ** 2)

    def get_re(self, u_ref, d_bar, mu_custom=None):
        """
        Calculate the Reynolds number.
        Works with both raw floats and PyTorch tensors.
        """
        mu = mu_custom if mu_custom is not None else self.mu_ref
        return (self.rho * u_ref * d_bar) / mu