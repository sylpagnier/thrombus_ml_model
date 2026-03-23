from dataclasses import dataclass, field
from typing import Dict, Optional
from src.utils.paths import get_project_root
from pathlib import Path


@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation (SI Units: m)."""
    tier: str = "tier1"
    project_root: Path = field(default_factory=get_project_root)
    template_path: Path = field(init=False)
    mesh_input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    graph_output_dir: Path = field(init=False)

    def __post_init__(self):
        """Dynamically resolve paths based on the selected tier."""
        self.template_path = self.project_root / "comsol_models/phase1_template.mph"

        # Explicitly handle tier 3 patient data directory mapping
        if self.tier == "tier3_patients":
            self.mesh_input_dir = self.project_root / "data/raw/tier3_patients"
            self.output_dir = self.project_root / "data/processed/cfd_results_tier3_patients"
            self.graph_output_dir = self.project_root / "data/processed/graphs_tier3_patients"
        else:
            self.mesh_input_dir = self.project_root / f"data/raw/{self.tier}"
            self.output_dir = self.project_root / f"data/processed/cfd_results_{self.tier}"
            self.graph_output_dir = self.project_root / f"data/processed/graphs_{self.tier}"

    # Mesh Settings
    mesh_size_factor: float = 0.75
    mesh_lc: float = 0.00008

    # Vessel Dimensions (Scaled for Portal Vein)
    base_length: float = 0.15
    width_min: float = 0.008
    width_max: float = 0.02
    curvature_amplitude: float = 0.0025

    # Pathology Constraints
    stenosis_factor_min: float = 0.1
    stenosis_factor_max: float = 0.2
    aneurysm_factor_min: float = 0.15
    aneurysm_factor_max: float = 0.2
    num_ctrl_pts: int = 11

    # Physical Group Tags
    TAGS: Dict[str, int] = field(default_factory=lambda: {
        "Inlet": 101,
        "Outlet_1": 102,
        "Walls": 103,
        "Fluid_Domain": 201
    })


@dataclass
class PhysicsConfig:
    """Fluid dynamics configuration (SI Units: kg, m, s, Pa)."""
    tier: str = "tier1"
    rho: float = 1106  # kg/m^3
    re_target: float = 450  # Reynolds number [-]

    viscosity_model: str = field(init=False)
    mu_ref: float = field(init=False)

    # Carreau-Yasuda parameters (SI)
    mu_inf: float = 0.0035  # Pa*s
    mu_0: float = 0.056  # Pa*s
    lam: float = 3.313  # Relaxation time [s]
    n: float = 0.358  # Power law index
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


@dataclass
class BiochemConfig:
    """Configuration for Tier 3 HiFi dynamic biochemical thrombosis simulations [SI units]"""
    tier: str = "tier3"

    # --- Initial Concentrations ---
    # plt/m^3
    # mol/m^3
    c_RP0: float = 2.5e14  # 2.5e8 plt/ml
    c_pT0: float = 0.0012  # 1.2 uM
    c_Fg0: float = 0.007  # 7.0 uM
    cAT0: float = 0.00284  # 2.84 uM

    # --- Agonist Critical Thresholds (mol/m^3) ---
    APScrit: float = 0.0006
    APRcrit: float = 0.002
    Tcrit: float = 5.0e-7

    # --- Activation & Adhesion Constants (m/s) ---
    t_act: float = 1.0
    shear_crit: float = 10000.0
    k_rs: float = 0.000037  # 3.7e-5 m/s
    k_as: float = 0.00045  # 4.5e-4 m/s

    # --- Thrombin Generation & Inhibition ---
    k_1t: float = 13.33
    c_H: float = 0.00025  # mol/m^3
    phi_at: float = 3.69e-6  # Adjusted for mol/m^3 in denominator
    phi_rt: float = 6.5e-7  # Adjusted for mol/m^3 in denominator
    beta: float = 9.11e-12  # mol/U

    # --- Agonist Synthesis & Inactivation ---
    lambda_adp: float = 2.4e-17  # mol/plt
    s_t: float = 9.5e-21  # mol/(s*plt)
    k_i: float = 0.0161

    # --- Fibrin Kinetics ---
    kfi: float = 59.0
    kmfi: float = 0.00316  # 3.16 mol/m^3

    # --- Surface Parameters & Diffusion (m and m^2/s) ---
    Minf: float = 7.0e10  # 7.0e6 plt/cm^2 -> 7.0e10 plt/m^2
    d_RBC: float = 5.5e-6  # 5.5e-4 cm -> 5.5e-6 m

    # Diffusion Coefficients: m^2/s
    D_RP: float = 1.58e-13
    D_AP: float = 1.58e-13
    D_APR: float = 2.57e-10
    D_APS: float = 2.14e-10
    D_PT: float = 3.32e-11
    D_T: float = 4.16e-11
    D_AT: float = 3.49e-11
    D_FI: float = 2.47e-11
    D_FG: float = 3.10e-11