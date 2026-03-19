from dataclasses import dataclass, field
from typing import Dict, Optional
from src.utils.paths import get_project_root
from pathlib import Path


@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation."""
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

    # Vessel Dimensions
    base_length: float = 0.015
    width_min: float = 0.0015
    width_max: float = 0.0025
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
    """Configuration for physical properties and fluid dynamics."""
    tier: str = "tier1"

    # Fluid Properties
    rho: float = 1106  # kg/m^3
    re_target: float = 75  # Reynolds number [-]

    viscosity_model: str = field(init=False)
    mu_ref: float = field(init=False)

    # Newtonian Reference (tier 1)
    mu_newtonian: float = 0.0035  # Pa*s (3.5 cP)

    # Relaxed Carreau-Yasuda Rheology (Mild Shear-Thinning Proxy)
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
    """Configuration for Tier 3 HiFi dynamic biochemical thrombosis simulations."""
    tier: str = "tier3"

    # --- Initial Concentrations ---
    c_RP0: float = 2.5e8  # Initial resting platelets [plt/ml]
    c_pT0: float = 1.2  # Initial prothrombin concentration [uM]
    c_Fg0: float = 7.0  # Initial fibrinogen concentration [uM]
    cAT0: float = 2.84  # Initial/Static background antithrombin concentration [uM]

    # --- Agonist Critical Thresholds ---
    APScrit: float = 0.6  # Thromboxane critical concentration [uM]
    APRcrit: float = 2.0  # ADP critical concentration for activation [uM]
    Tcrit: float = 5.0e-4  # Thrombin concentration for activation [uM]

    # --- Activation & Adhesion Constants ---
    t_act: float = 1.0  # Activation time [s]
    shear_crit: float = 10000.0  # Threshold for mechanical activation [1/s]
    k_rs: float = 0.0037  # Resting adhesion rate [cm/s]
    k_as: float = 0.045  # Activated adhesion rate [cm/s]

    # --- Thrombin Generation & Inhibition ---
    k_1t: float = 13.33  # Rate constant for AT [1/s]
    c_H: float = 0.25  # Heparin concentration [uM]
    K_at: float = 0.1  # Dissociation constant heparin-AT [uM]
    K_T: float = 0.035  # Dissociation constant heparin-Thrombin [uM]
    phi_at: float = 3.69e-9  # Thrombin generation rate (activated) [U/(plt*s*uM)]
    phi_rt: float = 6.5e-10  # Thrombin generation rate (resting) [U/(plt*s*uM)]
    beta: float = 9.11e-3  # Conversion factor for thrombin [nmol/U]

    # --- Agonist Synthesis & Inactivation ---
    lambda_adp: float = 2.4e-8  # Released ADP per activated platelet [nmol/plt] (renamed from lambda)
    s_t: float = 9.5e-12  # Rate of synthesis of TxA2 [nmol/(s*plt)]
    k_i: float = 0.0161  # Rate of TxA2 inactivation [1/s]

    # --- Fibrin Kinetics ---
    kfi: float = 59.0  # Reaction rate fibrinogen [1/s]
    kmfi: float = 3.16  # Rate constant fibrin reaction [uM]

    # --- Surface Parameters & Diffusion ---
    Minf: float = 7.0e6  # Total deposition capacity / max surface saturation [plt/cm^2]
    d_RBC: float = 5.5e-4  # Keller diffusion coefficient proxy (RBC diameter) [cm]

    # Diffusion Coefficients [cm^2/s]
    D_RP: float = 1.58e-9  # Resting platelets
    D_AP: float = 1.58e-9  # Activated platelets
    D_APR: float = 2.57e-6  # ADP agonist
    D_APS: float = 2.14e-6  # TxA2 agonist
    D_PT: float = 3.32e-7  # Prothrombin
    D_T: float = 4.16e-7  # Thrombin
    D_AT: float = 3.49e-7  # Antithrombin
    D_FI: float = 2.47e-7  # Fibrin
    D_FG: float = 3.10e-7  # Fibrinogen

    # --- Curriculum Learning Bounds ---
    mu_ratio_init: float = 2.0
    mu_ratio_max: float = 80.0

    def __post_init__(self):
        """Validate constraints on biochemical properties if needed."""
        if self.mu_ratio_max <= self.mu_ratio_init:
            raise ValueError("mu_ratio_max must be strictly greater than mu_ratio_init")