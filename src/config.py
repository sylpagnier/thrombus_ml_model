from dataclasses import dataclass, field
from typing import Dict, Union
from src.utils.paths import get_project_root
from pathlib import Path

# Channel index for effective viscosity in [u, v, p, mu_eff_nd, ...] state / label tensors.
# Convention: mu_eff_nd = mu_eff_si / PhysicsConfig.mu_viscosity_nd_scale (see that property).
STATE_CHANNEL_MU_EFF_ND = 3


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
        self.template_path = self.project_root / "comsol_models/phase1_template.mph"
        if self.tier == "tier3_patients":
            self.mesh_input_dir = self.project_root / "data/raw/tier3_patients"
            self.output_dir = self.project_root / "data/processed/cfd_results_tier3_patients"
            self.graph_output_dir = self.project_root / "data/processed/graphs_tier3_patients"
        else:
            self.mesh_input_dir = self.project_root / f"data/raw/{self.tier}"
            self.output_dir = self.project_root / f"data/processed/cfd_results_{self.tier}"
            self.graph_output_dir = self.project_root / f"data/processed/graphs_{self.tier}"

    mesh_size_factor: float = 0.75
    mesh_lc: float = 1/1000  # [m]

    # Vessel Dimensions
    base_length: float = 0.1  # [m]
    width_min: float = 0.008  # [m]
    width_max: float = 0.02  # [m]
    curvature_amplitude: float = 0.0025  # [m]

    # Pathology Constraints
    stenosis_factor_min: float = 0.1
    stenosis_factor_max: float = 0.2
    aneurysm_factor_min: float = 0.15
    aneurysm_factor_max: float = 0.2
    num_ctrl_pts: int = 50

    # Physical Group Tags
    TAGS: Dict[str, int] = field(default_factory=lambda: {
        "Inlet": 101, "Outlet_1": 102, "Walls": 103, "Fluid_Domain": 201
    })


@dataclass
class PhysicsConfig:
    """Configuration for physical properties and fluid dynamics.

    Reynolds / velocity reference uses ``mu_ref`` (``get_u_ref``, ``get_re``). Stored labels and
    predictions use ``mu_viscosity_nd_scale`` for channel ``STATE_CHANNEL_MU_EFF_ND``: non-dimensional
    viscosity is always ``mu_eff_si / mu_viscosity_nd_scale`` so Re scaling can change without
    relabeling graphs.
    """
    tier: str = "tier1"

    # --- Unit Conversion Scales (COMSOL CGS to SI) ---
    cm_to_m: float = 0.01
    cgs_p_to_pa: float = 0.1
    cgs_mu_to_pa_s: float = 0.1

    # Mesh nodes farther than this from a COMSOL export coordinate are not treated as labeled (Tier 3 patients).
    comsol_spatial_match_tol_m: float = 1e-4

    # Fluid Properties
    rho: float = 1106.0  # [kg/m^3]
    re_target: float = 450  # Reynolds number [-]
    viscosity_model: str = field(init=False)
    # Viscosity [Pa*s] used in Re / u_ref; may differ from mu_viscosity_nd_scale if extended.
    mu_ref: float = field(init=False)
    mu_newtonian: float = 0.0035  # [Pa*s]

    # Carreau-Yasuda Rheology (Mild Shear-Thinning Proxy)
    mu_inf: float = 0.0035  # [Pa*s]
    mu_0: float = 0.056  # [Pa*s]
    lam: float = 3.313  # Relaxation time [s]
    n: float = 0.358  # Power law index
    a: float = 2.0  # Yasuda parameter

    # Carreau momentum residual: if True, ∂μ/∂x does not backprop into predicted μ (PINN-style stability).
    detach_mu_for_ns_gradient: bool = True

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

    @property
    def mu_viscosity_nd_scale(self) -> float:
        """SI viscosity [Pa*s] that non-dimensionalizes ``mu_eff`` in labels (channel STATE_CHANNEL_MU_EFF_ND).

        Carreau: ``mu_inf``. Newtonian (tier 1): ``mu_newtonian``.
        """
        if self.viscosity_model == "carreau":
            return self.mu_inf
        return self.mu_newtonian

    def viscosity_si_to_nd(self, mu_si: Union[float, "torch.Tensor"]):
        """Effective viscosity SI [Pa*s] → non-dimensional label scale (same as stored in graphs)."""
        return mu_si / self.mu_viscosity_nd_scale

    def viscosity_nd_to_si(self, mu_nd: Union[float, "torch.Tensor"]):
        """Non-dimensional ``mu_eff`` (channel STATE_CHANNEL_MU_EFF_ND) → SI [Pa*s]."""
        return mu_nd * self.mu_viscosity_nd_scale

    def get_u_ref(self, d_bar) -> float:
        """Calculate the reference velocity for a given effective diameter."""
        return (self.re_target * self.mu_ref) / (self.rho * d_bar)

    def get_p_ref(self, u_ref) -> float:
        """Calculate the reference pressure scaling."""
        return self.rho * (u_ref ** 2)

    def get_re(self, u_ref, d_bar, mu_custom=None):
        """
        Calculate the Reynolds number
        """
        mu = mu_custom if mu_custom is not None else self.mu_ref
        return (self.rho * u_ref * d_bar) / mu


@dataclass
class BiochemConfig:
    """Tier 3 biochemical + rheology parameters aligned with the COMSOL phase-2 model.

    COMSOL global parameters are entered in a **CGS-style** convention for that file
    (lengths in cm, diffusion in cm^2/s, adhesion in cm/s, platelet density in plt/cm^2,
    many solute fields in uM). The **.mph does not non-dimensionalize** those species
    or transport inputs: consistency is by unit-aware parameters in COMSOL.

    This dataclass and ``physics_kernels_tier3`` use **SI** (m, m^2/s, m/s, mol/m^3, plt/m^2).
    Conversions are centralized (e.g. ``d_scale`` for D coefficients, ``cm_to_m`` for
    adhesion rates in kernels, ``bulk_scale`` / ``surface_scale`` for log1p species encoding).
    """

    tier: str = "tier3"

    # --- Centralized Scales ---
    bulk_scale: float = 1e6      # log1p encoding: nondim species * bulk_scale -> SI [mol/m^3] or plt/m^3
    surface_scale: float = 1e4   # wall species: nondim * surface_scale -> SI [plt/m^2]
    d_scale: float = 1e-4        # COMSOL D [cm^2/s] -> SI [m^2/s] (multiply by (cm_to_m)^2)

    # --- Temporal simulation (per-graph truth is authoritative) ---
    # Prefer storing COMSOL sample times on each graph as ``data.t`` [s]; lengths can differ
    # between patients/runs. These fields are fallbacks when ``data.t`` is missing.
    t_final: float = 6000  # Default horizon [s] if graph has no ``data.t``
    num_time_steps: int = 60  # Default count for synthetic linspace if not set by export

    # --- Initial Concentrations ---
    c_RP0: float = 2.5e14  # Initial resting platelets [plt/m^3]
    c_pT0: float = 1.2e-3  # Initial prothrombin concentration [mol/m^3]
    c_Fg0: float = 7.0e-3  # Initial fibrinogen concentration [mol/m^3]
    cAT0: float = 2.84e-3  # Initial/Static background antithrombin concentration [mol/m^3]

    # --- Agonist Critical Thresholds ---
    APScrit: float = 0.6e-3  # Thromboxane critical concentration [mol/m^3]
    APRcrit: float = 2.0e-3  # ADP critical concentration for activation [mol/m^3]
    Tcrit: float = 5.0e-7  # Thrombin concentration for activation [mol/m^3]

    # --- Activation & Adhesion Constants ---
    t_act: float = 1.0  # Activation time [s]
    shear_crit: float = 10000.0  # Threshold for mechanical activation [1/s]
    k_rs: float = 3.7e-5  # Resting adhesion rate [m/s]
    k_as: float = 4.5e-4  # Activated adhesion rate [m/s]

    # --- Thrombin Generation & Inhibition ---
    k_1t: float = 13.33  # Rate constant for AT [1/s]
    c_H: float = 0.25e-3  # Heparin concentration [mol/m^3]
    K_at: float = 0.1e-3  # Dissociation constant heparin-AT [mol/m^3]
    K_T: float = 0.035e-3  # Dissociation constant heparin-Thrombin [mol/m^3]
    phi_at: float = 3.69e-6  # Thrombin generation rate (activated) [U/(plt*s*(mol/m^3))]
    phi_rt: float = 6.5e-7  # Thrombin generation rate (resting) [U/(plt*s*(mol/m^3))]
    beta: float = 9.11e-12  # Conversion factor for thrombin [mol/U]

    # --- Agonist Synthesis & Inactivation ---
    lambda_adp: float = 2.4e-17  # Released ADP per activated platelet [mol/plt]
    s_t: float = 9.5e-21  # Rate of synthesis of TxA2 [mol/(s*plt)]
    k_i: float = 0.0161  # Rate of TxA2 inactivation [1/s]

    # --- Fibrin Kinetics ---
    kfi: float = 59.0  # Reaction rate fibrinogen [1/s]
    kmfi: float = 3.16e-3  # Rate constant fibrin reaction [mol/m^3]
    # SI scale for (1 - FI/C_max) taper; COMSOL "Sat" is often a 0–1 proxy — here use ~ fibrinogen scale.
    fi_reaction_saturation_si: float = 7.0e-3  # [mol/m^3], default matches c_Fg0

    # --- Surface Parameters & Diffusion ---
    Minf: float = 7.0e10  # Total deposition capacity / max surface saturation [plt/m^2]
    d_RBC: float = 5.5e-6  # Keller diffusion coefficient proxy (RBC diameter) [m]

    # Diffusion Coefficients [m^2/s]
    D_RP: float = 1.58e-13  # Resting platelets
    D_AP: float = 1.58e-13  # Activated platelets
    D_APR: float = 2.57e-10  # ADP agonist
    D_APS: float = 2.14e-10  # TxA2 agonist
    D_PT: float = 3.32e-11  # Prothrombin
    D_T: float = 4.16e-11  # Thrombin
    D_AT: float = 3.49e-11  # Antithrombin
    D_FI: float = 2.47e-11  # Fibrin
    D_FG: float = 3.10e-11  # Fibrinogen

    # Surface & Flow Pathology Constants mapped from COMSOL
    gamma_m: float = 150.0  # Reference shear rate for scaling [1/s]
    lss: float = 25.0  # Low shear rate threshold for stagnation [1/s]
    sgt: float = -750.0  # Spatial shear gradient threshold [1/(m*s)]
    L_char: float = 0.00075  # Characteristic length scale (converted 0.075 cm to m)
    k_aa: float = 0.045  # Adhesion rate for activated platelets on Mas [m/s]

    # Curriculum Learning Bounds (Corrected to match COMSOL mu1/mu2 max values)
    mu_ratio_init: float = 2.0
    mu_ratio_max: float = 80.0  # COMSOL mu1 and mu2 step functions max out at 80

    # Dual-trigger effective viscosity (COMSOL step proxies; shared by GNODE_Tier3 + penalty loss)
    viscosity_mat_crit: float = 2e7
    viscosity_fi_crit: float = 0.6
    viscosity_gnode_temp_mat: float = 1e6
    viscosity_gnode_temp_fi: float = 0.05
    # Soft-step temperatures in BiochemPhysicsKernels.compute_dual_viscosity_penalty
    viscosity_penalty_soft_temp_mat: float = 7e6
    viscosity_penalty_soft_temp_fi: float = 0.01

    # Soft-step temperatures for BiochemKinetics (STE forward / sigmoid backward). T_scale multiplies all.
    soft_step_T_omega: float = 0.05
    soft_step_T_shear: float = 500.0
    soft_step_T_grad: float = 50.0
    soft_step_T_low_shear: float = 5.0
    soft_step_T_scale: float = 1.0

    def __post_init__(self):
        """Validate constraints on biochemical properties if needed."""
        if self.mu_ratio_max <= self.mu_ratio_init:
            raise ValueError("mu_ratio_max must be strictly greater than mu_ratio_init")

    def get_species_scales(self, device='cpu'):
        """Dynamically generates the scaling tensor for ND transformations."""
        import torch
        return torch.tensor([
            self.c_RP0 * self.bulk_scale, self.c_RP0 * self.bulk_scale,
            self.APRcrit * self.bulk_scale, self.APScrit * self.bulk_scale,
            self.c_pT0 * self.bulk_scale, self.c_pT0 * self.bulk_scale,
            self.cAT0 * self.bulk_scale, self.c_Fg0 * self.bulk_scale,
            self.c_Fg0 * self.bulk_scale,
            self.Minf * self.surface_scale, self.Minf * self.surface_scale,
            self.Minf * self.surface_scale
        ], dtype=torch.float32, device=device)

    def get_adr_norm_scales(self, device='cpu'):
        """Scales for ADR residual Huber (SI concentration-like); thrombin uses Tcrit, not c_pT0."""
        import torch
        s = self.get_species_scales(device=device).clone()
        s[5] = max(self.Tcrit * self.bulk_scale, 1e-18)
        return s


@dataclass
class CurriculumConfig:
    """Training curriculum schedules (keeps magic numbers out of training loops)."""

    # Tier 2: Carreau index n during viscosity distillation (anneals toward PhysicsConfig.n)
    tier2_carreau_n_distill_start: float = 0.8

    # Tier 3: warmup / teacher forcing / T_scale schedule
    tier3_warmup_epochs: int = 10
    tier3_teacher_force_decay_epochs: int = 20
    tier3_t_scale_warmup_initial: float = 10.0
    tier3_t_scale_warmup_final: float = 8.0
    tier3_t_scale_coupled_initial: float = 8.0
    tier3_t_scale_coupled_final: float = 1.0

    # Tier 3: Kendall loss weighter — freeze during warmup; bound effective precisions
    tier3_weighter_freeze_during_warmup: bool = True
    # Cap exp(-log_var) for physics tasks (indices 0–5: ADR_F, ADR_S, W_Bio, W_Phy, Bio_IO, NS_mom).
    tier3_physics_precision_ceiling: float = 100.0
    # Floor exp(-log_var) for supervised tasks (indices 6–7: Data_Kine, Data_Bio).
    tier3_data_precision_floor: float = 0.12