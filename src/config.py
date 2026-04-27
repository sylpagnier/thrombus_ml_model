from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple, Union
from src.utils.paths import comsol_models_dir, data_root, get_project_root
from pathlib import Path

# Channel index for effective viscosity in [u, v, p, mu_eff_nd, ...] state / label tensors.
# Convention: mu_eff_nd = mu_eff_si / PhysicsConfig.mu_viscosity_nd_scale (see that property).
STATE_CHANNEL_MU_EFF_ND = 3


class PredChannels:
    """Canonical channel indices for kinematic predictor outputs."""

    U = 0
    V = 1
    P = 2
    MU_EFF_ND = STATE_CHANNEL_MU_EFF_ND
    WSS = 4
    UV = slice(U, V + 1)
    KINEMATICS = slice(U, P + 1)
    ALL_PHYSICS = slice(U, WSS + 1)


class NodeFeat:
    """Canonical node-feature slices for Kinematics/2 predictor inputs."""

    XY = slice(0, 2)
    SDF = slice(2, 3)
    SHEAR_POT = slice(3, 4)
    WALL_NORMAL = slice(4, 6)
    REST = slice(6, 11)
    UV_PRIOR = slice(11, 13)
    MU_PRIOR = slice(13, 14)
    WSS_PRIOR = slice(14, 15)
    # Local hydraulic width D(x) (non-dim) and derivatives along flow direction (WLS-consistent).
    WIDTH_ND = slice(15, 16)
    WIDTH_D1 = slice(16, 17)
    WIDTH_D2 = slice(17, 18)


class BiochemNodeFeat:
    """Canonical node-feature slices for Biochem graph inputs."""

    XY = slice(0, 2)
    SDF = slice(2, 3)
    WALL_NORMAL = slice(3, 5)


class BulkSpecies(IntEnum):
    RP = 0
    AP = 1
    APR = 2
    APS = 3
    PT = 4
    T = 5
    AT = 6
    FG = 7
    FI = 8


class WallSpecies(IntEnum):
    M = 0
    Mas = 1
    Mat = 2


BULK_SPECIES_ORDER: Tuple[BulkSpecies, ...] = (
    BulkSpecies.RP,
    BulkSpecies.AP,
    BulkSpecies.APR,
    BulkSpecies.APS,
    BulkSpecies.PT,
    BulkSpecies.T,
    BulkSpecies.AT,
    BulkSpecies.FG,
    BulkSpecies.FI,
)

SPECIES_GROUPS = {
    "fast": (
        BulkSpecies.RP,
        BulkSpecies.AP,
        BulkSpecies.APR,
        BulkSpecies.APS,
        BulkSpecies.T,
    ),
    "slow": (
        BulkSpecies.AT,
        BulkSpecies.FG,
        BulkSpecies.FI,
    ),
    "keller": (
        BulkSpecies.RP,
        BulkSpecies.AP,
        BulkSpecies.PT,
        BulkSpecies.T,
        BulkSpecies.AT,
    ),
    "solid": (BulkSpecies.FI,),
}

PHASE_DEFAULT_MESH_SIZE_FACTOR: Dict[str, float] = {
    "kinematics": 0.75,
}


def _map_phase_to_phase(value: str) -> str:
    v = (value or "").strip().lower()
    if v in ("kinematics", "biochem"):
        return v
    if v == "kinematics":
        return "kinematics"
    if v == "biochem":
        return "biochem"
    raise ValueError(f"Unknown phase: {value}")


@dataclass
class VesselConfig:
    """Central configuration for vessel geometry and mesh generation."""
    phase: str = "kinematics"
    project_root: Path = field(default_factory=get_project_root)
    template_path: Path = field(init=False)
    mesh_input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    graph_output_dir: Path = field(init=False)

    def __post_init__(self):
        self.template_path = comsol_models_dir() / "phase1_template.mph"
        self.phase = _map_phase_to_phase(self.phase)
        self.mesh_size_factor = PHASE_DEFAULT_MESH_SIZE_FACTOR.get(self.phase, self.mesh_size_factor)
        dr = data_root()
        if self.phase == "kinematics":
            self.mesh_input_dir = dr / "raw/kinematics/meshes"
            self.output_dir = dr / "processed/cfd_results_kinematics"
            self.graph_output_dir = dr / "processed/graphs_kinematics"
        else:
            self.mesh_input_dir = dr / "raw/biochem"
            self.output_dir = dr / "processed/cfd_results_biochem"
            self.graph_output_dir = dr / "processed/graphs_biochem"
        
        # --- RESTORED: Mesh Sweep Override ---
        import os
        if "GMSH_SIZE_FACTOR" in os.environ:
            self.mesh_size_factor = float(os.environ["GMSH_SIZE_FACTOR"])

    mesh_size_factor: float = 0.75
    mesh_lc: float = 1/1000  # [m]

    # Vessel Dimensions
    base_length: float = 0.1  # [m]
    width_min: float = 0.008  # [m]
    width_max: float = 0.02  # [m]
    curvature_amplitude: float = 0.0025  # [m]

    # Pathology Constraints (Gaussian depth scales vs nominal ``width``; not a lumen floor)
    stenosis_factor_min: float = 0.1
    stenosis_factor_max: float = 0.2
    # Hard floor on local lumen width vs nominal ``width`` (mesh / degeneracy guard; noise stacks on pathology)
    min_lumen_width_fraction: float = 0.2
    aneurysm_factor_min: float = 0.15
    aneurysm_factor_max: float = 0.2
    num_ctrl_pts: int = 50

    # ND hydraulic priors (``graph_velocity_priors``): characteristic half-width and worst-case radius
    # fraction vs nominal (e.g. severe stenosis). Used to set a physical floor on inferred R_nd.
    nominal_radius: float = 0.5
    min_radius_factor: float = 0.2

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
    phase: str = "kinematics"
    # Optional rheology override for kinematics phase ("newtonian" or "carreau").
    rheology: str | None = None

    # --- Unit Conversion Scales (COMSOL CGS to SI) ---
    cm_to_m: float = 0.01
    cgs_p_to_pa: float = 0.1
    cgs_mu_to_pa_s: float = 0.1

    # Mesh nodes farther than this from a COMSOL export coordinate are not treated as labeled (Biochem patients).
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
    # GINO graph-attention geometric modulation constants.
    gino_edge_decay_k: float = 5.0
    gino_curve_log_clamp_min: float = 1e-4
    gino_rheo_log_clamp_min: float = 1e-3
    gino_adv_log_clamp_min: float = 1e-3

    def __post_init__(self):
        """Automatically set the correct physics based on the project phase."""
        self.phase = _map_phase_to_phase(self.phase)
        if self.phase == "kinematics":
            mode_raw = (self.rheology or "carreau").strip().lower()
            if mode_raw not in {"newtonian", "carreau"}:
                raise ValueError(f"Unknown kinematics rheology: {self.rheology}")
            if mode_raw == "newtonian":
                self.viscosity_model = "newtonian"
                self.mu_ref = self.mu_newtonian
                self.n = 1.0
            else:
                self.viscosity_model = "carreau"
                self.mu_ref = self.mu_inf
        elif self.phase == "biochem":
            self.viscosity_model = "carreau"
            self.mu_ref = self.mu_inf
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    @property
    def mu_viscosity_nd_scale(self) -> float:
        """SI viscosity [Pa*s] that non-dimensionalizes ``mu_eff`` in labels (channel STATE_CHANNEL_MU_EFF_ND).

        Carreau: ``mu_inf``. Newtonian (phase 1): ``mu_newtonian``.
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
        """Reynolds number from stored reference scales: ``rho * u_ref * d_bar / mu``.

        This is **not** the same as reading ``re_target`` off the config: ``re_target`` is used when
        **building** graphs via ``get_u_ref(d_bar)`` so that, for consistent ``(u_ref, d_bar)``,
        this call returns ``re_target`` again. If a graph stores inconsistent or zero ``u_ref`` /
        ``d_bar``, this value can be zero or nonsensical regardless of ``re_target``.
        """
        mu = mu_custom if mu_custom is not None else self.mu_ref
        return (self.rho * u_ref * d_bar) / mu


@dataclass
class BiochemConfig:
    """Biochem biochemical + rheology parameters aligned with the COMSOL phase-2 model.

    COMSOL global parameters are entered in a **CGS-style** convention for that file
    (lengths in cm, diffusion in cm^2/s, adhesion in cm/s, platelet density in plt/cm^2,
    many solute fields in uM). The **.mph does not non-dimensionalize** those species
    or transport inputs: consistency is by unit-aware parameters in COMSOL.

    This dataclass and ``physics_kernels_biochem`` use **SI** (m, m^2/s, m/s, mol/m^3, plt/m^2).
    Conversions are centralized (e.g. ``d_scale`` for D coefficients, ``cm_to_m`` for
    adhesion rates in kernels, ``bulk_scale`` / ``surface_scale`` for log1p species encoding).
    """

    phase: str = "biochem"

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
    # COMSOL parameter is -750 [1/(cm*s)] -> SI is -7.5e4 [1/(m*s)].
    sgt: float = -7.5e4  # Spatial shear gradient threshold [1/(m*s)]
    # COMSOL parameter is 0.075 [cm] -> SI is 7.5e-4 [m].
    L_char: float = 7.5e-4  # Characteristic length scale [m]
    # COMSOL parameter is 0.045 [cm/s] -> SI is 4.5e-4 [m/s].
    k_aa: float = 4.5e-4  # Adhesion rate for activated platelets on Mas [m/s]

    # Curriculum Learning Bounds (Predictor-Corrector Architecture)
    mu_ratio_init: float = 1.0  # Kine phase: Rheologically neutral flow field
    mu_ratio_max: float = 80.0  # COMSOL mu1 and mu2 step functions max out at 80

    # Dual-trigger effective viscosity (COMSOL step proxies; shared by GNODE_Phase3 + penalty loss)
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
    # Shared Huber delta for Biochem biochemical residuals.
    biochem_huber_delta: float = 1.0
    # Optional non-zero slope keeps adhesion gradients alive when M_tot exceeds Minf.
    availability_negative_slope: float = 0.0

    def __post_init__(self):
        """Validate constraints on biochemical properties if needed."""
        self.phase = _map_phase_to_phase(self.phase)
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

    def _ensure_strictly_increasing_times(self, t):
        """Force strictly increasing timestamps with representable float32 deltas."""
        import torch

        dev = t.device
        dtype = t.dtype
        t1d = t.reshape(-1).contiguous().to(dtype).cpu()
        n = t1d.numel()
        if n <= 1:
            return t.reshape(-1).contiguous().to(dtype).to(dev)
        out = t1d.clone()
        pinf = torch.tensor(float("inf"), dtype=dtype)
        for i in range(1, n):
            lo = out[i - 1]
            if not bool((out[i] > lo).item()):
                out[i] = torch.nextafter(lo, pinf)
        return out.to(device=dev)

    def resolve_biochem_times(self, data, device):
        """Return physical timestamps [s] with length data.y.shape[0]."""
        import torch
        import warnings

        t_steps = int(data.y.shape[0])
        if hasattr(data, "t") and data.t is not None and data.t.numel() > 0:
            t = data.t.to(device=device, dtype=torch.float32).reshape(-1)
            if t.numel() == t_steps:
                return self._ensure_strictly_increasing_times(t)
            t_last = float(t[-1].item()) if t.numel() else float(self.t_final)
            warnings.warn(
                f"data.t length {t.numel()} != y time dim {t_steps}; "
                f"using linspace(0, {t_last:g}, {t_steps}). Re-export graphs with aligned t.",
                stacklevel=2,
            )
            return self._ensure_strictly_increasing_times(
                torch.linspace(0.0, t_last, steps=t_steps, device=device, dtype=torch.float32)
            )
        return self._ensure_strictly_increasing_times(
            torch.linspace(0.0, float(self.t_final), steps=t_steps, device=device, dtype=torch.float32)
        )


@dataclass
class CurriculumConfig:
    """Training curriculum schedules (keeps magic numbers out of training loops)."""

    # Kinematics: Carreau index n during viscosity distillation (anneals toward PhysicsConfig.n)
    kinematics_carreau_n_distill_start: float = 0.8

    # Biochem: warmup / teacher forcing / T_scale schedule
    biochem_warmup_epochs: int = 10
    biochem_teacher_force_decay_epochs: int = 20
    biochem_t_scale_warmup_initial: float = 10.0
    biochem_t_scale_warmup_final: float = 8.0
    biochem_t_scale_coupled_initial: float = 8.0
    biochem_t_scale_coupled_final: float = 1.0

    # Biochem: Kendall loss weighter — freeze during warmup; bound effective precisions
    biochem_weighter_freeze_during_warmup: bool = True
    # Cap exp(-log_var) for physics tasks (indices 0–5: ADR_F, ADR_S, W_Bio, W_Phy, Bio_IO, NS_mom).
    biochem_physics_precision_ceiling: float = 100.0
    # Floors for specific physics terms that must not be down-weighted too aggressively.
    # These apply to ADR_S (index 1) and W_Phy (index 3) only.
    biochem_adr_s_precision_floor: float = 1.0
    biochem_w_phys_precision_floor: float = 1.0
    # Floor exp(-log_var) for supervised tasks (indices 6–7: Data_Kine, Data_Bio).
    biochem_data_precision_floor: float = 0.12

    # Biochem: smoother curriculum than piecewise-linear (reduces loss cliffs).
    # ``linear`` | ``smoothstep`` | ``cosine`` — applies to mu_ratio ramp and T_scale segments.
    biochem_curriculum_easing: str = "smoothstep"
    # Minimum unique anchor graphs before validation Dice / WSS are treated as generalization metrics.
    biochem_min_anchors_for_trusted_metrics: int = 2
    # After main warmup, keep physics-task log_vars frozen for extra epochs (data heads tune first).
    biochem_weighter_physics_grace_epochs: int = 3
    # Divide ADR/wall/IO/NS residuals by sqrt(num_nodes) so graphs of different sizes are comparable.
    biochem_physics_geom_normalization: bool = True