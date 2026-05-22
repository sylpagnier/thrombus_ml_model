import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from torch_geometric.utils import degree
# ``odeint`` OOMs on large graphs; ``odeint_adjoint`` fixes that. For short TBPTT teacher
# windows, ``BIOCHEM_ODEINT_USE_ADJOINT=0`` uses dense ``odeint`` backward (more VRAM,
# often stabler than adjoint on stiff segments / low RK substep counts).
from torchdiffeq import odeint, odeint_adjoint
from src.architecture.ginodeq import GINOBlock, SpectralLinear
from src.config import BiochemConfig, NodeFeat
from src.core_physics.kinematics_clot_prior import clot_prior_features, clot_prior_score_flat
from src.utils.batching import get_batch_tensor

# Matches BiochemPhysicsKernels: species channels are log1p(species_nd).
_SPECIES_LOG1P_MIN = -10.0
_SPECIES_LOG1P_MAX = 8.0


def _biochem_ode_grad_checkpoint_enabled() -> bool:
    """Recompute GINO derivative block during backward to lower peak VRAM (more compute).

    Set ``BIOCHEM_ODE_GRADIENT_CHECKPOINT=1`` during biochem / teacher training on tight GPUs.
    """
    v = (os.environ.get("BIOCHEM_ODE_GRADIENT_CHECKPOINT", "0") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _biochem_gelation_prior_gate_enabled() -> bool:
    """When true (default), scale FI/Mat + learned gelation by wall-local kinematic clot-risk prior.

    Set ``BIOCHEM_GELATION_PRIOR_GATE=0`` to restore legacy behaviour (gelation can lift μ everywhere).
    """
    raw = (os.environ.get("BIOCHEM_GELATION_PRIOR_GATE", "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _biochem_delta_mu_head_enabled() -> bool:
    """Enable residual viscosity correction head on top of analytic rheology.

    Set ``BIOCHEM_USE_DELTA_MU_HEAD=1`` to apply a bounded multiplicative correction.
    """
    raw = (os.environ.get("BIOCHEM_USE_DELTA_MU_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _biochem_split_mu_regime_head_enabled() -> bool:
    """Enable split residual log-μ heads with trigger-gated bulk/tail mixing."""
    raw = (os.environ.get("BIOCHEM_USE_SPLIT_MU_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _biochem_wall_delta_head_enabled() -> bool:
    """Enable an extra near-wall residual log-μ correction branch."""
    raw = (os.environ.get("BIOCHEM_USE_WALL_DELTA_HEAD", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def biochem_truth_node_mask(batch, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Nodes whose entries in ``y`` are trusted COMSOL labels (per-node mask or graph-level ``is_anchor``)."""
    if not hasattr(batch, "is_anchor"):
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    m = batch.is_anchor
    if not torch.is_tensor(m):
        m = torch.tensor(m, dtype=torch.bool)
    m = m.reshape(-1)
    if m.numel() == 1:
        if bool(m.item()):
            return torch.ones(num_nodes, dtype=torch.bool, device=device)
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    batch_idx = getattr(batch, "batch", None)
    if batch_idx is not None:
        return m[batch_idx].to(device)
    if m.shape[0] != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    return m.to(device)


def _default_resting_species(num_nodes: int, device: torch.device, batch) -> torch.Tensor:
    current_species = torch.zeros(num_nodes, 12, dtype=torch.float32, device=device)
    # COMSOL-consistent resting blood chemistry in ND/log1p space.
    current_species[:, 0] = math.log1p(1.0)   # RP
    current_species[:, 1] = math.log1p(0.05)  # AP
    current_species[:, 4] = math.log1p(1.0)   # PT
    current_species[:, 6] = math.log1p(1.0)   # AT
    current_species[:, 7] = math.log1p(1.0)   # FG
    if hasattr(batch, "bio_inlet_bc"):
        mask_inlet = batch.mask_inlet.view(-1).bool()
        current_species[mask_inlet, 0:9] = batch.bio_inlet_bc[mask_inlet]
    return current_species


class BioODEFunc(nn.Module):
    """
    Calculates the temporal derivative dz/dt of the biochemical latent state.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # Processor to compute spatial interactions for the derivative
        # Plain Linear (no spectral norm): ODE inner loop runs many times per dopri5 step;
        # spectral_norm power iterations + odeint backprop peak GPU memory.
        self.derivative_processor = GINOBlock(latent_dim, use_spectral_norm=False)
        # Start from a near-steady system (dz/dt ~ 0) and let training grow dynamics.
        self.derivative_scale = nn.Parameter(torch.tensor(1e-5, dtype=torch.float32))
        # Memory-safe accumulator for latent derivative magnitude.
        # We only store detached scalar stats (not full tensors per ODE eval).
        self.derivative_energy_sum = 0.0
        self.derivative_eval_count = 0

    def _derivative_gino_block(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_idx: Tensor,
        mb: Tensor,
    ) -> Tensor:
        n_e = int(edge_index.shape[1])
        mod_dummy = torch.zeros(n_e, 1, dtype=torch.float32, device=z.device)
        return self.derivative_processor(z, edge_index, edge_attr, batch_idx, mb, mod_dummy, mod_dummy)

    def forward(self, t, z, edge_index, edge_attr, batch_idx, mod_biochem=None):
        # The derivative dz/dt depends strictly on the current biochemical state 'z' and frozen physics.
        z = torch.clamp(z, min=-20.0, max=20.0)
        n_e = int(edge_index.shape[1])
        mod_dummy = torch.zeros(n_e, 1, dtype=torch.float32, device=z.device)
        mod_in = mod_biochem if mod_biochem is not None else mod_dummy
        if self.training and _biochem_ode_grad_checkpoint_enabled():
            dz_raw = checkpoint(
                self._derivative_gino_block,
                z,
                edge_index,
                edge_attr,
                batch_idx,
                mod_in,
                use_reentrant=False,
            )
        else:
            dz_raw = self._derivative_gino_block(z, edge_index, edge_attr, batch_idx, mod_in)

        # Explicitly smooth latent states to suppress high-frequency spatial jitter during integration.
        row, col = edge_index
        deg = degree(row, z.size(0), dtype=z.dtype).clamp_(min=1.0)
        laplacian_smooth = torch.zeros_like(z)
        z_j = z[col]
        laplacian_smooth.scatter_add_(
            0,
            row.unsqueeze(-1).expand(-1, z.size(-1)),
            z_j / deg[col].unsqueeze(-1),
        )
        laplacian_smooth = laplacian_smooth - z
        dz_raw = dz_raw + 0.05 * laplacian_smooth

        dz_dt = self.derivative_scale * dz_raw
        dz_dt = torch.clamp(dz_dt, min=-10.0, max=10.0)
        if self.training:
            self.derivative_energy_sum += float(dz_dt.detach().pow(2).mean().item())
            self.derivative_eval_count += 1

        return dz_dt

class GNODE_Phase3(nn.Module):
    """
    Biochem Physics-Informed Graph Neural ODE for dynamic Thrombosis Simulation.
    Replaces the steady-state DEQ with a continuous-time latent ODE solver.
    """

    def __init__(
        self,
        phys_cfg,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=25,
        bio_encoder_prior_dim: int = 0,
        mu_ratio_max=80.0,
        mat_crit=2e7,
        fi_crit=0.6,
        temp_mat=1e6,
        temp_fi=0.05,
        rtol=1e-3,
        atol=1e-4,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_inner_iters = max_inner_iters
        self.rtol = rtol
        self.atol = atol
        self.phys_cfg = phys_cfg  # Store physics config internally
        # Micro-step ODE restart behavior:
        # - "rk4" is restart-friendly for short segments.
        # - set BIOCHEM_ODE_METHOD=implicit_adams to preserve legacy behavior.
        self.micro_ode_method = str(os.environ.get("BIOCHEM_ODE_METHOD", "rk4")).strip().lower()
        # Optional TBPTT-style truncation at macro boundaries.
        self.detach_macro_state_default = bool(int(os.environ.get("BIOCHEM_DETACH_MACRO_STATE", "0")))

        # COMSOL Step Function Approximations (Dual-Trigger)
        self.mu_ratio_max = mu_ratio_max
        self.mat_crit = mat_crit
        self.fi_crit = fi_crit

        # Temperature scaling for soft sigmoids
        self.temp_mat = temp_mat
        self.temp_fi = temp_fi
        self.T_scale = 1.0

        # ==========================================
        # 1. KINEMATICS BACKBONE (FROZEN)
        # ==========================================
        self.edge_decay_k = float(self.phys_cfg.gino_edge_decay_k)
        self.curve_log_clamp_min = float(self.phys_cfg.gino_curve_log_clamp_min)
        self.rheo_log_clamp_min = float(self.phys_cfg.gino_rheo_log_clamp_min)
        self.adv_log_clamp_min = float(self.phys_cfg.gino_adv_log_clamp_min)

        self.num_fourier_freqs = 8
        freqs = (2.0 ** torch.arange(self.num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 5 * self.num_fourier_freqs * 2
        encoded_channels = (15 - 5) + 5 + fourier_channels

        self.kin_encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.kin_processor = GINOBlock(latent_dim, edge_dim=3, num_global_tokens=16)
        self.kinematics_decoder = nn.Linear(latent_dim, 3)
        self.mu_encoder = nn.Linear(1, latent_dim)
        self.z_prior_proj = SpectralLinear(4, latent_dim)

        # ==========================================
        # 2. BIOCHEMISTRY NEURAL ODE
        # ==========================================
        # Initial Condition Encoder (Maps spatial config to z0)
        self.bio_encoder_prior_dim = max(0, int(bio_encoder_prior_dim))
        self._bio_cfg = BiochemConfig(phase="biochem")
        self.bio_encoder = SpectralLinear(
            in_features=in_channels + 3 + spatial_channels + self.bio_encoder_prior_dim,
            out_features=latent_dim,
        )
        _bio = self._bio_cfg
        self.sgt = _bio.sgt
        self.T_grad = _bio.soft_step_T_grad
        # 5x attention multiplier for edges in decelerating zones (log-space bias)
        self.biochem_attention_boost = math.log(5.0)

        # The ODE Function
        self.ode_func = BioODEFunc(latent_dim)

        # Physical Decoder
        self.biochem_decoder = SpectralLinear(in_features=latent_dim, out_features=12)

        # Kinematic baseline + biochem-only corrector: μ_eff = μ_kin * (1 + explicit_gel + learned_gel).
        # ``learned_clot_penalty`` sees only log1p species (12); Softplus ⇒ nonnegative learned gelation.
        self.learned_clot_penalty = nn.Sequential(
            nn.Linear(12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self._init_learned_clot_penalty_near_zero()
        # Optional residual log-space correction for μ; kept near-zero by default.
        self.mu_delta_head = nn.Sequential(
            nn.Linear(latent_dim + 12, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Optional split residual correction:
        # log(μ) = log(μ_base) + (1-g)*Δ_bulk + g*Δ_tail
        # where g∈[0,1] is a trigger gate from species + mechanics cues.
        self.mu_delta_bulk_head = nn.Sequential(
            nn.Linear(latent_dim + 12, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.mu_delta_tail_head = nn.Sequential(
            nn.Linear(latent_dim + 16, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.mu_trigger_gate_head = nn.Sequential(
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Optional near-wall residual branch.
        # Uses kinematics + species + wall cues to correct underfit wall viscosity.
        self.mu_delta_wall_head = nn.Sequential(
            nn.Linear(latent_dim + 16, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.mu_trigger_gate_temp = max(float(os.environ.get("BIOCHEM_MU_TRIGGER_GATE_TEMP", "0.20")), 1e-5)
        self.mu_delta_log_clip = max(float(os.environ.get("BIOCHEM_DELTA_MU_LOG_CLIP", "1.5")), 1e-6)
        self.mu_wall_gate_temp = max(float(os.environ.get("BIOCHEM_MU_WALL_GATE_TEMP", "0.18")), 1e-5)
        self.mu_wall_gate_center = float(os.environ.get("BIOCHEM_MU_WALL_GATE_CENTER", "0.55"))
        self.mu_wall_delta_gain = float(os.environ.get("BIOCHEM_MU_WALL_DELTA_GAIN", "0.65"))
        self.mu_wall_mask_mix = min(
            max(float(os.environ.get("BIOCHEM_MU_WALL_MASK_MIX", "0.80")), 0.0),
            1.0,
        )
        self._init_mu_delta_head_near_zero()

        self.register_buffer("species_si_scales", _bio.get_species_scales(device="cpu"))

    def _kinematics_prior_tail(self, batch, u_nd: torch.Tensor, v_nd: torch.Tensor) -> torch.Tensor | None:
        """Extra ``bio_encoder`` channels from kinematics clot prior (COMSOL-aligned cues)."""
        if self.bio_encoder_prior_dim <= 0:
            return None
        props = {
            "u_ref": batch.u_ref.view(-1, 1).to(dtype=torch.float32),
            "d_bar": batch.d_bar.view(-1, 1).to(dtype=torch.float32),
        }
        return clot_prior_features(
            batch,
            u_nd.reshape(-1),
            v_nd.reshape(-1),
            self._bio_cfg,
            props,
            n_features=self.bio_encoder_prior_dim,
        )

    def train(self, mode=True):
        """
        Override the default train method to ensure the frozen kinematic
        backbone strictly remains in evaluation mode.
        """
        super().train(mode)
        self.kin_encoder.eval()
        self.kin_processor.eval()
        self.kinematics_decoder.eval()
        self.mu_encoder.eval()

    def _apply_fourier_encoding(self, x, pos_nd=None):
        nodes_nd = pos_nd if pos_nd is not None else x[:, NodeFeat.XY]
        sdf_nd = x[:, NodeFeat.SDF]
        shear_pot = x[:, NodeFeat.SHEAR_POT]
        wall_normal = x[:, NodeFeat.WALL_NORMAL]
        rest = x[:, NodeFeat.REST]
        uv_prior = x[:, NodeFeat.UV_PRIOR]
        mu_prior = x[:, NodeFeat.MU_PRIOR]
        wss_prior = x[:, NodeFeat.WSS_PRIOR]

        features_to_encode = torch.cat([nodes_nd, sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat([
            shear_pot, features_to_encode, fourier_feats,
            rest, uv_prior, mu_prior, wss_prior
        ], dim=1)
        return encoded_x

    def _compute_kinematics_modulators(self, batch):
        """Computes physical edge modulators (advection, rheology, curvature) for GINO."""
        row, col = batch.edge_index
        edge_vec = batch.edge_attr[:, :2]
        wall_normals = batch.x[:, NodeFeat.WALL_NORMAL]

        e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
        n_dir_row = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
        n_dir_col = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir_row).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = batch.x[:, NodeFeat.SDF]
        sdf_edge = sdf_nd[row]
        decay_factor = torch.exp(-self.edge_decay_k * sdf_edge)

        curve_dot = (n_dir_row * n_dir_col).sum(dim=-1, keepdim=True)
        mod_curve = torch.log(torch.clamp(1.0 - curve_dot, min=self.curve_log_clamp_min, max=1.0)) * decay_factor
        mod_rheo = torch.log(torch.clamp(dot_prod, min=self.rheo_log_clamp_min, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=self.adv_log_clamp_min, max=1.0)) * decay_factor

        return mod_adv, mod_rheo, mod_curve

    def _decode_constrained_uvp(self, z_kin: torch.Tensor, kin_in: torch.Tensor, batch) -> torch.Tensor:
        """
        Decode latent kinematics directly to (u, v, p) using SDF hard constraints,
        matching the updated GINO_DEQ model.
        """
        u_v_p = self.kinematics_decoder(z_kin)
        sdf = kin_in[:, NodeFeat.SDF]
        uv_prior = kin_in[:, NodeFeat.UV_PRIOR]
        u_v_constrained = uv_prior + sdf * u_v_p[:, :2]
        return torch.cat([u_v_constrained, u_v_p[:, 2:3]], dim=1)

    def _decode_species_log1p(self, raw_species: torch.Tensor) -> torch.Tensor:
        """Decoder predicts log1p(species_nd) directly; clamp to a safe training range."""
        return torch.clamp(raw_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

    def species_log_nd_to_si(self, species_log: Tensor) -> Tensor:
        """Map stored ``log1p(c / c_scale)`` bulk+wall channels (12) to SI concentrations."""
        sc = self.species_si_scales.to(device=species_log.device, dtype=species_log.dtype)
        sp = torch.clamp(species_log, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
        return torch.expm1(sp) * sc

    def _init_learned_clot_penalty_near_zero(self) -> None:
        """Small random weights, zero biases, then shift last linear bias so Softplus output starts ~0."""
        with torch.no_grad():
            for m in self.learned_clot_penalty.modules():
                if isinstance(m, nn.Linear):
                    m.weight.uniform_(-1e-4, 1e-4)
                    if m.bias is not None:
                        m.bias.zero_()
            last_lin = self.learned_clot_penalty[2]
            if isinstance(last_lin, nn.Linear):
                last_lin.bias.fill_(-6.0)

    def _init_mu_delta_head_near_zero(self) -> None:
        """Keep residual μ correction neutral at startup (factor ~= 1)."""
        with torch.no_grad():
            for head in (
                self.mu_delta_head,
                self.mu_delta_bulk_head,
                self.mu_delta_tail_head,
                self.mu_delta_wall_head,
            ):
                for m in head.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.uniform_(-1e-4, 1e-4)
                        if m.bias is not None:
                            m.bias.zero_()
            for m in self.mu_trigger_gate_head.modules():
                if isinstance(m, nn.Linear):
                    m.weight.uniform_(-1e-4, 1e-4)
                    if m.bias is not None:
                        m.bias.zero_()

    def mu1_sigmoid(self, mat):
        """Soft logic for Platelet-driven viscosity multiplier with numerical safeguards."""
        safe_t_scale = max(self.T_scale, 1e-5)
        norm_val = (mat - self.mat_crit) / (self.temp_mat * safe_t_scale)
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)
        return (self.mu_ratio_max - 1.0) * torch.sigmoid(safe_val)

    def mu2_sigmoid(self, fi):
        """Soft logic for Fibrin-driven viscosity multiplier with numerical safeguards."""
        safe_t_scale = max(self.T_scale, 1e-5)
        norm_val = (fi - self.fi_crit) / (self.temp_fi * safe_t_scale)
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)
        return self.mu_ratio_max * torch.sigmoid(safe_val)

    def autoencode(self, batch):
        """
        Kinematics: Pure spatial representation learning (no ODE time integration).
        Predict biochemical state at t=0 from priors + kinematics that match the rollout path
        (resting species, DEQ velocities from frozen backbone — not ground-truth ``batch.y``).
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device
        current_species = _default_resting_species(num_nodes, device, batch)

        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        mu_inf = self.phys_cfg.mu_inf
        current_mu_eff = torch.full((num_nodes, 1), mu_inf, dtype=torch.float32, device=device)

        kin_in = batch.x[:, :15].clone()
        kin_in[:, 13:14] = current_mu_eff / mu_nd_scale
        kin_encoded = self._apply_fourier_encoding(kin_in)
        mu_enc = self.mu_encoder(current_mu_eff / mu_nd_scale)
        uv_prior = kin_in[:, NodeFeat.UV_PRIOR]
        p_prior = kin_in[:, NodeFeat.SHEAR_POT]
        mu_prior = kin_in[:, NodeFeat.MU_PRIOR]
        priors = torch.cat([uv_prior, p_prior, mu_prior], dim=1)
        z_kin = self.z_prior_proj(priors)

        mod_adv, mod_rheo, mod_curve = self._compute_kinematics_modulators(batch)

        def apply_kin_processor(x):
            batch_idx = get_batch_tensor(batch, num_nodes, device)
            return self.kin_processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)

        with torch.no_grad():
            for _ in range(self.max_inner_iters):
                injection = self.kin_encoder(kin_encoded) + mu_enc
                z_kin_next = apply_kin_processor(injection + z_kin)
                diff = torch.norm(z_kin_next - z_kin, p=2, dim=-1).mean()
                z_kin = 0.5 * z_kin + 0.5 * z_kin_next
                if diff < 1e-4:
                    break
        with torch.enable_grad():
            u_v_p = self._decode_constrained_uvp(z_kin, kin_in, batch)

        prior_tail = self._kinematics_prior_tail(batch, u_v_p[:, 0], u_v_p[:, 1])
        if prior_tail is None:
            bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15]], dim=-1)
        else:
            bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15], prior_tail], dim=-1)

        z = self.bio_encoder(bio_in)
        raw_species = self.biochem_decoder(z)
        next_species_flat = self._decode_species_log1p(raw_species)

        wall_mask_view = batch.mask_wall.view(-1, 1).float()
        surface_species = next_species_flat[:, 9:12] * wall_mask_view
        return torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)

    def forward(
        self,
        batch,
        evaluation_times,
        y_true_trajectory=None,
        teacher_forcing_ratio=0.0,
        start_idx=0,
        initial_species=None,
        detach_macro_state=None,
    ):
        """
        Forward pass for the Neural ODE with Two-Way Macro-Micro Coupling.

        Designed for long clinical time horizons (e.g., 10,000 - 30,000 seconds with ~150s
        macro-steps, yielding 66-200 steps per trajectory).

        Args:
            batch: PyG graph batch containing spatial features.
            evaluation_times: A 1D tensor of times [ 0.0, t1, t2, ..., t_n ] to evaluate the clot state.
            y_true_trajectory: Ground truth log-space species trajectory for scheduled sampling.
            teacher_forcing_ratio: Probability (0.0 to 1.0) of injecting the ground truth state at each step.
            start_idx: Starting index for Truncated BPTT (TBPTT) offset.
            initial_species: Optional detached state from a previous chunk for warm-starting TBPTT.
            detach_macro_state: If True, detaches the computational graph at each macro-step
                to prevent OOM on long 200-step rollouts.
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device
        if self.training:
            self.ode_func.derivative_energy_sum = 0.0
            self.ode_func.derivative_eval_count = 0
        self._last_mu_trigger_gate = None
        self._last_mu_delta_bulk = None
        self._last_mu_delta_tail = None
        self._last_mu_wall_gate = None
        self._last_mu_delta_wall = None

        truth_mask = biochem_truth_node_mask(batch, num_nodes, device)
        species_prior = _default_resting_species(num_nodes, device, batch)

        u_ref = batch.u_ref.view(-1, 1)
        d_bar = batch.d_bar.view(-1, 1)
        num_times = len(evaluation_times)

        # ==========================================
        # 1. INITIALIZE BIOCHEMICAL SPECIES
        # Default: physical resting prior everywhere. With TBPTT (start_idx > 0) or full-sequence TF,
        # anchor nodes take species from ``y`` / ``y_true_trajectory``; non-anchors keep the prior
        # (nodes without COMSOL-matched truth keep the resting prior). Synthetic graphs: all-prior at t=0.
        # Optional ``initial_species`` lets the caller warm-start from a pre-rolled state.
        # ==========================================
        if initial_species is not None:
            current_species = initial_species.to(device=device, dtype=species_prior.dtype)
        elif hasattr(batch, 'y') and batch.y is not None and start_idx > 0:
            safe_start_idx = min(start_idx, int(batch.y.shape[0]) - 1)
            gt_species = batch.y[safe_start_idx, :, 4:16].to(device)
            current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, species_prior)
        elif y_true_trajectory is not None and y_true_trajectory.shape[0] > 0:
            gt_species = y_true_trajectory[0, :, 4:16].to(device)
            current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, species_prior)
        else:
            current_species = species_prior
        current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

        # USE CENTRALIZED RHEOLOGY PARAMS
        mu_inf = self.phys_cfg.mu_inf
        mu_0 = self.phys_cfg.mu_0
        lam = self.phys_cfg.lam
        n_idx = self.phys_cfg.n
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale

        current_mu_eff = torch.full((num_nodes, 1), mu_inf, dtype=torch.float32, device=device)

        # Warm-start for the kinematic DEQ: carried detached between macro time steps so TBPTT
        # does not retain a cross-time graph through the fixed-point iteration.
        kin_in_ws = batch.x[:, :15]
        uv_prior_ws = kin_in_ws[:, NodeFeat.UV_PRIOR]
        p_prior_ws = kin_in_ws[:, NodeFeat.SHEAR_POT]
        mu_prior_ws = kin_in_ws[:, NodeFeat.MU_PRIOR]
        priors_ws = torch.cat([uv_prior_ws, p_prior_ws, mu_prior_ws], dim=1)
        z_kin_ws = self.z_prior_proj(priors_ws)

        mod_adv, mod_rheo, mod_curve = self._compute_kinematics_modulators(batch)

        def apply_kin_processor(x):
            batch_idx = get_batch_tensor(batch, num_nodes, device)
            return self.kin_processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)

        pred_trajectory = []
        detach_macro_state = self.detach_macro_state_default if detach_macro_state is None else bool(detach_macro_state)

        # ==========================================
        # 2. MACRO-MICRO STEPPING (Two-Way Coupling)
        # ==========================================
        for i in range(num_times):
            # --- A. MACRO STEP: SOLVE KINEMATICS (Frozen Biochemistry) ---
            kin_in = batch.x[:, :15].clone()

            # ND viscosity for kinematics matches label channel (mu_viscosity_nd_scale)
            kin_in[:, 13:14] = current_mu_eff / mu_nd_scale
            kin_encoded = self._apply_fourier_encoding(kin_in)

            mu_nd = current_mu_eff / mu_nd_scale
            mu_enc = self.mu_encoder(mu_nd)

            with torch.no_grad():
                zc = z_kin_ws
                for _ in range(self.max_inner_iters):
                    injection = self.kin_encoder(kin_encoded) + mu_enc
                    z_kin_next = apply_kin_processor(injection + zc)
                    diff = torch.norm(z_kin_next - zc, p=2, dim=-1).mean()
                    zc = 0.5 * zc + 0.5 * z_kin_next
                    if diff < 1e-4:
                        break
                z_kin_ws = zc

            injection = self.kin_encoder(kin_encoded) + mu_enc
            z_kin = apply_kin_processor(injection + z_kin_ws.detach())
            u_v_p = self._decode_constrained_uvp(z_kin, kin_in, batch)

            # --- B. UPDATE DYNAMIC RHEOLOGY FOR CURRENT TIME ---
            u_nd = u_v_p[:, 0:1]
            v_nd = u_v_p[:, 1:2]

            du_dx_nd = torch.sparse.mm(batch.G_x, u_nd)
            du_dy_nd = torch.sparse.mm(batch.G_y, u_nd)
            dv_dx_nd = torch.sparse.mm(batch.G_x, v_nd)
            dv_dy_nd = torch.sparse.mm(batch.G_y, v_nd)

            scale_grad = u_ref / d_bar
            gamma_dot = torch.sqrt(2 * ((du_dx_nd * scale_grad) ** 2 + (dv_dy_nd * scale_grad) ** 2) +
                                   ((du_dy_nd * scale_grad) + (dv_dx_nd * scale_grad)) ** 2 + 1e-8)

            # 1) Pure shear-thinning kinematic baseline (no species coupling).
            mu_kin_baseline = mu_inf + (mu_0 - mu_inf) * torch.pow(
                1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0
            )

            # 2) Biochem gelation: explicit FI/Mat gates + species-only learned nonnegative penalty.
            sp_safe = torch.clamp(current_species, _SPECIES_LOG1P_MIN, _SPECIES_LOG1P_MAX)
            species_si = self.species_log_nd_to_si(sp_safe)
            FI_si = species_si[:, 8:9]
            # STRICT COMSOL PARITY: viscosity depends on Mat (channel 11) only.
            Mat_si = species_si[:, 11:12]

            explicit_gelation = self.mu1_sigmoid(Mat_si) + self.mu2_sigmoid(FI_si)
            learned_gelation = self.learned_clot_penalty(sp_safe)
            gel_extra = explicit_gelation + learned_gelation
            if _biochem_gelation_prior_gate_enabled():
                # Wall-localised [0,1] map: keeps bulk lumen near pure Carreau (μ_kin_baseline) unless
                # kinematics indicate clot-risk (separation / low-shear × wall proximity).
                props_g = {"u_ref": u_ref.to(dtype=torch.float32), "d_bar": d_bar.to(dtype=torch.float32)}
                p_gate = clot_prior_score_flat(
                    batch,
                    u_nd.reshape(-1),
                    v_nd.reshape(-1),
                    self._bio_cfg,
                    props_g,
                )
                p_gate = p_gate.detach().clamp(0.0, 1.0).reshape(-1, 1).to(dtype=gel_extra.dtype)
                gel_extra = gel_extra * p_gate
            total_multiplier = 1.0 + gel_extra
            current_mu_eff = mu_kin_baseline * total_multiplier
            if _biochem_delta_mu_head_enabled():
                if _biochem_split_mu_regime_head_enabled():
                    # Trigger features (SI/physics aligned):
                    # [log1p species(12), FI_si, Mat_si, gamma_dot, wall_proximity]
                    sdf_nd = kin_in[:, NodeFeat.SDF]
                    wall_prox = torch.exp(-torch.abs(sdf_nd)).to(dtype=sp_safe.dtype)
                    trigger_feats = torch.cat(
                        [
                            sp_safe,
                            FI_si,
                            Mat_si,
                            gamma_dot.to(dtype=sp_safe.dtype),
                            wall_prox,
                        ],
                        dim=1,
                    )
                    gate_temp = max(self.mu_trigger_gate_temp * max(self.T_scale, 0.25), 1e-5)
                    gate_logits = self.mu_trigger_gate_head(trigger_feats)
                    gate = torch.sigmoid(torch.clamp(gate_logits / gate_temp, min=-50.0, max=50.0))
                    # Physics prior: suppress neural tail/wall corrections when no local clotting mass exists.
                    bio_suppressor_enabled = (
                        (os.environ.get("BIOCHEM_USE_BIO_GATE_SUPPRESSOR", "0") or "").strip().lower()
                        in ("1", "true", "yes", "on")
                    )
                    if bio_suppressor_enabled:
                        suppressor_thresh = max(
                            float(os.environ.get("BIOCHEM_BIO_SUPPRESSOR_THRESHOLD_SI", "1e-4")),
                            1e-8,
                        )
                        bio_signal = torch.clamp((FI_si + Mat_si) / suppressor_thresh, min=0.0, max=1.0)
                        # Detached to avoid rewarding artificial FI/Mat spikes to open the gate.
                        gate = gate * bio_signal.detach()
                    delta_bulk = self.mu_delta_bulk_head(torch.cat([z_kin, sp_safe], dim=1))
                    delta_tail = self.mu_delta_tail_head(torch.cat([z_kin, trigger_feats], dim=1))
                    delta_log_mu = ((1.0 - gate) * delta_bulk) + (gate * delta_tail)
                    if _biochem_wall_delta_head_enabled():
                        wall_mask = batch.mask_wall.view(-1, 1).to(dtype=sp_safe.dtype)
                        wall_signal_val = torch.maximum(
                            wall_prox,
                            self.mu_wall_mask_mix * wall_mask,
                        )
                        wall_logits = (wall_signal_val - self.mu_wall_gate_center) / max(
                            self.mu_wall_gate_temp * max(self.T_scale, 0.25), 1e-5
                        )
                        wall_gate = torch.sigmoid(torch.clamp(wall_logits, min=-50.0, max=50.0))
                        if bio_suppressor_enabled:
                            wall_gate = wall_gate * bio_signal.detach()
                        delta_wall = self.mu_delta_wall_head(torch.cat([z_kin, trigger_feats], dim=1))
                        delta_log_mu = delta_log_mu + (self.mu_wall_delta_gain * wall_gate * delta_wall)
                        self._last_mu_wall_gate = wall_gate
                        self._last_mu_delta_wall = delta_wall
                    # Expose lightweight diagnostics for training loss/metrics.
                    self._last_mu_trigger_gate = gate
                    self._last_mu_delta_bulk = delta_bulk
                    self._last_mu_delta_tail = delta_tail
                else:
                    delta_in = torch.cat([z_kin, sp_safe], dim=1)
                    delta_log_mu = self.mu_delta_head(delta_in)
                delta_log_mu = torch.clamp(delta_log_mu, min=-self.mu_delta_log_clip, max=self.mu_delta_log_clip)
                current_mu_eff = current_mu_eff * torch.exp(delta_log_mu)
            current_mu_eff = torch.clamp(current_mu_eff, min=1e-8)

            # ==========================================
            # BIOCHEM ATTENTION MODULATOR (Streamwise + detached)
            # ==========================================
            dshear_dx = torch.sparse.mm(batch.G_x, gamma_dot)
            dshear_dy = torch.sparse.mm(batch.G_y, gamma_dot)

            vel_mag = torch.sqrt(u_nd ** 2 + v_nd ** 2) + 1e-8
            u_dir = u_nd / vel_mag
            v_dir = v_nd / vel_mag

            dshear_ds = (u_dir * dshear_dx) + (v_dir * dshear_dy)
            dshear_ds_phys = dshear_ds / torch.clamp(d_bar, min=1e-8)

            row, _ = batch.edge_index
            dshear_edge = dshear_ds_phys[row]

            scaled_temp = max(self.T_grad * self.T_scale, 1e-5)
            separation_logits = -(dshear_edge - self.sgt) / scaled_temp
            is_separation_edge = torch.sigmoid(torch.clamp(separation_logits, min=-50.0, max=50.0))
            mod_separation = (is_separation_edge * self.biochem_attention_boost).detach()

            # --- C. RECORD COUPLED STATE (Record prediction first) ---
            current_mu_eff_nd = current_mu_eff / mu_nd_scale
            # Use the ND version for the recorded trajectory (species at this macro time, pre-TF).
            pred_step = torch.cat([u_v_p, current_mu_eff_nd, sp_safe], dim=-1)
            pred_trajectory.append(pred_step)

            # --- C.5 TEACHER FORCING INJECTION (For next ODE step only) ---
            if self.training and y_true_trajectory is not None and i > 0:
                # Scheduled sampling in log1p-space states for long trajectories (66-200 steps):
                # We DO NOT blend values linearly in log-space, as this physically skews concentrations.
                # Instead, choose 100% GT or 100% model state for anchor nodes via a probabilistic coin flip.
                tf = float(max(0.0, min(1.0, teacher_forcing_ratio)))
                if tf > 0.0:
                    if tf >= 1.0 - 1e-6:
                        use_ground_truth = True
                    else:
                        use_ground_truth = bool((torch.rand((), device=device) < tf).item())
                    if use_ground_truth:
                        gt_species = y_true_trajectory[i, :, 4:16].to(device)
                        current_species = torch.where(truth_mask.unsqueeze(-1), gt_species, current_species)
                        current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

            # --- D. MICRO STEP: INTEGRATE BIOCHEMISTRY (Frozen Kinematics) ---
            if i < num_times - 1:
                # Physical time [s] so dz/dt matches finite-difference d_pred_dt in training losses.
                t_span = evaluation_times[i: i + 2]

                # Encode current physical state into latent representation
                safe_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
                prior_tail = self._kinematics_prior_tail(batch, u_v_p[:, 0], u_v_p[:, 1])
                if prior_tail is None:
                    bio_in = torch.cat([safe_species, u_v_p, batch.x[:, :15]], dim=-1)
                else:
                    bio_in = torch.cat([safe_species, u_v_p, batch.x[:, :15], prior_tail], dim=-1)
                z_current = self.bio_encoder(bio_in)

                # Integrate ODE over the Delta t interval (adjoint: memory-safe backward).
                dt_seg = float((t_span[-1] - t_span[0]).abs().item())
                _min_dt = 1e-9
                if dt_seg < _min_dt:
                    # Duplicate/near-duplicate timestamps → no evolution.
                    z_next = z_current
                else:
                    def odefunc_wrapper(t, z):
                        batch_idx = get_batch_tensor(batch, num_nodes, device)
                        dz = self.ode_func(t, z, batch.edge_index, batch.edge_attr, batch_idx, mod_separation)
                        return torch.clamp(dz, min=-10.0, max=10.0)

                    # Use an explicit method (like "rk4") by default for large 150s jumps to avoid
                    # implicit solver history-drop penalties on restarted biochemical segment solves.
                    solver_method = self.micro_ode_method
                    ode_kwargs_base = dict(
                        method=solver_method,
                        adjoint_method=solver_method,
                        adjoint_params=tuple(self.ode_func.parameters()),
                        rtol=self.rtol,
                        atol=self.atol,
                        adjoint_rtol=self.rtol,
                        adjoint_atol=self.atol,
                    )
                    # ``evaluation_times`` are nondimensional (see ``to_t_nd(..., t_final)``); env cap is SI seconds.
                    max_step_env = (os.environ.get("BIOCHEM_ODE_MAX_STEP_S") or "").strip()
                    max_step_phys_s = max(float(max_step_env), 1e-9) if max_step_env else 10.0
                    t_int_ref = float(getattr(self._bio_cfg, "t_final", 30000.0))
                    max_step_nd = max_step_phys_s / max(t_int_ref, 1e-12)
                    rk_sub_env = (os.environ.get("BIOCHEM_ADJOINT_RK4_SUBSTEPS") or "").strip()
                    n_rk_sub = max(1, int(rk_sub_env) if rk_sub_env else 32)
                    use_adjoint = (os.environ.get("BIOCHEM_ODEINT_USE_ADJOINT", "1") or "").strip().lower() not in (
                        "0",
                        "false",
                        "no",
                        "off",
                    )

                    def _one_odeint(z0, span: torch.Tensor, step_hint: float) -> torch.Tensor:
                        ode_kwargs = dict(ode_kwargs_base)
                        if solver_method == "rk4":
                            # ``step_hint`` is the macro subsegment length; torchdiffeq's fixed ``rk4``
                            # would otherwise take a single step over the whole segment (unstable).
                            seg_len = abs(float(span[-1] - span[0]))
                            sh = max(seg_len / float(n_rk_sub), 1e-12)
                            ode_kwargs["options"] = {"step_size": sh}
                            ode_kwargs["adjoint_options"] = {"step_size": sh}
                        if use_adjoint:
                            z_out = odeint_adjoint(
                                odefunc_wrapper,
                                z0,
                                span,
                                **ode_kwargs,
                            )
                        else:
                            plain = dict(
                                method=solver_method,
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                            if solver_method == "rk4" and "options" in ode_kwargs:
                                plain["options"] = ode_kwargs["options"]
                            z_out = odeint(odefunc_wrapper, z0, span, **plain)
                        return z_out[1]

                    if dt_seg > max_step_nd:
                        t0 = float(t_span[0].detach().item())
                        t1 = float(t_span[-1].detach().item())
                        duration = t1 - t0
                        n_sub = max(1, int(math.ceil(abs(duration) / max_step_nd)))
                        sub_dt = duration / float(n_sub)
                        z_run = z_current
                        for j in range(n_sub):
                            ts0 = t0 + j * sub_dt
                            ts1 = t0 + (j + 1) * sub_dt
                            t_sub = torch.tensor([ts0, ts1], device=device, dtype=t_span.dtype)
                            z_run = _one_odeint(z_run, t_sub, step_hint=sub_dt)
                        z_next = z_run
                    else:
                        z_next = _one_odeint(z_current, t_span, step_hint=dt_seg)
                raw_species = self.biochem_decoder(z_next)

                next_species_flat = self._decode_species_log1p(raw_species)

                # Enforce surface species only on walls
                wall_mask_view = batch.mask_wall.view(-1, 1).float()
                surface_species = next_species_flat[:, 9:12] * wall_mask_view

                # Update species state for the next macro-step
                current_species = torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)
                current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
                if detach_macro_state:
                    # Prevent OOM during BPTT across long 66-200 step macro trajectories on
                    # finite-memory GPUs by severing the carried-state graph each macro step.
                    current_species = current_species.detach()
                    current_mu_eff = current_mu_eff.detach()

        # Stack into shape: [ Time, Nodes, 16 ]
        pred_series = torch.stack(pred_trajectory, dim=0)

        return pred_series