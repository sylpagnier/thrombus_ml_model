import torch
import torch.nn as nn
import math
from torch import Tensor
# ``odeint`` OOMs on large graphs; ``odeint_adjoint`` fixes that.
from torchdiffeq import odeint_adjoint
from src.architecture.ginodeq import GINOBlock, SpectralLinear
from src.config import BiochemConfig
from src.utils.math_operators import wls_derivatives
from src.utils.batching import get_batch_tensor

# Matches BiochemPhysicsKernels: species channels are log1p(species_nd).
_SPECIES_LOG1P_MIN = -10.0
_SPECIES_LOG1P_MAX = 8.0


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
    resting_indices = [0, 4, 6, 7]
    resting_value = math.log(2.0)
    for idx in resting_indices:
        current_species[:, idx] = resting_value
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

    def forward(self, t, z, edge_index, edge_attr, batch_idx):
        # The derivative dz/dt depends strictly on the current biochemical state 'z' and frozen physics.
        z = torch.clamp(z, min=-20.0, max=20.0)
        mod_dummy = torch.zeros(int(edge_index.shape[ 1 ]), 1, dtype=torch.float32, device=z.device)
        dz_raw = self.derivative_processor(z, edge_index, edge_attr, batch_idx, mod_dummy, mod_dummy, mod_dummy)
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

    def __init__(self, phys_cfg, in_channels=12, spatial_channels=15, latent_dim=64, max_inner_iters=25,
                 mu_ratio_max=80.0, mat_crit=2e7, fi_crit=0.6,
                 temp_mat=1e6, temp_fi=0.05, rtol=1e-3, atol=1e-4):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_inner_iters = max_inner_iters
        self.rtol = rtol
        self.atol = atol
        self.phys_cfg = phys_cfg  # Store physics config internally

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
        self.num_fourier_freqs = 8
        freqs = (2.0 ** torch.arange(self.num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 3 * self.num_fourier_freqs * 2
        encoded_channels = (15 - 3) + 1 + 3 + fourier_channels

        self.kin_encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.kin_processor = GINOBlock(latent_dim)
        # Stream-function formulation for kinematics: decoder predicts (psi, p).
        self.kinematics_decoder = nn.Linear(latent_dim, 2)
        self.mu_encoder = nn.Linear(1, latent_dim)

        # ==========================================
        # 2. BIOCHEMISTRY NEURAL ODE
        # ==========================================
        # Initial Condition Encoder (Maps spatial config to z0)
        self.bio_encoder = SpectralLinear(in_features=in_channels + 3 + spatial_channels, out_features=latent_dim)

        # The ODE Function
        self.ode_func = BioODEFunc(latent_dim)

        # Physical Decoder
        self.biochem_decoder = SpectralLinear(in_features=latent_dim, out_features=12)

        _bio = BiochemConfig(phase="biochem")
        self.register_buffer("species_si_scales", _bio.get_species_scales(device="cpu"))

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
        nodes_nd = pos_nd if pos_nd is not None else x[:, 0:2]
        sdf_nd = x[:, 2:3]
        shear_pot = torch.zeros_like(sdf_nd)
        wall_normal = x[:, 3:5]

        rest = x[:, 5:11]
        uv_prior = x[:, 11:13]
        mu_prior = x[:, 13:14]
        wss_prior = x[:, 14:15]

        features_to_encode = torch.cat([sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat([
            nodes_nd, shear_pot, features_to_encode, fourier_feats,
            rest, uv_prior, mu_prior, wss_prior
        ], dim=1)
        return encoded_x

    def _compute_wls_derivatives(self, field: Tensor, batch) -> Tensor:
        """Compute WLS derivatives [d/dx, d/dy, d2/dx2, d2/dxdy, d2/dy2] for a nodal scalar."""
        row, col = batch.edge_index
        num_nodes = batch.num_nodes
        V, W, M_inv = batch.V, batch.W, batch.M_inv
        edge_index = torch.stack([row, col], dim=0)

        boundary_mask = None
        boundary_normals = None
        if hasattr(batch, "mask_wall") or hasattr(batch, "mask_inlet") or hasattr(batch, "mask_outlet"):
            boundary_mask = torch.zeros(num_nodes, dtype=torch.bool, device=V.device)
            if hasattr(batch, "mask_wall"):
                boundary_mask |= batch.mask_wall.view(-1).to(device=V.device).bool()
            if hasattr(batch, "mask_inlet"):
                boundary_mask |= batch.mask_inlet.view(-1).to(device=V.device).bool()
            if hasattr(batch, "mask_outlet"):
                boundary_mask |= batch.mask_outlet.view(-1).to(device=V.device).bool()

            boundary_normals = torch.zeros((num_nodes, 2), dtype=V.dtype, device=V.device)
            if hasattr(batch, "x") and torch.is_tensor(batch.x) and batch.x.dim() == 2 and batch.x.shape[1] >= 5:
                boundary_normals = batch.x[:, 3:5].to(device=V.device, dtype=V.dtype).clone()
            if hasattr(batch, "outlet_normal") and batch.outlet_normal is not None and hasattr(batch, "mask_outlet"):
                om = batch.mask_outlet.view(-1).to(device=V.device).bool()
                on = batch.outlet_normal.to(device=V.device, dtype=V.dtype)
                if on.dim() == 2 and on.shape[1] >= 2 and on.shape[0] == num_nodes:
                    boundary_normals[om] = on[om, :2]
            nrm = torch.linalg.norm(boundary_normals, dim=1, keepdim=True)
            boundary_normals = boundary_normals / (nrm + 1e-12)

        return wls_derivatives(
            field,
            edge_index,
            num_nodes,
            V,
            W,
            M_inv,
            boundary_mask=boundary_mask,
            boundary_normals=boundary_normals,
        )

    def _stream_to_velocity(
        self,
        psi_raw: Tensor,
        p: Tensor,
        batch,
        sdf: Tensor,
        wall_normal: Tensor,
    ) -> torch.Tensor:
        """Convert (psi, p) -> (u, v, p) using WLS derivatives + manual product rule."""
        c_psi = self._compute_wls_derivatives(psi_raw, batch)
        psi_x = c_psi[:, 0:1, 0]
        psi_y = c_psi[:, 1:2, 0]
        n_x = wall_normal[:, 0:1]
        n_y = wall_normal[:, 1:2]
        u = (sdf ** 2) * psi_y + psi_raw * 2.0 * sdf * n_y
        v = -((sdf ** 2) * psi_x + psi_raw * 2.0 * sdf * n_x)
        return torch.cat([u, v, p], dim=1)

    def _decode_constrained_uvp(self, z_kin: torch.Tensor, kin_in: torch.Tensor, batch) -> torch.Tensor:
        """
        Decode latent kinematics and recover velocity analytically via product rule.
        """
        psi_p = self.kinematics_decoder(z_kin)
        psi_raw = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        sdf = kin_in[:, 2:3]
        wall_normal = kin_in[:, 3:5]
        return self._stream_to_velocity(psi_raw, p, batch, sdf, wall_normal)

    def _decode_species_log1p(self, raw_species: torch.Tensor) -> torch.Tensor:
        """Decoder predicts log1p(species_nd) directly; clamp to a safe training range."""
        return torch.clamp(raw_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

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

        z_kin = torch.zeros(num_nodes, self.latent_dim, device=device)

        def apply_kin_processor(x):
            batch_idx = get_batch_tensor(batch, num_nodes, device)
            mod_dummy = torch.zeros(int(batch.edge_index.shape[1]), 1, dtype=torch.float32, device=device)
            return self.kin_processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_dummy, mod_dummy, mod_dummy)

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

        bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15]], dim=-1)

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
    ):
        """
        Forward pass for the Neural ODE with Two-Way Macro-Micro Coupling.
        evaluation_times: A 1D tensor of times [ 0.0, t1, t2, ..., t_n ] to evaluate the clot state.
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device
        if self.training:
            self.ode_func.derivative_energy_sum = 0.0
            self.ode_func.derivative_eval_count = 0

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
        z_kin_ws = torch.zeros(num_nodes, self.latent_dim, device=device)

        def apply_kin_processor(x):
            batch_idx = get_batch_tensor(batch, num_nodes, device)
            mod_dummy = torch.zeros(int(batch.edge_index.shape[1]), 1, dtype=torch.float32, device=device)
            return self.kin_processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_dummy, mod_dummy, mod_dummy)

        def odefunc_wrapper(t, z):
            batch_idx = get_batch_tensor(batch, num_nodes, device)
            dz = self.ode_func(t, z, batch.edge_index, batch.edge_attr, batch_idx)
            return torch.clamp(dz, min=-10.0, max=10.0)

        pred_trajectory = []

        # ==========================================
        # 2. MACRO-MICRO STEPPING (Two-Way Coupling)
        # ==========================================
        for i in range(num_times):
            if self.training and y_true_trajectory is not None and i > 0:
                # Smooth teacher forcing: blend predicted and target anchor states.
                tf = float(max(0.0, min(1.0, teacher_forcing_ratio)))
                if tf > 0.0:
                    gt_species = y_true_trajectory[i, :, 4:16].to(device)
                    blended_species = (1.0 - tf) * current_species + tf * gt_species
                    current_species = torch.where(truth_mask.unsqueeze(-1), blended_species, current_species)
                    current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

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

            mu_base = mu_inf + (mu_0 - mu_inf) * torch.pow(1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0)

            # mu1/mu2 thresholds (mat_crit, fi_crit) are in SI units; convert log1p state to linear SI.
            sc = self.species_si_scales.to(device=device, dtype=current_species.dtype)
            sp_safe = torch.clamp(current_species, _SPECIES_LOG1P_MIN, _SPECIES_LOG1P_MAX)
            FI_si = torch.expm1(sp_safe[:, 8:9]) * sc[8:9]
            Mat_si = torch.expm1(sp_safe[:, 11:12]) * sc[11:12]

            total_multiplier = 1.0 + self.mu1_sigmoid(Mat_si) + self.mu2_sigmoid(FI_si)
            current_mu_eff = mu_base * total_multiplier

            # --- C. RECORD COUPLED STATE ---
            current_mu_eff_nd = current_mu_eff / mu_nd_scale
            # Use the ND version for the recorded trajectory
            safe_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
            pred_step = torch.cat([u_v_p, current_mu_eff_nd, safe_species], dim=-1)
            pred_trajectory.append(pred_step)

            # --- D. MICRO STEP: INTEGRATE BIOCHEMISTRY (Frozen Kinematics) ---
            if i < num_times - 1:
                # Physical time [s] so dz/dt matches finite-difference d_pred_dt in training losses.
                t_span = evaluation_times[i: i + 2]

                # Encode current physical state into latent representation
                safe_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)
                bio_in = torch.cat([safe_species, u_v_p, batch.x[:, :15]], dim=-1)
                z_current = self.bio_encoder(bio_in)

                # Integrate ODE over the Delta t interval (adjoint: memory-safe backward).
                dt_seg = float((t_span[-1] - t_span[0]).abs().item())
                _min_dt = 1e-9
                if dt_seg < _min_dt:
                    # Duplicate/near-duplicate timestamps → no evolution.
                    z_next = z_current
                else:
                    z_out = odeint_adjoint(
                        odefunc_wrapper,
                        z_current,
                        t_span,
                        method="implicit_adams",
                        adjoint_method="implicit_adams",
                        adjoint_params=tuple(self.ode_func.parameters()),
                        # Keep forward/backward solver tolerances aligned with model config.
                        rtol=self.rtol,
                        atol=self.atol,
                        adjoint_rtol=self.rtol,
                        adjoint_atol=self.atol,
                    )
                    z_next = z_out[1]
                raw_species = self.biochem_decoder(z_next)

                next_species_flat = self._decode_species_log1p(raw_species)

                # Enforce surface species only on walls
                wall_mask_view = batch.mask_wall.view(-1, 1).float()
                surface_species = next_species_flat[:, 9:12] * wall_mask_view

                # Update species state for the next macro-step
                current_species = torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)
                current_species = torch.clamp(current_species, min=_SPECIES_LOG1P_MIN, max=_SPECIES_LOG1P_MAX)

        # Stack into shape: [ Time, Nodes, 16 ]
        pred_series = torch.stack(pred_trajectory, dim=0)

        return pred_series