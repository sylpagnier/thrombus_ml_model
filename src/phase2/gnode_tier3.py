import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from src.phase1.physics.ginodeq import GINOBlock, SpectralLinear


class BioODEFunc(nn.Module):
    """
    Calculates the temporal derivative dz/dt of the biochemical latent state.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # Processor to compute spatial interactions for the derivative
        self.derivative_processor = GINOBlock(latent_dim)

    def forward(self, t, z, edge_index, edge_attr, batch_idx):
        # The derivative dz/dt depends strictly on the current biochemical state 'z' and frozen physics.
        mod_dummy = torch.zeros(int(edge_index.shape[ 1 ]), 1, dtype=torch.float32, device=z.device)
        dz_dt = self.derivative_processor(z, edge_index, edge_attr, batch_idx, mod_dummy, mod_dummy)

        return dz_dt

class GNODE_Tier3(nn.Module):
    """
    Tier 3 Physics-Informed Graph Neural ODE for dynamic Thrombosis Simulation.
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
        self.kinematics_decoder = nn.Linear(latent_dim, 3)
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

    def _apply_fourier_encoding(self, x):
        nodes_nd = x[:, 0:2]
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

    def forward(self, batch, evaluation_times):
        """
        Forward pass for the Neural ODE with Two-Way Macro-Micro Coupling.
        evaluation_times: A 1D tensor of times [ 0.0, t1, t2, ..., t_n ] to evaluate the clot state.
        """
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device

        u_ref = batch.u_ref.view(-1, 1)
        d_bar = batch.d_bar.view(-1, 1)
        num_times = len(evaluation_times)

        # ==========================================
        # 1. INITIALIZE STATES (t=0)
        # ==========================================
        current_species = torch.zeros(num_nodes, 12, dtype=torch.float32, device=device)
        if hasattr(batch, 'bio_inlet_bc'):
            mask_inlet = batch.mask_inlet.view(-1).bool()
            current_species[mask_inlet, 0:9] = batch.bio_inlet_bc[mask_inlet]

        # USE CENTRALIZED RHEOLOGY PARAMS
        mu_inf = self.phys_cfg.mu_inf
        mu_0 = self.phys_cfg.mu_0
        lam = self.phys_cfg.lam
        n_idx = self.phys_cfg.n

        current_mu_eff = torch.full((num_nodes, 1), mu_inf, dtype=torch.float32, device=device)

        # Warm-start tensor for Kinematics DEQ to ensure rapid convergence in the loop
        z_kin = torch.zeros(num_nodes, self.latent_dim, device=device)

        def apply_kin_processor(x):
            batch_idx = batch.batch if hasattr(batch, 'batch') and batch.batch is not None else torch.zeros(num_nodes,
                                                                                                            dtype=torch.long,
                                                                                                            device=device)
            mod_dummy = torch.zeros(int(batch.edge_index.shape[1]), 1, dtype=torch.float32, device=device)
            return self.kin_processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_dummy, mod_dummy)

        def odefunc_wrapper(t, z):
            batch_idx = batch.batch if hasattr(batch, 'batch') and batch.batch is not None else torch.zeros(num_nodes,
                                                                                                            dtype=torch.long,
                                                                                                            device=device)
            return self.ode_func(t, z, batch.edge_index, batch.edge_attr, batch_idx)

        pred_trajectory = []

        # ==========================================
        # 2. MACRO-MICRO STEPPING (Two-Way Coupling)
        # ==========================================
        for i in range(num_times):

            # --- A. MACRO STEP: SOLVE KINEMATICS (Frozen Biochemistry) ---
            kin_in = batch.x[:, :15].clone()

            # Scale by mu_inf, NOT mu_0 to match Tier 2 training distribution
            kin_in[:, 13:14] = current_mu_eff / mu_inf
            kin_encoded = self._apply_fourier_encoding(kin_in)

            # FIX: Scale by mu_inf here as well
            mu_nd = current_mu_eff / mu_inf
            mu_enc = self.mu_encoder(mu_nd)

            with torch.no_grad():
                for _ in range(self.max_inner_iters):
                    injection = self.kin_encoder(kin_encoded) + mu_enc
                    z_kin_next = apply_kin_processor(injection + z_kin)
                    diff = torch.norm(z_kin_next - z_kin, p=2, dim=-1).mean()
                    z_kin = 0.5 * z_kin + 0.5 * z_kin_next

                    if diff < 1e-4: break

            u_v_p = self.kinematics_decoder(z_kin)[:, :3]

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

            FI_current = current_species[:, 8:9]
            Mat_current = current_species[:, 11:12]

            total_multiplier = 1.0 + self.mu1_sigmoid(Mat_current) + self.mu2_sigmoid(FI_current)
            current_mu_eff = mu_base * total_multiplier

            # --- C. RECORD COUPLED STATE ---
            current_mu_eff_nd = current_mu_eff / mu_inf
            # Use the ND version for the recorded trajectory
            pred_step = torch.cat([u_v_p, current_mu_eff_nd, current_species], dim=-1)
            pred_trajectory.append(pred_step)

            # --- D. MICRO STEP: INTEGRATE BIOCHEMISTRY (Frozen Kinematics) ---
            if i < num_times - 1:
                t_span = evaluation_times[i: i + 2]

                # FIX 1: Normalize the time span to [0, ~0.016] to prevent the untrained
                # Neural ODE from exploding and stalling the adaptive solver.
                t_final_safe = evaluation_times[-1].clamp(min=1.0)
                t_span_nd = t_span / t_final_safe

                # Encode current physical state into latent representation
                bio_in = torch.cat([current_species, u_v_p, batch.x[:, :15]], dim=-1)
                z_current = self.bio_encoder(bio_in)

                # Integrate ODE over the Delta t interval
                z_out = odeint(
                    odefunc_wrapper,
                    z_current,
                    t_span_nd,  # Pass the normalized time
                    method='dopri5',
                    rtol=self.rtol,
                    atol=self.atol,
                    adjoint_params=tuple(self.ode_func.parameters())
                )

                # Decode the next state trajectory
                z_next = z_out[1]
                raw_species = self.biochem_decoder(z_next)

                # Enforce an upper bound on the log1p space (e.g., max value of 15.0)
                next_species_flat = 15.0 * torch.sigmoid(raw_species)

                # Enforce surface species only on walls
                wall_mask_view = batch.mask_wall.view(-1, 1).float()
                surface_species = next_species_flat[:, 9:12] * wall_mask_view

                # Update species state for the next macro-step
                current_species = torch.cat([next_species_flat[:, 0:9], surface_species], dim=1)

        # Stack into shape: [ Time, Nodes, 16 ]
        pred_series = torch.stack(pred_trajectory, dim=0)

        return pred_series