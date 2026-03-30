import torch
import torch.nn as nn
from src.phase1.physics.ginodeq import GINOBlock, SpectralLinear
import torch.nn.functional as F


class GINO_DEQ_Tier3(nn.Module):
    """
    Tier 3 Deep Equilibrium Model for Thrombosis Simulation.
    Segregated Block-Seidel solver adapted for 12-species (9 bulk + 3 surface)
    incorporating COMSOL's Fibrin kinetics and dual-trigger rheology.
    """

    def __init__(self, in_channels=12, spatial_channels=16, latent_dim=64, max_outer_iters=3, max_inner_iters=25,
                 mu_ratio_max=80.0, mat_crit=2e7, fi_crit=0.6,
                 temp_mat=1e6, temp_fi=0.05, tol=1e-4):
        super().__init__()

        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.latent_dim = latent_dim

        # COMSOL Step Function Approximations (Dual-Trigger)
        self.mu_ratio_max = mu_ratio_max
        self.mat_crit = mat_crit  # COMSOL mu1 critical threshold (Platelets)
        self.fi_crit = fi_crit  # COMSOL mu2 critical threshold (Fibrin)

        # Temperature scaling for soft sigmoids to ensure differentiable backprop
        self.temp_mat = temp_mat
        self.temp_fi = temp_fi
        self.tol = tol
        self.T_scale = 1.0  # Default scale for temperature annealing

        # ==========================================
        # 1. KINEMATICS BACKBONE (FROZEN)
        # ==========================================
        # Must strictly match the Tier 2 Architecture
        self.num_fourier_freqs = 8
        freqs = (2.0 ** torch.arange(self.num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 3 * self.num_fourier_freqs * 2
        encoded_channels = (15 - 3) + 3 + fourier_channels

        self.kin_encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.kin_processor = GINOBlock(latent_dim)
        self.kinematics_decoder = nn.Linear(latent_dim, 3)  # Back to 3 channels (u, v, p)
        self.mu_encoder = nn.Linear(1, latent_dim)

        # BIOCHEMISTRY SOLVER
        # Biochemistry takes 12 species + 3 kinematics (u, v, p) + 16 spatial features = 31 features
        self.bio_encoder = SpectralLinear(in_features=in_channels + 3 + spatial_channels, out_features=latent_dim)
        self.bio_processor = GINOBlock(latent_dim)

        # Decoder strictly matched to the updated 12 species count
        self.biochem_decoder = SpectralLinear(in_features=latent_dim, out_features=12)

    def train(self, mode=True):
        """
        Override the default train method to ensure the frozen kinematic
        backbone strictly remains in evaluation mode.
        """
        super().train(mode)  # Sets the whole model to train mode

        # Force the frozen layers back into deterministic eval mode
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
        # 1. Prevent Division by Zero if T_scale aggressively anneals
        safe_t_scale = max(self.T_scale, 1e-5)

        # 2. Calculate the raw normalized value
        norm_val = (mat - self.mat_crit) / (self.temp_mat * safe_t_scale)

        # 3. Clamp the argument to prevent FP16/FP32 precision overflow in the sigmoid
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)

        # FIX: Subtract 1.0 from the max ratio here.
        # When 1.0 is added in the forward pass, this will strictly peak at mu_ratio_max (80.0x)
        return (self.mu_ratio_max - 1.0) * torch.sigmoid(safe_val)

    def mu2_sigmoid(self, fi):
        """Soft logic for Fibrin-driven viscosity multiplier with numerical safeguards."""
        safe_t_scale = max(self.T_scale, 1e-5)
        norm_val = (fi - self.fi_crit) / (self.temp_fi * safe_t_scale)
        safe_val = torch.clamp(norm_val, min=-50.0, max=50.0)

        return self.mu_ratio_max * torch.sigmoid(safe_val)

    def forward(self, batch):
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device

        # Extract scaling factors for dimensionalizing derivatives
        u_ref = batch.u_ref.view(-1, 1)
        d_bar = batch.d_bar.view(-1, 1)

        bio_init = torch.zeros(num_nodes, 12, dtype=torch.float32, device=device)

        if hasattr(batch, 'bio_inlet_bc'):
            mask_inlet = batch.mask_inlet.view(-1).bool()
            bio_init[mask_inlet, 0:9] = batch.bio_inlet_bc[mask_inlet]

        # Initialize with mu_inf instead of a static Newtonian constant
        mu_inf = 0.0035
        mu_0 = 0.056
        lam = 3.313
        n_idx = 0.358

        # Initial guess uses mu_inf
        mu_eff = torch.full((num_nodes, 1), mu_inf, dtype=torch.float32, device=device)

        z_kin = torch.zeros(num_nodes, self.latent_dim, device=device)
        z_bio = torch.zeros(num_nodes, self.latent_dim, device=device)

        def apply_processor(processor, x):
            batch_idx = batch.batch if hasattr(batch, 'batch') and batch.batch is not None else torch.zeros(num_nodes,
                                                                                                            dtype=torch.long,
                                                                                                            device=device)
            mod_dummy = torch.zeros(int(batch.edge_index.shape[1]), 1, dtype=torch.float32, device=device)
            return processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_dummy, mod_dummy)

        # ==========================================
        # BLOCK-SEIDEL OUTER LOOP
        # ==========================================
        for outer_idx in range(self.max_outer_iters):
            def f_kinematics(z):
                # Slice only the first 15 channels (the foundational physics layout)
                kin_in = batch.x[:, :15].clone()

                # Update the dynamic viscosity channel (Index 13 in the new layout)
                kin_in[:, 13:14] = mu_eff / 0.056

                # Now _apply_fourier_encoding will correctly find:
                # nodes_nd at [0:2], sdf at [2:3], normals at [3:5]
                kin_encoded = self._apply_fourier_encoding(kin_in)

                # Correctly encode the dynamic viscosity for the latent sum
                mu_nd = mu_eff / 0.056
                mu_enc = self.mu_encoder(mu_nd)

                return apply_processor(self.kin_processor, self.kin_encoder(kin_encoded) + mu_enc + z)

            with torch.no_grad():
                for _ in range(self.max_inner_iters):
                    z_kin_next = f_kinematics(z_kin)
                    diff = torch.norm(z_kin_next - z_kin, p=2, dim=-1).mean()
                    z_kin = z_kin_next
                    if diff < self.tol: break

                    # Slice only the [u, v, p] kinematics from the 5-channel decoder output
                    u_v_p = self.kinematics_decoder(z_kin)[:, :3]

            # --- 2. Biochemistry Inner Loop ---
            def f_biochem(z):
                # Concatenate geometric features (batch.x) so the bio_encoder is spatially aware
                bio_in = torch.cat([bio_init, u_v_p.detach(), batch.x], dim=-1)
                return apply_processor(self.bio_processor, self.bio_encoder(bio_in) + z)

            with torch.no_grad():
                for _ in range(self.max_inner_iters):
                    z_bio_next = f_biochem(z_bio)
                    diff = torch.norm(z_bio_next - z_bio, p=2, dim=-1).mean()
                    z_bio = z_bio_next
                    if diff < self.tol: break

            raw_species = F.softplus(self.biochem_decoder(z_bio))
            wall_mask_view = batch.mask_wall.view(-1, 1).float()

            surface_species_loop = raw_species[:, 9:12] * wall_mask_view
            current_species = torch.cat([raw_species[:, 0:9], surface_species_loop], dim=1)

            # --- 3. DYNAMIC CARREAU RHEOLOGY UPDATE ---
            u_nd = u_v_p[:, 0:1]
            v_nd = u_v_p[:, 1:2]

            # Compute Non-dimensional gradients using WLS operators
            du_dx_nd = torch.sparse.mm(batch.G_x, u_nd)
            du_dy_nd = torch.sparse.mm(batch.G_y, u_nd)
            dv_dx_nd = torch.sparse.mm(batch.G_x, v_nd)
            dv_dy_nd = torch.sparse.mm(batch.G_y, v_nd)

            # Re-dimensionalize gradients to get physical shear rate
            scale_grad = u_ref / d_bar
            du_dx, du_dy = du_dx_nd * scale_grad, du_dy_nd * scale_grad
            dv_dx, dv_dy = dv_dx_nd * scale_grad, dv_dy_nd * scale_grad

            gamma_dot = torch.sqrt(2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2 + 1e-8)

            # Base Carreau Viscosity
            mu_base = mu_inf + (mu_0 - mu_inf) * torch.pow(1.0 + (lam * gamma_dot) ** 2, (n_idx - 1.0) / 2.0)

            # Apply Dual-Trigger Rheology (Max multiplier is 80 + 80 = 160x, mapping COMSOL's exact logic)
            FI = current_species[:, 8:9]
            Mat = current_species[:, 11:12]

            # Add 1.0 so the base fluid remains 1.0x when no clot is present
            total_multiplier = 1.0 + self.mu1_sigmoid(Mat) + self.mu2_sigmoid(FI)
            mu_eff = mu_base * total_multiplier

        # ==========================================
        # FINAL DECODING & LOSS PREPARATION
        # ==========================================
        if self.training:
            # Final Kinematics pass (connected to autograd)
            kin_in = batch.x[:, :15].clone()

            if kin_in.shape[-1] == 15:
                kin_in[:, 13] = mu_eff.squeeze(-1) / 0.056

            # Apply Fourier encoding and encode the dynamic viscosity
            kin_encoded = self._apply_fourier_encoding(kin_in)
            mu_nd_train = mu_eff / 0.056
            mu_enc_train = self.mu_encoder(mu_nd_train)

            # Reconstruct the exact 3-part latent sum
            z_kin = apply_processor(self.kin_processor, self.kin_encoder(kin_encoded) + mu_enc_train + z_kin)

            # Ensure we slice only the [u, v, p] kinematics (3 channels)
            u_v_p = self.kinematics_decoder(z_kin)[:, :3]

            # Final Biochemistry pass
            bio_in = torch.cat([bio_init, u_v_p, batch.x], dim=-1)
            z_bio = apply_processor(self.bio_processor, self.bio_encoder(bio_in) + z_bio)

            # Enforce non-negativity
            raw_final_species = self.biochem_decoder(z_bio)
            final_species = F.softplus(raw_final_species)

            # FIX 1: Use batch.mask_wall
            wall_mask_view = batch.mask_wall.view(-1, 1).float()
            surface_species = final_species[:, 9:12] * wall_mask_view
            final_species = torch.cat([final_species[:, 0:9], surface_species], dim=1)

            # FIX 2: Recompute the dynamic base Carreau viscosity for the gradient graph
            u_nd_train = u_v_p[:, 0:1]
            v_nd_train = u_v_p[:, 1:2]

            du_dx_nd_t = torch.sparse.mm(batch.G_x, u_nd_train)
            du_dy_nd_t = torch.sparse.mm(batch.G_y, u_nd_train)
            dv_dx_nd_t = torch.sparse.mm(batch.G_x, v_nd_train)
            dv_dy_nd_t = torch.sparse.mm(batch.G_y, v_nd_train)

            scale_grad = u_ref / d_bar
            gamma_dot_train = torch.sqrt(2 * ((du_dx_nd_t * scale_grad) ** 2 + (dv_dy_nd_t * scale_grad) ** 2) +
                                         ((du_dy_nd_t * scale_grad) + (dv_dx_nd_t * scale_grad)) ** 2 + 1e-8)

            mu_base_train = mu_inf + (mu_0 - mu_inf) * torch.pow(1.0 + (lam * gamma_dot_train) ** 2,
                                                                 (n_idx - 1.0) / 2.0)

            FI = final_species[:, 8:9]
            Mat = final_species[:, 11:12]

            # FIX 3: Apply total multiplier to mu_base_train, not mu_cy
            total_multiplier = 1.0 + self.mu1_sigmoid(Mat) + self.mu2_sigmoid(FI)
            mu_eff = mu_base_train * total_multiplier
        else:
            final_species = current_species
            u_v_p = u_v_p  # from the last loop iteration

            # Combine into [Nodes, 16] output
        pred = torch.cat([u_v_p, mu_eff, final_species], dim=-1)
        return pred