import torch
import torch.nn as nn
from src.phase1.physics.ginodeq import GINOBlock, SpectralLinear


class GINO_DEQ_Tier3(nn.Module):
    """
    Tier 3 Deep Equilibrium Model for Thrombosis Simulation.
    Segregated Block-Seidel solver adapted for 12-species (9 bulk + 3 surface)
    incorporating COMSOL's Fibrin kinetics and dual-trigger rheology.
    """

    def __init__(self, in_channels=12, latent_dim=64, max_outer_iters=3, max_inner_iters=25,
                 mu_ratio_max=80.0, mat_crit=2e7, fi_crit=0.6,
                 temp_mat=1e6, temp_fi=0.05, lora_rank=4, lora_alpha=1.0):
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

        # ==========================================
        # 1. KINEMATICS BACKBONE (FROZEN)
        # ==========================================
        # Kinematics takes u, v, p + mu_eff (4 features)
        self.kin_encoder = SpectralLinear(in_features=4, out_features=latent_dim)
        self.kin_processor = GINOBlock(latent_dim)
        self.kinematics_decoder = SpectralLinear(in_features=latent_dim, out_features=3)

        # ==========================================
        # 2. BIOCHEMISTRY SOLVER
        # ==========================================
        # Biochemistry takes 12 species + newly updated kinematics (u, v, p) (15 features)
        self.bio_encoder = SpectralLinear(in_features=in_channels + 3, out_features=latent_dim)
        self.bio_processor = GINOBlock(latent_dim)

        # Decoder strictly matched to the updated 12 species count
        self.biochem_decoder = SpectralLinear(in_features=latent_dim, out_features=12)

    def mu1_sigmoid(self, mat):
        """Soft logic for Platelet-driven viscosity multiplier."""
        return self.mu_ratio_max * torch.sigmoid((mat - self.mat_crit) / self.temp_mat)

    def mu2_sigmoid(self, fi):
        """Soft logic for Fibrin-driven viscosity multiplier."""
        return self.mu_ratio_max * torch.sigmoid((fi - self.fi_crit) / self.temp_fi)

    def forward(self, batch, anderson_beta=1.0, anderson_warmup=5):
        # 1. Extract dimensions natively and FORCE standard integer type
        num_nodes = int(batch.x.shape[0])
        device = batch.x.device

        # 2. Setup Initial Guesses & Boundary Conditions
        # (Pass dimensions directly rather than as a tuple)
        kin_init = torch.zeros(num_nodes, 3, dtype=torch.float32, device=device)  # u, v, p
        bio_init = torch.zeros(num_nodes, 12, dtype=torch.float32, device=device)  # 12 species

        # torch.full still requires a tuple or list for the size parameter
        mu_cy = torch.full((num_nodes, 1), 0.0035, dtype=torch.float32, device=device)

        # Wall mask formatted for element-wise multiplication
        wall_mask = batch.mask_wall.view(-1, 1).float()

        # Initialize hidden states
        z_kin = torch.zeros(num_nodes, self.latent_dim, device=device)
        z_bio = torch.zeros(num_nodes, self.latent_dim, device=device)

        # Initialize Effective Viscosity
        mu_eff = mu_cy.clone()

        # Helper to pass all required topology and modulation args to GINOBlock
        def apply_processor(processor, x):
            # 1. Get batch indices
            batch_idx = batch.batch if hasattr(batch, 'batch') and batch.batch is not None else torch.zeros(num_nodes,
                                                                                                            dtype=torch.long,
                                                                                                            device=device)

            # 2. Match PyG's expected edge dimensions for the unused modulators
            num_edges = int(batch.edge_index.shape[1])
            mod_adv = torch.zeros(num_edges, 1, dtype=torch.float32, device=device)
            mod_rheo = torch.zeros(num_edges, 1, dtype=torch.float32, device=device)

            # 3. Call GINOBlock with its strict signature
            return processor(x, batch.edge_index, batch.edge_attr, batch_idx, mod_adv, mod_rheo)

        # ==========================================
        # BLOCK-SEIDEL OUTER LOOP
        # ==========================================
        for outer_idx in range(self.max_outer_iters):

            # --- 1. Kinematics Inner Loop ---
            def f_kinematics(z):
                kin_in = torch.cat([kin_init, mu_eff], dim=-1)
                return apply_processor(self.kin_processor, self.kin_encoder(kin_in) + z)

            with torch.no_grad():
                for _ in range(self.max_inner_iters):
                    z_kin = f_kinematics(z_kin)

            u_v_p = self.kinematics_decoder(z_kin)

            # --- 2. Biochemistry Inner Loop ---
            def f_biochem(z):
                bio_in = torch.cat([bio_init, u_v_p.detach()], dim=-1)
                return apply_processor(self.bio_processor, self.bio_encoder(bio_in) + z)

            with torch.no_grad():
                for _ in range(self.max_inner_iters):
                    z_bio = f_biochem(z_bio)

            current_species = self.biochem_decoder(z_bio)

            # Zero out surface species (M, Mas, Mat) away from the wall
            current_species[:, 9:12] = current_species[:, 9:12] * wall_mask

            # --- 3. Dual-Trigger Rheology Update ---
            FI = current_species[:, 8:9]  # 9th bulk species
            Mat = current_species[:, 11:12]

            mu_eff = mu_cy * (self.mu1_sigmoid(Mat) + self.mu2_sigmoid(FI))

        # ==========================================
        # FINAL DECODING & LOSS PREPARATION
        # ==========================================
        if self.training:
            # Re-engage autograd graph for the final pass to calculate loss
            z_kin = f_kinematics(z_kin)
            u_v_p = self.kinematics_decoder(z_kin)

            z_bio = f_biochem(z_bio)
            final_species = self.biochem_decoder(z_bio)
            final_species[:, 9:12] = final_species[:, 9:12] * wall_mask

            FI = final_species[:, 8:9]
            Mat = final_species[:, 11:12]
            mu_eff = mu_cy * (self.mu1_sigmoid(Mat) + self.mu2_sigmoid(FI))
        else:
            final_species = current_species

        # Combine into a single [Nodes, 16] output tensor: u, v, p, mu, 12x species
        pred = torch.cat([u_v_p, mu_eff, final_species], dim=-1)

        return pred