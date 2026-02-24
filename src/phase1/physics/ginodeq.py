import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax
from src.phase1.physics.anderson import anderson_acceleration


class GlobalMixingBlock(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(latent_dim, latent_dim))
        )

    def forward(self, x, batch):
        global_context = global_mean_pool(x, batch)
        global_update = self.global_mlp(global_context)
        return global_update[batch]


class MultiHeadPhysicsGATConv(MessagePassing):
    """
    Physics-Informed Multi-Head Graph Attention Network.
    Separates streamwise (advection) and cross-stream (rheology/shear)
    message passing to prevent over-smoothing of physical gradients.
    """

    def __init__(self, latent_dim, edge_dim=2):
        super().__init__(aggr='add', node_dim=0)

        self.edge_proj = spectral_norm(nn.Linear(edge_dim, latent_dim))

        # --- Head 1: Advection (Streamwise) ---
        self.att_adv = spectral_norm(nn.Linear(2 * latent_dim, 1))
        self.val_adv = spectral_norm(nn.Linear(latent_dim, latent_dim // 2))

        # --- Head 2: Rheology/Shear (Cross-stream) ---
        self.att_rheo = spectral_norm(nn.Linear(2 * latent_dim, 1))
        self.val_rheo = spectral_norm(nn.Linear(latent_dim, latent_dim // 2))

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(latent_dim, latent_dim))
        )

    def forward(self, x, edge_index, edge_attr, mod_adv, mod_rheo):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             mod_adv=mod_adv, mod_rheo=mod_rheo)
        return self.mlp(out)

    def message(self, x_i, x_j, edge_attr, mod_adv, mod_rheo, index, ptr, size_i):
        edge_emb = self.edge_proj(edge_attr)
        msg_base = x_j + edge_emb

        # Combined features for attention scoring
        alpha_feat = torch.cat([x_i, msg_base], dim=-1)

        # --- Calculate Advection Messages ---
        e_adv = self.leaky_relu(self.att_adv(alpha_feat))
        e_adv = e_adv + mod_adv  # Apply streamwise structural prior
        alpha_adv = softmax(e_adv, index, ptr, size_i)
        out_adv = alpha_adv * self.val_adv(msg_base)

        # --- Calculate Rheology Messages ---
        e_rheo = self.leaky_relu(self.att_rheo(alpha_feat))
        e_rheo = e_rheo + mod_rheo  # Apply cross-stream structural prior
        alpha_rheo = softmax(e_rheo, index, ptr, size_i)
        out_rheo = alpha_rheo * self.val_rheo(msg_base)

        # Concatenate the split latent space back together
        return torch.cat([out_adv, out_rheo], dim=-1)


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        # Ensure latent_dim is even so it splits cleanly into the two heads
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for multi-head split"

        self.conv = MultiHeadPhysicsGATConv(latent_dim, edge_dim=edge_dim)
        self.global_mixer = GlobalMixingBlock(latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr, batch, mod_adv, mod_rheo):
        local_out = self.conv(z, edge_index, edge_attr, mod_adv, mod_rheo)
        global_out = self.global_mixer(z, batch)
        return self.norm(self.relu(z + local_out + global_out))


class GINO_DEQ(nn.Module):
    def __init__(self, in_channels=11, out_channels=4, latent_dim=64, max_iters=25, num_fourier_freqs=8, outer_iters=3):
        super().__init__()
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.num_fourier_freqs = num_fourier_freqs

        freqs = (2.0 ** torch.arange(num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 3 * num_fourier_freqs * 2
        encoded_channels = (in_channels - 3) + 3 + fourier_channels

        self.encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.core = GINOBlock(latent_dim, edge_dim=2)

        self.kinematics_decoder = nn.Linear(latent_dim, 3)

        self.mu_decoder = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

        self.mu_encoder = nn.Linear(1, latent_dim)

    def _apply_fourier_encoding(self, x):
        nodes_nd = x[:, 0:2]
        sdf_nd = x[:, 2:3]
        shear_pot = x[:, 3:4]
        wall_normal = x[:, 4:6]

        # Extract the prior (assuming it's the last 2 columns based on mesh_to_graph)
        uv_prior = x[:, -2:]
        rest = x[:, 6:-2]

        features_to_encode = torch.cat([sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # We keep uv_prior in the encoded input so the network can "see" it
        encoded_x = torch.cat([nodes_nd, shear_pot, features_to_encode, fourier_feats, rest, uv_prior], dim=1)
        return encoded_x, uv_prior

    def forward(self, data, solver="anderson", anderson_beta=0.8):
        x_encoded, uv_prior = self._apply_fourier_encoding(data.x)
        x_enc = self.encoder(x_encoded)
        z = x_enc.clone()

        row, col = data.edge_index
        edge_attr = data.x[col, :2] - data.x[row, :2]

        # Safe batch extraction for PyG
        batch_idx = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        # --- PRECOMPUTE DUAL STATIC PHYSICS MODULATORS ---
        wall_normals = data.x[:, 4:6]
        e_dir = F.normalize(edge_attr, p=2, dim=-1, eps=1e-8)
        n_dir = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        mod_rheo = torch.log(torch.clamp(dot_prod, min=1e-3, max=1.0))
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=1e-3, max=1.0))

        # -------------------------------------------------

        # --- THE COUPLED DEQ STEP ---
        # Instead of an outer loop, we define the complete non-linear step z_{k+1} = f(z_k).
        # This allows Anderson Acceleration to continuously build its m=5 history.
        def f_coupled(curr_z):
            # 1. Enforce the physical bottleneck: decode latent state to physical mu
            mu_raw = self.mu_decoder(curr_z)
            mu = F.softplus(mu_raw) + 1.0

            # 2. Re-encode mu to inject into the feature space
            mu_enc = self.mu_encoder(mu)

            # 3. Form the combined input
            z_in = curr_z + x_enc + mu_enc

            # 4. Pass through the multi-head physics core
            return self.core(z_in, data.edge_index, edge_attr, batch_idx, mod_adv, mod_rheo)

        z_init = z.unsqueeze(0) if z.ndim == 2 else z

        # --- SINGLE CONTINUOUS SOLVE ---
        if solver == "picard":
            z_star = z_init
            for _ in range(self.max_iters):
                z_star = f_coupled(z_star)
            z = z_star.squeeze(0)
        else:
            # Anderson now runs for the full max_iters, preserving the contraction mapping
            z_star = anderson_acceleration(
                f_coupled, z_init, batch_idx=batch_idx,
                max_iter=self.max_iters, beta=anderson_beta
            )
            z = z_star.squeeze(0)

        # --- FINAL DECODE AFTER CONVERGENCE ---
        mu_raw = self.mu_decoder(z)
        mu = F.softplus(mu_raw) + 1.0

        u_v_p_residual = self.kinematics_decoder(z)
        uv_res = u_v_p_residual[:, :2]
        p = u_v_p_residual[:, 2:3]

        uv_final = uv_res + uv_prior

        u_v_p = torch.cat([uv_final, p], dim=1)
        return torch.cat([u_v_p, mu], dim=1)