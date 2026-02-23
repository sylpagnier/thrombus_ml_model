import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax
from src.phase1.physics.anderson import anderson_acceleration


class GlobalMixingBlock(nn.Module):
    """
    Communicates information globally across the mesh (Inlet <-> Outlet).
    Wraps layers in spectral_norm to ensure contraction mapping for DEQ stability.
    """

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


class ModulatedGATConv(MessagePassing):
    """
    Physics-Informed Graph Attention Network.
    Modulates standard attention weights based on a precomputed static edge
    modulator to penalize cross-stream message passing.
    """

    def __init__(self, latent_dim, edge_dim=2):
        super().__init__(aggr='add', node_dim=0)

        self.edge_proj = spectral_norm(nn.Linear(edge_dim, latent_dim))

        self.att = spectral_norm(nn.Linear(2 * latent_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(latent_dim, latent_dim))
        )

    def forward(self, x, edge_index, edge_attr, physics_modulator):
        # Propagate now accepts the precomputed static modulator
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, physics_modulator=physics_modulator)
        return self.mlp(out)

    def message(self, x_i, x_j, edge_attr, physics_modulator, index, ptr, size_i):
        edge_emb = self.edge_proj(edge_attr)
        alpha_feat = torch.cat([x_i, x_j + edge_emb], dim=-1)

        e = self.att(alpha_feat)
        e = self.leaky_relu(e)

        # Apply the static physics modulator (log transform is already precomputed)
        e = e + physics_modulator

        alpha = softmax(e, index, ptr, size_i)
        return alpha * (x_j + edge_emb)


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        self.conv = ModulatedGATConv(latent_dim, edge_dim=edge_dim)
        self.global_mixer = GlobalMixingBlock(latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr, batch, physics_modulator):
        local_out = self.conv(z, edge_index, edge_attr, physics_modulator)
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
        rest = x[:, 6:]

        features_to_encode = torch.cat([sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)

        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return torch.cat([nodes_nd, shear_pot, features_to_encode, fourier_feats, rest], dim=1)

    def forward(self, data, solver="anderson", anderson_beta=0.8):
        x_encoded = self._apply_fourier_encoding(data.x)

        x_enc = self.encoder(x_encoded)
        z = x_enc.clone()  # Kinematics latent state

        row, col = data.edge_index
        edge_attr = data.x[col, :2] - data.x[row, :2]
        batch_idx = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.size(0),
                                                                                                     dtype=torch.long,
                                                                                                     device=data.x.device)

        # --- PRECOMPUTE STATIC PHYSICS MODULATOR ---
        # Target nodes in message passing are indexed by 'col' in PyG (x_i)
        wall_normals = data.x[:, 4:6]
        e_dir = F.normalize(edge_attr, p=2, dim=-1, eps=1e-8)
        n_dir = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        # Precompute the log addition so the Anderson loop only does addition
        physics_modulator = torch.log(dot_prod + 0.1 + 1e-8)
        # ---------------------------------------------

        # Initialize Viscosity to 1.0 (Newtonian reference prior)
        mu = torch.ones((data.x.size(0), 1), dtype=data.x.dtype, device=data.x.device)

        # Distribute the total DEQ iterations across the segregated loops
        inner_iters = max(1, self.max_iters // self.outer_iters)

        # --- OUTER LOOP (Property Update) ---
        for outer_step in range(self.outer_iters):

            # 1. Freeze mu and encode it for injection
            mu_enc = self.mu_encoder(mu)

            # 2. INNER LOOP (Kinematics DEQ)
            def f_inner(curr_z):
                bsz, n, d = curr_z.shape
                z_flat = curr_z.reshape(bsz * n, d)

                # Inject Geometry/BCs + Current Kinematics + FROZEN Viscosity
                z_in = z_flat + x_enc + mu_enc

                # Pass the precomputed physics_modulator instead of wall_normals
                out_flat = self.core(z_in, data.edge_index, edge_attr, batch_idx, physics_modulator)
                return out_flat.reshape(bsz, n, d)

            if solver == "picard":
                z_star = z.unsqueeze(0) if z.ndim == 2 else z
                for _ in range(inner_iters):
                    z_star = f_inner(z_star)
                z = z_star.squeeze(0)
            else:
                z_init = z.unsqueeze(0) if z.ndim == 2 else z

                z_star = anderson_acceleration(
                    f_inner, z_init,
                    max_iter=inner_iters,
                    beta=anderson_beta
                )
                z = z_star.squeeze(0)

            # 3. Explicit Sub-Network Update for Viscosity
            mu_raw = self.mu_decoder(z)
            mu = F.softplus(mu_raw) + 1.0 # +1 since  mu_inf is mu_ref for ND

            # Final decode
        u_v_p = self.kinematics_decoder(z)

        return torch.cat([u_v_p, mu], dim=1)