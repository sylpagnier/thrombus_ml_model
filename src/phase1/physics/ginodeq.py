import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import GINEConv, global_mean_pool
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


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                spectral_norm(nn.Linear(latent_dim, latent_dim)),
                nn.ReLU(),
                spectral_norm(nn.Linear(latent_dim, latent_dim))
            ),
            edge_dim=edge_dim
        )
        self.global_mixer = GlobalMixingBlock(latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr, batch):
        local_out = self.conv(z, edge_index, edge_attr)
        global_out = self.global_mixer(z, batch)
        return self.norm(self.relu(z + local_out + global_out))


class GINO_DEQ(nn.Module):
    def __init__(self, in_channels=11, out_channels=4, latent_dim=64, max_iters=25):
        super().__init__()
        self.max_iters = max_iters
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.core = GINOBlock(latent_dim, edge_dim=2)
        self.decoder = nn.Linear(latent_dim, out_channels)

    def forward(self, data, solver="anderson", anderson_beta=0.8):
        """
        solver: "picard" for warmup epochs, "anderson" for accelerated DEQ.
        anderson_beta: Damping factor for Anderson (values < 1.0 improve stability).
        """
        # Store the encoded geometry to inject into the solver loop
        x_enc = self.encoder(data.x)

        z = x_enc.clone()
        row, col = data.edge_index
        edge_attr = data.x[col, :2] - data.x[row, :2]

        batch_idx = getattr(data, 'batch', None)
        if batch_idx is None:
            # Create a mock batch array [0, 0, ..., 0] for a single graph
            batch_idx = torch.zeros(data.num_nodes, dtype=torch.long, device=z.device)

        def f_fixed(curr_z):
            # The anderson solver adds a dummy batch dim [1, N, d].
            # We reshape to [N, d] for PyG, then back to [1, N, d] for the solver.
            bsz, n, d = curr_z.shape
            z_flat = curr_z.reshape(bsz * n, d)

            # Additive Injection of Geometric Features
            z_in = z_flat + x_enc

            out_flat = self.core(z_in, data.edge_index, edge_attr, batch_idx)

            return out_flat.reshape(bsz, n, d)

        # Dynamic Solver Routing
        if solver == "picard":
            # Stable warmup: blindly apply the fixed-point function
            z_star = z.unsqueeze(0) if z.ndim == 2 else z
            for _ in range(self.max_iters):
                z_star = f_fixed(z_star)
            z_star = z_star.squeeze(0)
        else:
            # Accelerated DEQ with damping
            # Ensure z has the dummy batch dimension expected by f_fixed
            z_init = z.unsqueeze(0) if z.ndim == 2 else z

            z_star = anderson_acceleration(
                f_fixed, z_init,
                max_iter=self.max_iters,
                beta=anderson_beta
            )

            # Remove the dummy batch dimension before passing to the decoder
            if z_star.ndim == 3:
                z_star = z_star.squeeze(0)

        raw_out = self.decoder(z_star)
        raw_out[:, 3] = F.softplus(raw_out[:, 3])
        return raw_out