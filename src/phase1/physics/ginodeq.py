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
        # Applying spectral_norm to the GNN's MLP to stabilize fixed-point iterations
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
        # ResNet connection with both Local and Global updates
        return self.norm(self.relu(z + local_out + global_out))

class rGINO_DEQ(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_dim=64, max_iters=25):
        super().__init__()
        self.max_iters = max_iters
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.core = GINOBlock(latent_dim, edge_dim=2)
        self.decoder = nn.Linear(latent_dim, out_channels)

    def forward(self, data):
        # x_in is now already [nodes_nd (2), sdf_nd (1), shear_pot_nd (1)] = 4 channels
        z = self.encoder(data.x)

        row, col = data.edge_index
        # Use only spatial coordinates (first 2 columns) for edge features
        edge_attr = data.x[col, :2] - data.x[row, :2]

        def f_fixed(curr_z):
            bsz, n, d = curr_z.shape
            z_flat = curr_z.reshape(bsz * n, d)
            out_flat = self.core(z_flat, data.edge_index, edge_attr, data.batch)
            return out_flat.reshape(bsz, n, d)

        z_star = anderson_acceleration(f_fixed, z, max_iter=self.max_iters)
        return self.decoder(z_star)