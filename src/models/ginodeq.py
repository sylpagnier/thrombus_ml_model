import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from anderson import anderson_acceleration

class GINOBlock(nn.Module):
    """The iterative GNN block that finds the equilibrium state Z*."""

    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        # FIX: Added edge_dim to project [dx, dy] into the latent_dim space
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            ),
            edge_dim=edge_dim
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr):
        # Residual update: z_{i+1} = z_i + GNN(z_i, geometry)
        res = self.conv(z, edge_index, edge_attr)
        return self.norm(self.relu(z + res))


class rGINO_DEQ(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, latent_dim=64, max_iters=25):
        super().__init__()
        self.max_iters = max_iters
        self.encoder = nn.Sequential(...)  # (Same as before)
        self.core = GINOBlock(latent_dim, edge_dim=2)
        self.decoder = nn.Linear(latent_dim, out_channels)

    def forward(self, data):
        x_in = torch.cat([data.x, data.sdf, data.shear_pot], dim=-1)
        z = self.encoder(x_in)

        row, col = data.edge_index
        edge_attr = data.x[col] - data.x[row]

        # Define the fixed-point function for the accelerator
        def f_fixed(current_z):
            return self.core(current_z, data.edge_index, edge_attr)

        # Replace the 'for' loop with Anderson Acceleration
        # Using a simple view to handle the node-level latent states as a 'batch'
        z_final = anderson_acceleration(f_fixed, z.unsqueeze(0), max_iter=self.max_iters)

        return self.decoder(z_final.squeeze(0))