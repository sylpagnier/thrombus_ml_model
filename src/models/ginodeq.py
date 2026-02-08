import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from src.models.anderson import anderson_acceleration


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)),
            edge_dim=edge_dim
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr):
        res = self.conv(z, edge_index, edge_attr)
        return self.norm(self.relu(z + res))


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
        # x_in: [x, y, sdf, shear_pot]
        x_in = torch.cat([data.x, data.sdf, data.shear_pot], dim=-1)
        z = self.encoder(x_in)

        row, col = data.edge_index
        edge_attr = data.x[col] - data.x[row]

        def f_fixed(curr_z):
            # anderson_acceleration handles the unsqueeze internally
            return self.core(curr_z.squeeze(0), data.edge_index, edge_attr).unsqueeze(0)

        z_star = anderson_acceleration(f_fixed, z, max_iter=self.max_iters)
        return self.decoder(z_star)