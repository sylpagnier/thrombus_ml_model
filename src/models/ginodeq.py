import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from src.models.anderson import anderson_acceleration


class GlobalMixingBlock(nn.Module):
    """
    Communicates information globally across the mesh (Inlet <-> Outlet)
    in a single step. Essential for pressure propagation in fluid solvers.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, batch):
        # 1. Aggregate: Pool info from all nodes to a single global vector
        # (batch_size, latent_dim)
        global_context = global_mean_pool(x, batch)

        # 2. Process: "Think" about the global state (Physics constraints)
        global_update = self.global_mlp(global_context)

        # 3. Broadcast: Send global info back to every specific node
        # expands (batch_size, latent_dim) -> (num_nodes, latent_dim)
        # We index using 'batch' to map the global vec back to the correct nodes
        return global_update[batch]


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=2):
        super().__init__()
        # Local Physics (Viscosity, Wall interactions)
        self.conv = GINEConv(
            nn.Sequential(nn.Linear(latent_dim, latent_dim),
                          nn.ReLU(),
                          nn.Linear(latent_dim, latent_dim)),
            edge_dim=edge_dim
        )
        # Global Physics (Pressure, Continuity)
        self.global_mixer = GlobalMixingBlock(latent_dim)

        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr, batch):
        # Local Message Passing
        local_out = self.conv(z, edge_index, edge_attr)

        # Global Context Mixing
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
        # x_in: [x, y, sdf, shear_pot]
        x_in = torch.cat([data.x, data.sdf, data.shear_pot], dim=-1)
        z = self.encoder(x_in)

        row, col = data.edge_index
        edge_attr = data.x[col] - data.x[row]

        # The Fixed Point Function
        def f_fixed(curr_z):
            # Curr_z comes in shape (Batch, Nodes, Latent) from Anderson
            # We need to reshape for PyG GNNs -> (Total_Nodes, Latent)
            bsz, n, d = curr_z.shape
            z_flat = curr_z.reshape(bsz * n, d)

            # Run the GINO Block (Local + Global)
            # Note: data.batch is required for global pooling
            out_flat = self.core(z_flat, data.edge_index, edge_attr, data.batch)

            # Reshape back for Anderson -> (Batch, Nodes, Latent)
            return out_flat.reshape(bsz, n, d)

        # Anderson Solver
        z_star = anderson_acceleration(f_fixed, z, max_iter=self.max_iters)
        return self.decoder(z_star)