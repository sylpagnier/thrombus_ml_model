import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.anderson import anderson_acceleration
from src.config import PhysicsConfig


def setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=100):
    graph_list = []
    for _ in range(num_graphs):
        pos = torch.randn((num_nodes_per_graph, 2))  # [0:2] x, y
        sdf = torch.rand((num_nodes_per_graph, 1))  # [2] SDF
        shear_pot = torch.rand((num_nodes_per_graph, 1))  # [3] Potential
        normals = torch.nn.functional.normalize(torch.randn((num_nodes_per_graph, 2)), dim=1)  # [4:6]
        rest = torch.zeros((num_nodes_per_graph, 5))  # [6:11]

        x_full = torch.cat([pos, sdf, shear_pot, normals, rest], dim=-1)
        graph_list.append(
            Data(x=x_full, edge_index=torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 5))))
    return Batch.from_data_list(graph_list)


def run_physics_audit(tier="tier1", max_iters=40):
    print("\n" + "=" * 55)
    print(f"   GINO-DEQ Physics & Solver Audit: {tier.upper()}")
    print("=" * 55)

    phys_cfg = PhysicsConfig(tier=tier)
    # Tier 1 outputs [u, v, p, mu] where mu is constant 1.0
    # Tier 2 outputs [u, v, p, mu] where mu is predicted
    out_channels = 4

    batch_data = setup_synthetic_batch()
    model = GINO_DEQ(in_channels=11, out_channels=out_channels, latent_dim=64, max_iters=max_iters)
    model.eval()

    print(f"[Setup] Viscosity Model: {phys_cfg.viscosity_model}")

    with torch.no_grad():
        # A. Apply Fourier encoding (Transforms inputs for High-Frequency awareness)
        x_fourier = model._apply_fourier_encoding(batch_data.x)
        x_enc = model.encoder(x_fourier)

        z0 = x_enc.clone()
        row, col = batch_data.edge_index
        edge_attr = batch_data.x[col, :2] - batch_data.x[row, :2]
        wall_normals = batch_data.x[:, 4:6]

        # B. Newtonian prior for the inner loop audit
        mu = torch.ones((batch_data.x.size(0), 1))
        mu_enc = model.mu_encoder(mu)

        def f_fixed_point(z):
            if z.ndim == 3: z = z.squeeze(0)
            z_in = z + x_enc + mu_enc  # Segregated injection logic
            return model.core(z_in, batch_data.edge_index, edge_attr, batch_data.batch, wall_normals)

        # --- Test Picard vs Anderson ---
        z_picard = z0.clone()
        res_picard = []
        for _ in range(max_iters):
            z_next = f_fixed_point(z_picard)
            res_picard.append((z_next - z_picard).norm().item() / (z_picard.norm().item() + 1e-8))
            z_picard = z_next

        z_anderson = anderson_acceleration(f_fixed_point, z0.unsqueeze(0), max_iter=max_iters, tol=1e-5).squeeze(0)
        res_and = (f_fixed_point(z_anderson) - z_anderson).norm().item() / (z_anderson.norm().item() + 1e-8)

        print(f"   > Picard Final Residual: {res_picard[-1]:.4e}")
        print(f"   > Anderson Final Residual: {res_and:.4e}")

        # --- Test Full Forward ---
        out = model(batch_data)
        print(f"   > Full Forward Success! Shape: {out.shape}")

    # Plotting per tier
    plt.figure(figsize=(8, 4))
    plt.plot(res_picard, label='Picard', color='tab:blue')
    plt.axhline(y=res_and, color='tab:red', linestyle='--', label=f'Anderson ({res_and:.1e})')
    plt.yscale('log')
    plt.title(f"Convergence: {tier.upper()} ({phys_cfg.viscosity_model})")
    plt.xlabel("Iteration")
    plt.ylabel("Rel. Residual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Using non-blocking show if running sequentially
    plt.show(block=True)


if __name__ == "__main__":
    # Now explicitly calling both
    run_physics_audit(tier="tier1")
    run_physics_audit(tier="tier2")