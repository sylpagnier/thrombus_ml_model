import torch
import matplotlib.pyplot as plt
import time
from torch_geometric.data import Data, Batch

from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.anderson import anderson_acceleration
from src.config import PhysicsConfig


def setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=100):
    """
    Creates a batch of graphs to test Global Mixing and Batching logic.
    Incorporates Phase 1 features: [x, y], SDF, and Log-Shear Potential.
    """
    graph_list = []
    for _ in range(num_graphs):
        # 1. Create Nodes and Features
        pos = torch.randn((num_nodes_per_graph, 2))

        # SDF: Distance to the vessel wall
        sdf = torch.rand((num_nodes_per_graph, 1))

        # Log-Shear Potential (Step 1.2 spec): log(1 + 1/SDF)
        # Provides a geometric approximation of shear while preventing exploding gradients
        shear_raw = 1.0 / (sdf + 1e-6)
        log_shear_pot = torch.log(1 + shear_raw)

        # Full Input: [x, y] (2) + [sdf] (1) + [log_shear_pot] (1) = 4 Channels
        x_full = torch.cat([pos, sdf, log_shear_pot], dim=-1)

        # 2. Create Random Edges (Spatial)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 5))

        data = Data(x=x_full, edge_index=edge_index)
        graph_list.append(data)

    return Batch.from_data_list(graph_list)


def run_physics_audit(tier="tier1", max_iters=40):
    print("\n=======================================================")
    print(f"   GINO-DEQ Physics & Solver Audit: {tier.upper()}")
    print("=======================================================")

    # Use the central config to dictate physics logic (DRY principle)
    phys_cfg = PhysicsConfig(tier=tier)

    # Tier 1 outputs [u, v, p] (3 channels)
    # Tier 2 outputs [u, v, p, \mu] (4 channels) as \mu becomes a latent variable
    out_channels = 3 if phys_cfg.viscosity_model == "newtonian" else 4

    # 1. Setup Data & Model
    batch_data = setup_synthetic_batch()

    # Initialize the core model with dynamic output channels
    model = GINO_DEQ(
        in_channels=4,
        out_channels=out_channels,
        latent_dim=64,
        max_iters=max_iters
    )
    model.eval()

    print(f"[Setup] Viscosity Model: {phys_cfg.viscosity_model}")
    print(f"[Setup] Output Channels: {out_channels}")
    print(f"[Setup] Created Batch with {batch_data.num_graphs} graphs.")
    print(f"[Setup] Total Nodes: {batch_data.num_nodes}, Input Channels: 4")

    # 2. Pre-compute Fixed Components
    with torch.no_grad():
        z0 = model.encoder(batch_data.x)

        row, col = batch_data.edge_index
        edge_attr = batch_data.x[col, :2] - batch_data.x[row, :2]

        def f_fixed_point(z):
            """Wrapper for DEQ solver: z_{k+1} = GINO(z_k, X)"""
            # Strip the dummy batch dimension added by the Anderson solver
            # to restore PyG's expected [total_nodes, features] shape
            if z.ndim == 3:
                z = z.squeeze(0)

            return model.core(z, batch_data.edge_index, edge_attr, batch_data.batch)

        # --- TEST A: Naive Picard Iteration ---
        print("\n--- Test A: Naive Picard Iteration ---")
        z_picard = z0.clone()
        residuals_naive = []
        start_time = time.time()
        for _ in range(max_iters):
            z_next = f_fixed_point(z_picard)
            res = (z_next - z_picard).norm(dim=-1).mean().item()
            residuals_naive.append(res)
            z_picard = z_next
        picard_time = time.time() - start_time
        print(f"   > Picard Final Residual: {residuals_naive[-1]:.4e} (Time: {picard_time:.3f}s)")

        # --- TEST B: Anderson Acceleration ---
        print("\n--- Test B: Anderson Acceleration ---")
        start_time = time.time()
        z_anderson = anderson_acceleration(f_fixed_point, z0, m=5, max_iter=max_iters, tol=1e-5)
        anderson_time = time.time() - start_time

        final_res_anderson = (f_fixed_point(z_anderson) - z_anderson).norm(dim=-1).mean().item()
        print(f"   > Anderson Final Residual: {final_res_anderson:.4e} (Time: {anderson_time:.3f}s)")

        # --- TEST C: Full Model Forward Pass ---
        print("\n--- Test C: Full Model Forward Pass ---")
        try:
            # Replaces the internal default out_channels with our dynamic one during integration check
            model.decoder = torch.nn.Linear(64, out_channels)
            out = model(batch_data)
            print(f"   > Success! Output shape: {out.shape}")
        except Exception as e:
            print(f"   > Failed! Error: {e}")

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(residuals_naive)), residuals_naive, label='Naive Picard', linewidth=2, linestyle='--')
    plt.axhline(y=final_res_anderson, color='r', linestyle='-', label=f'Anderson Final ({final_res_anderson:.2e})')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Residual ||f(z) - z|| / ||z||')
    plt.title(f'Physics Convergence Audit ({tier.capitalize()}): Naive vs Anderson')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Sequentially audit both active tiers
    run_physics_audit(tier="tier1")
    run_physics_audit(tier="tier2")