import torch
import matplotlib.pyplot as plt
import time
from torch_geometric.data import Data, Batch
from src.phase1.physics.ginodeq import rGINO_DEQ
from src.phase1.physics.anderson import anderson_acceleration


def setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=100):
    """
    Creates a batch of graphs to test Global Mixing and Batching logic.
    """
    graph_list = []
    for _ in range(num_graphs):
        # 1. Create Nodes and Features
        # Features: [x, y] (2) + [sdf] (1) + [shear_pot] (1) = 4 Channels
        pos = torch.randn((num_nodes_per_graph, 2))
        sdf = torch.rand((num_nodes_per_graph, 1))
        shear = torch.rand((num_nodes_per_graph, 1))

        # Concatenate to make the full input feature vector 'x'
        x_full = torch.cat([pos, sdf, shear], dim=-1)

        # 2. Create Random Edges (Spatial)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 5))

        data = Data(x=x_full, edge_index=edge_index)
        graph_list.append(data)

    # Batch them together (PyG handles batch vector creation automatically)
    return Batch.from_data_list(graph_list)


def run_physics_audit(max_iters=40):
    print("=======================================================")
    print("   GINO-DEQ Physics & Solver Audit")
    print("=======================================================")

    # 1. Setup Data & Model
    batch_data = setup_synthetic_batch()
    # Note: in_channels=4 matches our synthetic data (pos+sdf+shear)
    model = rGINO_DEQ(in_channels=4, latent_dim=64, max_iters=max_iters)
    model.eval()

    print(f"[Setup] Created Batch with {batch_data.num_graphs} graphs.")
    print(f"[Setup] Total Nodes: {batch_data.num_nodes}, Input Channels: 4")

    # 2. Pre-compute Fixed Components
    with torch.no_grad():
        # Run Encoder once to get initial Latent State z0
        z0 = model.encoder(batch_data.x)

        # Pre-compute edge attributes (relative positions) for the core
        row, col = batch_data.edge_index
        # Assuming first 2 channels are coordinates
        edge_attr = batch_data.x[col, :2] - batch_data.x[row, :2]

        # Define the Fixed-Point Function f(z) wrapper
        # This maps exactly to what happens inside rGINO_DEQ.forward
        def f_solver(curr_z):
            # curr_z shape: [Batch_Size (1), Total_Nodes, Latent_Dim]
            # Reshape for Geometric Layer: [Total_Nodes, Latent_Dim]
            bsz, n, d = curr_z.shape
            z_flat = curr_z.reshape(bsz * n, d)

            # The core requires the batch vector to do Global Mixing correctly
            out_flat = model.core(z_flat, batch_data.edge_index, edge_attr, batch_data.batch)

            return out_flat.reshape(bsz, n, d)

        # ---------------------------------------------------------
        # TEST A: Naive Picard Iteration (Baseline Physics Check)
        # ---------------------------------------------------------
        print("\n--- Test A: Naive Fixed-Point Iteration (Baseline) ---")
        z_naive = z0.unsqueeze(0)  # Add fake batch dim for consistency
        residuals_naive = []

        start_t = time.time()
        for i in range(max_iters):
            z_next = f_solver(z_naive)
            res = (z_next - z_naive).norm().item() / (z_naive.norm().item() + 1e-6)
            residuals_naive.append(res)
            z_naive = z_next
        time_naive = time.time() - start_t

        print(f"   > Time: {time_naive:.4f}s")
        print(f"   > Final Relative Residual: {residuals_naive[-1]:.2e}")

        # ---------------------------------------------------------
        # TEST B: Anderson Acceleration (Solver Check)
        # ---------------------------------------------------------
        print("\n--- Test B: Anderson Acceleration (Solver Integration) ---")

        start_t = time.time()
        # We call your imported anderson function directly to audit it
        z_anderson = anderson_acceleration(
            f_solver,
            z0.unsqueeze(0),
            m=5,
            lam=1e-4,
            max_iter=max_iters,
            tol=1e-4
        )
        time_anderson = time.time() - start_t

        # Check Equilibrium Quality: || f(z*) - z* ||
        z_star_next = f_solver(z_anderson.unsqueeze(0)).squeeze(0)
        final_res_anderson = (z_star_next - z_anderson).norm().item() / (z_anderson.norm().item() + 1e-6)

        print(f"   > Time: {time_anderson:.4f}s")
        print(f"   > Final Relative Residual: {final_res_anderson:.2e}")

        # ---------------------------------------------------------
        # TEST C: Full Model Forward Pass (Integration Check)
        # ---------------------------------------------------------
        print("\n--- Test C: Full Model Forward Pass ---")
        try:
            out = model(batch_data)
            print(f"   > Success! Output shape: {out.shape}")
        except Exception as e:
            print(f"   > Failed! Error: {e}")

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(residuals_naive)), residuals_naive, label='Naive Picard', linewidth=2, linestyle='--')

    # We plot the final Anderson point as a horizontal line or single point for comparison
    # (Since we didn't track Anderson history inside the function, we show the final result)
    plt.axhline(y=final_res_anderson, color='r', linestyle='-', label=f'Anderson Final ({final_res_anderson:.2e})')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Residual ||f(z) - z|| / ||z||')
    plt.title('Physics Convergence Audit: Naive vs Anderson')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nAudit Conclusion:")
    if final_res_anderson < residuals_naive[-1]:
        print("Anderson Acceleration is converging faster/better than Naive.")
    else:
        print("Anderson is performing similarly or worse. Check hyperparameters (m, lam) or spectral norms.")


if __name__ == "__main__":
    run_physics_audit()