import torch
import matplotlib.pyplot as plt
import time
from src.models.ginodeq import rGINO_DEQ
from torch_geometric.data import Data


def run_convergence_audit(max_iters=30):
    print("--- Project PIRON: DEQ Convergence & Performance Audit ---")

    # 1. Setup Synthetic Graph (N=200)
    num_nodes = 200
    data = Data(
        x=torch.randn((num_nodes, 2)),
        sdf=torch.rand((num_nodes, 1)),
        shear_pot=torch.rand((num_nodes, 1)),
        edge_index=torch.randint(0, num_nodes, (2, 1000))
    )

    # 2. Initialize Model
    model = rGINO_DEQ(latent_dim=64, max_iters=max_iters)
    model.eval()

    # 3. Track Residuals per Iteration
    # We temporarily bypass the Anderson wrapper to see raw GNN stability
    residuals = []

    with torch.no_grad():
        # Encoder pass
        x_in = torch.cat([data.x, data.sdf, data.shear_pot], dim=-1)
        z = model.encoder(x_in)

        row, col = data.edge_index
        edge_attr = data.x[col] - data.x[row]

        # Iterative loop with debugging
        start_time = time.time()
        for i in range(max_iters):
            z_next = model.core(z, data.edge_index, edge_attr)
            res = torch.norm(z_next - z).item()
            residuals.append(res)
            z = z_next

            if i % 5 == 0:
                print(f"Iteration {i:02d} | Residual: {res:.6f}")

        total_time = time.time() - start_time

    # 4. Final Inference through Decoder
    out = model.decoder(z)

    print(f"\nAudit Results:")
    print(f"- Total Inference Time: {total_time:.4f}s")
    print(f"- Final Residual: {residuals[-1]:.8f}")
    print(f"- Output Shape: {out.shape}")

    # 5. Visualization: Convergence Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(residuals)), residuals, marker='o', color='tab:blue', label='Latent Residual')
    plt.yscale('log')  # Residuals should drop exponentially
    plt.xlabel('Iteration (k)')
    plt.ylabel('||z_k - z_{k-1}||')
    plt.title('rGINO-DEQ Fixed-Point Convergence Audit')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_convergence_audit()