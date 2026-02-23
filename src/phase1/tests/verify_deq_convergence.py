import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from src.config import PhysicsConfig
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.anderson import anderson_acceleration


def setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=100):
    """Generates deterministic-shaped dummy data for testing."""
    graph_list = []
    for _ in range(num_graphs):
        pos = torch.randn((num_nodes_per_graph, 2))
        sdf = torch.rand((num_nodes_per_graph, 1))
        shear_pot = torch.rand((num_nodes_per_graph, 1))
        normals = torch.nn.functional.normalize(torch.randn((num_nodes_per_graph, 2)), dim=1)
        rest = torch.zeros((num_nodes_per_graph, 5))

        x_full = torch.cat([pos, sdf, shear_pot, normals, rest], dim=-1)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 5))
        graph_list.append(Data(x=x_full, edge_index=edge_index))

    return Batch.from_data_list(graph_list)


def test_anderson_standalone_convergence():
    """
    TEST 1: Pure Solver Math
    Tests the Anderson solver on a known, deterministic contraction mapping
    to ensure it actually finds the correct root.
    Equation: f(x) = 0.5 * x + 1.0  => The true fixed point is x = 2.0
    """

    def f_affine(x):
        return 0.5 * x + 1.0

    # [bsz, n, d] format expected by anderson.py
    z0 = torch.zeros((1, 10, 5))

    z_fixed = anderson_acceleration(f_affine, z0, max_iter=20, tol=1e-5)

    # Assert the solver actually converges to the mathematical truth
    assert torch.allclose(z_fixed, torch.tensor(2.0), atol=1e-4), \
        f"Anderson solver failed to find the true fixed point. Got {z_fixed.mean().item()}"


def test_ginodeq_forward_api():
    """
    TEST 2: Model Plumbing & Forward Pass
    Tests that the actual model's forward pass correctly routes data through
    the operator splitting logic, outer/inner loops, and both solvers without NaNs.
    """
    batch_data = setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=50)

    # Use smaller dims to speed up the test suite
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=16, max_iters=4, outer_iters=2)
    model.eval()

    for solver in ["picard", "anderson"]:
        with torch.no_grad():
            out = model(batch_data, solver=solver)

        # 1. Assert Output Shape: Expecting [N, 4] for [u, v, p, mu]
        expected_shape = (batch_data.x.size(0), 4)
        assert out.shape == expected_shape, \
            f"Shape mismatch for {solver}. Expected {expected_shape}, got {out.shape}"

        # 2. Assert No NaNs or Infs
        assert torch.isfinite(out).all(), \
            f"NaNs or Infs detected in the {solver} forward pass!"

        # 3. Assert Viscosity Bounds: softplus(mu) + 1.0 means mu must be >= 1.0
        mu_pred = out[:, 3]
        assert (mu_pred >= 1.0).all(), \
            f"Viscosity (mu) dropped below 1.0 reference bound in {solver} solver!"


def test_ginodeq_backward_pass():
    """
    TEST 3: Gradient Flow
    Ensures autograd can trace through the DEQ loop without in-place mutation crashes.
    """
    batch_data = setup_synthetic_batch(num_graphs=1, num_nodes_per_graph=30)
    model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=16, max_iters=3)
    model.train()

    out = model(batch_data, solver="anderson")

    # Create a dummy MSE loss against random targets
    target = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out, target)

    loss.backward()

    # Verify gradients actually reached the earliest parts of the network
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.norm(param.grad) > 0:
            has_grad = True
            break

    assert has_grad, "Backward pass failed! Gradients did not propagate through the DEQ loop."


def test_visualize_convergence_audit():
    """
    TEST 4: Visual Diagnostics
    Recreates the inner-loop physics mapping to extract step-by-step residuals.
    Pops up an interactive matplotlib window with subplots for both Tier 1 and Tier 2.
    """
    max_iters = 30
    batch_data = setup_synthetic_batch(num_graphs=1, num_nodes_per_graph=50)

    # Create a side-by-side plot (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DEQ Solver Convergence Audit: Picard vs Anderson", fontsize=14, y=1.05)

    tiers = ["tier1", "tier2"]

    for idx, tier in enumerate(tiers):
        phys_cfg = PhysicsConfig(tier=tier)
        ax = axes[idx]

        # Initialize a fresh model for each tier to ensure independent graphs
        model = GINO_DEQ(in_channels=11, out_channels=4, latent_dim=16, max_iters=max_iters)
        model.eval()

        with torch.no_grad():
            x_fourier = model._apply_fourier_encoding(batch_data.x)
            x_enc = model.encoder(x_fourier)
            z0 = x_enc.clone()

            row, col = batch_data.edge_index
            edge_attr = batch_data.x[col, :2] - batch_data.x[row, :2]
            wall_normals = batch_data.x[:, 4:6]

            # Baseline Viscosity
            mu = torch.ones((batch_data.x.size(0), 1))
            mu_enc = model.mu_encoder(mu)

            def f_fixed_point(z):
                if z.ndim == 3: z = z.squeeze(0)
                z_in = z + x_enc + mu_enc
                return model.core(z_in, batch_data.edge_index, edge_attr, batch_data.batch, wall_normals)

            # 1. Track Picard History
            z_picard = z0.clone()
            res_picard = []
            for _ in range(max_iters):
                z_next = f_fixed_point(z_picard)
                res_picard.append((z_next - z_picard).norm().item() / (z_picard.norm().item() + 1e-8))
                z_picard = z_next

            # 2. Track Anderson History
            z_anderson, res_anderson = anderson_acceleration(
                f_fixed_point, z0.unsqueeze(0),
                max_iter=max_iters,
                tol=1e-5,
                return_history=True
            )

        # --- Plotting for this specific tier ---
        ax.plot(res_picard, label='Picard', color='tab:blue', marker='o', markersize=4)

        # Anderson curve (shifted by 2 because the history loop starts at k=2)
        iterations_anderson = range(2, len(res_anderson) + 2)
        ax.plot(iterations_anderson, res_anderson, label='Anderson', color='tab:red', marker='x', markersize=4)

        ax.set_yscale('log')
        ax.set_title(f"{tier.upper()} ({phys_cfg.viscosity_model.capitalize()})")
        ax.set_xlabel("Iteration")

        # Only add the Y-axis label to the leftmost plot to keep it clean
        if idx == 0:
            ax.set_ylabel("Relative Residual (Log Scale)")

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("   Running Robust DEQ Test Suite")
    print("=" * 55)

    test_anderson_standalone_convergence()
    print(" [✓] Anderson Standalone Math Verification")

    test_ginodeq_forward_api()
    print(" [✓] GINO-DEQ Forward API & Shape Verification")

    test_ginodeq_backward_pass()
    print(" [✓] GINO-DEQ Autograd & Backward Pass Verification")

    print("\n [!] Launching Visual Convergence Audit...")
    print("     (Close the plot window to finish the script)")
    test_visualize_convergence_audit()

    print("\nAll tests passed successfully! 🚀")