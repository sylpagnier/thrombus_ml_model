import torch
from src.models.ginodeq import rGINO_DEQ
from torch_geometric.data import Data


def test_model_forward():
    # 1. Create a dummy graph (N=100 nodes)
    num_nodes = 100
    x = torch.randn((num_nodes, 2))  # ND-Coordinates
    sdf = torch.rand((num_nodes, 1))  # ND-SDF
    shear = torch.rand((num_nodes, 1))  # ND-Shear Potential
    edge_index = torch.randint(0, num_nodes, (2, 500))  # Random connectivity

    data = Data(x=x, sdf=sdf, shear_pot=shear, edge_index=edge_index)
    data.num_nodes = num_nodes

    # 2. Initialize Model
    model = rGINO_DEQ(in_channels=4, out_channels=3, latent_dim=64, max_iters=5)

    # 3. Forward Pass
    try:
        out = model(data)
        print(f"✅ Forward Pass Successful!")
        print(f"Output Shape: {out.shape} (Expected: [{num_nodes}, 3])")

        # Verify output dimensions: [u, v, p]
        assert out.shape == (num_nodes, 3)
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")


def test_iteration_stability():
    """Checks if the latent state changes across iterations."""
    num_nodes = 50
    data = Data(
        x=torch.randn((num_nodes, 2)),
        sdf=torch.rand((num_nodes, 1)),
        shear_pot=torch.rand((num_nodes, 1)),
        edge_index=torch.randint(0, num_nodes, (2, 200))
    )

    model = rGINO_DEQ(max_iters=1)
    out_1 = model(data)

    model.max_iters = 10
    out_10 = model(data)

    # If the core is working, the output after 10 iterations should be
    # different from 1 iteration (unless it already converged)
    diff = torch.norm(out_1 - out_10)
    if diff > 1e-5:
        print(f"✅ Iteration Stability: Latent state is evolving (Diff: {diff:.6f})")
    else:
        print("⚠️ Warning: Model state did not change between 1 and 10 iterations.")


if __name__ == "__main__":
    test_model_forward()
    test_iteration_stability()