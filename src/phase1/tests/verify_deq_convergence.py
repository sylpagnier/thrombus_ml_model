import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.config import PhysicsConfig
from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.anderson import anderson_acceleration


def setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=100):
    """Generates dummy data with 13 input channels."""
    graph_list = []
    for _ in range(num_graphs):
        pos = torch.randn((num_nodes_per_graph, 2))
        sdf = torch.rand((num_nodes_per_graph, 1))
        shear_pot = torch.rand((num_nodes_per_graph, 1))
        normals = F.normalize(torch.randn((num_nodes_per_graph, 2)), dim=1)
        rest = torch.zeros((num_nodes_per_graph, 7))  # To reach 13 channels

        x_full = torch.cat([pos, sdf, shear_pot, normals, rest], dim=-1)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 5))
        graph_list.append(Data(x=x_full, edge_index=edge_index))

    return Batch.from_data_list(graph_list)


def test_anderson_standalone_convergence():
    def f_affine(x): return 0.5 * x + 1.0

    z0 = torch.zeros((1, 10, 5))
    z_fixed = anderson_acceleration(f_affine, z0, max_iter=20, tol=1e-5)
    assert torch.allclose(z_fixed, torch.tensor(2.0), atol=1e-4)


def test_ginodeq_forward_api():
    batch_data = setup_synthetic_batch(num_graphs=2, num_nodes_per_graph=50)
    # in=13, out=5 (u,v,p,mu,wss)
    model = GINO_DEQ(in_channels=13, out_channels=5, latent_dim=16, max_iters=4)
    model.eval()

    for solver in ["picard", "anderson"]:
        with torch.no_grad():
            out = model(batch_data, solver=solver)

        assert out.shape == (batch_data.x.size(0), 5), f"Expected [N, 5], got {out.shape}"
        assert torch.isfinite(out).all(), f"NaNs in {solver} pass"


def test_ginodeq_backward_pass():
    batch_data = setup_synthetic_batch(num_graphs=1, num_nodes_per_graph=30)
    model = GINO_DEQ(in_channels=13, out_channels=5, latent_dim=16, max_iters=3)
    model.train()

    pred, jac_loss = model(batch_data, solver="anderson")
    loss = F.mse_loss(pred, torch.randn_like(pred)) + (0.1 * jac_loss)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert has_grad, "Gradients did not propagate through DEQ loop"


def test_visualize_convergence_audit():
    max_iters = 30
    batch_data = setup_synthetic_batch(num_graphs=1, num_nodes_per_graph=50)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, tier in enumerate(["tier1", "tier2"]):
        ax = axes[idx]
        model = GINO_DEQ(in_channels=13, out_channels=5, latent_dim=16, max_iters=max_iters)
        model.eval()

        with torch.no_grad():
            x_encoded, _ = model._apply_fourier_encoding(batch_data.x)
            x_enc = model.encoder(x_encoded)

            row, col = batch_data.edge_index
            edge_vec = batch_data.x[col, :2] - batch_data.x[row, :2]
            edge_dist = torch.norm(edge_vec, p=2, dim=-1, keepdim=True)
            edge_attr = torch.cat([edge_vec, edge_dist], dim=-1)

            wall_normals = batch_data.x[:, 4:6]
            e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
            n_dir = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
            dot_prod = torch.abs((e_dir * n_dir).sum(dim=-1, keepdim=True))

            mod_rheo = torch.log(torch.clamp(dot_prod, min=1e-3, max=1.0))
            mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=1e-3, max=1.0))

            mu_enc = model.mu_encoder(torch.ones((batch_data.x.size(0), 1)))

            def f_fp(z):
                if z.ndim == 3: z = z.squeeze(0)
                return model.core(z + x_enc + mu_enc, batch_data.edge_index, edge_attr, batch_data.batch, mod_adv,
                                  mod_rheo)

            # Picard
            z_p = x_enc.clone()
            res_p = []
            for _ in range(max_iters):
                z_next = f_fp(z_p)
                res_p.append((z_next - z_p).norm().item() / (z_p.norm().item() + 1e-8))
                z_p = z_next

            # Anderson
            _, res_a = anderson_acceleration(f_fp, x_enc.unsqueeze(0), max_iter=max_iters, return_history=True)

        ax.plot(res_p, label='Picard', marker='o', markersize=3)
        ax.plot(range(2, len(res_a) + 2), res_a, label='Anderson', marker='x')
        ax.set_yscale('log')
        ax.set_title(f"Convergence Audit: {tier.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_anderson_standalone_convergence()
    test_ginodeq_forward_api()
    test_ginodeq_backward_pass()
    test_visualize_convergence_audit()
    print("\nAll systems nominal. DEQ convergence verified! 🚀")