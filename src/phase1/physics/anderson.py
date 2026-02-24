import torch


def anderson_acceleration(f, z0, batch_idx=None, m=5, lam=1e-4, max_iter=50, tol=1e-3, beta=1.0, return_history=False):
    """
    Robust Anderson Acceleration for Deep Equilibrium Models.
    Minimizes the residual norm over a history of size m.
    Supports PyG batching by ensuring all subgraphs meet the tolerance.
    """
    if z0.ndim == 2:
        z0 = z0.unsqueeze(0)

    bsz, n, d = z0.shape
    X = torch.zeros(bsz, m, n * d, dtype=z0.dtype, device=z0.device)
    F = torch.zeros(bsz, m, n * d, dtype=z0.dtype, device=z0.device)

    z0_flat = z0.view(bsz, -1)
    X[:, 0] = z0_flat
    F[:, 0] = f(z0).view(bsz, -1)

    X[:, 1] = F[:, 0]
    F[:, 1] = f(F[:, 0].view(bsz, n, d)).view(bsz, -1)

    res = []

    I_max = torch.eye(m, dtype=z0.dtype, device=z0.device).unsqueeze(0).expand(bsz, -1, -1)
    Y_max = torch.ones(bsz, m, 1, dtype=z0.dtype, device=z0.device)

    slot = 1
    for k in range(2, max_iter):
        n_history = min(k, m)
        G = F[:, :n_history] - X[:, :n_history]

        G_flat = G.view(bsz, n_history, -1)
        H = torch.bmm(G_flat, G_flat.transpose(1, 2))
        H = H + lam * I_max[:, :n_history, :n_history]

        y_slice = Y_max[:, :n_history, :]
        alpha = torch.linalg.lstsq(H, y_slice).solution
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        alpha = alpha.view(bsz, n_history, 1)

        combined_X = (alpha * X[:, :n_history]).sum(dim=1)
        combined_F = (alpha * F[:, :n_history]).sum(dim=1)

        z_next = beta * combined_F + (1 - beta) * combined_X

        combined_X_node = combined_X.view(bsz, n, d)
        combined_F_node = combined_F.view(bsz, n, d)

        diff_norm = (combined_F_node - combined_X_node).norm(p=2, dim=-1)  # Shape: [bsz, n]
        x_norm = combined_X_node.norm(p=2, dim=-1)  # Shape: [bsz, n]

        # Node-level residuals
        node_res = diff_norm / (x_norm + 1e-8)  # Shape: [bsz, n]

        # PyG Batched Evaluation
        if batch_idx is not None and bsz == 1:
            node_res_flat = node_res.squeeze(0)

            # Calculate the mean residual for each distinct graph in the batch
            num_graphs = batch_idx.max().item() + 1
            graph_res_sum = torch.zeros(num_graphs, dtype=node_res.dtype, device=node_res.device)
            graph_res_sum.scatter_add_(0, batch_idx, node_res_flat)

            graph_node_counts = torch.bincount(batch_idx).to(node_res.dtype)
            graph_res = graph_res_sum / graph_node_counts

            # The batch only converges if the worst-performing graph meets the tolerance
            current_res = graph_res.max()
        else:
            current_res = node_res.mean()

        res.append(current_res.item())

        if current_res < tol:
            out = z_next.view(bsz, n, d).squeeze(0)
            return (out, res) if return_history else out

        slot = k % m
        new_f = f(z_next.view(bsz, n, d)).view(bsz, -1)

        X[:, slot] = z_next.view(bsz, -1)
        F[:, slot] = new_f.view(bsz, -1)

    out = F[:, slot].view(bsz, n, d).squeeze(0)
    return (out, res) if return_history else out