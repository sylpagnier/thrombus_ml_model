import torch


def anderson_acceleration(f, z0, batch_idx=None, m=5, lam=1e-4, max_iter=50, tol=1e-3, beta=1.0, return_history=False):
    """
    Robust Anderson Acceleration for Deep Equilibrium Models.
    Minimizes the residual norm over a history of size m.
    """
    if z0.ndim == 2:
        z0 = z0.unsqueeze(0)

    bsz, n, d = z0.shape

    # Use lists instead of pre-allocated tensors to avoid inplace modifications
    X_history = []
    F_history = []

    z0_flat = z0.view(bsz, -1)
    X_history.append(z0_flat)

    f0_flat = f(z0).view(bsz, -1)
    F_history.append(f0_flat)

    X_history.append(f0_flat)
    F_history.append(f(f0_flat.view(bsz, n, d)).view(bsz, -1))

    res = []

    for k in range(2, max_iter):
        n_history = len(X_history)

        # Out-of-place stack creates a new tensor for the computational graph
        X = torch.stack(X_history, dim=1)
        F = torch.stack(F_history, dim=1)

        G = F - X
        G_flat = G.view(bsz, n_history, -1)

        H = torch.bmm(G_flat, G_flat.transpose(1, 2))
        I_max = torch.eye(n_history, dtype=z0.dtype, device=z0.device).unsqueeze(0).expand(bsz, -1, -1)
        H = H + lam * I_max

        y_slice = torch.ones(bsz, n_history, 1, dtype=z0.dtype, device=z0.device)
        alpha = torch.linalg.lstsq(H, y_slice).solution
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        alpha = alpha.view(bsz, n_history, 1)

        combined_X = (alpha * X).sum(dim=1)
        combined_F = (alpha * F).sum(dim=1)

        z_next = beta * combined_F + (1 - beta) * combined_X

        # --- Subgraph Residual Tracking ---
        combined_X_node = combined_X.view(bsz, n, d)
        combined_F_node = combined_F.view(bsz, n, d)

        diff_norm = (combined_F_node - combined_X_node).norm(p=2, dim=-1)
        x_norm = combined_X_node.norm(p=2, dim=-1)
        node_res = diff_norm / (x_norm + 1e-8)

        if batch_idx is not None and bsz == 1:
            node_res_flat = node_res.squeeze(0)
            num_graphs = batch_idx.max().item() + 1
            graph_res_sum = torch.zeros(num_graphs, dtype=node_res.dtype, device=node_res.device)
            graph_res_sum.scatter_add_(0, batch_idx, node_res_flat)

            graph_node_counts = torch.bincount(batch_idx).to(node_res.dtype)
            graph_res = graph_res_sum / graph_node_counts
            current_res = graph_res.max()
        else:
            current_res = node_res.mean()

        res.append(current_res.item())

        if current_res < tol:
            out = z_next.view(bsz, n, d).squeeze(0)
            return (out, res) if return_history else out

        # Evaluate the next step
        new_f = f(z_next.view(bsz, n, d)).view(bsz, -1)

        # Update history out-of-place by appending
        X_history.append(z_next.view(bsz, -1))
        F_history.append(new_f)

        # Enforce history length 'm' by popping the oldest entry
        if len(X_history) > m:
            X_history.pop(0)
            F_history.pop(0)

    # If max_iter is reached, return the latest F evaluation
    out = F_history[-1].view(bsz, n, d).squeeze(0)
    return (out, res) if return_history else out