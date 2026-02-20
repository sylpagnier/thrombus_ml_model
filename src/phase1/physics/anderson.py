import torch


def anderson_acceleration(f, z0, m=5, lam=1e-4, max_iter=50, tol=1e-3, beta=1.0):
    """
    Robust Anderson Acceleration for Deep Equilibrium Models.
    Minimizes the residual norm over a history of size m.
    """
    # Ensure batch dimension exists (handling PyG's [Nodes, Features] shape)
    if z0.ndim == 2:
        z0 = z0.unsqueeze(0)

    bsz, n, d = z0.shape

    # History buffers
    # X: Input values (arguments)
    # F: Output values (f(x))
    X = torch.zeros(bsz, m, n * d, dtype=z0.dtype, device=z0.device)
    F = torch.zeros(bsz, m, n * d, dtype=z0.dtype, device=z0.device)

    z0_flat = z0.view(bsz, -1)
    X[:, 0] = z0_flat
    F[:, 0] = f(z0).view(bsz, -1)

    # Initial step (Standard Picard)
    X[:, 1] = F[:, 0]
    F[:, 1] = f(F[:, 0].view(bsz, n, d)).view(bsz, -1)

    res = []

    # Pre-allocate solver tensors outside the loop to save GPU memory
    I_max = torch.eye(m, dtype=z0.dtype, device=z0.device).unsqueeze(0).expand(bsz, -1, -1)
    Y_max = torch.ones(bsz, m, 1, dtype=z0.dtype, device=z0.device)

    slot = 1  # Fallback if loop doesn't execute
    for k in range(2, max_iter):
        n_history = min(k, m)

        # Current residual (F - X)
        G = F[:, :n_history] - X[:, :n_history]

        # 1. Form the Least Squares Problem
        G_flat = G.view(bsz, n_history, -1)
        H = torch.bmm(G_flat, G_flat.transpose(1, 2))

        # Slice the pre-allocated identity matrix
        H = H + lam * I_max[:, :n_history, :n_history]

        # Slice the pre-allocated ones vector
        y_slice = Y_max[:, :n_history, :]
        alpha = torch.linalg.lstsq(H, y_slice).solution

        # Normalize alpha so sum(alpha) = 1
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        # 2. Compute Next Step
        alpha = alpha.view(bsz, n_history, 1)  # [bsz, m, 1]

        # Weighted sum of History terms
        combined_X = (alpha * X[:, :n_history]).sum(dim=1)
        combined_F = (alpha * F[:, :n_history]).sum(dim=1)

        z_next = beta * combined_F + (1 - beta) * combined_X

        # 3. Check Convergence via Relative Residual
        diff_norm = (combined_F - combined_X).norm(p=2, dim=-1)
        x_norm = combined_X.norm(p=2, dim=-1)

        # Relative residual handles varying activation scales across the latent feature space
        current_res = (diff_norm / (x_norm + 1e-8)).mean()
        res.append(current_res.item())

        if current_res < tol:
            return z_next.view(bsz, n, d).squeeze(0)  # Remove dummy batch

        # 4. Update History
        # Rolling buffer update
        slot = k % m
        new_f = f(z_next.view(bsz, n, d)).view(bsz, -1)

        # Reconstruct the buffers using concatenation to avoid in-place mutation
        X = torch.cat([X[:, :slot], z_next.unsqueeze(1), X[:, slot + 1:]], dim=1)
        F = torch.cat([F[:, :slot], new_f.unsqueeze(1), F[:, slot + 1:]], dim=1)

    return F[:, slot].view(bsz, n, d).squeeze(0)