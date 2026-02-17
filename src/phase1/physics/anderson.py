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

    for k in range(2, max_iter):
        n_history = min(k, m)

        # Current residual (F - X)
        G = F[:, :n_history] - X[:, :n_history]

        # 1. Form the Least Squares Problem
        # We want to find alphas that minimize || G * alpha ||
        # This is equivalent to solving (G^T G) * alpha = 0 subject to sum(alpha)=1

        # Efficient batch-wise construction of G^T G
        # G_flat: [bsz, n_history, n*d]
        G_flat = G.view(bsz, n_history, -1)

        # H = G^T G (Gram Matrix) -> [bsz, n_history, n_history]
        H = torch.bmm(G_flat, G_flat.transpose(1, 2))

        # Ridge Regression (Regularization) for stability
        H = H + lam * torch.eye(n_history, dtype=z0.dtype, device=z0.device).unsqueeze(0)

        # Solve H * alpha = y (where y is all ones, theoretically, but we use a constrained solver trick)
        # Trick: Minimize || \sum gamma_i * delta_g_i - g_k ||
        # Here is the standard DEQ efficient solver:

        # Solve linear system H * x = 1 (to enforce sum(alpha)=1 later)
        try:
            # y is vector of ones [bsz, n_history, 1]
            y = torch.ones(bsz, n_history, 1, dtype=z0.dtype, device=z0.device)
            alpha = torch.linalg.solve(H, y)

            # Normalize alpha so sum(alpha) = 1
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        except RuntimeError:
            # Fallback if matrix is singular: Average the history
            alpha = torch.ones(bsz, n_history, 1, dtype=z0.dtype, device=z0.device) / n_history

        # 2. Compute Next Step
        # z_{k+1} = \sum alpha_i * (beta * F_i + (1-beta) * X_i)
        # Usually beta=1.0 for pure Anderson

        alpha = alpha.view(bsz, n_history, 1)  # [bsz, m, 1]

        # Weighted sum of History terms
        combined_X = (alpha * X[:, :n_history]).sum(dim=1)
        combined_F = (alpha * F[:, :n_history]).sum(dim=1)

        z_next = beta * combined_F + (1 - beta) * combined_X

        # 3. Check Convergence
        current_res = (combined_F - combined_X).norm(p=2, dim=-1).mean()
        res.append(current_res.item())
        if current_res < tol:
            return z_next.view(bsz, n, d).squeeze(0)

        # 4. Update History
        # Rolling buffer update
        slot = k % m
        new_f = f(z_next.view(bsz, n, d)).view(bsz, -1)

        # Reconstruct the buffers using concatenation to avoid in-place mutation
        X = torch.cat([X[:, :slot], z_next.unsqueeze(1), X[:, slot + 1:]], dim=1)
        F = torch.cat([F[:, :slot], new_f.unsqueeze(1), F[:, slot + 1:]], dim=1)

    return F[:, slot].view(bsz, n, d).squeeze(0)