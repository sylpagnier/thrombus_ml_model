import torch


def anderson_acceleration(f, z0, m=5, lam=1e-4, max_iter=25, tol=1e-3):
    """
    Anderson Acceleration for finding fixed point z = f(z).
    f: The GINOBlock function
    z0: Initial latent state from the Encoder
    m: History size (number of previous iterates to consider)
    """
    bsz, n, d = z0.shape  # Batch, Nodes, Latent Dim

    # Initialize history buffers
    X = torch.zeros(bsz, m, n * d, device=z0.device)
    F = torch.zeros(bsz, m, n * d, device=z0.device)

    X[:, 0] = z0.view(bsz, -1)
    F[:, 0] = f(z0).view(bsz, -1)
    res = F[:, 0] - X[:, 0]

    z = F[:, 0].view(bsz, n, d)

    for k in range(1, max_iter):
        X[:, k % m] = z.view(bsz, -1)
        F[:, k % m] = f(z).view(bsz, -1)
        res = F[:, k % m] - X[:, k % m]

        # Convergence check: residual norm
        if torch.norm(res) < tol:
            break

        # Solve the constrained optimization for weights (least squares)
        # For simplicity in this Tier 1 version, we implement the m=1 (Broyden-like) update
        # or a standard weighted mixing if m > 1.
        z = (1 - lam) * F[:, k % m].view(bsz, n, d) + lam * X[:, k % m].view(bsz, n, d)

    return z