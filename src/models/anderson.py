import torch

def anderson_acceleration(f, z0, m=5, lam=0.1, max_iter=25, tol=1e-3):
    """
    Anderson Acceleration for finding fixed point z = f(z).
    Stability Patch: Increased lam to 0.1 to stabilize high-Re transitions.
    """
    if len(z0.shape) == 2:  # Add batch dim if missing
        z0 = z0.unsqueeze(0)

    bsz, n, d = z0.shape
    X = torch.zeros(bsz, m, n * d, device=z0.device)
    F = torch.zeros(bsz, m, n * d, device=z0.device)

    X[:, 0] = z0.reshape(bsz, -1)
    F[:, 0] = f(z0).reshape(bsz, -1)

    z = F[:, 0].reshape(bsz, n, d)

    for k in range(1, max_iter):
        X[:, k % m] = z.reshape(bsz, -1)
        F[:, k % m] = f(z).reshape(bsz, -1)
        res = F[:, k % m] - X[:, k % m]

        if torch.norm(res) < tol:
            break

        # Mixing for Tier 1 stability
        z = (1 - lam) * F[:, k % m].reshape(bsz, n, d) + lam * X[:, k % m].reshape(bsz, n, d)

    return z.squeeze(0)