import torch
import torch.nn.functional as F

from src.utils.math_operators import wls_derivatives


def stream_to_velocity(
    psi_raw: torch.Tensor,
    p: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    V: torch.Tensor,
    W: torch.Tensor,
    M_inv: torch.Tensor,
    sdf: torch.Tensor,
    wall_normal: torch.Tensor,
    k_env: torch.Tensor = None,
):
    """
    Convert stream function to velocity with a shared envelope:
    envelope = (1 - exp(-k*sdf))^2, or sdf^2 when k_env is not provided.
    """
    c_psi = wls_derivatives(psi_raw, edge_index, num_nodes, V, W, M_inv)
    psi_x = c_psi[:, 0:1, 0]
    psi_y = c_psi[:, 1:2, 0]
    n_x = wall_normal[:, 0:1]
    n_y = wall_normal[:, 1:2]

    if k_env is None:
        envelope = sdf ** 2
        env_grad = 2.0 * sdf
    else:
        k_safe = F.softplus(k_env) + 1e-3
        base_env = 1.0 - torch.exp(-k_safe * sdf)
        envelope = base_env * base_env
        env_grad = 2.0 * base_env * (k_safe * torch.exp(-k_safe * sdf))

    u = envelope * psi_y + psi_raw * env_grad * n_y
    v = -(envelope * psi_x + psi_raw * env_grad * n_x)
    return torch.cat([u, v, p], dim=1)
