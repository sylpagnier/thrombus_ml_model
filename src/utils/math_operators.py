import torch


def scatter_add(src, index, dim=0, dim_size=None):
    """Standalone replacement for torch_scatter.scatter_add."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out_size = list(src.size())
    out_size[dim] = dim_size
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index = index.view(view_shape).expand_as(src)
    return out.scatter_add_(dim, index, src)


def wls_derivatives(field, edge_index, num_nodes, V, W, M_inv):
    """Compute WLS derivatives [x, y, xx, xy, yy] for nodal fields."""
    row, col = edge_index

    if M_inv.dim() == 4 and M_inv.shape[1] == 1:
        M_inv = M_inv.squeeze(1)

    u = field if field.dim() == 2 else field.unsqueeze(-1)
    if u.dim() != 2:
        raise ValueError(f"wls_derivatives expects [N] or [N,C], got {tuple(field.shape)}")
    if u.shape[0] != int(num_nodes):
        raise ValueError(f"wls_derivatives expected N={num_nodes}, got {u.shape[0]}")

    du = u[col] - u[row]
    b_e = W.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), du.unsqueeze(1))
    channels = u.shape[1]
    b_flat = scatter_add(b_e.view(-1, 5 * channels), row, dim=0, dim_size=num_nodes)
    b = b_flat.view(num_nodes, 5, channels)
    return torch.bmm(M_inv, b)


def sparse_gradient(field, G_x, G_y):
    """Compute sparse gradient components for a scalar nodal field."""
    col = field.unsqueeze(1) if field.dim() == 1 else field
    gx = torch.sparse.mm(G_x, col).squeeze(1)
    gy = torch.sparse.mm(G_y, col).squeeze(1)
    return gx, gy
