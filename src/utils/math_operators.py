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


def _effective_upwind_weights(row, V, W, boundary_mask, boundary_normals):
    """
    Build one-sided/upwind WLS edge weights for boundary rows.
    For boundary node i with outward normal n_i, keep edges (i->j) with (x_j-x_i)·n_i <= 0.
    """
    w_eff = W.clone()
    if boundary_mask is None or boundary_normals is None:
        return w_eff
    if boundary_mask.numel() == 0 or (not boundary_mask.any()):
        return w_eff

    bmask = boundary_mask.view(-1).bool()
    normals = boundary_normals
    if normals.dim() != 2 or normals.shape[1] != 2:
        return w_eff

    edge_on_boundary = bmask[row]
    if not edge_on_boundary.any():
        return w_eff

    n_row = normals[row]
    n_norm = torch.linalg.norm(n_row, dim=1)
    valid_normal = n_norm > 1e-12
    dot = V[:, 0] * n_row[:, 0] + V[:, 1] * n_row[:, 1]
    # Keep tangential and interior-pointing displacements, suppress outward-pointing edges.
    keep = (dot <= 1e-12) | (~edge_on_boundary) | (~valid_normal)
    w_eff = torch.where(keep, w_eff, torch.zeros_like(w_eff))
    return w_eff


def _recompute_boundary_minv(row, V, w_eff, M_inv, boundary_mask):
    """Rebuild local normal equations for boundary rows after one-sided edge filtering."""
    if boundary_mask is None or boundary_mask.numel() == 0 or (not boundary_mask.any()):
        return M_inv

    M_inv_eff = M_inv.clone()
    v_col = V.unsqueeze(2)
    v_row = V.unsqueeze(1)
    M_e = w_eff.view(-1, 1, 1) * torch.bmm(v_col, v_row)
    M_flat = scatter_add(M_e.view(-1, 25), row, dim=0, dim_size=M_inv.shape[0]).view(-1, 5, 5)

    bmask = boundary_mask.view(-1).bool()
    if bmask.shape[0] != M_flat.shape[0]:
        return M_inv_eff

    eps = 1e-6
    I = torch.eye(5, dtype=M_flat.dtype, device=M_flat.device).unsqueeze(0)
    M_sel = M_flat[bmask] + eps * I.expand(int(bmask.sum().item()), 5, 5)
    M_inv_eff[bmask] = torch.linalg.pinv(M_sel, rcond=1e-5)
    return M_inv_eff


def wls_derivatives(
    field,
    edge_index,
    num_nodes,
    V,
    W,
    M_inv,
    boundary_mask=None,
    boundary_normals=None,
):
    """Compute WLS derivatives [x, y, xx, xy, yy] for nodal fields."""
    row, col = edge_index

    if M_inv.dim() == 4 and M_inv.shape[1] == 1:
        M_inv = M_inv.squeeze(1)

    u = field if field.dim() == 2 else field.unsqueeze(-1)
    if u.dim() != 2:
        raise ValueError(f"wls_derivatives expects [N] or [N,C], got {tuple(field.shape)}")
    if u.shape[0] != int(num_nodes):
        raise ValueError(f"wls_derivatives expected N={num_nodes}, got {u.shape[0]}")

    w_eff = _effective_upwind_weights(row, V, W, boundary_mask, boundary_normals)
    M_inv_eff = _recompute_boundary_minv(row, V, w_eff, M_inv, boundary_mask)

    du = u[col] - u[row]
    b_e = w_eff.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), du.unsqueeze(1))
    channels = u.shape[1]
    b_flat = scatter_add(b_e.view(-1, 5 * channels), row, dim=0, dim_size=num_nodes)
    b = b_flat.view(num_nodes, 5, channels)
    return torch.bmm(M_inv_eff, b)


def sparse_gradient(field, G_x, G_y):
    """Compute sparse gradient components for a scalar nodal field."""
    col = field.unsqueeze(1) if field.dim() == 1 else field
    gx = torch.sparse.mm(G_x, col).squeeze(1)
    gy = torch.sparse.mm(G_y, col).squeeze(1)
    return gx, gy
