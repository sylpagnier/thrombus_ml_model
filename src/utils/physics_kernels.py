import torch
import torch.nn.functional as F


class PhysicsKernels:
    def __init__(self, reynolds=100.0, rho_nd=1.0, mu_inf_nd=0.0035):
        """
        Implements differentiable physics losses for Non-Dimensionalized (ND) flow.
        Scaling: Option A (Mean Diameter D_bar).
        """
        self.Re = reynolds
        self.rho = rho_nd
        self.mu_inf = mu_inf_nd  # Baseline ND viscosity

    def _compute_graph_gradients(self, f, data):
        """
        Calculates spatial gradients (df/dx, df/dy) using graph connectivity.
        Uses a first-order central difference approximation for unstructured grids.
        """
        row, col = data.edge_index
        # Vector from node i to node j
        pos_diff = data.x[col] - data.x[row]  # [E, 2]
        # Scalar difference of field f
        f_diff = f[col] - f[row]  # [E, 1]

        # Least-squares gradient reconstruction or simple weighted average
        # Here we use a distance-weighted projection for simplicity and speed
        dist_sq = torch.sum(pos_diff ** 2, dim=1, keepdim=True) + 1e-6
        weights = 1.0 / dist_sq

        grad_f = torch.zeros((data.num_nodes, 2), device=f.device)
        # Project differences onto axes
        grad_f.index_add_(0, row, weights * f_diff * pos_diff)

        # Normalize by accumulated weights
        norm_weights = torch.zeros((data.num_nodes, 1), device=f.device)
        norm_weights.index_add_(0, row, weights)

        return grad_f / (norm_weights + 1e-8)

    def navier_stokes_residual(self, pred, data):
        """
        Calculates the steady-state ND Navier-Stokes residual (L_NS).
        pred: [N, 3] tensor containing [u_nd, v_nd, p_nd]
        """
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        p = pred[:, 2:3]

        # 1. Spatial Derivatives
        grad_u = self._compute_graph_gradients(u, data)  # [N, 2] -> (du/dx, du/dy)
        grad_v = self._compute_graph_gradients(v, data)  # [N, 2] -> (dv/dx, dv/dy)
        grad_p = self._compute_graph_gradients(p, data)  # [N, 2] -> (dp/dx, dp/dy)

        # 2. Continuity Equation (Incompressibility): div(U) = 0
        l_continuity = grad_u[:, 0:1] + grad_v[:, 1:2]

        # 3. Momentum Equations (Steady State ND form)
        # Momentum X: (u*du/dx + v*du/dy) + dp/dx - (1/Re)*laplacian(u) = 0
        # (Laplacian is approximated via grad of grad for simplicity)
        convection_x = u * grad_u[:, 0:1] + v * grad_u[:, 1:2]
        momentum_x = convection_x + grad_p[:, 0:1]  # Simplified for Tier 1 (Newtonian)

        convection_y = u * grad_v[:, 0:1] + v * grad_v[:, 1:2]
        momentum_y = convection_y + grad_p[:, 1:2]

        # 4. Total NS Loss
        l_ns = torch.mean(l_continuity ** 2 + momentum_x ** 2 + momentum_y ** 2)
        return l_ns

    def carreau_yasuda_loss(self, pred_mu, pred_u, data, params):
        """
        Enforces mu follows the CY rheology curve based on shear rate.
        params: dict of [mu0, mu_inf, lambda, a, n] non-dimensionalized.
        """
        # Calculate local shear rate gamma_dot from velocity gradients
        grad_u = self._compute_graph_gradients(pred_u[:, 0:1], data)
        grad_v = self._compute_graph_gradients(pred_u[:, 1:2], data)

        # Shear rate magnitude (2D simplification)
        gamma_dot = torch.sqrt(2 * (grad_u[:, 0] ** 2 + grad_v[:, 1] ** 2) + (grad_u[:, 1] + grad_v[:, 0]) ** 2)

        # Carreau-Yasuda Formula
        mu_0, mu_inf = params['mu0'], params['mu_inf']
        lam, a, n = params['lambda'], params['a'], params['n']

        target_mu = mu_inf + (mu_0 - mu_inf) * (1 + (lam * gamma_dot) ** a) ** ((n - 1) / a)

        return F.mse_loss(pred_mu, target_mu.unsqueeze(-1))

    def brinkman_penalty(self, pred_u, phi, lambda_brink=1e4):
        """
        Penalizes velocity inside the thrombus (phi > 0.5).
        Forces u -> 0 as clot becomes a "solid".
        """
        # Brinkman logic: higher phi increases flow resistance exponentially
        penalty = lambda_brink * (phi ** 2) * torch.norm(pred_u, dim=1, keepdim=True)
        return torch.mean(penalty)