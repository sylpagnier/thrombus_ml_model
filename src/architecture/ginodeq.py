import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union
from torch import Tensor

from src.core_physics.anderson import anderson_acceleration
from src.architecture.lora_injection import LoRAParametrization, SpectralLinear
from src.core_physics.physics_kernels import scatter_add


def _spectral_or_plain_linear(in_features: int, out_features: int, bias: bool, spectral: bool) -> nn.Module:
    if spectral:
        return SpectralLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def _make_activation(name: str) -> nn.Module:
    mode = (name or "relu").strip().lower()
    if mode == "silu":
        return nn.SiLU()
    if mode == "gelu":
        return nn.GELU()
    return nn.ReLU()


class GlobalMixingBlock(nn.Module):
    def __init__(self, latent_dim, use_spectral_norm: bool = True, activation_fn: str = "relu"):
        super().__init__()
        self.global_mlp = nn.Sequential(
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
            _make_activation(activation_fn),
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
        )

    def forward(self, x, batch):
        global_context = global_mean_pool(x, batch)
        global_update = self.global_mlp(global_context)
        return global_update[batch]


class MultiHeadPhysicsGATConv(MessagePassing):
    """
    Physics-Informed Multi-Head Graph Attention Network.
    Strictly typed to satisfy IDE linters and PyG's message passing dispatcher.
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int = 3,
        temperature: float = 1.5,
        use_spectral_norm: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('node_dim', 0)
        super().__init__(**kwargs)

        self.temperature = temperature
        self.edge_proj = _spectral_or_plain_linear(edge_dim, latent_dim, True, use_spectral_norm)

        self.lin_src = _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm)
        self.lin_dst = _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm)
        self.att = _spectral_or_plain_linear(latent_dim, 1, True, use_spectral_norm)

    def forward(self,
                x: Union[Tensor, Tuple[Tensor, Tensor]],
                edge_index: Tensor,
                edge_attr: Tensor,
                mod_adv: Tensor,
                mod_rheo: Tensor,
                mod_curve: Tensor,
                size: Optional[Tuple[int, int]] = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        x_src = self.lin_src(x[0])
        x_dst = self.lin_dst(x[1])

        alpha_src = self.att(x_src)
        alpha_dst = self.att(x_dst)

        out = self.propagate(
            edge_index,
            size=size,
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_attr=edge_attr,
            mod_adv=mod_adv,
            mod_rheo=mod_rheo,
            mod_curve=mod_curve
        )
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_attr: Tensor, mod_adv: Tensor, mod_rheo: Tensor, mod_curve: Tensor,
                index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = (alpha_j + alpha_i) / self.temperature
        # Bias pre-softmax logits with flow-wall directional modulators and curvature.
        alpha = alpha + mod_adv + mod_rheo + mod_curve
        alpha = alpha * self.edge_proj(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=3, use_spectral_norm: bool = True, activation_fn: str = "relu"):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for multi-head split"

        self.conv = MultiHeadPhysicsGATConv(
            latent_dim, edge_dim=edge_dim, use_spectral_norm=use_spectral_norm
        )
        self.global_mixer = GlobalMixingBlock(
            latent_dim,
            use_spectral_norm=use_spectral_norm,
            activation_fn=activation_fn,
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.activation = _make_activation(activation_fn)

    def forward(self, z, edge_index, edge_attr, batch, mod_adv, mod_rheo, mod_curve):
        local_out = self.conv(z, edge_index, edge_attr, mod_adv, mod_rheo, mod_curve)
        global_out = self.global_mixer(z, batch)
        return self.norm(self.activation(z + local_out + global_out))


class GINO_DEQ(nn.Module):
    def __init__(
        self,
        in_channels=11,
        out_channels=5,
        latent_dim=64,
        max_iters=25,
        num_fourier_freqs=8,
        outer_iters=3,
        mu_inf_nd=0.03,
        mu_0_nd=1.0,
        kinematics_mode: str = "direct_uvp",
        activation_fn: str = "silu",
        fourier_base: float = 2.0,
    ):
        super().__init__()
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.num_fourier_freqs = num_fourier_freqs
        self.mu_inf_nd = mu_inf_nd
        self.mu_0_nd = mu_0_nd
        self.activation_fn = (activation_fn or "relu").strip().lower()
        self.fourier_base = float(fourier_base)

        self.wss_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, 1)  # Non-recurrent output projection
        )
        self.kinematics_mode = (kinematics_mode or "direct_uvp").strip().lower()
        if self.kinematics_mode not in ("stream", "direct_uvp"):
            raise ValueError(
                f"Unsupported kinematics_mode={self.kinematics_mode!r}; expected 'stream' or 'direct_uvp'."
            )

        freqs = (self.fourier_base ** torch.arange(num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 5 * num_fourier_freqs * 2
        encoded_channels = (in_channels - 5) + 5 + fourier_channels

        self.encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, latent_dim)
        )

        self.core = GINOBlock(latent_dim, edge_dim=3, activation_fn=self.activation_fn)
        # Stream-function formulation: decoder predicts (psi, p), then u,v are derived from psi.
        # Direct formulation: decoder predicts (u, v, p) directly (used to avoid WLS-on-WLS differentiation).
        self.kinematics_decoder = nn.Linear(latent_dim, 2 if self.kinematics_mode == "stream" else 3)

        self.mu_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, 1)
        )
        self.mu_encoder = nn.Linear(1, latent_dim)
        self.k_env = nn.Parameter(torch.tensor(5.0))

    def prepare_for_tier3_lora(self, rank: int = 4, alpha: float = 1.0):
        """
        Iterates through the model's architecture and dynamically injects LoRA
        into all SpectralLinear modules while rigorously maintaining Lipschitz bounds.
        """
        for module in self.modules():
            if isinstance(module, SpectralLinear):
                module.inject_lora(rank=rank, alpha=alpha)

    def _apply_fourier_encoding(self, x, pos_nd=None):
        nodes_nd = pos_nd if pos_nd is not None else x[:, 0:2]
        sdf_nd = x[:, 2:3]
        shear_pot = x[:, 3:4]
        wall_normal = x[:, 4:6]

        rest = x[:, 6:11]
        uv_prior = x[:, 11:13]
        mu_prior = x[:, 13:14]
        wss_prior = x[:, 14:15]

        features_to_encode = torch.cat([nodes_nd, sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat(
            [shear_pot, features_to_encode, fourier_feats, rest, uv_prior, mu_prior, wss_prior], dim=1)
        return encoded_x, uv_prior

    def _compute_wls_derivatives(self, field: Tensor, data) -> Tensor:
        """Compute WLS derivatives [d/dx, d/dy, d2/dx2, d2/dxdy, d2/dy2] for a nodal scalar."""
        row, col = data.edge_index
        num_nodes = data.num_nodes
        V, W, M_inv = data.V, data.W, data.M_inv

        u = field if field.dim() == 2 else field.unsqueeze(-1)
        du = u[col] - u[row]

        w = W.view(-1, 1, 1)
        v = V.unsqueeze(2)
        du_unsq = du.unsqueeze(1)
        b_e = w * torch.bmm(v, du_unsq)

        c = u.shape[1]
        b_flat = scatter_add(b_e.view(-1, 5 * c), row, dim=0, dim_size=num_nodes)
        b = b_flat.view(num_nodes, 5, c)
        return torch.bmm(M_inv, b)

    def _stream_to_velocity(
        self,
        psi_raw: Tensor,
        p: Tensor,
        data,
        sdf: Tensor,
        wall_normal: Tensor,
    ) -> Tensor:
        """
        Derive velocity from stream function using WLS spatial derivatives and product rule.
        d/dy (psi * envelope) = envelope * d(psi)/dy + psi * d(envelope)/dy
        """
        c_psi = self._compute_wls_derivatives(psi_raw, data)
        psi_x = c_psi[:, 0:1, 0]
        psi_y = c_psi[:, 1:2, 0]
        n_x = wall_normal[:, 0:1]
        n_y = wall_normal[:, 1:2]

        # Squared saturating envelope makes both envelope and its gradient vanish at the wall.
        # This enforces hard no-slip analytically for the stream-function branch.
        k_safe = F.softplus(self.k_env) + 1e-3
        base_env = 1.0 - torch.exp(-k_safe * sdf)
        envelope = base_env * base_env
        env_grad = 2.0 * base_env * (k_safe * torch.exp(-k_safe * sdf))
        u = envelope * psi_y + psi_raw * env_grad * n_y
        v = -(envelope * psi_x + psi_raw * env_grad * n_x)
        return torch.cat([u, v, p], dim=1)

    @torch.enable_grad()
    def forward(self, data, solver="anderson", anderson_beta=0.8, anderson_warmup_iters=5):
        x_encoded, _ = self._apply_fourier_encoding(data.x)
        x_enc = self.encoder(x_encoded)
        z = x_enc.clone()

        row, col = data.edge_index

        # Pull precomputed edge attributes directly instead of recalculating
        edge_attr = data.edge_attr
        edge_vec = edge_attr[:, :2]

        batch_idx = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        wall_normals = data.x[:, 4:6]
        e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
        n_dir_row = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
        n_dir_col = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir_row).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = data.x[:, 2:3]
        sdf_edge = sdf_nd[row]

        k_decay = 5.0
        decay_factor = torch.exp(-k_decay * sdf_edge)
        curve_dot = (n_dir_row * n_dir_col).sum(dim=-1, keepdim=True)
        mod_curve = torch.log(torch.clamp(1.0 - curve_dot, min=1e-4, max=1.0)) * decay_factor

        mod_rheo = torch.log(torch.clamp(dot_prod, min=1e-3, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=1e-3, max=1.0)) * decay_factor

        def f_coupled(curr_z):
            curr_z_flat = curr_z.squeeze(0) if curr_z.ndim == 3 else curr_z
            mu_raw = self.mu_decoder(curr_z_flat)
            mu = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw)
            mu_enc = self.mu_encoder(mu)
            z_in = curr_z_flat + x_enc + mu_enc
            out = self.core(z_in, data.edge_index, edge_attr, batch_idx, mod_adv, mod_rheo, mod_curve)
            return out.unsqueeze(0) if curr_z.ndim == 3 else out

        z_init = z.unsqueeze(0) if z.ndim == 2 else z

        # Solve for equilibrium without unrolling autograd through all fixed-point steps.
        with torch.no_grad():
            if solver == "picard":
                z_star = z_init
                for _ in range(self.max_iters):
                    z_star = f_coupled(z_star)
            else:
                # Pass the warmup iters down to the Anderson solver.
                z_star = anderson_acceleration(
                    f_coupled, z_init, batch_idx=batch_idx,
                    max_iter=self.max_iters, beta=anderson_beta, warmup_iters=anderson_warmup_iters
                )

        # Re-attach once so kinematics keep a differentiable path to coordinates.
        z_star_req = z_star.detach().requires_grad_(self.training)
        z_out = f_coupled(z_star_req)
        if self.training:
            eps = torch.randn_like(z_out)
            vjp = torch.autograd.grad(z_out, z_star_req, grad_outputs=eps, create_graph=True)[0]
            jac_loss = torch.mean(vjp ** 2)
            z = z_out.squeeze(0)
        else:
            z = z_out.squeeze(0)
            jac_loss = torch.tensor(0.0, device=z.device)

        mu_raw = self.mu_decoder(z)
        mu = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw)

        kinematics_out = self.kinematics_decoder(z)
        if self.kinematics_mode == "direct_uvp":
            u_v_p = kinematics_out[:, 0:3]
        else:
            psi_raw = kinematics_out[:, 0:1]
            p = kinematics_out[:, 1:2]
            sdf = data.sdf_wall if hasattr(data, "sdf_wall") else data.x[:, 2:3]
            wall_normal = data.x[:, 4:6]
            u_v_p = self._stream_to_velocity(
                psi_raw,
                p,
                data,
                sdf,
                wall_normal,
            )
        wss_pred = self.wss_decoder(z)
        pred = torch.cat([u_v_p, mu, wss_pred], dim=1)

        return (pred, jac_loss) if self.training else pred