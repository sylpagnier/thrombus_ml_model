import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union
from torch import Tensor

from src.phase1.physics.anderson import anderson_acceleration


class LoRAParametrization(nn.Module):
    """
    Sub-module that computes the Low-Rank Adaptation (LoRA) additive weight.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / rank

    def forward(self, original_weight):
        # The returned tensor is the mathematical sum of the frozen base and the active LoRA matrices
        return original_weight + (self.lora_B @ self.lora_A) * self.scaling


class SpectralLinear(nn.Module):
    """
    A Linear layer pre-configured for strictly bounded Lipschitz Operator Splitting.
    Allows for dynamic, computationally-safe LoRA injection during Tier 3 adaptation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Apply spectral norm to the base layer immediately for Tier 1 & 2
        spectral_norm(self.linear)

    def forward(self, x):
        return self.linear(x)

    def inject_lora(self, rank: int = 4, alpha: float = 1.0):
        """
        Safely injects LoRA into the parameterization chain BEFORE the spectral norm.
        This guarantees that the SVD power iteration computes the spectral radius
        of the COMBINED (frozen + LoRA) weight matrix, maintaining the Lipschitz bound.
        """
        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # 1. Temporarily remove spectral norm to access the base parameters
        torch.nn.utils.remove_spectral_norm(self.linear)

        # 2. Register LoRA parameterization first
        parametrize.register_parametrization(
            self.linear, "weight",
            LoRAParametrization(in_features, out_features, rank, alpha)
        )

        # 3. Re-apply spectral norm so it wraps the LoRA-augmented weight sum
        spectral_norm(self.linear)

        # 4. Freeze base weights, leaving only LoRA parameters (A and B) trainable
        self.linear.parametrizations.weight.original.requires_grad = False


def _spectral_or_plain_linear(in_features: int, out_features: int, bias: bool, spectral: bool) -> nn.Module:
    if spectral:
        return SpectralLinear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


class GlobalMixingBlock(nn.Module):
    def __init__(self, latent_dim, use_spectral_norm: bool = True):
        super().__init__()
        self.global_mlp = nn.Sequential(
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
            nn.ReLU(),
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
            mod_rheo=mod_rheo
        )
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_attr: Tensor, mod_adv: Tensor, mod_rheo: Tensor,
                index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = (alpha_j + alpha_i) / self.temperature
        alpha = alpha * self.edge_proj(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha


class GINOBlock(nn.Module):
    def __init__(self, latent_dim=64, edge_dim=3, use_spectral_norm: bool = True):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for multi-head split"

        self.conv = MultiHeadPhysicsGATConv(
            latent_dim, edge_dim=edge_dim, use_spectral_norm=use_spectral_norm
        )
        self.global_mixer = GlobalMixingBlock(latent_dim, use_spectral_norm=use_spectral_norm)
        self.norm = nn.LayerNorm(latent_dim)
        self.relu = nn.ReLU()

    def forward(self, z, edge_index, edge_attr, batch, mod_adv, mod_rheo):
        local_out = self.conv(z, edge_index, edge_attr, mod_adv, mod_rheo)
        global_out = self.global_mixer(z, batch)
        return self.norm(self.relu(z + local_out + global_out))


class GINO_DEQ(nn.Module):
    def __init__(self, in_channels=11, out_channels=5, latent_dim=64, max_iters=25, num_fourier_freqs=8, outer_iters=3,
                 mu_inf_nd=0.03, mu_0_nd=1.0):
        super().__init__()
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.num_fourier_freqs = num_fourier_freqs
        self.mu_inf_nd = mu_inf_nd
        self.mu_0_nd = mu_0_nd

        self.wss_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)  # Non-recurrent output projection
        )

        freqs = (2.0 ** torch.arange(num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 3 * num_fourier_freqs * 2
        encoded_channels = (in_channels - 3) + 3 + fourier_channels

        self.encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.core = GINOBlock(latent_dim, edge_dim=3)
        self.kinematics_decoder = nn.Linear(latent_dim, 3)

        self.mu_decoder = nn.Sequential(
            SpectralLinear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        self.mu_encoder = nn.Linear(1, latent_dim)

    def prepare_for_tier3_lora(self, rank: int = 4, alpha: float = 1.0):
        """
        Iterates through the model's architecture and dynamically injects LoRA
        into all SpectralLinear modules while rigorously maintaining Lipschitz bounds.
        """
        for module in self.modules():
            if isinstance(module, SpectralLinear):
                module.inject_lora(rank=rank, alpha=alpha)

    def _apply_fourier_encoding(self, x):
        nodes_nd = x[:, 0:2]
        sdf_nd = x[:, 2:3]
        shear_pot = x[:, 3:4]
        wall_normal = x[:, 4:6]

        rest = x[:, 6:11]
        uv_prior = x[:, 11:13]
        mu_prior = x[:, 13:14]
        wss_prior = x[:, 14:15]

        features_to_encode = torch.cat([sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat(
            [nodes_nd, shear_pot, features_to_encode, fourier_feats, rest, uv_prior, mu_prior, wss_prior], dim=1)
        return encoded_x, uv_prior

    def forward(self, data, solver="anderson", anderson_beta=0.8, anderson_warmup_iters=5):
        x_encoded, uv_prior = self._apply_fourier_encoding(data.x)
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
        n_dir = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = data.x[:, 2:3]
        sdf_edge = sdf_nd[row]

        k_decay = 5.0
        decay_factor = torch.exp(-k_decay * sdf_edge)

        mod_rheo = torch.log(torch.clamp(dot_prod, min=1e-3, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=1e-3, max=1.0)) * decay_factor

        def f_coupled(curr_z):
            curr_z_flat = curr_z.squeeze(0) if curr_z.ndim == 3 else curr_z
            mu_raw = self.mu_decoder(curr_z_flat)
            mu = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw)
            mu_enc = self.mu_encoder(mu)
            z_in = curr_z_flat + x_enc + mu_enc
            out = self.core(z_in, data.edge_index, edge_attr, batch_idx, mod_adv, mod_rheo)
            return out.unsqueeze(0) if curr_z.ndim == 3 else out

        z_init = z.unsqueeze(0) if z.ndim == 2 else z

        if solver == "picard":
            z_star = z_init
            for _ in range(self.max_iters):
                z_star = f_coupled(z_star)
            z = z_star.squeeze(0)
        else:
            # Pass the warmup iters down to the Anderson solver
            z_star = anderson_acceleration(
                f_coupled, z_init, batch_idx=batch_idx,
                max_iter=self.max_iters, beta=anderson_beta, warmup_iters=anderson_warmup_iters
            )
            z = z_star.squeeze(0)

        jac_loss = torch.tensor(0.0, device=z.device)
        if self.training:
            z_star_req = z_star.detach().requires_grad_(True)
            f_z = f_coupled(z_star_req)
            eps = torch.randn_like(f_z)
            vjp = torch.autograd.grad(f_z, z_star_req, grad_outputs=eps, create_graph=True)[0]
            jac_loss = torch.mean(vjp ** 2)

        mu_raw = self.mu_decoder(z)
        mu = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * torch.sigmoid(mu_raw)

        u_v_p_residual = self.kinematics_decoder(z)
        uv_res = u_v_p_residual[:, :2]
        p = u_v_p_residual[:, 2:3]
        uv_final = uv_res + uv_prior

        u_v_p = torch.cat([uv_final, p], dim=1)
        wss_pred = self.wss_decoder(z)
        pred = torch.cat([u_v_p, mu, wss_pred], dim=1)

        return (pred, jac_loss) if self.training else pred