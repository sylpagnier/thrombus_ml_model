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
from src.architecture.siren_decoder import SIRENDecoder
from src.config import NodeFeat, PhysicsConfig, PredChannels
from src.utils.batching import get_batch_tensor
from src.utils.kinematics import stream_to_velocity


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


class AttentionGlobalMixingBlock(nn.Module):
    """
    Perceiver-style bottleneck: global tokens read the graph via cross-attention,
    reason with an MLP, then broadcast back to nodes. Runs **per graph** in the PyG batch.
    """

    def __init__(
        self,
        latent_dim: int,
        num_global_tokens: int = 16,
        num_heads: int = 4,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})")
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, latent_dim))
        self.cross_att_read = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )
        self.global_mlp = nn.Sequential(
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
            nn.SiLU(),
            _spectral_or_plain_linear(latent_dim, latent_dim, True, use_spectral_norm),
        )
        self.cross_att_broadcast = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        num_graphs = int(batch.max().item()) + 1
        device = x.device
        dtype = x.dtype
        gt_all = self.global_tokens.to(device=device, dtype=dtype).expand(num_graphs, -1, -1).contiguous()
        for b in range(num_graphs):
            idx = batch == b
            seq_x = x[idx].unsqueeze(0)
            if seq_x.size(1) == 0:
                continue
            read_tokens, _ = self.cross_att_read(gt_all[b : b + 1], seq_x, seq_x)
            processed_tokens = self.global_mlp(read_tokens)
            broadcast_update, _ = self.cross_att_broadcast(seq_x, processed_tokens, processed_tokens)
            out[idx] = broadcast_update.squeeze(0)
        return out


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
    def __init__(
        self,
        latent_dim=64,
        edge_dim=3,
        use_spectral_norm: bool = True,
        activation_fn: str = "relu",
        global_pool_mode: str = "mean",
        num_global_tokens: int = 16,
    ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be divisible by 2 for multi-head split"

        self.global_pool_mode = (global_pool_mode or "mean").strip().lower()
        self.conv = MultiHeadPhysicsGATConv(
            latent_dim, edge_dim=edge_dim, use_spectral_norm=use_spectral_norm
        )
        if self.global_pool_mode == "attention":
            self.global_mixer = AttentionGlobalMixingBlock(
                latent_dim,
                num_global_tokens=num_global_tokens,
                use_spectral_norm=use_spectral_norm,
            )
        elif self.global_pool_mode == "mean":
            self.global_mixer = GlobalMixingBlock(
                latent_dim,
                use_spectral_norm=use_spectral_norm,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                f"Unsupported global_pool_mode={global_pool_mode!r}; expected 'mean' or 'attention'."
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
        mu_inf_nd: Optional[float] = None,
        mu_0_nd: Optional[float] = None,
        phys_cfg: Optional[PhysicsConfig] = None,
        kinematics_mode: str = "direct_uvp",
        activation_fn: str = "silu",
        fourier_base: float = 2.0,
        use_hard_bcs: bool = False,
        global_pool_mode: str = "mean",
        num_global_tokens: int = 16,
        use_siren_decoder: bool = False,
        use_width_priors: bool = False,
    ):
        super().__init__()
        self.max_iters = max_iters
        self.outer_iters = outer_iters
        self.num_fourier_freqs = num_fourier_freqs
        if phys_cfg is not None:
            mu_scale = float(phys_cfg.mu_viscosity_nd_scale)
            default_mu_inf_nd = float(phys_cfg.mu_inf / mu_scale)
            default_mu_0_nd = float(phys_cfg.mu_0 / mu_scale)
            self.edge_decay_k = float(phys_cfg.gino_edge_decay_k)
            self.curve_log_clamp_min = float(phys_cfg.gino_curve_log_clamp_min)
            self.rheo_log_clamp_min = float(phys_cfg.gino_rheo_log_clamp_min)
            self.adv_log_clamp_min = float(phys_cfg.gino_adv_log_clamp_min)
        else:
            default_mu_inf_nd = 0.03
            default_mu_0_nd = 1.0
            self.edge_decay_k = 5.0
            self.curve_log_clamp_min = 1e-4
            self.rheo_log_clamp_min = 1e-3
            self.adv_log_clamp_min = 1e-3
        self.mu_inf_nd = float(default_mu_inf_nd if mu_inf_nd is None else mu_inf_nd)
        self.mu_0_nd = float(default_mu_0_nd if mu_0_nd is None else mu_0_nd)
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
        self.use_hard_bcs = bool(use_hard_bcs)
        self.use_siren_decoder = bool(use_siren_decoder) and self.kinematics_mode == "direct_uvp"
        self.use_width_priors = bool(use_width_priors)
        self.global_pool_mode = (global_pool_mode or "mean").strip().lower()

        freqs = (self.fourier_base ** torch.arange(num_fourier_freqs)) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        fourier_channels = 5 * num_fourier_freqs * 2
        width_extra = 3 if self.use_width_priors else 0
        encoded_channels = (in_channels - 5) + 5 + fourier_channels + width_extra

        self.encoder = nn.Sequential(
            nn.Linear(encoded_channels, latent_dim),
            _make_activation(self.activation_fn),
            nn.Linear(latent_dim, latent_dim)
        )

        self.core = GINOBlock(
            latent_dim,
            edge_dim=3,
            activation_fn=self.activation_fn,
            global_pool_mode=self.global_pool_mode,
            num_global_tokens=num_global_tokens,
        )
        # Stream-function formulation: decoder predicts (psi, p), then u,v are derived from psi.
        # Direct formulation: decoder predicts (u, v, p) directly (used to avoid WLS-on-WLS differentiation).
        if self.kinematics_mode == "stream":
            self.kinematics_decoder = nn.Linear(latent_dim, 2)
            self.siren_decoder = None
        elif self.use_siren_decoder:
            self.siren_decoder = SIRENDecoder(latent_dim)
            self.kinematics_decoder = None
        else:
            self.kinematics_decoder = nn.Linear(latent_dim, 3)
            self.siren_decoder = None

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
        # Canonical Tier-1 layout is 15 channels; optional width priors append three more (see NodeFeat).
        xb = x[:, :15] if x.size(1) >= 15 else x
        nodes_nd = pos_nd if pos_nd is not None else xb[:, NodeFeat.XY]
        sdf_nd = xb[:, NodeFeat.SDF]
        shear_pot = xb[:, NodeFeat.SHEAR_POT]
        wall_normal = xb[:, NodeFeat.WALL_NORMAL]

        rest = xb[:, NodeFeat.REST]
        uv_prior = xb[:, NodeFeat.UV_PRIOR]
        mu_prior = xb[:, NodeFeat.MU_PRIOR]
        wss_prior = xb[:, NodeFeat.WSS_PRIOR]

        features_to_encode = torch.cat([nodes_nd, sdf_nd, wall_normal], dim=1)
        N, C = features_to_encode.shape

        x_proj = (features_to_encode.unsqueeze(-1) * self.fourier_freqs).contiguous()
        x_proj = x_proj.view(N, -1)
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        encoded_x = torch.cat(
            [shear_pot, features_to_encode, fourier_feats, rest, uv_prior, mu_prior, wss_prior], dim=1)
        if getattr(self, "use_width_priors", False):
            if x.size(1) >= NodeFeat.WIDTH_D2.stop:
                width_features = x[:, NodeFeat.WIDTH_ND.start : NodeFeat.WIDTH_D2.stop]
            else:
                width_features = torch.zeros(x.size(0), 3, device=x.device, dtype=x.dtype)
            encoded_x = torch.cat([encoded_x, width_features], dim=1)
        return encoded_x, uv_prior

    @torch.enable_grad()
    def forward(self, data, solver="anderson", anderson_beta=0.8, anderson_warmup_iters=5):
        x_encoded, _ = self._apply_fourier_encoding(data.x)
        x_enc = self.encoder(x_encoded)
        z = x_enc.clone()

        row, col = data.edge_index

        # Pull precomputed edge attributes directly instead of recalculating
        edge_attr = data.edge_attr
        edge_vec = edge_attr[:, :2]

        batch_idx = get_batch_tensor(data, data.x.size(0), data.x.device)

        wall_normals = data.x[:, NodeFeat.WALL_NORMAL]
        e_dir = F.normalize(edge_vec, p=2, dim=-1, eps=1e-8)
        n_dir_row = F.normalize(wall_normals[row], p=2, dim=-1, eps=1e-8)
        n_dir_col = F.normalize(wall_normals[col], p=2, dim=-1, eps=1e-8)

        dot_prod = torch.abs((e_dir * n_dir_row).sum(dim=-1, keepdim=True))
        dot_prod = torch.clamp(dot_prod, max=1.0)

        sdf_nd = data.x[:, NodeFeat.SDF]
        sdf_edge = sdf_nd[row]

        decay_factor = torch.exp(-self.edge_decay_k * sdf_edge)
        curve_dot = (n_dir_row * n_dir_col).sum(dim=-1, keepdim=True)
        mod_curve = torch.log(torch.clamp(1.0 - curve_dot, min=self.curve_log_clamp_min, max=1.0)) * decay_factor

        mod_rheo = torch.log(torch.clamp(dot_prod, min=self.rheo_log_clamp_min, max=1.0)) * decay_factor
        mod_adv = torch.log(torch.clamp((1.0 - dot_prod), min=self.adv_log_clamp_min, max=1.0)) * decay_factor

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

        if self.kinematics_mode == "direct_uvp" and self.siren_decoder is not None:
            pos_nd = data.x[:, NodeFeat.XY]
            uvp, _ = self.siren_decoder(z, pos_nd)
            u_v_p = uvp[:, PredChannels.KINEMATICS]
        elif self.kinematics_mode == "direct_uvp":
            assert self.kinematics_decoder is not None
            kinematics_out = self.kinematics_decoder(z)
            u_v_p = kinematics_out[:, PredChannels.KINEMATICS]
        else:
            assert self.kinematics_decoder is not None
            kinematics_out = self.kinematics_decoder(z)
            psi_raw = kinematics_out[:, 0:1]
            p = kinematics_out[:, 1:2]
            sdf = data.sdf_wall if hasattr(data, "sdf_wall") else data.x[:, NodeFeat.SDF]
            wall_normal = data.x[:, NodeFeat.WALL_NORMAL]
            u_v_p = stream_to_velocity(
                psi_raw=psi_raw,
                p=p,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                V=data.V,
                W=data.W,
                M_inv=data.M_inv,
                sdf=sdf,
                wall_normal=wall_normal,
                k_env=self.k_env,
            )

        if self.kinematics_mode == "direct_uvp" and self.use_hard_bcs:
            # SDF is already [N, 1]; do not add another singleton (would break broadcast with [N, 2]).
            sdf = data.x[:, NodeFeat.SDF]
            uv_prior = data.x[:, NodeFeat.UV_PRIOR]
            u_v_constrained = uv_prior + sdf * u_v_p[:, :2]
            u_v_p = torch.cat([u_v_constrained, u_v_p[:, 2:3]], dim=1)

        wss_pred = self.wss_decoder(z)
        pred = torch.cat([u_v_p, mu, wss_pred], dim=1)

        return (pred, jac_loss) if self.training else pred