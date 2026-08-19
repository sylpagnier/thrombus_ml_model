"""Differentiable physics ODE rollout for wall clot prediction.

Implements a fully differentiable PyTorch forward pass of:
  1. Soft flow-derived deposition gates (low-shear stagnation and separation).
  2. Local stagnation wake feedback (reducing shear near committed tissue).
  3. Continuous surface ODE integration (resting/activated platelet adhesion and autocatalysis).
  4. Soft gelation thresholding and smooth graph-front growth.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d
from src.core_physics.physics_wall_model import wall_platelet_constants
from src.core_physics.shear_redistribution import build_crosssection_operator, sdf_nd
from src.differentiable_wall_model.parameters import (
    GlobalPhysicsParameters,
    ParameterMap,
    ParameterProvider,
)

M_TO_CM = 100.0
PER_M3_TO_PER_CM3 = 1.0e-6
PER_M2_TO_PER_CM2 = 1.0e-4


def _scipy_csr_to_torch_sparse(mat: sp.csr_matrix, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    coo = mat.tocoo()
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


class DifferentiableWallModel(nn.Module):
    """Differentiable wall-clot forward solver.

    Wraps a ParameterProvider to enable end-to-end backpropagation from clot loss
    directly into physical parameters (Level 1.1) or local neural encoders (Level 1.2).
    """

    def __init__(
        self,
        bio_cfg: BiochemConfig | None = None,
        parameter_provider: ParameterProvider | None = None,
        *,
        default_grow_hops: int = 6,
        default_blockage_every: int = 5,
        default_mls_hops: int = 4,
        diffusion_module: nn.Module | None = None,
    ):
        super().__init__()
        self.bio_cfg = bio_cfg or BiochemConfig(phase="biochem")
        self.param_provider = parameter_provider or GlobalPhysicsParameters()
        self.default_grow_hops = default_grow_hops
        self.default_blockage_every = default_blockage_every
        self.default_mls_hops = default_mls_hops
        self.diffusion_module = diffusion_module
        self.chem_estimator = None  # To be set later or passed in init

        # Physical constants in CGS / COMSOL units
        self.k_rs = float(self.bio_cfg.k_rs) * M_TO_CM
        self.k_as = float(self.bio_cfg.k_as) * M_TO_CM
        self.k_aa = float(self.bio_cfg.k_aa) * M_TO_CM
        self.minf = float(self.bio_cfg.Minf) * PER_M2_TO_PER_CM2
        self.surface_da = float(self.bio_cfg.surface_damkohler)
        self.mat_crit = float(self.bio_cfg.viscosity_mat_crit)
        self.L_char_cm = float(self.bio_cfg.L_char) * M_TO_CM
        self.gamma_m = float(self.bio_cfg.gamma_m)
        self.gate_s = float(self.bio_cfg.surface_time_gate_s)
        self.gate_slope = float(self.bio_cfg.surface_time_gate_slope)

    def compute_flow_fields(
        self,
        data,
        device: torch.device,
        *,
        flow_source: str = "pred",
        mls_hops: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute base shear rate [1/s] and shear x-gradient [1/(s*cm)] at t=0."""
        hops = mls_hops or (3 if flow_source == "gt" else self.default_mls_hops)
        pos = node_positions(data)
        ei = data.edge_index.detach().cpu().numpy()
        u_ref = float(data.u_ref.reshape(-1)[0])
        d_bar = float(data.d_bar.reshape(-1)[0])

        Dx, Dy = build_mls_gradient(pos, ei, hops=hops)
        if flow_source == "pred":
            if getattr(data, "u0_pred", None) is None:
                raise ValueError("Graph has no u0_pred; cannot use flow_source='pred'")
            u = data.u0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
            v = data.v0_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            u = data.y[0, :, 0].detach().cpu().numpy().astype(np.float64)
            v = data.y[0, :, 1].detach().cpu().numpy().astype(np.float64)

        sr_np = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)
        dsrx_np = (Dx @ sr_np) / (d_bar * M_TO_CM)

        sr = torch.tensor(sr_np, dtype=torch.float32, device=device)
        dsrx = torch.tensor(dsrx_np, dtype=torch.float32, device=device)
        return sr, dsrx

    def compute_soft_gates(
        self,
        sr: torch.Tensor,
        dsrx: torch.Tensor,
        params: ParameterMap,
    ) -> torch.Tensor:
        """Compute soft, differentiable bracket prefactor for deposition law."""
        # Low shear gate: sr < lss
        temp_low = torch.clamp(params.tau_low * params.lss, min=1e-4)
        g_low = torch.sigmoid((params.lss - sr) / temp_low)

        # Separation gate: dsrx < sgt_cgs
        temp_sep = torch.clamp(params.tau_sep * torch.abs(params.sgt_cgs), min=1e-4)
        g_sep = torch.sigmoid((params.sgt_cgs - dsrx) / temp_sep)

        coef = self.L_char_cm / self.gamma_m
        gate = g_sep * coef * torch.abs(dsrx) + g_low
        return gate

    def build_graph_operators(self, data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cross-section operator B and wall adjacency operator A."""
        pos = node_positions(data)
        wall_np = data.mask_wall.reshape(-1).bool().cpu().numpy()
        sdf = sdf_nd(data)
        B_sp = build_crosssection_operator(pos, sdf, wall_np, radius_mult=0.30)
        B_tensor = _scipy_csr_to_torch_sparse(B_sp, device)

        ei = data.edge_index.cpu().numpy()
        n = len(wall_np)
        A_sp = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A_sp = ((A_sp + A_sp.T) > 0).astype(np.float32)
        # Row-normalize adjacency
        deg = np.asarray(A_sp.sum(axis=1)).reshape(-1)
        deg[deg == 0] = 1.0
        A_norm_sp = sp.diags(1.0 / deg) @ A_sp
        A_tensor = _scipy_csr_to_torch_sparse(A_norm_sp, device)
        return B_tensor, A_tensor

    def forward(
        self,
        data,
        *,
        flow_source: str = "pred",
        grow_hops: int | None = None,
        blockage_every: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run differentiable ODE simulation on graph data.

        Returns a dictionary containing:
          - 'prob_clot': [N] differentiable clot probability (0 to 1).
          - 'mat_final': [N] final continuous Mat values in CGS units.
          - 'mat_traj':  [T, N] trajectory of Mat across timesteps.
          - 'gate_init': [N] initial bracket prefactor gate.
          - 'params':    ParameterMap evaluated for this graph.
        """
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        num_nodes = int(data.num_nodes if hasattr(data, "num_nodes") else len(data.mask_wall))
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        params = self.param_provider(data, num_nodes, device)
        grow_hops = grow_hops if grow_hops is not None else self.default_grow_hops
        every = blockage_every if blockage_every is not None else self.default_blockage_every

        sr, dsrx = self.compute_flow_fields(data, device, flow_source=flow_source)
        gate0 = self.compute_soft_gates(sr, dsrx, params) * wall_mask

        # Platelet initial conditions (treated as constants unless chem_estimator active)
        names = data.y_channel_names.split(",")
        scales = self.bio_cfg.get_species_scales(device=device)
        rp_nd = torch.expm1(data.y[0, :, names.index("RP_log1p_nd")].to(device).clamp(-10, 8))
        ap_nd = torch.expm1(data.y[0, :, names.index("AP_log1p_nd")].to(device).clamp(-10, 8))
        rp_initial = rp_nd * float(scales[0]) * PER_M3_TO_PER_CM3
        ap_initial = ap_nd * float(scales[1]) * PER_M3_TO_PER_CM3
        rp_current = rp_initial
        ap_current = ap_initial

        t = data.t.reshape(-1).to(device=device, dtype=torch.float32)
        n_steps = len(t)

        B_tensor, A_tensor = self.build_graph_operators(data, device)

        mas = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        mat = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        traj_list = [mat]

        da_eff = self.surface_da * params.da_scale

        current_gate = gate0
        for i in range(n_steps - 1):
            h = t[i + 1] - t[i]
            t_curr = t[i]
            step2t = torch.sigmoid((t_curr - self.gate_s) * self.gate_slope)

            if (i > 0) and (i % every == 0):
                # Continuous soft commitment for wake evaluation
                phi_occ = torch.sigmoid((mat - self.mat_crit) / (self.mat_crit * 0.1)) * wall_mask
                # Sparse matvec: B @ phi_occ
                occ_frac = torch.clamp(
                    torch.sparse.mm(B_tensor, phi_occ.unsqueeze(1)).squeeze(1),
                    min=0.0,
                    max=0.85,
                )
                amp = torch.clamp(1.0 - params.wake * occ_frac, min=0.02, max=1.0)
                sr_eff = sr * amp
                dsrx_eff = dsrx * amp
                g_updated = self.compute_soft_gates(sr_eff, dsrx_eff, params) * wall_mask
                # Already-committed nodes continue to deposit
                current_gate = torch.where(phi_occ > 0.5, torch.maximum(g_updated, gate0), g_updated)

                # Update chemical state periodically if estimator is present
                if hasattr(self, "chem_estimator") and self.chem_estimator is not None:
                    ap_frac, rp_frac = self.chem_estimator(mat, sr_eff, data, device)
                    ap_current = ap_initial * (1.0 - ap_frac)
                    rp_current = rp_initial * (1.0 - rp_frac)

            sat = torch.clamp(1.0 - mas / self.minf, min=0.0, max=1.0)
            dep = sat * (self.k_rs * rp_current + self.k_as * ap_current)
            auto = (mas / self.minf) * self.k_aa * ap_current

            # Forward Euler step
            d_mas = da_eff * current_gate * dep * step2t
            d_mat = da_eff * current_gate * (dep + auto) * step2t

            mas = mas + h * d_mas
            mat = mat + h * d_mat
            traj_list.append(mat)

        traj = torch.stack(traj_list, dim=0)  # [T, N]

        # Gelation / Clot readout
        # Soft normalized thresholding: sigmoid((mat / mat_crit - 1.0) / temp)
        p_seed = torch.sigmoid((mat / self.mat_crit - 1.0) / params.phi_temp) * wall_mask

        # Soft Graph-Front Growth
        adm_thresh = params.lss * params.relax
        adm_temp = torch.clamp(params.lss * 0.1, min=1e-4)
        p_adm = torch.sigmoid((adm_thresh - sr) / adm_temp) * wall_mask

        if hasattr(self, "diffusion_module") and self.diffusion_module is not None:
            # GNO predicts unconstrained raw growth, multiplied by physics admission gate
            raw_growth = self.diffusion_module(data, p_seed, device)
            p_cur = torch.clamp(p_seed + raw_growth * p_adm, min=0.0, max=1.0)
        else:
            p_cur = p_seed
            if grow_hops > 0:
                for _ in range(grow_hops):
                    # Diffuse from committed neighbors
                    p_diff = torch.sparse.mm(A_tensor, p_cur.unsqueeze(1)).squeeze(1)
                    # Boost probability if admitted
                    p_cur = torch.clamp(p_cur + (1.0 - p_cur) * p_diff * p_adm, min=0.0, max=1.0)

        return {
            "prob_clot": p_cur * wall_mask,
            "mat_final": mat,
            "mat_traj": traj,
            "gate_init": gate0,
            "params": params,
        }
