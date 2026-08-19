"""Temporal Neural PDE Architectures for Wall Clot Prediction.
Implements 10 proxy architectures for zero-touch integration into the sweep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv

from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.core_physics.physics_wall_model import PER_M3_TO_PER_CM3

# =============================================================================
# Core Hook Wrapper
# =============================================================================
class TemporalDifferentiableWallModel(DifferentiableWallModel):
    """Overrides forward to accept an optional temporal_corrector that runs inside the Euler loop."""
    def __init__(self, *args, temporal_corrector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_corrector = temporal_corrector

    def forward(self, data, *, flow_source="pred", grow_hops=None, blockage_every=None, device=None) -> dict[str, torch.Tensor]:
        device = device or data.x.device if hasattr(data, "x") else torch.device("cpu")
        num_nodes = int(data.num_nodes if hasattr(data, "num_nodes") else len(data.mask_wall))
        wall_mask = data.mask_wall.reshape(-1).float().to(device)

        params = self.param_provider(data, num_nodes, device)
        grow_hops = grow_hops if grow_hops is not None else self.default_grow_hops
        every = blockage_every if blockage_every is not None else self.default_blockage_every

        sr, dsrx = self.compute_flow_fields(data, device, flow_source=flow_source)
        gate0 = self.compute_soft_gates(sr, dsrx, params) * wall_mask

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
        
        # Give corrector a chance to initialize hidden state
        hidden_state = None
        if self.temporal_corrector is not None:
            if hasattr(self.temporal_corrector, "init_hidden"):
                hidden_state = self.temporal_corrector.init_hidden(data, device)

        for i in range(n_steps - 1):
            h = t[i + 1] - t[i]
            t_curr = t[i]
            step2t = torch.sigmoid((t_curr - self.gate_s) * self.gate_slope)

            if (i > 0) and (i % every == 0):
                phi_occ = torch.sigmoid((mat - self.mat_crit) / (self.mat_crit * 0.1)) * wall_mask
                occ_frac = torch.clamp(torch.sparse.mm(B_tensor, phi_occ.unsqueeze(1)).squeeze(1), min=0.0, max=0.85)
                amp = torch.clamp(1.0 - params.wake * occ_frac, min=0.02, max=1.0)
                sr_eff = sr * amp
                dsrx_eff = dsrx * amp
                g_updated = self.compute_soft_gates(sr_eff, dsrx_eff, params) * wall_mask
                current_gate = torch.where(phi_occ > 0.5, torch.maximum(g_updated, gate0), g_updated)

            sat = torch.clamp(1.0 - mas / self.minf, min=0.0, max=1.0)
            dep = sat * (self.k_rs * rp_current + self.k_as * ap_current)
            auto = (mas / self.minf) * self.k_aa * ap_current

            d_mas = da_eff * current_gate * dep * step2t
            d_mat = da_eff * current_gate * (dep + auto) * step2t

            mas = mas + h * d_mas
            mat = mat + h * d_mat
            
            # --- NEW HOOK ---
            if self.temporal_corrector is not None:
                mat, mas, hidden_state = self.temporal_corrector(
                    mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask
                )
            
            traj_list.append(mat)

        traj = torch.stack(traj_list, dim=0)

        # FNO runs here post-loop
        if self.temporal_corrector is not None and hasattr(self.temporal_corrector, "post_process"):
            traj = self.temporal_corrector.post_process(traj, data, device, wall_mask)
            mat = traj[-1]

        p_seed = torch.sigmoid((mat / self.mat_crit - 1.0) / params.phi_temp) * wall_mask
        adm_thresh = params.lss * params.relax
        adm_temp = torch.clamp(params.lss * 0.1, min=1e-4)
        p_adm = torch.sigmoid((adm_thresh - sr) / adm_temp) * wall_mask
        p_cur = p_seed
        
        if grow_hops > 0:
            for _ in range(grow_hops):
                p_diff = torch.sparse.mm(A_tensor, p_cur.unsqueeze(1)).squeeze(1)
                p_cur = torch.clamp(p_cur + (1.0 - p_cur) * p_diff * p_adm, min=0.0, max=1.0)

        return {
            "prob_clot": p_cur * wall_mask,
            "mat_final": mat,
            "mat_traj": traj,
            "gate_init": gate0,
            "params": params,
        }

# =============================================================================
# 10 Proxy Architectures
# =============================================================================

class PredictorCorrectorGNN(nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.conv = SAGEConv(2, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        x = torch.stack([mat, mas], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, data.edge_index.to(x.device)))
        delta = self.head(feat).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        # Act as a high-frequency filter
        mat = mat + delta * 0.01 * wall_mask
        return mat, mas, hidden_state

class TGCN(nn.Module):
    def __init__(self, hidden_channels=4):
        super().__init__()
        self.conv_z = SAGEConv(2 + hidden_channels, hidden_channels)
        self.conv_r = SAGEConv(2 + hidden_channels, hidden_channels)
        self.conv_h = SAGEConv(2 + hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.hc = hidden_channels
        
    def init_hidden(self, data, device):
        return torch.zeros((data.num_nodes, self.hc), device=device)
        
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        x = torch.stack([mat, mas], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=0.0, neginf=0.0)
        
        cat_x = torch.cat([x, hidden_state], dim=-1)
        
        z = torch.sigmoid(self.conv_z(cat_x, ei))
        r = torch.sigmoid(self.conv_r(cat_x, ei))
        
        cat_r = torch.cat([x, r * hidden_state], dim=-1)
        h_tilde = torch.tanh(self.conv_h(cat_r, ei))
        
        hidden_state = (1 - z) * hidden_state + z * h_tilde
        delta = self.head(hidden_state).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        mat = mat + delta * 0.01 * wall_mask
        return mat, mas, hidden_state

class UNetODEPeriodic(nn.Module):
    def __init__(self, period=5, hidden_channels=16):
        super().__init__()
        self.period = period
        self.conv = SAGEConv(2, hidden_channels)
        self.head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        if i % self.period == 0:
            ei = data.edge_index.to(mat.device)
            x = torch.stack([mat, mas], dim=-1)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            feat = F.relu(self.conv(x, ei))
            delta = self.head(feat).squeeze(-1)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            mat = mat + delta * 0.05 * wall_mask
        return mat, mas, hidden_state

class GraphormerLite(nn.Module):
    def __init__(self):
        super().__init__()
        # Proxy: standard attention over local graph, pretending temporal window
        self.conv = GATv2Conv(2, 4, heads=1)
        self.head = nn.Linear(4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        x = torch.stack([mat, t_curr.expand_as(mat)], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        delta = self.head(feat).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        mat = mat + delta * 0.01 * wall_mask
        return mat, mas, hidden_state

class PseudoSDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(1, 16)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        x = torch.nan_to_num(mat.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        sigma = torch.sigmoid(self.head(feat).squeeze(-1))
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        # Add spatially correlated noise
        noise = torch.randn_like(mat) * sigma * 0.1
        mat = mat + noise * wall_mask
        return mat, mas, hidden_state

class FNOProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1, 16))
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        return mat, mas, hidden_state
    def post_process(self, traj, data, device, wall_mask):
        # T x N -> mix in time
        w = torch.sigmoid(self.w)
        # Simplistic temporal filter as proxy
        return traj * (1.0 + w.mean() * 0.01)

class PseudoCGNODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(2, 16)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        # Predict continuous derivative
        ei = data.edge_index.to(mat.device)
        x = torch.stack([mat, mas], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        d_mat_gnn = self.head(feat).squeeze(-1)
        d_mat_gnn = torch.nan_to_num(d_mat_gnn, nan=0.0, posinf=0.0, neginf=0.0)
        # Integrate GNN derivative
        mat = mat + h * d_mat_gnn * 0.1 * wall_mask
        return mat, mas, hidden_state

class PseudoCDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(3, 16)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        # Use physics gradient dsrx as control
        x = torch.stack([mat, mas, dsrx], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        ctrl = self.head(feat).squeeze(-1)
        ctrl = torch.nan_to_num(ctrl, nan=0.0, posinf=0.0, neginf=0.0)
        mat = mat + ctrl * d_mat * 0.1 * wall_mask # mod d_mat
        return mat, mas, hidden_state

class TrajectoryMatchingMPC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(1, 16)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        x = torch.nan_to_num(mat.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        action = torch.tanh(self.head(feat).squeeze(-1))
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        # Control action acts as artificial mass injection
        mat = mat + action * 0.05 * wall_mask
        return mat, mas, hidden_state

class ThresholdGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(2, 16)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -5.0)
    def forward(self, mat, mas, d_mat, data, t_curr, h, i, hidden_state, sr, dsrx, wall_mask):
        ei = data.edge_index.to(mat.device)
        x = torch.stack([mat, sr], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feat = F.relu(self.conv(x, ei))
        # dynamically boost mat if sr conditions met
        boost = torch.sigmoid(self.head(feat).squeeze(-1))
        boost = torch.nan_to_num(boost, nan=0.0, posinf=0.0, neginf=0.0)
        mat = mat + boost * 0.05 * wall_mask
        return mat, mas, hidden_state

def get_temporal_model_by_name(name):
    MODELS = {
        "PredictorCorrectorGNN": PredictorCorrectorGNN,
        "TGCN": TGCN,
        "UNetODEPeriodic": UNetODEPeriodic,
        "GraphormerLite": GraphormerLite,
        "PseudoSDE": PseudoSDE,
        "FNOProxy": FNOProxy,
        "PseudoCGNODE": PseudoCGNODE,
        "PseudoCDE": PseudoCDE,
        "TrajectoryMatchingMPC": TrajectoryMatchingMPC,
        "ThresholdGNN": ThresholdGNN,
    }
    return MODELS[name]
