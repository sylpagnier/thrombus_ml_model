"""Kinematic baseline + species-only learned gelation (learned_clot_penalty)."""

import math

import torch

from src.architecture.gnode_biochem import GNODE_Phase3
from src.config import BiochemConfig, PhysicsConfig


def test_learned_clot_penalty_is_nonneg_and_small_at_init():
    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")
    model = GNODE_Phase3(
        phys_cfg=phys_cfg,
        latent_dim=16,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    )
    # Near-resting log1p species (same order of magnitude as _default_resting_species).
    sp = torch.zeros(4, 12, dtype=torch.float32)
    sp[:, 0] = math.log1p(1.0)
    sp[:, 1] = math.log1p(0.05)
    sp[:, 4] = math.log1p(1.0)
    sp[:, 6] = math.log1p(1.0)
    sp[:, 7] = math.log1p(1.0)
    out = model.learned_clot_penalty(sp)
    assert out.shape == (4, 1)
    assert (out >= 0.0).all()
    assert float(out.max().item()) < 0.05


def test_forward_one_step_with_gelation_head():
    from torch_geometric.data import Data

    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")

    n = 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    e = edge_index.shape[1]
    eye = torch.eye(n, dtype=torch.float32).to_sparse_coo()

    x = torch.zeros(n, 15, dtype=torch.float32)
    x[:, 2] = 0.1
    x[:, 3] = 0.0
    x[:, 4:6] = torch.tensor([1.0, 0.0])
    x[:, 11:13] = 0.01
    x[:, 13:14] = 1.0
    x[:, 14:15] = 0.0

    batch = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.zeros(e, 3, dtype=torch.float32),
        G_x=eye,
        G_y=eye,
        mask_wall=torch.zeros(n, dtype=torch.bool),
        u_ref=torch.tensor(0.1, dtype=torch.float32),
        d_bar=torch.tensor(0.01, dtype=torch.float32),
        is_anchor=torch.tensor(False),
    )

    model = GNODE_Phase3(
        phys_cfg=phys_cfg,
        latent_dim=16,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    )
    model.eval()
    times = torch.tensor([0.0], dtype=torch.float32)

    with torch.no_grad():
        traj = model(batch, times, detach_macro_state=False)

    assert traj.shape == (1, n, 16)
    mu_nd = traj[0, :, 3:4]
    assert torch.isfinite(mu_nd).all()
