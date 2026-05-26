"""Physics oracle mu/phi from GT species."""

import os

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import (
    build_clot_phi_step,
    physics_mu_eff_si,
    species_log1p_nd_to_si,
)


def test_species_log_to_si_shape():
    bio = BiochemConfig(phase="biochem")
    sp = torch.zeros(10, 12)
    si = species_log1p_nd_to_si(sp, bio)
    assert si.shape == (10, 12)


def test_physics_mu_positive():
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "1.0"
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig(phase="biochem")
    paths = list(__import__("pathlib").Path("data/processed/graphs_biochem_anchors").glob("*.pt"))
    if not paths:
        return
    data = torch.load(str(paths[0]), weights_only=False)
    device = torch.device("cpu")
    step = build_clot_phi_step(data, 0, phys, bio, device)
    mu = physics_mu_eff_si(step.mu_c_si, step.species_log_gt, bio, device=device)
    assert bool((mu > 0).all())
