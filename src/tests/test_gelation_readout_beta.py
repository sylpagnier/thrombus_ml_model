"""Gelation readout gain (beta) on the graded clot label.

Before this wiring ``rollout_t0_clot_phi`` accepted ``gelation_beta`` and immediately
``del``'d it, so every ``deploy_clot_*`` number -- and any sweep over
``SPECIES_GELATION_BETA_OVERRIDE`` -- was graded at an effective beta of 1.0.
See docs/WALL_MODEL_PLAN.md s1.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from src.config import BiochemConfig
from src.core_physics.clot_phi_simple import apply_gelation_beta_to_gel
from src.core_physics.species_gelation_readout import differentiable_mu_eff_from_species12
from src.core_physics.species_viscosity_calibration import resolve_clot_readout_beta
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi


@pytest.fixture
def _clear_beta_env():
    prev = os.environ.pop("SPECIES_GELATION_BETA_OVERRIDE", None)
    yield
    if prev is None:
        os.environ.pop("SPECIES_GELATION_BETA_OVERRIDE", None)
    else:
        os.environ["SPECIES_GELATION_BETA_OVERRIDE"] = prev


def test_beta_one_is_identity_in_both_gel_conventions():
    """comsol legs are multiplicative (identity 1), carreau legs additive (identity 0)."""
    gel = torch.tensor([1.0, 2.5, 8.0])
    for mode in ("comsol_carreau", "comsol", "carreau", "blood"):
        assert torch.allclose(apply_gelation_beta_to_gel(gel, 1.0, base_mode=mode), gel)


def test_beta_scales_excess_over_the_no_clot_identity():
    gel = torch.tensor([1.0, 3.0])
    comsol = apply_gelation_beta_to_gel(gel, 0.5, base_mode="comsol_carreau")
    # 1 stays 1 (no gelation there); 3 -> 1 + 0.5*(3-1) = 2.
    assert comsol.tolist() == pytest.approx([1.0, 2.0])
    carreau = apply_gelation_beta_to_gel(gel, 0.5, base_mode="carreau")
    assert carreau.tolist() == pytest.approx([0.5, 1.5])


def test_closed_loop_mu_eff_honours_beta():
    bio = BiochemConfig(phase="biochem")
    sp = torch.zeros(3, 12)
    mu_c = torch.full((3,), 4.0e-3)
    phi = torch.tensor([0.0, 0.5, 1.0])
    base = differentiable_mu_eff_from_species12(sp, mu_c, phi, bio)
    assert torch.allclose(
        differentiable_mu_eff_from_species12(sp, mu_c, phi, bio, gelation_beta=1.0), base
    )
    half = differentiable_mu_eff_from_species12(sp, mu_c, phi, bio, gelation_beta=0.5)
    # Baseline (phi=0) is untouched; the viscosity rise above it is halved.
    assert half[0].item() == pytest.approx(base[0].item())
    assert (half[1:] - mu_c[1:]).tolist() == pytest.approx(
        (0.5 * (base[1:] - mu_c[1:])).tolist(), rel=1e-5
    )


def test_rollout_forwards_beta_instead_of_discarding_it():
    src = inspect.getsource(rollout_t0_clot_phi)
    assert "del gamma_mode, gelation_beta" not in src
    assert "gelation_beta=gelation_beta" in src


def test_readout_beta_requires_an_explicit_override(_clear_beta_env):
    """No override => None => historical grading, never the stale on-disk t=53 beta."""
    assert resolve_clot_readout_beta() is None
    os.environ["SPECIES_GELATION_BETA_OVERRIDE"] = "0.4949"
    assert resolve_clot_readout_beta() == pytest.approx(0.4949)


def test_readout_beta_rejects_out_of_range(_clear_beta_env):
    os.environ["SPECIES_GELATION_BETA_OVERRIDE"] = "5.0"
    with pytest.raises(ValueError):
        resolve_clot_readout_beta()
