"""Mat/FI channel decode for COMSOL mu1(Mat) / mu2(FI) gelation steps."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig
from src.core_physics.clot_phi_simple import (
    fi_si_for_gelation_from_log1p,
    mat_si_for_gelation_from_log1p,
    mu1_comsol_from_mat_si,
    mu2_comsol_from_fi_si,
    species_log1p_nd_to_si,
)


def test_mat_gelation_decode_removes_surface_scale_factor():
    """Graph ND encodes log1p(raw/Minf); legacy decode was raw*surface_scale."""
    bio = BiochemConfig(phase="biochem")
    raw = torch.tensor([0.0, 5.9067e4, 2.5454986e7, 3.0e7])
    nd = torch.log1p(raw / bio.Minf)
    legacy = torch.expm1(nd) * bio.Minf * bio.surface_scale
    fixed = mat_si_for_gelation_from_log1p(nd, bio)
    assert torch.allclose(fixed, raw, rtol=1e-4, atol=1.0)
    ratio = (legacy / fixed.clamp(min=1.0)).median()
    assert abs(float(ratio.item()) - bio.surface_scale) / bio.surface_scale < 0.01


def test_species_log1p_mat_channel_uses_gelation_units():
    bio = BiochemConfig(phase="biochem")
    raw_mat = 5.9067e4
    sp = torch.zeros(4, 12)
    sp[:, 11] = torch.log1p(torch.tensor(raw_mat / bio.Minf))
    si = species_log1p_nd_to_si(sp, bio)
    assert abs(float(si[0, 11].item()) - raw_mat) / raw_mat < 1e-4
    # M/Mas still use legacy surface_scale decode (deposition channels).
    legacy_m = float(torch.expm1(sp[0, 9]) * bio.Minf * bio.surface_scale)
    assert si[0, 9].item() == pytest.approx(legacy_m)


def test_mu1_step_threshold_at_comsol_crit():
    bio = BiochemConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    below = torch.tensor([crit * 0.5])
    above = torch.tensor([crit * 1.5])
    mu_below = mu1_comsol_from_mat_si(below, bio, bio.mu_ratio_max)
    mu_above = mu1_comsol_from_mat_si(above, bio, bio.mu_ratio_max)
    assert float(mu_below.item()) == pytest.approx(1.0, rel=0.01)
    assert float(mu_above.item()) == pytest.approx(float(bio.mu_ratio_max), rel=0.01)


def _fi_nd_for_working(bio: BiochemConfig, working: float) -> torch.Tensor:
    scale_fi = float(bio.get_species_scales(device=torch.device("cpu"))[8])
    return torch.log1p(torch.tensor(working / scale_fi))


def test_fi_gelation_decode_returns_uM_not_working():
    """COMSOL mu2(FI) steps at 0.6 uM; decode must convert working -> uM (=working*1e3/bulk_scale)."""
    bio = BiochemConfig(phase="biochem")
    for working in (1.0, 700.0, 7000.0):
        nd = _fi_nd_for_working(bio, working)
        fi_uM = fi_si_for_gelation_from_log1p(nd, bio)
        expected_uM = working * 1e3 / float(bio.bulk_scale)
        assert float(fi_uM.item()) == pytest.approx(expected_uM, rel=1e-4)


def test_species_log1p_fi_channel_uses_uM():
    bio = BiochemConfig(phase="biochem")
    working = 7000.0
    sp = torch.zeros(3, 12)
    sp[:, 8] = _fi_nd_for_working(bio, working)
    si = species_log1p_nd_to_si(sp, bio)
    expected_uM = working * 1e3 / float(bio.bulk_scale)
    assert float(si[0, 8].item()) == pytest.approx(expected_uM, rel=1e-4)


def test_fi_below_physical_crit_does_not_gel(monkeypatch):
    """FI in working units above 0.6 but below 0.6 uM must NOT gel (was ~1e3x too lenient)."""
    monkeypatch.setenv("CLOT_PHI_PHYSICS_HARD_STEP", "1")
    bio = BiochemConfig(phase="biochem")
    # working = 100 -> would gel under the old working-unit compare (100 > 0.6),
    # but uM = 0.1 < 0.6, so the unit-correct mu2 must stay inert.
    fi_uM = fi_si_for_gelation_from_log1p(_fi_nd_for_working(bio, 100.0), bio)
    assert float(fi_uM.item()) < float(bio.viscosity_fi_crit)
    mu2 = mu2_comsol_from_fi_si(fi_uM, bio, bio.mu_ratio_max)
    assert float(mu2.reshape(-1)[0].item()) == pytest.approx(0.0)


def test_fi_above_physical_crit_gels(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_PHYSICS_HARD_STEP", "1")
    bio = BiochemConfig(phase="biochem")
    # uM target = 1.0 (>= 0.6 crit) -> working = uM * bulk_scale / 1e3.
    working = 1.0 * float(bio.bulk_scale) / 1e3
    fi_uM = fi_si_for_gelation_from_log1p(_fi_nd_for_working(bio, working), bio)
    assert float(fi_uM.item()) >= float(bio.viscosity_fi_crit)
    mu2 = mu2_comsol_from_fi_si(fi_uM, bio, bio.mu_ratio_max)
    assert float(mu2.reshape(-1)[0].item()) == pytest.approx(float(bio.mu_ratio_max), rel=1e-3)
