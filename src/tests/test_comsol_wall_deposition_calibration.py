"""Regression test: canonical surface-deposition law vs COMSOL exports.

Pins the unit-consistent COMSOL phase-2 surface platelet-deposition law against a
compact fixture extracted from the patient007 calibration export
(``src/tests/fixtures/comsol_wall_deposition_patient007.csv``). Guards that:

  1. Reconstructing ``J0_Mat`` from the exported inputs (CGS) recovers the
     Damkohler number ``Da == surface_damkohler == 1e-4`` to machine precision,
     using COMSOL's own exported gate columns.
  2. The thrombin source ``J0_th = beta*phi_at*Mat*PT*step2t`` is reproduced.

If this fails, the COMSOL unit mapping in
``src/core_physics/comsol_surface_deposition.py`` / ``BiochemConfig`` drifted.
See ``docs/COMSOL_PHYSICS_VALIDATION.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.config import BiochemConfig
from src.core_physics.comsol_surface_deposition import (
    M_TO_CM,
    MS_TO_CMS,
    PER_M2_TO_PER_CM2,
    PER_M3_TO_PER_CM3,
    PER_MS_TO_PER_CMS,
    j0_mat_cgs,
    j0_mat_si,
    j0_thrombin_cgs,
    recover_damkohler_cgs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "comsol_wall_deposition_patient007.csv"


def _load_fixture() -> dict[str, torch.Tensor]:
    with FIXTURE.open() as fh:
        reader = csv.DictReader(fh)
        cols = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v))
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in cols.items()}


def test_recovered_damkohler_matches_config():
    cfg = BiochemConfig()
    f = _load_fixture()
    Da = recover_damkohler_cgs(
        j0_mat_exported=f["J0_Mat"],
        sat_m=f["Sat_M"],
        shear_sr=f["sr"],
        dsrx=f["dsrx"],
        rp=f["rp"],
        ap=f["ap"],
        mas=f["Mas"],
        step2t=f["step2t"],
        gate_low=f["sr_lt_lss"],
        gate_sep=f["dsrx_lt_sgt"],
        cfg=cfg,
    )
    assert Da.numel() > 100, "fixture should contain many active deposition points"
    median = float(Da.median())
    # Da must equal surface_damkohler (1e-4) to machine precision, and be constant.
    assert abs(median - cfg.surface_damkohler) / cfg.surface_damkohler < 1e-6
    cv = float(Da.std() / Da.mean())
    assert cv < 1e-6, f"recovered Da not constant (CV={cv:.3g}) -> unit drift"


def test_j0_mat_reconstruction_matches_export():
    cfg = BiochemConfig()
    f = _load_fixture()
    j0_recon = j0_mat_cgs(
        sat_m=f["Sat_M"],
        shear_sr=f["sr"],
        dsrx=f["dsrx"],
        rp=f["rp"],
        ap=f["ap"],
        mas=f["Mas"],
        step2t=f["step2t"],
        cfg=cfg,
        gate_low_temp=None,  # exact COMSOL booleans via thresholds
        gate_sep_temp=None,
    )
    exp = f["J0_Mat"]
    denom = exp.abs().sum() + 1e-30
    rel = float((j0_recon - exp).abs().sum() / denom)
    assert rel < 1e-6, f"J0_Mat reconstruction rel-err {rel:.3e}"


def test_j0_thrombin_reconstruction_matches_export():
    cfg = BiochemConfig()
    f = _load_fixture()
    j0_th = j0_thrombin_cgs(mat=f["Mat"], pt=f["PT"], step2t=f["step2t"], cfg=cfg)
    exp = f["J0_th"]
    m = exp.abs() > 0
    assert int(m.sum()) > 50
    rel = float((j0_th[m] - exp[m]).abs().sum() / (exp[m].abs().sum() + 1e-30))
    assert rel < 1e-6, f"J0_th reconstruction rel-err {rel:.3e}"


def test_si_law_matches_export_and_cgs():
    """The SI law (biochem_wall_residual convention) reproduces the CGS exports.

    Converts the CGS fixture inputs to SI, runs ``j0_mat_si``, and checks it equals
    ``j0_mat_cgs`` * 1e4 (surface rate plt/(cm^2 s) -> plt/(m^2 s)). This proves the
    kernel's SI unit system is consistent with the COMSOL CGS ground truth.
    """
    cfg = BiochemConfig()
    f = _load_fixture()
    # CGS -> SI input conversions (divide by the SI->CGS factors).
    si = dict(
        sat_m=f["Sat_M"],
        shear_sr=f["sr"],
        dsrx=f["dsrx"] / PER_MS_TO_PER_CMS,   # 1/(s*cm) -> 1/(s*m)
        rp=f["rp"] / PER_M3_TO_PER_CM3,        # plt/cm^3 -> plt/m^3
        ap=f["ap"] / PER_M3_TO_PER_CM3,
        mas=f["Mas"] / PER_M2_TO_PER_CM2,      # plt/cm^2 -> plt/m^2
        step2t=f["step2t"],
    )
    j_si = j0_mat_si(cfg=cfg, **si)
    j_cgs = j0_mat_cgs(
        sat_m=f["Sat_M"], shear_sr=f["sr"], dsrx=f["dsrx"], rp=f["rp"], ap=f["ap"],
        mas=f["Mas"], step2t=f["step2t"], cfg=cfg,
    )
    # surface rate: 1 plt/(cm^2 s) = 1e4 plt/(m^2 s)
    denom = (j_cgs * 1e4).abs().sum() + 1e-30
    rel = float((j_si - j_cgs * 1e4).abs().sum() / denom)
    assert rel < 1e-6, f"SI law != CGS law (rel={rel:.3e}) -> kernel unit drift"


def test_soft_gates_track_hard_gates():
    """Differentiable surrogate (finite temps) stays close to the exact-gate law."""
    cfg = BiochemConfig()
    f = _load_fixture()
    hard = j0_mat_cgs(
        sat_m=f["Sat_M"], shear_sr=f["sr"], dsrx=f["dsrx"], rp=f["rp"], ap=f["ap"],
        mas=f["Mas"], step2t=f["step2t"], cfg=cfg,
    )
    soft = j0_mat_cgs(
        sat_m=f["Sat_M"], shear_sr=f["sr"], dsrx=f["dsrx"], rp=f["rp"], ap=f["ap"],
        mas=f["Mas"], step2t=f["step2t"], cfg=cfg,
        gate_low_temp=1.0, gate_sep_temp=50.0,
    )
    denom = hard.abs().sum() + 1e-30
    rel = float((soft - hard).abs().sum() / denom)
    assert rel < 0.5, f"soft-gate law diverges from hard-gate law (rel={rel:.3f})"
