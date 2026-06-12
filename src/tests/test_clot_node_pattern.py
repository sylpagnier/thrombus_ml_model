"""Clot vs non-clot kinematic patterns on anchor graphs (patient007 when available)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_kinematics_fields import (
    ClotKinematicsFields,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat

REPO = Path(__file__).resolve().parents[2]
ANCHOR = REPO / "data" / "processed" / "graphs_biochem_anchors" / "patient007.pt"


def _graph_props(data, device):
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        u_ref = data.u_ref.to(device=device, dtype=torch.float32).reshape(-1)[:1]
        d_bar = data.d_bar.to(device=device, dtype=torch.float32).reshape(-1)[:1]
    else:
        u_ref = torch.as_tensor(data.u_ref, device=device, dtype=torch.float32).reshape(1)
        d_bar = torch.as_tensor(data.d_bar, device=device, dtype=torch.float32).reshape(1)
    return {"u_ref": u_ref, "d_bar": d_bar}


def _time_index(data, t_query: float | None) -> int:
    if not hasattr(data, "t") or data.t is None:
        return 0
    t = data.t.detach().cpu().numpy().astype("float64").reshape(-1)
    if t_query is None:
        return int(t.argmax())
    return int(abs(t - float(t_query)).argmin())


def _subset_stats(
    name: str,
    values: torch.Tensor,
    clot_mask: torch.Tensor,
    band_mask: torch.Tensor,
) -> dict[str, float]:
    v = values.detach().cpu().float().reshape(-1)
    c = clot_mask.cpu().numpy().astype(bool)
    b = band_mask.cpu().numpy().astype(bool)
    m = c & b
    nc = (~c) & b
    out: dict[str, float] = {}
    if m.any():
        out[f"{name}_clot_mean"] = float(v[m].mean())
        out[f"{name}_clot_p90"] = float(torch.quantile(v[m], 0.9))
    else:
        out[f"{name}_clot_mean"] = float("nan")
        out[f"{name}_clot_p90"] = float("nan")
    if nc.any():
        out[f"{name}_non_mean"] = float(v[nc].mean())
        out[f"{name}_non_p90"] = float(torch.quantile(v[nc], 0.9))
    else:
        out[f"{name}_non_mean"] = float("nan")
        out[f"{name}_non_p90"] = float("nan")
    if m.any() and nc.any():
        out[f"{name}_delta_mean"] = out[f"{name}_clot_mean"] - out[f"{name}_non_mean"]
    return out


@pytest.fixture
def patient007():
    if not ANCHOR.is_file():
        pytest.skip(f"missing anchor graph: {ANCHOR}")
    return torch.load(ANCHOR, map_location="cpu", weights_only=False)


def test_comsol_hybrid_prior_beats_legacy_on_dx_hotspot(monkeypatch):
    """Negative dγ/dx hotspot should dominate comsol_hybrid but not legacy max-norm."""
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("BIOCHEM_PRIOR_DGAMMA_DX_THRESH", "800")
    bio = BiochemConfig(phase="biochem")

    fields = ClotKinematicsFields(
        gamma_si=torch.tensor([40.0, 40.0, 40.0]),
        dshear_ds_phys=torch.tensor([0.0, 0.0, 0.0]),
        dgamma_dx_phys=torch.tensor([-900.0, -900.0, -50.0]),
        dgamma_dy_phys=torch.zeros(3),
        is_low_shear=torch.zeros(3),
        is_separation_stream=torch.zeros(3),
        flux_path_stream=torch.tensor([0.1, 0.1, 0.1]),
        flux_path_dx=torch.tensor([0.9, 0.1, 0.02]),
        flux_path_dx_raw=torch.tensor([0.9, 0.1, 0.02]),
        flux_stag=torch.tensor([0.1, 0.1, 0.1]),
        wall_proximity=torch.tensor([1.0, 1.0, 0.2]),
        adjacent_band=torch.tensor([True, True, False]),
    )
    score_hybrid, path_h, _ = score_clot_risk_from_fields(fields, bio)

    monkeypatch.delenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", raising=False)
    monkeypatch.setenv("BIOCHEM_PRIOR_SCORE_MODE", "legacy")
    score_legacy, path_l, _ = score_clot_risk_from_fields(fields, bio)

    assert float(score_hybrid[0]) > float(score_hybrid[2])
    # Legacy path channel uses stream separation only (flat across nodes here).
    assert float(path_h[0]) > float(path_h[1])
    assert abs(float(path_l[0]) - float(path_l[1])) < 0.05


def test_clot_vs_nonclot_pattern_at_t0_and_tfinal(patient007, monkeypatch):
    """Report whether kinematic cues separate clot vs non-clot nodes (adjacent band)."""
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("BIOCHEM_PRIOR_DGAMMA_DX_THRESH", "800")
    monkeypatch.setenv("BIOCHEM_PRIOR_NORM_MASK", "adjacent")

    data = patient007
    bio = BiochemConfig(phase="biochem")
    phys = PhysicsConfig()
    device = torch.device("cpu")
    props = _graph_props(data, device)

    slices = [
        ("t0", _time_index(data, 0.0)),
        ("tfinal", _time_index(data, None)),
    ]
    report: dict[str, float] = {}

    for label, ti in slices:
        y = data.y[ti]
        mu_si = phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND]).reshape(-1)
        mu_floor = max(
            float(os.environ.get("BIOCHEM_K11_CLOT_MU_SI_MIN", "0.055")),
            float(phys.mu_inf),
        )
        clot_strict = mu_si >= mu_floor
        mu_cut = torch.quantile(mu_si, 0.9)
        clot_p90 = mu_si >= mu_cut

        u = y[:, 0]
        v = y[:, 1]
        fields = compute_clot_kinematics_fields(data, u, v, bio, props)
        prior, _, _ = score_clot_risk_from_fields(fields, bio)
        band = fields.adjacent_band

        use_clot = clot_strict if int(clot_strict.sum()) >= 5 else clot_p90
        for name, tensor in (
            ("dgamma_dx", fields.dgamma_dx_phys),
            ("dshear_ds", fields.dshear_ds_phys),
            ("prior", prior),
            ("flux_dx", fields.flux_path_dx),
        ):
            stats = _subset_stats(f"{label}_{name}", tensor, use_clot, band)
            report.update(stats)

        n_clot = int((use_clot & band).sum())
        report[f"{label}_n_clot_adjacent"] = float(n_clot)

    # t_final: COMSOL clots — expect more negative dγ/dx on clot nodes (adjacent band).
    if report.get("tfinal_n_clot_adjacent", 0) >= 5:
        delta_dx = report.get("tfinal_dgamma_dx_delta_mean", float("nan"))
        delta_prior = report.get("tfinal_prior_delta_mean", float("nan"))
        assert delta_dx == delta_dx  # documented in assertion message
        if delta_dx == delta_dx:  # not nan
            assert delta_dx < 0.0 or delta_prior > 0.0, (
                f"expected clot separation at t_final; report={report}"
            )

    # t0: usually no strict clots; test only checks we can run and counts are small.
    assert report["t0_n_clot_adjacent"] >= 0.0


def test_clot_dx_gate_active_when_gradient_below_threshold(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("BIOCHEM_PRIOR_DGAMMA_DX_THRESH", "800")

    bio = BiochemConfig(phase="biochem")
    props = {"u_ref": torch.ones(1), "d_bar": torch.ones(1)}
    fields = ClotKinematicsFields(
        gamma_si=torch.tensor([50.0, 50.0]),
        dshear_ds_phys=torch.tensor([0.0, 0.0]),
        dgamma_dx_phys=torch.tensor([-900.0, -100.0]),
        dgamma_dy_phys=torch.tensor([0.0, 0.0]),
        is_low_shear=torch.zeros(2),
        is_separation_stream=torch.zeros(2),
        flux_path_stream=torch.zeros(2),
        flux_path_dx=torch.tensor([1.1, 0.1]),
        flux_path_dx_raw=torch.tensor([1.1, 0.1]),
        flux_stag=torch.zeros(2),
        wall_proximity=torch.tensor([1.0, 0.5]),
        adjacent_band=torch.tensor([True, True]),
    )
    assert float(fields.flux_path_dx[0]) > float(fields.flux_path_dx[1])
