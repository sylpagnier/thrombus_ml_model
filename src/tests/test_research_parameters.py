"""Unit tests for transferable research-parameter pack."""

from __future__ import annotations

import numpy as np

from src.evaluation.research_parameters import (
    SCHEMA_VERSION,
    compute_research_timeseries,
    research_parameters_from_trajectory,
    summarize_research_timeseries,
)


class _FakeTraj:
    """Minimal CustomerTrajectory-like object for metric tests."""

    def __init__(self):
        n = 10
        self.n_steps = 3
        self.pos = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
        self.mask_wall = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        self.mask_inlet = np.zeros(n, dtype=bool)
        self.mask_outlet = np.zeros(n, dtype=bool)
        self.hop_from_wall = np.array([0, 0, 1, 2, 3, 4, 3, 2, 1, 1], dtype=np.int32)
        self.meta = {"include_velocity": False, "velocity_indices": []}
        # Growing lumen clot: frame0 none; frame1 hop1-2; frame2 hop1-4 (full).
        self._phis = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float),
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
        ]
        # 0, 4, 8 UI hours at 3750 s/h
        self._t = [0.0, 15000.0, 30000.0]

    def has_velocity_at(self, _i: int) -> bool:
        return False

    def frame(self, i: int) -> dict:
        return {
            "t_sec": float(self._t[i]),
            "phi": self._phis[i],
            "vel_mag": np.zeros(10),
        }


def test_research_timeseries_matches_scientific_keys():
    rows = compute_research_timeseries(_FakeTraj(), seconds_per_ui_hour=3750.0)
    assert len(rows) == 3
    assert abs(rows[-1]["t_h"] - 8.0) < 1e-9
    assert "vessel_clot_pct" in rows[0]
    assert "wall_clot_pct" in rows[0]
    assert "max_occlusion_pct" in rows[0]


def test_summarize_peak_final_auc_and_time_to_occ():
    rows = compute_research_timeseries(_FakeTraj(), seconds_per_ui_hour=3750.0)
    summary = summarize_research_timeseries(rows)
    assert summary["n_frames"] == 3.0
    assert summary["max_occlusion_pct_peak"] >= summary["max_occlusion_pct_final"] - 1e-9
    # Frame2 clot reaches hop 4 of max lumen hop 4 -> 100%
    assert abs(summary["max_occlusion_pct_final"] - 100.0) < 1e-6
    assert abs(summary["t_h_to_occ_50"] - 4.0) < 1e-6 or summary["t_h_to_occ_50"] <= 8.0
    assert np.isfinite(summary["vessel_clot_pct_auc_h"])
    assert np.isfinite(summary["max_occlusion_pct_early_rate_per_h"])
    assert "t_h_to_first_wall_clot" in summary
    assert "t_h_to_first_lumen_clot" in summary
    assert "clot_front_speed_early_per_h" in summary
    assert "clot_frac_hop_ge2_pct_final" in summary


def test_research_parameters_from_trajectory_pack_shape():
    pack = research_parameters_from_trajectory(_FakeTraj(), seconds_per_ui_hour=3750.0)
    assert pack["schema_version"] == SCHEMA_VERSION
    assert pack["definitions"]["coverage_basis"] == "node_fraction"
    assert len(pack["timeseries"]) == 3
    assert "max_occlusion_pct_peak" in pack["summary"]
    assert "t_h_to_occ_75" in pack["summary"]
    assert "clot_mass_prox_pct" in pack["timeseries"][0]
    assert "open_lumen_residual_pct" in pack["timeseries"][0]
    assert "clot_front_speed_per_h" in pack["timeseries"][0]


def test_summarize_empty_rows():
    assert summarize_research_timeseries([]) == {"n_frames": 0.0}
