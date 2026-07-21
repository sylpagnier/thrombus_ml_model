"""Unit tests for customer predict scientific metrics."""

from __future__ import annotations

import numpy as np

from src.tools.customer_predict_metrics import (
    frame_scientific_metrics,
    max_lumen_hop_occlusion_pct,
    trajectory_scientific_table,
    vessel_axis_coordinate,
    wall_hop_distances_numpy,
    write_scientific_csv,
)


def test_vessel_axis_coordinate_projects_along_long_axis():
    pos = np.column_stack([np.linspace(0.0, 1.0, 20), np.zeros(20)])
    s = vessel_axis_coordinate(pos)
    assert s.shape == (20,)
    assert float(s[-1] - s[0]) > 0.5


def test_wall_hop_distances_numpy_bfs():
    # Line graph: 0(wall)-1-2-3
    edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    wall = np.array([True, False, False, False])
    hops = wall_hop_distances_numpy(edge_index, wall, 4)
    assert list(hops) == [0, 1, 2, 3]


def test_max_lumen_hop_occlusion_tracks_penetration():
    # Wall at 0; lumen hops 1..4. Clot only to hop 2 -> 50% of max lumen hop 4.
    hops = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    wall = np.array([1, 0, 0, 0, 0], dtype=bool)
    phi = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    out = max_lumen_hop_occlusion_pct(
        phi, hop_from_wall=hops, mask_wall=wall, threshold=0.5
    )
    assert out["max_clot_lumen_hop"] == 2.0
    assert out["max_lumen_hop"] == 4.0
    assert abs(out["max_occlusion_pct"] - 50.0) < 1e-6


def test_frame_scientific_metrics_wall_and_vessel_pct(tmp_path):
    n = 10
    pos = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    phi = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    wall = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    inlet = np.zeros(n, dtype=bool)
    outlet = np.zeros(n, dtype=bool)
    hops = np.array([0, 0, 1, 2, 3, 3, 2, 1, 1, 1], dtype=np.int32)
    row = frame_scientific_metrics(
        pos=pos,
        phi=phi,
        vel_mag=None,
        mask_wall=wall,
        mask_inlet=inlet,
        mask_outlet=outlet,
        t_sec=30000.0,
        hop_from_wall=hops,
        threshold=0.5,
    )
    assert abs(row["wall_clot_pct"] - 100.0) < 1e-6
    assert abs(row["vessel_clot_pct"] - 20.0) < 1e-6
    assert row["n_clot_nodes"] == 2.0
    # Clot on wall only (hops 0) -> no lumen penetration.
    assert row["max_clot_lumen_hop"] == 0.0
    assert row["max_occlusion_pct"] == 0.0
    assert "mean_vel_open_lumen" in row
    assert np.isnan(row["mean_vel_open_lumen"])

    class _Traj:
        n_steps = 1
        meta = {"include_velocity": False, "velocity_indices": []}

        def __init__(self):
            self.pos = pos
            self.mask_wall = wall
            self.mask_inlet = inlet
            self.mask_outlet = outlet
            self.hop_from_wall = hops

        def has_velocity_at(self, _i):
            return False

        def frame(self, _i):
            return {"t_sec": 30000.0, "phi": phi, "vel_mag": np.zeros(n)}

    rows = trajectory_scientific_table(_Traj(), seconds_per_ui_hour=3750.0)
    assert len(rows) == 1
    assert abs(rows[0]["t_h"] - 8.0) < 1e-9

    out = tmp_path / "m.csv"
    write_scientific_csv(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "wall_clot_pct" in text
    assert "max_occlusion_pct" in text
    assert "max_clot_lumen_hop" in text
    assert "mean_vel_open_lumen" in text


def test_bookend_velocity_csv_schema_stable(tmp_path):
    """Regression: velocity only at first/last must not KeyError on CSV write."""
    n = 6
    pos = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    wall = np.array([1, 0, 0, 0, 0, 1], dtype=bool)
    inlet = np.zeros(n, dtype=bool)
    outlet = np.zeros(n, dtype=bool)
    hops = np.array([0, 1, 2, 2, 1, 0], dtype=np.int32)
    phi0 = np.zeros(n)
    phi1 = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    phi2 = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    vel = np.linspace(0.1, 1.0, n)

    class _BookendTraj:
        n_steps = 3
        meta = {"include_velocity": True, "velocity_indices": [0, 2]}

        def __init__(self):
            self.pos = pos
            self.mask_wall = wall
            self.mask_inlet = inlet
            self.mask_outlet = outlet
            self.hop_from_wall = hops
            self._phi = [phi0, phi1, phi2]

        def has_velocity_at(self, i):
            return int(i) in (0, 2)

        def frame(self, i):
            return {
                "t_sec": float(i) * 10000.0,
                "phi": self._phi[i],
                "vel_mag": vel,
            }

    rows = trajectory_scientific_table(_BookendTraj(), seconds_per_ui_hour=3750.0)
    assert len(rows) == 3
    assert set(rows[0].keys()) == set(rows[1].keys()) == set(rows[2].keys())
    assert "mean_vel_open_lumen" in rows[0]
    assert not np.isnan(rows[0]["mean_vel_open_lumen"])
    assert np.isnan(rows[1]["mean_vel_open_lumen"])
    assert not np.isnan(rows[2]["mean_vel_open_lumen"])

    out = tmp_path / "bookend.csv"
    write_scientific_csv(out, rows)  # must not raise KeyError
    text = out.read_text(encoding="utf-8")
    lines = [ln for ln in text.strip().splitlines() if ln]
    assert len(lines) == 4  # header + 3 rows
    assert "mean_vel_open_lumen" in lines[0]


def test_write_scientific_csv_tolerates_uneven_keys(tmp_path):
    """Defensive: older uneven row dicts still write without KeyError."""
    rows = [
        {"t_s": 0.0, "t_h": 0.0, "mean_vel_open_lumen": 1.2},
        {"t_s": 1.0, "t_h": 0.5},  # missing velocity key (legacy bug shape)
    ]
    out = tmp_path / "uneven.csv"
    write_scientific_csv(out, rows)
    assert "mean_vel_open_lumen" in out.read_text(encoding="utf-8")
