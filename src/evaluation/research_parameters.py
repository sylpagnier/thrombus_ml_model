"""Transferable research-parameter pack for geometry-sensitivity sweeps.

Reuses customer Scientific definitions (node coverage + lumen-hop occlusion
+ axial / hop / residual extras) and adds trajectory summaries so multiple
scripts share one schema.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.tools.customer_predict_metrics import (
    DEFAULT_CLOT_PHI_THRESHOLD,
    DEFAULT_SECONDS_PER_UI_HOUR,
    frame_scientific_metrics,
    max_lumen_hop_occlusion_pct,
    trajectory_scientific_table,
    vessel_axis_coordinate,
    wall_hop_distances_numpy,
    write_scientific_csv,
)

SCHEMA_VERSION = 2

# Keys summarized across arms (pct / normalized series).
PCT_SERIES_KEYS: tuple[str, ...] = (
    "wall_clot_pct",
    "vessel_clot_pct",
    "lumen_clot_pct",
    "open_lumen_pct",
    "max_occlusion_pct",
    "open_lumen_residual_pct",
    "clot_frac_hop0_pct",
    "clot_frac_hop1_pct",
    "clot_frac_hop_ge2_pct",
    "clot_mass_prox_pct",
    "clot_mass_mid_pct",
    "clot_mass_dist_pct",
    "clot_cov_prox_pct",
    "clot_cov_mid_pct",
    "clot_cov_dist_pct",
    "clot_axis_span_norm",
)

OCC_THRESHOLDS_PCT: tuple[float, ...] = (25.0, 50.0, 75.0)
EARLY_GROWTH_HOURS = 2.0


def compute_research_timeseries(
    traj: Any,
    *,
    threshold: float = DEFAULT_CLOT_PHI_THRESHOLD,
    seconds_per_ui_hour: float = DEFAULT_SECONDS_PER_UI_HOUR,
) -> list[dict[str, float]]:
    """Per-frame research metrics (same schema as customer Scientific CSV)."""
    return trajectory_scientific_table(
        traj,
        threshold=threshold,
        seconds_per_ui_hour=seconds_per_ui_hour,
    )


def _series(rows: Sequence[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=np.float64)


# np.trapz renamed to np.trapezoid in NumPy 2.x.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz"))


def _trapz_auc(t_h: np.ndarray, y: np.ndarray) -> float:
    """AUC of y vs UI hours; ignores NaNs by dropping non-finite samples."""
    if t_h.size < 2:
        return 0.0
    mask = np.isfinite(t_h) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    tt = t_h[mask]
    yy = y[mask]
    order = np.argsort(tt)
    return float(_trapezoid(yy[order], tt[order]))


def _time_to_threshold(
    t_h: np.ndarray,
    y: np.ndarray,
    thresh: float,
) -> float:
    """First time (UI hours) where y >= thresh; NaN if never reached."""
    for i in range(int(y.size)):
        if np.isfinite(y[i]) and float(y[i]) >= float(thresh):
            return float(t_h[i]) if np.isfinite(t_h[i]) else float("nan")
    return float("nan")


def _time_to_flag(t_h: np.ndarray, flag: np.ndarray) -> float:
    """First time where flag > 0.5; NaN if never."""
    for i in range(int(flag.size)):
        if np.isfinite(flag[i]) and float(flag[i]) > 0.5:
            return float(t_h[i]) if np.isfinite(t_h[i]) else float("nan")
    return float("nan")


def _early_growth_rate(
    t_h: np.ndarray,
    y: np.ndarray,
    *,
    early_hours: float = EARLY_GROWTH_HOURS,
) -> float:
    """(y_at_early - y0) / dt_h over the first ``early_hours`` of UI time.

    Uses the last sample with ``t_h <= t0 + early_hours``. If that is still
    the first frame (coarse time grid), advances to the first later sample.
    """
    if t_h.size < 2 or not np.isfinite(y[0]):
        return float("nan")
    t0 = float(t_h[0]) if np.isfinite(t_h[0]) else 0.0
    target = t0 + float(early_hours)
    idx = 0
    for i in range(int(t_h.size)):
        if np.isfinite(t_h[i]) and float(t_h[i]) <= target + 1e-12:
            idx = i
    if idx == 0:
        for i in range(1, int(t_h.size)):
            if np.isfinite(t_h[i]) and np.isfinite(y[i]):
                idx = i
                break
    dt = float(t_h[idx]) - t0
    if dt <= 1e-12 or not np.isfinite(y[idx]):
        return float("nan")
    return float(y[idx] - y[0]) / dt


def summarize_research_timeseries(
    rows: Sequence[dict[str, float]],
    *,
    occ_thresholds: Sequence[float] = OCC_THRESHOLDS_PCT,
    early_hours: float = EARLY_GROWTH_HOURS,
) -> dict[str, float]:
    """Peak / final / AUC / time-to-threshold / onset / front speed for series."""
    out: dict[str, float] = {
        "n_frames": float(len(rows)),
    }
    if not rows:
        return out

    t_h = _series(rows, "t_h")
    out["t_h_final"] = float(t_h[-1]) if np.isfinite(t_h[-1]) else float("nan")
    out["t_s_final"] = float(rows[-1].get("t_s", float("nan")))

    for key in PCT_SERIES_KEYS:
        y = _series(rows, key)
        finite = y[np.isfinite(y)]
        out[f"{key}_peak"] = float(finite.max()) if finite.size else float("nan")
        out[f"{key}_final"] = float(y[-1]) if np.isfinite(y[-1]) else float("nan")
        out[f"{key}_auc_h"] = _trapz_auc(t_h, y)
        out[f"{key}_early_rate_per_h"] = _early_growth_rate(
            t_h, y, early_hours=early_hours
        )

    occ = _series(rows, "max_occlusion_pct")
    for thr in occ_thresholds:
        tag = f"{int(thr)}" if float(thr) == int(thr) else f"{thr:g}"
        out[f"t_h_to_occ_{tag}"] = _time_to_threshold(t_h, occ, float(thr))

    # Onset: first wall clot and first lumen (hop>=1) clot.
    out["t_h_to_first_wall_clot"] = _time_to_flag(t_h, _series(rows, "has_wall_clot"))
    out["t_h_to_first_lumen_clot"] = _time_to_flag(t_h, _series(rows, "has_lumen_clot"))

    # Early axial front speed from clot_axis_span_norm.
    span = _series(rows, "clot_axis_span_norm")
    out["clot_front_speed_early_per_h"] = _early_growth_rate(
        t_h, span, early_hours=early_hours
    )
    front = _series(rows, "clot_front_speed_per_h")
    finite_front = front[np.isfinite(front)]
    out["clot_front_speed_peak_per_h"] = (
        float(finite_front.max()) if finite_front.size else float("nan")
    )

    # Bookend velocity (optional; NaN when clot-only).
    vel = _series(rows, "mean_vel_open_lumen")
    finite_vel = vel[np.isfinite(vel)]
    if finite_vel.size:
        out["mean_vel_open_lumen_final"] = (
            float(vel[-1]) if np.isfinite(vel[-1]) else float("nan")
        )
        out["mean_vel_open_lumen_peak"] = float(finite_vel.max())
        # Prefer first finite as baseline for drop.
        v0 = float(finite_vel[0])
        v1 = float(vel[-1]) if np.isfinite(vel[-1]) else float("nan")
        if v0 > 1e-12 and np.isfinite(v1):
            out["vel_open_lumen_drop_pct_final"] = 100.0 * (v0 - v1) / v0
        else:
            out["vel_open_lumen_drop_pct_final"] = float("nan")
    else:
        out["mean_vel_open_lumen_final"] = float("nan")
        out["mean_vel_open_lumen_peak"] = float("nan")
        out["vel_open_lumen_drop_pct_final"] = float("nan")

    return out


def research_parameters_from_trajectory(
    traj: Any,
    *,
    threshold: float = DEFAULT_CLOT_PHI_THRESHOLD,
    seconds_per_ui_hour: float = DEFAULT_SECONDS_PER_UI_HOUR,
    occ_thresholds: Sequence[float] = OCC_THRESHOLDS_PCT,
    early_hours: float = EARLY_GROWTH_HOURS,
) -> dict[str, Any]:
    """Full transferable pack: timeseries + summary + schema version."""
    rows = compute_research_timeseries(
        traj,
        threshold=threshold,
        seconds_per_ui_hour=seconds_per_ui_hour,
    )
    summary = summarize_research_timeseries(
        rows,
        occ_thresholds=occ_thresholds,
        early_hours=early_hours,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "vessel_clot_pct": "100 * (# interior nodes with phi>=thr) / (# interior)",
            "wall_clot_pct": "100 * (# wall nodes with phi>=thr) / (# wall)",
            "lumen_clot_pct": "100 * (# lumen nodes with phi>=thr) / (# lumen)",
            "open_lumen_pct": "100 * (# open lumen nodes) / (# lumen)",
            "max_occlusion_pct": "100 * max_clot_lumen_hop / max_lumen_hop (wall hop0 excluded)",
            "open_lumen_residual_pct": "100 * max_open_lumen_hop / max_lumen_hop",
            "clot_frac_hop*_pct": "% of interior clot nodes at hop 0 / 1 / >=2",
            "clot_mass_*_pct": "% of vessel clot nodes in proximal/mid/distal thirds",
            "clot_cov_*_pct": "% coverage within each axial third",
            "clot_axis_span_norm": "axial clot extent / vessel span",
            "clot_front_speed_per_h": "d(clot_axis_span_norm)/d(t_h)",
            "coverage_basis": "node_fraction",
            "phi_threshold": float(threshold),
            "seconds_per_ui_hour": float(seconds_per_ui_hour),
        },
        "timeseries": rows,
        "summary": summary,
    }


__all__ = [
    "SCHEMA_VERSION",
    "PCT_SERIES_KEYS",
    "OCC_THRESHOLDS_PCT",
    "EARLY_GROWTH_HOURS",
    "DEFAULT_CLOT_PHI_THRESHOLD",
    "DEFAULT_SECONDS_PER_UI_HOUR",
    "compute_research_timeseries",
    "summarize_research_timeseries",
    "research_parameters_from_trajectory",
    "frame_scientific_metrics",
    "max_lumen_hop_occlusion_pct",
    "trajectory_scientific_table",
    "vessel_axis_coordinate",
    "wall_hop_distances_numpy",
    "write_scientific_csv",
]
