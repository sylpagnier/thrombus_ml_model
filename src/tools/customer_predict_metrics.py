"""Scientific time-series metrics for the customer predict app."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_CLOT_PHI_THRESHOLD = 0.5
# Matches customer UI: 8 h <-> 30000 s.
DEFAULT_SECONDS_PER_UI_HOUR = 30000.0 / 8.0


def _as_bool(mask: np.ndarray | None, n: int) -> np.ndarray:
    if mask is None:
        return np.zeros(n, dtype=bool)
    return np.asarray(mask, dtype=bool).reshape(-1)


def vessel_axis_coordinate(pos: np.ndarray) -> np.ndarray:
    """Scalar along-vessel coordinate via first principal component of XY."""
    xy = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[0], dtype=np.float64)
    centered = xy - xy.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    return centered @ axis


def wall_hop_distances_numpy(
    edge_index: np.ndarray,
    wall_mask: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """BFS hop distance from wall nodes (unreachable -> 99)."""
    hops = np.full(int(num_nodes), -1, dtype=np.int32)
    wall = _as_bool(wall_mask, num_nodes)
    hops[wall] = 0
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        hops[hops < 0] = 99
        return hops
    row, col = ei[0], ei[1]
    current = wall.copy()
    cur_h = 0
    while True:
        nbr = np.zeros(num_nodes, dtype=bool)
        nbr[col[current[row]]] = True
        nxt = nbr & (hops < 0)
        if not bool(nxt.any()):
            break
        cur_h += 1
        hops[nxt] = cur_h
        current = nxt
    hops[hops < 0] = 99
    return hops


def max_lumen_hop_occlusion_pct(
    phi: np.ndarray,
    *,
    hop_from_wall: np.ndarray,
    mask_wall: np.ndarray | None,
    mask_inlet: np.ndarray | None = None,
    mask_outlet: np.ndarray | None = None,
    threshold: float = DEFAULT_CLOT_PHI_THRESHOLD,
) -> dict[str, float]:
    """Occlusion from how far clot penetrates into the lumen (wall-hop depth).

    ``max_occlusion_pct = 100 * max_clot_lumen_hop / max_lumen_hop``.

    Wall nodes are hop 0 and do not count as lumen penetration. This tracks
    radial growth into the vessel better than along-axis bin clot fractions.
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    hops = np.asarray(hop_from_wall, dtype=np.int32).reshape(-1)
    n = int(phi.shape[0])
    wall = _as_bool(mask_wall, n)
    inlet = _as_bool(mask_inlet, n)
    outlet = _as_bool(mask_outlet, n)
    lumen = ~(wall | inlet | outlet)
    reachable = lumen & (hops < 99) & (hops > 0)
    clot_lumen = reachable & (phi >= float(threshold))

    max_lumen_hop = int(hops[reachable].max()) if reachable.any() else 0
    max_clot_hop = int(hops[clot_lumen].max()) if clot_lumen.any() else 0
    if max_lumen_hop <= 0:
        occ = 0.0
    else:
        occ = 100.0 * float(max_clot_hop) / float(max_lumen_hop)
    return {
        "max_occlusion_pct": occ,
        "max_clot_lumen_hop": float(max_clot_hop),
        "max_lumen_hop": float(max_lumen_hop),
    }


def frame_scientific_metrics(
    *,
    pos: np.ndarray,
    phi: np.ndarray,
    vel_mag: np.ndarray | None,
    mask_wall: np.ndarray | None,
    mask_inlet: np.ndarray | None,
    mask_outlet: np.ndarray | None,
    t_sec: float,
    hop_from_wall: np.ndarray | None = None,
    threshold: float = DEFAULT_CLOT_PHI_THRESHOLD,
) -> dict[str, float]:
    """Compute one-step scientific summary for CSV / plotting."""
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    n = int(phi.shape[0])
    wall = _as_bool(mask_wall, n)
    inlet = _as_bool(mask_inlet, n)
    outlet = _as_bool(mask_outlet, n)
    interior = ~(inlet | outlet)
    lumen = interior & ~wall
    clot = phi >= float(threshold)

    n_wall = int(wall.sum())
    n_interior = int(interior.sum())
    n_lumen = int(lumen.sum())
    wall_clot = int((wall & clot).sum())
    vessel_clot = int((interior & clot).sum())
    lumen_clot = int((lumen & clot).sum())

    wall_pct = 100.0 * wall_clot / max(n_wall, 1)
    vessel_pct = 100.0 * vessel_clot / max(n_interior, 1)
    lumen_pct = 100.0 * lumen_clot / max(n_lumen, 1)

    if hop_from_wall is not None and hop_from_wall.size == n:
        hop_stats = max_lumen_hop_occlusion_pct(
            phi,
            hop_from_wall=hop_from_wall,
            mask_wall=wall,
            mask_inlet=inlet,
            mask_outlet=outlet,
            threshold=threshold,
        )
    else:
        hop_stats = {
            "max_occlusion_pct": 0.0,
            "max_clot_lumen_hop": 0.0,
            "max_lumen_hop": 0.0,
        }

    mean_phi = float(phi[interior].mean()) if n_interior else 0.0
    mean_phi_clot = float(phi[interior & clot].mean()) if vessel_clot else 0.0

    out: dict[str, float] = {
        "t_s": float(t_sec),
        "t_h": float(t_sec) / DEFAULT_SECONDS_PER_UI_HOUR,
        "wall_clot_pct": wall_pct,
        "vessel_clot_pct": vessel_pct,
        "lumen_clot_pct": lumen_pct,
        "max_occlusion_pct": float(hop_stats["max_occlusion_pct"]),
        "max_clot_lumen_hop": float(hop_stats["max_clot_lumen_hop"]),
        "max_lumen_hop": float(hop_stats["max_lumen_hop"]),
        "mean_phi_interior": mean_phi,
        "mean_phi_clot_nodes": mean_phi_clot,
        "n_clot_nodes": float(vessel_clot),
        "n_wall_clot_nodes": float(wall_clot),
        # Always present so bookend-only velocity rows share one CSV schema.
        "mean_vel_open_lumen": float("nan"),
        "mean_vel_lumen": float("nan"),
    }
    if vel_mag is not None:
        vel = np.asarray(vel_mag, dtype=np.float64).reshape(-1)
        open_lumen = lumen & ~clot
        if open_lumen.any():
            out["mean_vel_open_lumen"] = float(vel[open_lumen].mean())
        else:
            out["mean_vel_open_lumen"] = 0.0
        if lumen.any():
            out["mean_vel_lumen"] = float(vel[lumen].mean())
        else:
            out["mean_vel_lumen"] = 0.0
    return out


def trajectory_scientific_table(
    traj: Any,
    *,
    threshold: float = DEFAULT_CLOT_PHI_THRESHOLD,
    seconds_per_ui_hour: float = DEFAULT_SECONDS_PER_UI_HOUR,
) -> list[dict[str, float]]:
    """Build per-step metric rows from a CustomerTrajectory-like object."""
    rows: list[dict[str, float]] = []
    hops = getattr(traj, "hop_from_wall", None)
    for i in range(int(traj.n_steps)):
        fr = traj.frame(i)
        has_vel = False
        if hasattr(traj, "has_velocity_at"):
            has_vel = bool(traj.has_velocity_at(i))
        else:
            has_vel = bool((getattr(traj, "meta", None) or {}).get("include_velocity", False))
        vel = np.asarray(fr["vel_mag"], dtype=np.float64) if has_vel else None
        row = frame_scientific_metrics(
            pos=traj.pos,
            phi=np.asarray(fr["phi"], dtype=np.float64),
            vel_mag=vel,
            mask_wall=getattr(traj, "mask_wall", None),
            mask_inlet=getattr(traj, "mask_inlet", None),
            mask_outlet=getattr(traj, "mask_outlet", None),
            t_sec=float(fr["t_sec"]),
            hop_from_wall=hops,
            threshold=threshold,
        )
        row["t_h"] = float(fr["t_sec"]) / float(seconds_per_ui_hour)
        rows.append(row)
    return rows


def _format_csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_scientific_csv(path: str | Any, rows: list[dict[str, float]]) -> None:
    """Write metric rows to CSV (stdlib csv)."""
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("t_s,t_h\n", encoding="utf-8")
        return
    # Union keys so bookend velocity columns never KeyError on middle steps.
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_cell(row.get(k)) for k in keys})
