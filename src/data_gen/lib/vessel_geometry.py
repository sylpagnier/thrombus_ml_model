"""Pure NumPy vessel wall/centerline geometry (no Gmsh)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

from src.data_gen.lib.vessel_generator import (
    _centerline_arc,
    _centerline_s_curve,
    _centerline_straight,
    resolve_bend_sign_mode,
)


class GeometryValidationError(ValueError):
    """Raised when wall/centerline geometry fails validation checks."""


@dataclass
class VesselGeometry:
    idx: int
    n: int
    unit: str
    d_bar: float
    d_inlet: float
    centerline_pts: np.ndarray
    centerline_tangents: np.ndarray
    top_coords: np.ndarray
    bot_coords: np.ndarray
    top_wall_tangents: np.ndarray
    bot_wall_tangents: np.ndarray
    top_wall_normals: np.ndarray
    bot_wall_normals: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    base_length: float = 0.0


def _wall_tangents_normals(top_coords: np.ndarray, bot_coords: np.ndarray):
    top_tangents = np.gradient(top_coords, axis=0)
    top_tangents = top_tangents / np.maximum(np.linalg.norm(top_tangents, axis=1, keepdims=True), 1e-9)
    bot_tangents = np.gradient(bot_coords, axis=0)
    bot_tangents = bot_tangents / np.maximum(np.linalg.norm(bot_tangents, axis=1, keepdims=True), 1e-9)
    top_normals = np.column_stack([-top_tangents[:, 1], top_tangents[:, 0]])
    bot_normals = np.column_stack([bot_tangents[:, 1], -bot_tangents[:, 0]])
    return top_tangents, bot_tangents, top_normals, bot_normals


def _build_meta(
    *,
    idx: int,
    params: Mapping[str, Any],
    unit: str,
    d_bar: float,
    d_inlet: float,
    pts: np.ndarray,
    tangents: np.ndarray,
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
    top_tangents: np.ndarray,
    bot_tangents: np.ndarray,
    top_normals: np.ndarray,
    bot_normals: np.ndarray,
) -> Dict[str, Any]:
    curve_type = str(params.get("curve_type", "straight"))
    v_type = str(params.get("v_type", "straight"))
    meta = {
        "id": idx,
        "type": f"{v_type}_{curve_type}",
        "curve": curve_type,
        "level": int(params.get("level", 0)),
        "bend_sign": float(params.get("bend_sign", 1.0)),
        "bend_sign_mode": str(params.get("bend_sign_mode", resolve_bend_sign_mode())),
        "unit": unit,
        "d_bar": float(d_bar),
        "d_inlet": float(d_inlet),
        "num_outlets": 1,
        "centerline_pts": (pts / d_bar).tolist(),
        "centerline_tangents": tangents.tolist(),
        "top_wall_pts": (top_coords / d_bar).tolist(),
        "bot_wall_pts": (bot_coords / d_bar).tolist(),
        "top_wall_tangents": top_tangents.tolist(),
        "bot_wall_tangents": bot_tangents.tolist(),
        "top_wall_normals": top_normals.tolist(),
        "bot_wall_normals": bot_normals.tolist(),
    }

    # Inject boundary-layer patch metadata for the COMSOL mu-clot painter.
    for key in (
        "clot_x_center",
        "clot_height",
        "clot_width",
        "clot_edge_width",
        "clot_mu_peak",
        "inlet_shear_rate",
    ):
        if key in params:
            meta[key] = float(params[key])
    if "clot_shape" in params:
        meta["clot_shape"] = str(params["clot_shape"])

    return meta


def compute_geometry_from_params(params: Dict[str, Any], cfg_dict: Dict[str, Any]) -> VesselGeometry:
    """Build wall polylines from parametric vessel params (steps 1-4 of legacy mesh builder)."""
    idx = int(params["idx"])
    n = int(cfg_dict["num_ctrl_pts"])
    L = float(cfg_dict["base_length"])
    min_lumen_frac = float(cfg_dict["min_lumen_width_fraction"])
    curve_type = str(params["curve_type"])
    path_loc = int(params["path_loc"])
    width = float(params["width"])
    offsets = np.asarray(params["offsets"], dtype=float)

    if curve_type == "straight":
        pts, tangents = _centerline_straight(n, L, np.zeros(n - 4))
    elif curve_type in ("arc", "hook"):
        pts, tangents = _centerline_arc(
            n, L, float(params["angle_span"]), bend_sign=float(params.get("bend_sign", 1.0))
        )
    else:
        pts, tangents = _centerline_s_curve(n, L, float(params["amplitude"]))

    tortuosity = np.asarray(params.get("tortuosity", np.zeros(n - 4)), dtype=float)
    if tortuosity.size and np.any(tortuosity):
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        pts[2 : n - 2] += normals[2 : n - 2] * tortuosity[:, np.newaxis]
        tangents = np.gradient(pts, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-9)

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    top_offsets = offsets if path_loc in (0, 2) else np.zeros(n)
    bot_offsets = offsets if path_loc in (1, 2) else np.zeros(n)
    top_offsets = top_offsets + np.asarray(params.get("noise_top", np.zeros(n)), dtype=float)
    bot_offsets = bot_offsets + np.asarray(params.get("noise_bot", np.zeros(n)), dtype=float)

    top_dist = (width / 2.0) + top_offsets
    bot_dist = (width / 2.0) + bot_offsets
    cross_widths = top_dist + bot_dist
    d_bar = float(np.mean(cross_widths))

    top_coords = pts + normals * top_dist[:, np.newaxis]
    bot_coords = pts - normals * bot_dist[:, np.newaxis]

    unit = str(cfg_dict.get("unit", "m"))
    unit_scale = 100.0 if unit == "cm" else 1.0
    if unit_scale != 1.0:
        top_coords = top_coords * unit_scale
        bot_coords = bot_coords * unit_scale
        pts = pts * unit_scale
        d_bar *= unit_scale
        L *= unit_scale

    top_t, bot_t, top_n, bot_n = _wall_tangents_normals(top_coords, bot_coords)
    d_inlet = float(np.linalg.norm(top_coords[0] - bot_coords[0]))
    meta = _build_meta(
        idx=idx,
        params=params,
        unit=unit,
        d_bar=d_bar,
        d_inlet=d_inlet,
        pts=pts,
        tangents=tangents,
        top_coords=top_coords,
        bot_coords=bot_coords,
        top_tangents=top_t,
        bot_tangents=bot_t,
        top_normals=top_n,
        bot_normals=bot_n,
    )

    geom = VesselGeometry(
        idx=idx,
        n=n,
        unit=unit,
        d_bar=d_bar,
        d_inlet=d_inlet,
        centerline_pts=pts,
        centerline_tangents=tangents,
        top_coords=top_coords,
        bot_coords=bot_coords,
        top_wall_tangents=top_t,
        bot_wall_tangents=bot_t,
        top_wall_normals=top_n,
        bot_wall_normals=bot_n,
        meta=meta,
        base_length=L,
    )
    validate_geometry(geom, cfg_dict, reference_width=width * unit_scale)
    return geom


def compute_geometry_from_walls(
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
    *,
    idx: int = 0,
    unit: str = "m",
    params: Mapping[str, Any] | None = None,
    base_length: float | None = None,
) -> VesselGeometry:
    """Build centerline + meta from edited top/bottom wall station polylines (SI coords)."""
    top_coords = np.asarray(top_coords, dtype=np.float64)
    bot_coords = np.asarray(bot_coords, dtype=np.float64)
    if top_coords.shape != bot_coords.shape or top_coords.ndim != 2 or top_coords.shape[1] != 2:
        raise ValueError(f"top/bot coords must match shape (n, 2); got {top_coords.shape} vs {bot_coords.shape}")
    n = int(top_coords.shape[0])
    pts = 0.5 * (top_coords + bot_coords)
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)
    cross_widths = np.linalg.norm(top_coords - bot_coords, axis=1)
    d_bar = float(np.mean(cross_widths))
    d_inlet = float(np.linalg.norm(top_coords[0] - bot_coords[0]))
    top_t, bot_t, top_n, bot_n = _wall_tangents_normals(top_coords, bot_coords)

    p = dict(params or {})
    p.setdefault("curve_type", "edited")
    p.setdefault("v_type", "edited")
    p.setdefault("level", 0)
    meta = _build_meta(
        idx=idx,
        params=p,
        unit=unit,
        d_bar=d_bar,
        d_inlet=d_inlet,
        pts=pts,
        tangents=tangents,
        top_coords=top_coords,
        bot_coords=bot_coords,
        top_tangents=top_t,
        bot_tangents=bot_t,
        top_normals=top_n,
        bot_normals=bot_n,
    )
    L = float(base_length) if base_length is not None else float(np.max(pts[:, 0]) - np.min(pts[:, 0]) + 1e-9)
    return VesselGeometry(
        idx=idx,
        n=n,
        unit=unit,
        d_bar=d_bar,
        d_inlet=d_inlet,
        centerline_pts=pts,
        centerline_tangents=tangents,
        top_coords=top_coords,
        bot_coords=bot_coords,
        top_wall_tangents=top_t,
        bot_wall_tangents=bot_t,
        top_wall_normals=top_n,
        bot_wall_normals=bot_n,
        meta=meta,
        base_length=L,
    )


def validate_geometry(
    geom: VesselGeometry,
    cfg_dict: Dict[str, Any],
    *,
    reference_width: float | None = None,
    baseline_top: np.ndarray | None = None,
    baseline_bot: np.ndarray | None = None,
    max_wall_displacement_m: float | None = None,
) -> None:
    """Raise ``GeometryValidationError`` on degenerate or invalid wall geometry."""
    n = geom.n
    L = float(geom.base_length or cfg_dict.get("base_length", 0.0))
    min_lumen_frac = float(cfg_dict.get("min_lumen_width_fraction", 0.05))
    ref_w = reference_width
    if ref_w is None:
        ref_w = float(np.mean(np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)))

    cross_widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
    if geom.d_bar < 1e-5 or np.any(cross_widths < 0):
        raise GeometryValidationError(f"Degenerate geometry: d_bar={geom.d_bar:.2e}")
    if np.any(cross_widths < (ref_w * min_lumen_frac)):
        raise GeometryValidationError("Geometry too narrow at a control point.")

    for coords in (geom.top_coords, geom.bot_coords):
        step_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if np.any(step_lengths < (L / max(n, 1)) * 0.1):
            raise GeometryValidationError("Self-intersection detected: boundary polyline collapsed.")

    if geom.top_coords[-1, 0] < (L / 3.0) or geom.bot_coords[-1, 0] < (L / 3.0):
        raise GeometryValidationError("Geometry rejected: outlet curled back past L/3.")

    if max_wall_displacement_m is not None and baseline_top is not None and baseline_bot is not None:
        bt = np.asarray(baseline_top, dtype=float)
        bb = np.asarray(baseline_bot, dtype=float)
        for i in range(n):
            if np.linalg.norm(geom.top_coords[i] - bt[i]) > max_wall_displacement_m:
                raise GeometryValidationError(f"Top wall handle {i} exceeds max displacement.")
            if np.linalg.norm(geom.bot_coords[i] - bb[i]) > max_wall_displacement_m:
                raise GeometryValidationError(f"Bottom wall handle {i} exceeds max displacement.")


def geometry_to_params_override(geom: VesselGeometry) -> Dict[str, Any]:
    """Serialize edited walls into a params dict for ``build_vessel_mesh``."""
    return {
        "geometry_mode": "edited_walls",
        "idx": int(geom.idx),
        "n": int(geom.n),
        "top_coords": geom.top_coords.tolist(),
        "bot_coords": geom.bot_coords.tolist(),
        "level": int(geom.meta.get("level", 0)),
        "curve_type": str(geom.meta.get("curve", "edited")),
        "v_type": str(geom.meta.get("type", "edited")).split("_")[0],
    }


def smooth_wall_curve(coords: np.ndarray, n_dense: int = 400) -> np.ndarray:
    """Dense wall polyline via B-spline (matches Gmsh BSpline preview; no natural-spline overshoot)."""
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if float(s[-1]) < 1e-12:
        return pts.copy()
    try:
        from scipy.interpolate import make_interp_spline

        k = min(3, n - 1)
        t = np.linspace(0.0, float(s[-1]), int(max(n_dense, n * 8)))
        bx = make_interp_spline(s, pts[:, 0], k=k)
        by = make_interp_spline(s, pts[:, 1], k=k)
        return np.column_stack([bx(t), by(t)])
    except Exception:
        return pts.copy()


def relax_wall_coords(
    coords: np.ndarray,
    fixed: frozenset[int] | set[int],
    *,
    n_iters: int = 4,
    omega: float = 0.38,
) -> np.ndarray:
    """Laplacian smooth interior stations; fixed indices stay pinned."""
    out = np.asarray(coords, dtype=np.float64).copy()
    n = out.shape[0]
    pinned = {int(i): out[int(i)].copy() for i in fixed if 0 <= int(i) < n}
    for _ in range(max(1, int(n_iters))):
        nxt = out.copy()
        for i in range(1, n - 1):
            if i in pinned:
                continue
            nxt[i] = (1.0 - omega) * out[i] + omega * 0.5 * (out[i - 1] + out[i + 1])
        out = nxt
        for i, v in pinned.items():
            out[i] = v
    return out


def _drag_influence_weights(
    n: int,
    index: int,
    fixed: frozenset[int] | set[int],
    sigma_stations: float,
) -> np.ndarray:
    """Gaussian drag weights with soft taper near fixed stations (reduces kinks at gold nodes)."""
    j = np.arange(n, dtype=np.float64)
    sigma = max(float(sigma_stations), 1.0)
    w = np.exp(-0.5 * ((j - float(index)) / sigma) ** 2)
    barrier = np.ones(n, dtype=np.float64)
    for fi in fixed:
        fi = int(fi)
        if 0 <= fi < n:
            w[fi] = 0.0
            d = np.abs(j - float(fi))
            barrier *= 1.0 - np.exp(-0.5 * (d / max(2.0, sigma * 0.5)) ** 2)
    w *= np.clip(barrier, 0.0, 1.0)
    return w


def default_fixed_wall_indices(n: int) -> frozenset[int]:
    """Inlet/outlet stations pinned by default."""
    if n < 2:
        return frozenset()
    return frozenset({0, n - 1})


def apply_wall_handle_drag(
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
    *,
    index: int,
    side: str,
    xy: tuple[float, float] | np.ndarray,
    sigma_stations: float = 7.0,
    fixed_top: frozenset[int] | set[int] | None = None,
    fixed_bot: frozenset[int] | set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread handle motion to neighboring stations (Gaussian); fixed stations do not move."""
    top = np.asarray(top_coords, dtype=np.float64).copy()
    bot = np.asarray(bot_coords, dtype=np.float64).copy()
    n = top.shape[0]
    ft = fixed_top if fixed_top is not None else default_fixed_wall_indices(n)
    fb = fixed_bot if fixed_bot is not None else default_fixed_wall_indices(n)
    wall = top if side == "top" else bot
    fixed = ft if side == "top" else fb
    i = int(index)
    if i in fixed:
        return top, bot
    target = np.asarray(xy, dtype=np.float64)
    delta = target - wall[i]
    w = _drag_influence_weights(n, i, fixed, sigma_stations)
    wall += w[:, np.newaxis] * delta
    if side == "top":
        top = relax_wall_coords(wall, ft)
    else:
        bot = relax_wall_coords(wall, fb)
    return top, bot


def default_max_wall_displacement_m(
    reference_width: float,
    cfg_dict: Mapping[str, Any],
) -> float:
    """Generous drag leash (wider than initial 0.5*width)."""
    L = float(cfg_dict.get("base_length", 0.02))
    return max(6.0 * float(reference_width), 0.85 * L)


def subsample_handle_indices(n: int, *, stride: int = 5, exclude_ends: bool = True) -> np.ndarray:
    """Station indices that receive visible drag handles (never inlet/outlet)."""
    lo = 2 if exclude_ends else 0
    hi = n - 3 if exclude_ends else n - 1
    if hi < lo:
        return np.array([], dtype=int)
    return np.arange(lo, hi + 1, stride, dtype=int)


def snapshot_walls_from_params(params: Dict[str, Any], cfg_dict: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: parametric wall polylines for entering edit mode."""
    geom = compute_geometry_from_params(params, cfg_dict)
    return geom.top_coords.copy(), geom.bot_coords.copy()
