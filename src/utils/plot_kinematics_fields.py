"""Matplotlib scalar-field plotting for kinematics u/v/p (shared by demo and eval)."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.config import PredChannels
from src.data_gen.lib.vessel_geometry import smooth_wall_curve


def plot_field(
    fig,
    ax,
    pos,
    val,
    title,
    cmap,
    vmin=None,
    vmax=None,
    *,
    tight_axes: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Plot a scalar field on an unstructured mesh using tripcolor."""
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])

    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)

    mask = max_edge_sq > (np.median(max_edge_sq) * 10.0)
    triang.set_mask(mask)

    tc = ax.tripcolor(triang, val, cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=6)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if tight_axes:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("equal")
    ax.axis("off")
    old_cbar = getattr(ax, "_kin_field_cbar", None)
    if old_cbar is not None:
        try:
            old_cbar.remove()
        except Exception:
            pass
    cbar = fig.colorbar(tc, ax=ax, fraction=0.042, pad=0.02, shrink=0.88)
    ax._kin_field_cbar = cbar


def _si_or_nd_label(base: str, *, si_scale: float | None, unit: str) -> str:
    if si_scale is not None:
        return f"{base} (x {si_scale:.3g} {unit})"
    return f"{base} (ND)"


def plot_wall_outline(
    ax,
    top_coords: np.ndarray,
    bot_coords: np.ndarray,
    *,
    highlight_handles: np.ndarray | None = None,
    fixed_top: set[int] | frozenset[int] | None = None,
    fixed_bot: set[int] | frozenset[int] | None = None,
    return_artists: bool = False,
):
    """Draw top/bot wall polylines and lumen fill; optional handle scatter."""
    top = np.asarray(top_coords, dtype=float)
    bot = np.asarray(bot_coords, dtype=float)
    top_draw = smooth_wall_curve(top)
    bot_draw = smooth_wall_curve(bot)
    ax.clear()
    top_line, = ax.plot(top_draw[:, 0], top_draw[:, 1], "b-", lw=1.4, label="top wall")
    bot_line, = ax.plot(bot_draw[:, 0], bot_draw[:, 1], "r-", lw=1.4, label="bot wall")
    ring = np.vstack([top_draw, bot_draw[::-1]])
    fill = ax.fill(ring[:, 0], ring[:, 1], facecolor="lightblue", alpha=0.35, edgecolor="none")[0]
    from src.data_gen.lib.vessel_geometry import default_fixed_wall_indices

    n = top.shape[0]
    ft = set(fixed_top) if fixed_top is not None else set(default_fixed_wall_indices(n))
    fb = set(fixed_bot) if fixed_bot is not None else set(default_fixed_wall_indices(n))
    top_handles = bot_handles = None
    top_fixed_art = bot_fixed_art = None
    if highlight_handles is not None and len(highlight_handles):
        hi = np.asarray(highlight_handles, dtype=int)
        top_move = [i for i in hi if i not in ft]
        top_fix = [i for i in hi if i in ft]
        bot_move = [i for i in hi if i not in fb]
        bot_fix = [i for i in hi if i in fb]
        if top_move:
            ti = np.asarray(top_move, dtype=int)
            top_handles = ax.scatter(
                top[ti, 0],
                top[ti, 1],
                c="dodgerblue",
                s=120,
                marker="o",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
                picker=8,
            )
        if top_fix:
            ti = np.asarray(top_fix, dtype=int)
            top_fixed_art = ax.scatter(
                top[ti, 0],
                top[ti, 1],
                c="gold",
                s=100,
                marker="D",
                edgecolors="black",
                linewidths=1.2,
                zorder=11,
                picker=8,
            )
        if bot_move:
            bi = np.asarray(bot_move, dtype=int)
            bot_handles = ax.scatter(
                bot[bi, 0],
                bot[bi, 1],
                c="crimson",
                s=120,
                marker="s",
                edgecolors="white",
                linewidths=1.5,
                zorder=10,
                picker=8,
            )
        if bot_fix:
            bi = np.asarray(bot_fix, dtype=int)
            bot_fixed_art = ax.scatter(
                bot[bi, 0],
                bot[bi, 1],
                c="gold",
                s=100,
                marker="D",
                edgecolors="black",
                linewidths=1.2,
                zorder=11,
                picker=8,
            )
    ax.set_aspect("equal")
    ax.axis("off")
    if return_artists:
        return top_line, bot_line, fill, top_handles, bot_handles, top_fixed_art, bot_fixed_art
    return None


def plot_kinematics_uvp(
    fig,
    axes,
    pos: np.ndarray,
    pred: np.ndarray,
    *,
    channel_indices: dict | None = None,
    si_scale: tuple[float, float] | None = None,
    show_si: bool = False,
) -> None:
    """Four-panel u, v, p, speed plot on axes[0..3]."""
    ch = channel_indices or {
        "u": PredChannels.U,
        "v": PredChannels.V,
        "p": PredChannels.P,
    }
    u = pred[:, ch["u"]]
    v = pred[:, ch["v"]]
    p = pred[:, ch["p"]]
    speed = np.hypot(u, v)

    u_ref, p_ref = (None, None)
    if show_si and si_scale is not None:
        u_ref, p_ref = si_scale

    panels = [
        (u, "u", "RdBu_r", _si_or_nd_label("u", si_scale=u_ref, unit="m/s")),
        (v, "v", "RdBu_r", _si_or_nd_label("v", si_scale=u_ref, unit="m/s")),
        (p, "p", "coolwarm", _si_or_nd_label("p", si_scale=p_ref, unit="Pa")),
        (speed, "|U|", "jet", _si_or_nd_label("|U|", si_scale=u_ref, unit="m/s")),
    ]

    for ax, (values, _key, cmap, title) in zip(axes, panels):
        plot_field(fig, ax, pos, values, title, cmap)


def plot_kinematics_speed_pressure(
    fig,
    axes,
    pos: np.ndarray,
    pred: np.ndarray,
    *,
    channel_indices: dict | None = None,
    si_scale: tuple[float, float] | None = None,
    show_si: bool = False,
) -> None:
    """Two-panel |U| and p plot on axes[0..1]."""
    ch = channel_indices or {
        "u": PredChannels.U,
        "v": PredChannels.V,
        "p": PredChannels.P,
    }
    u = pred[:, ch["u"]]
    v = pred[:, ch["v"]]
    p = pred[:, ch["p"]]
    speed = np.hypot(u, v)

    u_ref, p_ref = (None, None)
    if show_si and si_scale is not None:
        u_ref, p_ref = si_scale

    panels = [
        (speed, "|U|", "jet", _si_or_nd_label("|U|", si_scale=u_ref, unit="m/s")),
        (p, "p", "coolwarm", _si_or_nd_label("p", si_scale=p_ref, unit="Pa")),
    ]

    for ax, (values, _key, cmap, title) in zip(axes, panels):
        plot_field(fig, ax, pos, values, title, cmap)
