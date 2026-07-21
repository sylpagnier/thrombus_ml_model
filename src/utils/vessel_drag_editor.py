"""Matplotlib wall control-point drag editor (no torch/Gmsh)."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from src.data_gen.lib.vessel_geometry import (
    GeometryValidationError,
    VesselGeometry,
    apply_wall_handle_drag,
    compute_geometry_from_walls,
    default_fixed_wall_indices,
    subsample_handle_indices,
    validate_geometry,
)
from src.utils.plot_kinematics_fields import plot_wall_outline


class WallControlPointEditor:
    """Drag interior wall station handles on a geometry axes."""

    def __init__(
        self,
        fig,
        ax,
        geom: VesselGeometry,
        *,
        cfg_dict: dict,
        on_change: Callable[[VesselGeometry], None],
        baseline_top: np.ndarray | None = None,
        baseline_bot: np.ndarray | None = None,
        max_wall_displacement_m: float | None = None,
        drag_sigma_stations: float = 8.5,
        pick_radius_px: float = 28.0,
        fill_facecolor: str = "lightblue",
        fill_alpha: float = 0.35,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.cfg_dict = cfg_dict
        self.on_change = on_change
        self.baseline_top = baseline_top
        self.baseline_bot = baseline_bot
        self.max_wall_displacement_m = max_wall_displacement_m
        self.drag_sigma_stations = float(drag_sigma_stations)
        self.pick_radius_px = pick_radius_px
        self.fill_facecolor = fill_facecolor
        self.fill_alpha = float(fill_alpha)
        self.geom = geom
        self.handle_indices = subsample_handle_indices(geom.n)
        n = geom.n
        self.fixed_top: set[int] = set(default_fixed_wall_indices(n))
        self.fixed_bot: set[int] = set(default_fixed_wall_indices(n))
        self._drag_state: dict | None = None
        self._cids: list[int] = []
        self._top_line: Line2D | None = None
        self._bot_line: Line2D | None = None
        self._fill: Polygon | None = None
        self._top_handles = None
        self._bot_handles = None
        self._top_fixed_art = None
        self._bot_fixed_art = None
        self._last_error: str | None = None
        self.set_geometry(geom)
        self.connect()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def n_fixed_handles(self) -> int:
        hi = set(int(i) for i in self.handle_indices)
        return len((self.fixed_top | self.fixed_bot) & hi)

    def set_geometry(self, geom: VesselGeometry) -> None:
        self.geom = geom
        n = geom.n
        ends = default_fixed_wall_indices(n)
        self.fixed_top |= set(ends)
        self.fixed_bot |= set(ends)
        self._redraw_polylines()

    def _redraw_polylines(self) -> None:
        for art in (
            self._top_line,
            self._bot_line,
            self._fill,
            self._top_handles,
            self._bot_handles,
            self._top_fixed_art,
            self._bot_fixed_art,
        ):
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
        artists = plot_wall_outline(
            self.ax,
            self.geom.top_coords,
            self.geom.bot_coords,
            highlight_handles=self.handle_indices,
            fixed_top=self.fixed_top,
            fixed_bot=self.fixed_bot,
            return_artists=True,
            fill_facecolor=self.fill_facecolor,
            fill_alpha=self.fill_alpha,
        )
        (
            self._top_line,
            self._bot_line,
            self._fill,
            self._top_handles,
            self._bot_handles,
            self._top_fixed_art,
            self._bot_fixed_art,
        ) = artists
        self.ax.autoscale_view()
        self.ax.set_aspect("equal")

    def _handle_scatter_groups(self) -> Iterable[tuple[str, object, np.ndarray]]:
        hi = self.handle_indices
        if self._top_handles is not None:
            move = [i for i in hi if i not in self.fixed_top]
            if move:
                yield "top", self._top_handles, np.asarray(move, dtype=int)
        if self._top_fixed_art is not None:
            fix = [i for i in hi if i in self.fixed_top]
            if fix:
                yield "top", self._top_fixed_art, np.asarray(fix, dtype=int)
        if self._bot_handles is not None:
            move = [i for i in hi if i not in self.fixed_bot]
            if move:
                yield "bot", self._bot_handles, np.asarray(move, dtype=int)
        if self._bot_fixed_art is not None:
            fix = [i for i in hi if i in self.fixed_bot]
            if fix:
                yield "bot", self._bot_fixed_art, np.asarray(fix, dtype=int)

    def _pick_handle(self, event) -> Tuple[int, str] | None:
        if event.inaxes is not self.ax or event.x is None or event.y is None:
            return None
        best: Tuple[float, int, str] | None = None
        for side, scatter, indices in self._handle_scatter_groups():
            if scatter is None:
                continue
            xy = scatter.get_offsets()
            for j, idx in enumerate(indices):
                if j >= len(xy):
                    continue
                px, py = self.ax.transData.transform(tuple(xy[j]))
                dist = float(np.hypot(px - event.x, py - event.y))
                if dist <= self.pick_radius_px and (best is None or dist < best[0]):
                    best = (dist, int(idx), side)
        return (best[1], best[2]) if best else None

    def _toggle_fixed(self, idx: int, side: str) -> None:
        n = self.geom.n
        if idx in (0, n - 1):
            return
        fixed = self.fixed_top if side == "top" else self.fixed_bot
        if idx in fixed:
            fixed.discard(idx)
        else:
            fixed.add(idx)
        self._last_error = None
        self._redraw_polylines()
        self.on_change(self.geom)

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        picked = self._pick_handle(event)
        if picked is None:
            return
        idx, side = picked
        if getattr(event, "button", 1) == 3:
            self._toggle_fixed(idx, side)
            return
        if getattr(event, "button", 1) != 1:
            return
        fixed = self.fixed_top if side == "top" else self.fixed_bot
        if idx in fixed:
            return
        self._drag_state = {
            "idx": idx,
            "side": side,
            "prev_top": self.geom.top_coords.copy(),
            "prev_bot": self.geom.bot_coords.copy(),
        }

    def _on_motion(self, event) -> None:
        if self._drag_state is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._apply_drag(self._drag_state["idx"], self._drag_state["side"], (event.xdata, event.ydata))

    def _on_release(self, _event) -> None:
        self._drag_state = None

    def _apply_drag(self, i: int, side: str, xy: Tuple[float, float]) -> None:
        fixed = self.fixed_top if side == "top" else self.fixed_bot
        if i in fixed:
            return
        prev_top = self.geom.top_coords.copy()
        prev_bot = self.geom.bot_coords.copy()
        top, bot = apply_wall_handle_drag(
            prev_top,
            prev_bot,
            index=i,
            side=side,
            xy=xy,
            sigma_stations=self.drag_sigma_stations,
            fixed_top=frozenset(self.fixed_top),
            fixed_bot=frozenset(self.fixed_bot),
        )
        try:
            new_geom = compute_geometry_from_walls(
                top,
                bot,
                idx=self.geom.idx,
                unit=self.geom.unit,
                params=self.geom.meta,
                base_length=self.geom.base_length,
            )
            validate_geometry(
                new_geom,
                self.cfg_dict,
                baseline_top=self.baseline_top,
                baseline_bot=self.baseline_bot,
                max_wall_displacement_m=self.max_wall_displacement_m,
            )
            self.geom = new_geom
            self._last_error = None
            self._redraw_polylines()
            self.on_change(self.geom)
        except GeometryValidationError as exc:
            self._last_error = str(exc)
            self.geom = compute_geometry_from_walls(
                prev_top,
                prev_bot,
                idx=self.geom.idx,
                unit=self.geom.unit,
                params=self.geom.meta,
                base_length=self.geom.base_length,
            )
            self._redraw_polylines()

    def connect(self) -> None:
        self.disconnect()
        self._cids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_press),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def disconnect(self) -> None:
        for cid in self._cids:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []
        self._drag_state = None
