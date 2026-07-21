"""HemoRGP Customer Predict App (matplotlib desktop).

Professional light chrome + white graphics viewport:
  - left control rail
  - right graphics: geometry preview (editable in Parametric mode) until Run finishes,
    then mode-specific result views + timeline scrubber (Clot mode)

Launch::

    python -m src.tools.customer_predict_app
    powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\go_customer_predict.ps1
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider

from src.config import VesselConfig
from src.data_gen.lib.customer_geometry_import import (
    DEFAULT_RE,
    CustomerGeometryError,
    build_parametric_customer_graph,
    copy_into_inbox,
    ensure_inbox,
    list_inbox,
    load_customer_geometry,
    preview_points_from_graph,
)
from src.data_gen.lib.vessel_generator import VesselGenerator, make_vessel_params
from src.data_gen.lib.vessel_geometry import (
    VesselGeometry,
    compute_geometry_from_params,
    default_max_wall_displacement_m,
    geometry_to_params_override,
)
from src.inference.customer_pipeline import CustomerDeployPipeline, CustomerTrajectory
from src.tools.customer_predict_metrics import (
    trajectory_scientific_table,
    write_scientific_csv,
)
from src.utils.paths import get_project_root
from src.utils.plot_kinematics_fields import plot_wall_outline
from src.utils.vessel_drag_editor import WallControlPointEditor

# Default horizon: 8 h -> 30000 s (COMSOL / BiochemConfig.t_final).
DEFAULT_HORIZON_H = 8.0
DEFAULT_HORIZON_S = 30000.0
_SECONDS_PER_UI_HOUR = DEFAULT_HORIZON_S / DEFAULT_HORIZON_H

# Light research-tool palette (COMSOL / MATLAB-app inspired).
C = {
    "bg": "#e8edf2",
    "panel": "#ffffff",
    "panel2": "#f2f5f8",
    "border": "#c5d0dc",
    "text": "#1b2430",
    "muted": "#5b6b7c",
    "accent": "#0f766e",
    "accent_dim": "#0d9488",
    "accent_hover": "#115e59",
    "btn_secondary": "#e8eef4",
    "btn_secondary_hover": "#d7e0ea",
    "warn": "#b45309",
    "err": "#b91c1c",
    "ok": "#047857",
    "slider": "#0f766e",
    "viewport": "#ffffff",
    "viewport_edge": "#b8c5d4",
    "plot_text": "#1b2430",
    "plot_muted": "#5b6b7c",
    "wall": "#1f6f9f",
    "fluid": "#6baed6",
    "inlet": "#2ca02c",
    "outlet": "#d62728",
    "clot_cmap": "RdBu_r",
}

# Labels sit above slider tracks so they are not clipped by the rail edge.
_SLIDER_CAPTION_GAP = 0.003
_SLIDER_TRACK_H = 0.018
_SLIDER_SLOT_H = 0.052  # caption + track + padding
_RAIL_RECT = (0.015, 0.075, 0.265, 0.835)  # x, y, w, h in figure fraction


def _fig_to_rail_y(fig_y: float) -> float:
    """Convert figure Y to ax_rail axes fraction."""
    _x, y0, _w, h = _RAIL_RECT
    return (float(fig_y) - y0) / h


def _add_captioned_slider(
    fig: Any,
    rect: list[float],
    caption: str,
    *slider_args: Any,
    **slider_kwargs: Any,
) -> tuple[Any, Slider, Any]:
    """Create a full-width Slider with an empty side label and a caption above."""
    ax = fig.add_axes(rect)
    slider = Slider(ax, "", *slider_args, **slider_kwargs)
    slider.label.set_text("")
    slider.label.set_visible(False)
    x, y, _w, h = rect
    cap = fig.text(
        x,
        y + h + _SLIDER_CAPTION_GAP,
        caption,
        fontsize=8,
        color=C["muted"],
        va="bottom",
        ha="left",
    )
    return ax, slider, cap


def _place_caption(cap: Any, rect: list[float]) -> None:
    x, y, _w, h = rect
    cap.set_position((x, y + h + _SLIDER_CAPTION_GAP))


def _slider_slot(y_bottom: float, *, x: float = 0.042, w: float = 0.215) -> list[float]:
    """Figure-fraction rect for a captioned slider track."""
    return [x, y_bottom, w, _SLIDER_TRACK_H]


# Soft blue lumen -> bright red clot (high-contrast on white viewport).
_CLOT_CMAP = LinearSegmentedColormap.from_list(
    "hemo_clot_pop",
    ["#9ecae1", "#6baed6", "#deebf7", "#fc9272", "#ef3b2c", "#a50f15"],
    N=256,
)
_CLOT_OPEN_COLOR = "#6baed6"
_CLOT_PHI_THRESHOLD = 0.45

_DEMO_PT = (
    get_project_root()
    / "data"
    / "phase_comparison_test"
    / "graphs_biochem"
    / "vessel_0.pt"
)

ViewMode = Literal["preview", "results"]
RunMode = Literal["clot", "clot_velocity", "scientific"]

_RUN_MODE_LABELS = ("Clot", "Clot + Velocity", "Scientific")
_RUN_MODE_BY_LABEL = {
    "Clot": "clot",
    "Clot + Velocity": "clot_velocity",
    "Clot + velocity": "clot_velocity",  # legacy label
    "Scientific": "scientific",
}


def _hours_to_seconds(hours: float) -> float:
    """UI horizon in hours -> seconds. Default 8 h -> 30000 s."""
    return max(float(hours), 0.1) * _SECONDS_PER_UI_HOUR


def _parse_run_mode(label: str) -> RunMode:
    key = str(label).strip()
    if key in _RUN_MODE_BY_LABEL:
        return _RUN_MODE_BY_LABEL[key]  # type: ignore[return-value]
    low = key.lower()
    if "scientific" in low:
        return "scientific"
    if "velocity" in low or "vel" in low:
        return "clot_velocity"
    return "clot"


def _seed_inbox_demo() -> None:
    d = ensure_inbox()
    if list_inbox():
        return
    if _DEMO_PT.is_file():
        dest = d / "vessel_0_demo.pt"
        if not dest.exists():
            shutil.copy2(_DEMO_PT, dest)
            print(f"[i] Seeded inbox with demo graph: {dest.name}", flush=True)


def _style_button(btn: Button, *, primary: bool = False) -> None:
    face = C["accent"] if primary else C["btn_secondary"]
    btn.ax.set_facecolor(face)
    btn.color = face
    btn.hovercolor = C["accent_hover"] if primary else C["btn_secondary_hover"]
    btn.label.set_color("#ffffff" if primary else C["text"])
    btn.label.set_fontsize(9 if primary else 8.5)
    btn.label.set_fontweight("bold" if primary else "normal")
    for spine in btn.ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(1.0)


def _style_slider(slider: Slider, *, label_size: float = 8.5) -> None:
    slider.ax.set_facecolor(C["panel2"])
    slider.label.set_visible(False)
    slider.valtext.set_color(C["text"])
    slider.valtext.set_fontsize(8.5)
    slider.poly.set_color(C["slider"])
    for spine in slider.ax.spines.values():
        spine.set_color(C["border"])


def _style_radio(radio: RadioButtons) -> None:
    radio.ax.set_facecolor(C["panel"])
    for spine in radio.ax.spines.values():
        spine.set_color(C["border"])
    try:
        radio.set_label_props({"color": C["text"], "fontsize": 9})
    except Exception:
        for label in getattr(radio, "labels", []):
            label.set_color(C["text"])
            label.set_fontsize(9)
    try:
        radio.set_radio_props({"edgecolor": C["accent"], "facecolor": C["panel2"]})
    except Exception:
        pass
    try:
        radio.activecolor = C["accent"]
    except Exception:
        pass


def _open_folder(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin":
            import subprocess

            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess

            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"[WARN] could not open folder: {exc}", flush=True)


def _style_viewport(ax: Any, *, title: str | None = None) -> None:
    """Opaque white graphics canvas; titles sit inside the canvas for contrast."""
    ax.set_facecolor(C["viewport"])
    try:
        ax.patch.set_facecolor(C["viewport"])
        ax.patch.set_edgecolor(C["viewport_edge"])
        ax.patch.set_linewidth(1.15)
        ax.patch.set_alpha(1.0)
        ax.patch.set_visible(True)
    except Exception:
        pass
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C["viewport_edge"])
        spine.set_linewidth(1.15)
    # Avoid matplotlib's external title (lands on dark chrome and becomes unreadable).
    ax.set_title("")
    if title:
        ax.text(
            0.5,
            0.985,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=C["plot_text"],
            fontsize=11,
            fontweight="bold",
            zorder=20,
        )


def _style_light_colorbar(cbar: Any, *, label: str) -> None:
    cbar.set_label(label, color=C["plot_muted"], fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color=C["plot_muted"], labelcolor=C["plot_muted"], labelsize=8)
    cbar.ax.set_facecolor(C["viewport"])
    try:
        cbar.outline.set_edgecolor(C["viewport_edge"])  # type: ignore[attr-defined]
    except Exception:
        pass
    for spine in cbar.ax.spines.values():
        spine.set_color(C["viewport_edge"])



class PredictApp:
    def __init__(self, *, require_cuda: bool = True) -> None:
        self.require_cuda = require_cuda
        self.pipeline: CustomerDeployPipeline | None = None
        self.traj: CustomerTrajectory | None = None
        self.view_mode: ViewMode = "preview"
        self.inbox_files: list[Path] = []
        self.selected_inbox_idx = 0
        self.geom_mode = "Inbox"
        self.field_mode: RunMode = "clot"
        self.status = "Ready. Preview shows the loaded geometry - run a prediction when ready."
        self._busy = False
        self._cbars: list[Any] = []
        self._science_rows: list[dict[str, float]] = []
        self._science_csv_path: Path | None = None
        self._science_fig: Any | None = None
        self._section_labels: dict[str, Any] = {}
        self._rail_widgets: dict[str, Any] = {}
        self._slider_captions: dict[str, Any] = {}

        # Parametric / edit state
        self.vessel_cfg = VesselConfig(phase="kinematics")
        self.cfg_dict = dict(VesselGenerator(phase="kinematics")._cfg_dict())
        self.cfg_dict["unit"] = "m"
        self.param_geom: VesselGeometry | None = None
        self.baseline_geom: VesselGeometry | None = None
        self.drag_editor: WallControlPointEditor | None = None
        self._param_sliders: dict[str, Slider] = {}
        self._param_axes: list[Any] = []
        self._suppress_param_cb = False

        _seed_inbox_demo()
        self._refresh_inbox()

        fig_w, fig_h = 14.0, 8.2
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            dpi = 100.0
            fig_w = max(11.0, min(18.0, sw * 0.82 / dpi))
            fig_h = max(6.8, min(11.0, sh * 0.78 / dpi))
        except Exception:
            pass

        plt.rcParams.update(
            {
                "figure.facecolor": C["bg"],
                "axes.facecolor": C["viewport"],
                "axes.edgecolor": C["border"],
                "text.color": C["text"],
                "axes.labelcolor": C["muted"],
                "xtick.color": C["muted"],
                "ytick.color": C["muted"],
                "font.size": 10,
            }
        )

        self.fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C["bg"])
        try:
            self.fig.canvas.manager.set_window_title("HemoRGP Predict")  # type: ignore[union-attr]
        except Exception:
            pass

        self._build_layout()
        self._set_view_mode("preview")
        self._refresh_preview()
        self.fig.canvas.mpl_connect("resize_event", lambda _e: self.fig.canvas.draw_idle())

    def _add_panel(self, rect: list[float]) -> Any:
        ax = self.fig.add_axes(rect)
        ax.set_facecolor(C["panel"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C["border"])
            spine.set_linewidth(1.2)
        return ax

    def _section_label(self, ax: Any, y: float, text: str) -> Any:
        return ax.text(
            0.055,
            y,
            text,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=C["accent"],
            va="top",
            ha="left",
        )

    def _build_layout(self) -> None:
        # Header
        self.ax_header = self._add_panel([0.015, 0.925, 0.97, 0.055])
        self.ax_header.text(
            0.018, 0.52, "HemoRGP", transform=self.ax_header.transAxes,
            fontsize=15, fontweight="bold", color=C["accent"], va="center",
        )
        self.ax_header.text(
            0.132, 0.52, "Predict", transform=self.ax_header.transAxes,
            fontsize=15, fontweight="bold", color=C["text"], va="center",
        )
        self.ax_header.text(
            0.245, 0.52, "Vessel clot forecast", transform=self.ax_header.transAxes,
            fontsize=9, color=C["muted"], va="center",
        )
        self.status_text = self.ax_header.text(
            0.985, 0.52, self.status, transform=self.ax_header.transAxes,
            fontsize=8.5, color=C["muted"], ha="right", va="center",
        )

        # Control rail
        self.ax_rail = self._add_panel([0.015, 0.075, 0.265, 0.835])
        self._section_labels["geometry"] = self._section_label(self.ax_rail, 0.975, "Geometry")

        ax_mode = self.fig.add_axes([0.035, 0.805, 0.225, 0.065])
        ax_mode.set_facecolor(C["panel"])
        self.radio_mode = RadioButtons(ax_mode, ("Inbox", "Parametric"), active=0)
        _style_radio(self.radio_mode)
        self.radio_mode.on_clicked(self._on_mode)
        self._rail_widgets["mode"] = ax_mode

        self.inbox_label = self.fig.text(
            0.035, 0.790, self._inbox_label_text(), fontsize=8, color=C["text"], va="top",
        )

        ax_prev = self.fig.add_axes([0.035, 0.745, 0.068, 0.028])
        ax_next = self.fig.add_axes([0.110, 0.745, 0.068, 0.028])
        ax_browse = self.fig.add_axes([0.185, 0.745, 0.075, 0.028])
        ax_folder = self.fig.add_axes([0.035, 0.708, 0.112, 0.028])
        ax_refresh = self.fig.add_axes([0.155, 0.708, 0.105, 0.028])
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_browse = Button(ax_browse, "Browse")
        self.btn_folder = Button(ax_folder, "Open folder")
        self.btn_refresh = Button(ax_refresh, "Refresh")
        for b in (self.btn_prev, self.btn_next, self.btn_browse, self.btn_folder, self.btn_refresh):
            _style_button(b)
            b.label.set_fontsize(8)
        self.btn_prev.on_clicked(lambda _e: self._cycle_inbox(-1))
        self.btn_next.on_clicked(lambda _e: self._cycle_inbox(1))
        self.btn_browse.on_clicked(self._on_browse)
        self.btn_folder.on_clicked(lambda _e: self._on_open_folder())
        self.btn_refresh.on_clicked(lambda _e: self._on_refresh())

        self.hint_text = self.fig.text(
            0.035, 0.688,
            "Browse opens the Geometries folder.\nUse Open folder to add mesh files.",
            fontsize=7.5, color=C["muted"], va="top",
        )

        # Shape (parametric only) — initial rects; final positions from _layout_rail
        self._section_labels["shape"] = self._section_label(self.ax_rail, 0.62, "Shape")
        ax_w, s_w, cap_w = _add_captioned_slider(
            self.fig, _slider_slot(0.55), "Width (m)",
            0.004, 0.012, valinit=0.008, valstep=0.0005, color=C["slider"],
        )
        ax_b, s_b, cap_b = _add_captioned_slider(
            self.fig, _slider_slot(0.50), "Bend (deg)",
            0.0, 90.0, valinit=20.0, valstep=1.0, color=C["slider"],
        )
        ax_a, s_a, cap_a = _add_captioned_slider(
            self.fig, _slider_slot(0.45), "S-amp (m)",
            0.0, 0.012, valinit=0.0, valstep=0.0005, color=C["slider"],
        )
        self._param_axes = [ax_w, ax_b, ax_a]
        self._param_sliders = {"width": s_w, "bend": s_b, "amp": s_a}
        self._slider_captions.update({"width": cap_w, "bend": cap_b, "amp": cap_a})
        for s in self._param_sliders.values():
            _style_slider(s)
            s.on_changed(self._on_param_slider)

        self.param_hint = self.fig.text(
            0.035, 0.40,
            "S-amp > 0: S-curve (overrides bend). Drag handles to edit.",
            fontsize=7.5, color=C["muted"], va="top",
        )

        # Conditions + run mode (positions set by _layout_rail)
        self._section_labels["conditions"] = self._section_label(self.ax_rail, 0.33, "Conditions")
        ax_re, self.slider_re, cap_re = _add_captioned_slider(
            self.fig, _slider_slot(0.30), "Inlet Re",
            100.0, 900.0, valinit=DEFAULT_RE, valstep=10.0, color=C["slider"],
        )
        _style_slider(self.slider_re)
        ax_h, self.slider_hours, cap_h = _add_captioned_slider(
            self.fig, _slider_slot(0.25), "Sim time (hrs)",
            1.0, 50.0, valinit=DEFAULT_HORIZON_H, valstep=0.5, color=C["slider"],
        )
        _style_slider(self.slider_hours)
        self._rail_widgets["re"] = ax_re
        self._rail_widgets["hours"] = ax_h
        self._slider_captions["re"] = cap_re
        self._slider_captions["hours"] = cap_h

        self._section_labels["run"] = self._section_label(self.ax_rail, 0.185, "Run mode")
        ax_field = self.fig.add_axes([0.035, 0.105, 0.225, 0.072])
        ax_field.set_facecolor(C["panel"])
        self.radio_field = RadioButtons(ax_field, _RUN_MODE_LABELS, active=0)
        _style_radio(self.radio_field)
        self.radio_field.on_clicked(self._on_field)
        self._rail_widgets["field"] = ax_field

        ax_run = self.fig.add_axes([0.035, 0.085, 0.225, 0.034])
        self.btn_run = Button(ax_run, "Run prediction")
        _style_button(self.btn_run, primary=True)
        self.btn_run.on_clicked(self._on_run)
        self._rail_widgets["run"] = ax_run

        # Download in timeline strip (Scientific mode)
        ax_dl = self.fig.add_axes([0.80, 0.095, 0.165, 0.032])
        self.btn_download = Button(ax_dl, "Download CSV")
        _style_button(self.btn_download, primary=False)
        self.btn_download.on_clicked(self._on_download)
        self.btn_download.ax.set_visible(False)

        # Graphics viewports
        self.ax_preview = self.fig.add_axes([0.305, 0.205, 0.665, 0.695])
        self.ax_left = self.fig.add_axes([0.305, 0.205, 0.320, 0.695])
        self.ax_right = self.fig.add_axes([0.650, 0.205, 0.320, 0.695])
        self.ax_science = self.fig.add_axes([0.305, 0.205, 0.665, 0.695])
        for ax in (self.ax_preview, self.ax_left, self.ax_right):
            _style_viewport(ax)
        self.ax_science.set_facecolor(C["viewport"])
        self.ax_science.set_visible(False)
        for spine in self.ax_science.spines.values():
            spine.set_color(C["viewport_edge"])

        self.ax_time_panel = self._add_panel([0.305, 0.075, 0.665, 0.105])
        self.ax_time_panel.text(
            0.02, 0.72, "Timeline", transform=self.ax_time_panel.transAxes,
            fontsize=8.5, fontweight="bold", color=C["accent"], va="center",
        )
        ax_t, self.slider_time, cap_t = _add_captioned_slider(
            self.fig, [0.38, 0.095, 0.40, 0.028], "Frame",
            0.0, 1.0, valinit=0.0, valstep=1.0, color=C["slider"],
        )
        _style_slider(self.slider_time)
        self.slider_time.on_changed(self._on_time)
        self.slider_time.set_active(False)
        self._slider_captions["frame"] = cap_t
        self.time_readout = self.ax_time_panel.text(
            0.98, 0.72, "Preview", transform=self.ax_time_panel.transAxes,
            fontsize=9, color=C["text"], ha="right", va="center",
        )

        self._update_param_widgets_visibility()

    def _set_section_fig_y(self, name: str, fig_y_top: float) -> None:
        """Place a rail section header using figure Y (avoids axes/figure coord mixups)."""
        self._section_labels[name].set_position((0.055, _fig_to_rail_y(fig_y_top)))

    def _place_slider_pair(self, key: str, y_bottom: float) -> None:
        rect = _slider_slot(y_bottom)
        self._rail_widgets[key].set_position(rect)
        _place_caption(self._slider_captions[key], rect)

    def _place_shape_slider(self, key: str, axis_idx: int, y_bottom: float) -> None:
        rect = _slider_slot(y_bottom)
        self._param_axes[axis_idx].set_position(rect)
        _place_caption(self._slider_captions[key], rect)

    def _layout_rail(self, *, parametric: bool) -> None:
        """Single stacked layout in figure coordinates for Inbox and Parametric modes."""
        # Leave clear air under the Geometry block / browse hint.
        cursor = 0.640 if parametric else 0.650
        self.hint_text.set_visible(not parametric)

        if parametric:
            self._set_section_fig_y("shape", cursor)
            cursor -= 0.026
            for i, key in enumerate(("width", "bend", "amp")):
                cursor -= _SLIDER_SLOT_H
                self._place_shape_slider(key, i, cursor)
            cursor -= 0.010
            self.param_hint.set_position((0.035, cursor))
            cursor -= 0.024  # one-line hint

        self._set_section_fig_y("conditions", cursor)
        cursor -= 0.032  # header clearance before first caption
        cursor -= _SLIDER_SLOT_H
        self._place_slider_pair("re", cursor)
        cursor -= _SLIDER_SLOT_H
        self._place_slider_pair("hours", cursor)

        cursor -= 0.022  # clear the last track before Run mode
        self._set_section_fig_y("run", cursor)
        cursor -= 0.014
        radio_h = 0.100 if not parametric else 0.068
        btn_h = 0.034
        gap = 0.008
        radio_y = max(0.118, cursor - radio_h)
        btn_y = max(0.085, radio_y - gap - btn_h)
        self._rail_widgets["field"].set_position([0.035, radio_y, 0.225, radio_h])
        self._rail_widgets["run"].set_position([0.035, btn_y, 0.225, btn_h])

    # --- view mode --------------------------------------------------------------

    def _set_view_mode(self, mode: ViewMode) -> None:
        self.view_mode = mode
        show_prev = mode == "preview"
        self.ax_preview.set_visible(show_prev)
        self.ax_science.set_visible(False)
        if show_prev:
            self.ax_left.set_visible(False)
            self.ax_right.set_visible(False)
            self.time_readout.set_text("Preview")
            self.slider_time.set_active(False)
            self.btn_download.ax.set_visible(False)
        else:
            self._apply_results_layout()
        self.fig.canvas.draw_idle()

    def _apply_results_layout(self) -> None:
        mode = self.field_mode
        self.ax_preview.set_visible(False)
        self.ax_science.set_visible(False)
        if mode == "clot_velocity":
            self.ax_left.set_visible(True)
            self.ax_right.set_visible(True)
            self.ax_left.set_position([0.305, 0.205, 0.320, 0.695])
            self.ax_right.set_position([0.650, 0.205, 0.320, 0.695])
            self.slider_time.set_active(False)
            self.btn_download.ax.set_visible(False)
        else:
            # Clot and Scientific: full-width clot field + timeline scrubber.
            self.ax_left.set_visible(False)
            self.ax_right.set_visible(True)
            self.ax_right.set_position([0.305, 0.205, 0.665, 0.695])
            self.slider_time.set_active(True)
            self.btn_download.ax.set_visible(mode == "scientific")

    def _disconnect_drag(self) -> None:
        if self.drag_editor is not None:
            self.drag_editor.disconnect()
            self.drag_editor = None

    def _update_param_widgets_visibility(self) -> None:
        show = self.geom_mode == "Parametric"
        for ax in self._param_axes:
            ax.set_visible(show)
        self.param_hint.set_visible(show)
        self._section_labels["shape"].set_visible(show)
        for key in ("width", "bend", "amp"):
            self._slider_captions[key].set_visible(show)
        self._layout_rail(parametric=show)

    # --- inbox ------------------------------------------------------------------

    def _refresh_inbox(self) -> None:
        self.inbox_files = list_inbox()
        if self.selected_inbox_idx >= len(self.inbox_files):
            self.selected_inbox_idx = max(0, len(self.inbox_files) - 1)

    def _inbox_label_text(self) -> str:
        if not self.inbox_files:
            return "No files yet - browse or open the folder"
        p = self.inbox_files[self.selected_inbox_idx]
        return f"{self.selected_inbox_idx + 1} / {len(self.inbox_files)}   {p.name}"

    def _set_status(self, msg: str, *, tone: str = "muted") -> None:
        self.status = msg
        color = {"ok": C["ok"], "err": C["err"], "warn": C["warn"], "accent": C["accent"]}.get(
            tone, C["muted"]
        )
        shown = msg if len(msg) < 90 else msg[:87] + "..."
        self.status_text.set_text(shown)
        self.status_text.set_color(color)
        self.fig.canvas.draw_idle()
        print(msg, flush=True)

    def _invalidate_results(self) -> None:
        """Geometry/conditions changed -- leave results view for preview."""
        self.traj = None
        self._science_rows = []
        self._science_csv_path = None
        self._close_science_fig()
        self._set_view_mode("preview")

    def _close_science_fig(self) -> None:
        if self._science_fig is not None:
            try:
                plt.close(self._science_fig)
            except Exception:
                pass
            self._science_fig = None

    def _on_mode(self, label: str) -> None:
        self.geom_mode = label
        self._update_param_widgets_visibility()
        self._invalidate_results()
        self._refresh_preview()
        if label == "Parametric":
            self._set_status(
                "Parametric mode: edit wall handles, then run a prediction.",
                tone="accent",
            )
        else:
            self._set_status("Inbox mode: select a geometry file, then run a prediction.")

    def _on_field(self, label: str) -> None:
        self.field_mode = _parse_run_mode(label)
        if self.view_mode == "results" and self.traj is not None:
            needs_vel = self.field_mode in ("clot_velocity", "scientific")
            has_vel = bool((self.traj.meta or {}).get("include_velocity", False))
            if needs_vel and not has_vel:
                self._set_status(
                    "Re-run the prediction for Clot + Velocity or Scientific mode.",
                    tone="warn",
                )
                return
            if self.field_mode == "scientific" and not self._science_rows:
                self._prepare_scientific_outputs()
            self._apply_results_layout()
            self._render_results()
        else:
            tips = {
                "clot": "clot field over time",
                "clot_velocity": "initial vs final velocity",
                "scientific": "clot timeline + metrics window",
            }
            self._set_status(f"Run mode set to {label} ({tips[self.field_mode]}).", tone="accent")

    def _cycle_inbox(self, delta: int) -> None:
        if self.geom_mode != "Inbox":
            self._set_status("Switch to Inbox mode to change files.", tone="warn")
            return
        if not self.inbox_files:
            self._set_status("Inbox is empty - browse or open the folder.", tone="warn")
            return
        self.selected_inbox_idx = (self.selected_inbox_idx + delta) % len(self.inbox_files)
        self.inbox_label.set_text(self._inbox_label_text())
        self._invalidate_results()
        self._refresh_preview()

    def _on_refresh(self) -> None:
        self._refresh_inbox()
        self.inbox_label.set_text(self._inbox_label_text())
        self._invalidate_results()
        self._refresh_preview()
        n = len(self.inbox_files)
        self._set_status(f"Refreshed geometries: {n} file{'s' if n != 1 else ''}.", tone="ok")

    def _on_open_folder(self) -> None:
        inbox = ensure_inbox()
        _open_folder(inbox)
        self._set_status(f"Opened geometries folder: {inbox}", tone="accent")

    def _on_browse(self, _event: Any) -> None:
        inbox = ensure_inbox()
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            self._set_status(f"Browse unavailable ({exc}). Use Open folder instead.", tone="err")
            return
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(
            title="Select vessel geometry",
            initialdir=str(inbox),
            filetypes=[
                ("Vessel geometries", "*.pt *.msh *.nas"),
                ("HemoRGP graph (.pt)", "*.pt"),
                ("Gmsh mesh (.msh)", "*.msh"),
                ("Nastran (.nas)", "*.nas"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not path:
            return
        try:
            dest = copy_into_inbox(path)
            self._refresh_inbox()
            for i, p in enumerate(self.inbox_files):
                if p.resolve() == dest.resolve():
                    self.selected_inbox_idx = i
                    break
            self.inbox_label.set_text(self._inbox_label_text())
            self.geom_mode = "Inbox"
            try:
                self.radio_mode.set_active(0)
            except Exception:
                pass
            self._update_param_widgets_visibility()
            self._invalidate_results()
            self._refresh_preview()
            self._set_status(f"Loaded {dest.name}.", tone="ok")
        except CustomerGeometryError as exc:
            self._set_status(str(exc), tone="err")

    # --- preview ----------------------------------------------------------------

    def _on_param_slider(self, _val: float) -> None:
        if self._suppress_param_cb or self.geom_mode != "Parametric":
            return
        self._invalidate_results()
        self._rebuild_parametric_geom(from_sliders=True)

    def _rebuild_parametric_geom(self, *, from_sliders: bool) -> None:
        try:
            width = float(self._param_sliders["width"].val)
            bend = math.radians(float(self._param_sliders["bend"].val))
            amp = float(self._param_sliders["amp"].val)
            overrides: dict[str, Any] = {"width": width, "angle_span": bend, "amplitude": amp}
            # S amp drives an S-curve centerline; previously bend>0 forced arc and ignored amp.
            if amp > 1e-9:
                overrides["curve_type"] = "sine"
            elif abs(bend) > 1e-6:
                overrides["curve_type"] = "arc"
            else:
                overrides["curve_type"] = "straight"
            params = make_vessel_params(idx=0, level=0, cfg=self.vessel_cfg, **overrides)
            geom = compute_geometry_from_params(params, self.cfg_dict)
            self.param_geom = geom
            self.baseline_geom = compute_geometry_from_params(params, self.cfg_dict)
            self._draw_parametric_preview(enable_drag=True)
        except Exception as exc:
            self._set_status(f"Parametric preview failed: {exc}", tone="err")

    def _draw_parametric_preview(self, *, enable_drag: bool) -> None:
        self._disconnect_drag()
        self.ax_preview.clear()
        _style_viewport(self.ax_preview)
        if self.param_geom is None:
            self.ax_preview.text(
                0.5, 0.5, "No parametric geometry", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["plot_muted"],
            )
            self.fig.canvas.draw_idle()
            return

        def on_change(geom: VesselGeometry) -> None:
            self.param_geom = geom
            self._invalidate_results()
            if self.drag_editor and self.drag_editor.last_error:
                self._set_status(self.drag_editor.last_error, tone="warn")
            else:
                self._set_status("Walls edited. Run a prediction to remesh and forecast.", tone="accent")

        if enable_drag and self.baseline_geom is not None:
            ref_w = float(self._param_sliders["width"].val)
            self.drag_editor = WallControlPointEditor(
                self.fig,
                self.ax_preview,
                self.param_geom,
                cfg_dict=self.cfg_dict,
                on_change=on_change,
                baseline_top=self.baseline_geom.top_coords,
                baseline_bot=self.baseline_geom.bot_coords,
                max_wall_displacement_m=default_max_wall_displacement_m(ref_w, self.cfg_dict),
                fill_facecolor=C["fluid"],
                fill_alpha=0.9,
            )
        else:
            plot_wall_outline(
                self.ax_preview,
                self.param_geom.top_coords,
                self.param_geom.bot_coords,
                fill_facecolor=C["fluid"],
                fill_alpha=0.9,
            )
        _style_viewport(self.ax_preview, title="Geometry preview - drag handles to reshape")
        self.fig.canvas.draw_idle()

    def _draw_inbox_preview(self) -> None:
        self._disconnect_drag()
        self.ax_preview.clear()
        _style_viewport(self.ax_preview)
        if not self.inbox_files:
            self.ax_preview.text(
                0.5, 0.52, "No geometry loaded", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["plot_muted"], fontsize=13, fontweight="bold",
            )
            self.ax_preview.text(
                0.5, 0.42, "Browse or Open folder, then select a file",
                transform=self.ax_preview.transAxes, ha="center", va="center",
                color=C["plot_muted"], fontsize=9,
            )
            self.fig.canvas.draw_idle()
            return
        path = self.inbox_files[self.selected_inbox_idx]
        try:
            # Lightweight load for preview (short timeline)
            data = load_customer_geometry(path, re_target=float(self.slider_re.val), t_final_s=100.0, n_steps=2)
            pos, inlet, outlet, wall = preview_points_from_graph(data)
            fluid = ~(inlet | outlet | wall)
            if fluid.any():
                self.ax_preview.scatter(
                    pos[fluid, 0], pos[fluid, 1], c=C["fluid"], s=4.0, alpha=0.95,
                    linewidths=0, rasterized=True,
                )
            if wall.any():
                self.ax_preview.scatter(
                    pos[wall, 0], pos[wall, 1], c=C["wall"], s=5.0, label="wall",
                    linewidths=0, rasterized=True,
                )
            if inlet.any():
                self.ax_preview.scatter(
                    pos[inlet, 0], pos[inlet, 1], c=C["inlet"], s=14, label="inlet", zorder=3,
                )
            if outlet.any():
                self.ax_preview.scatter(
                    pos[outlet, 0], pos[outlet, 1], c=C["outlet"], s=14, label="outlet", zorder=3,
                )
            _style_viewport(self.ax_preview, title=f"Geometry preview - {path.name}")
            self.ax_preview.legend(
                loc="upper right", fontsize=8, frameon=True,
                facecolor="#f7fafc", edgecolor=C["viewport_edge"], labelcolor=C["plot_text"],
            )
            self._set_status(f"Preview: {path.name} ({pos.shape[0]:,} nodes).", tone="ok")
        except Exception as exc:
            self.ax_preview.text(
                0.5, 0.5, f"Preview failed:\n{exc}", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["err"], fontsize=9, wrap=True,
            )
            _style_viewport(self.ax_preview)
            self._set_status(str(exc), tone="err")
        self.fig.canvas.draw_idle()

    def _refresh_preview(self) -> None:
        if self.view_mode != "preview":
            self._set_view_mode("preview")
        if self.geom_mode == "Parametric":
            self._rebuild_parametric_geom(from_sliders=True)
        else:
            self._draw_inbox_preview()

    # --- run --------------------------------------------------------------------

    def _ensure_pipeline(self) -> CustomerDeployPipeline:
        if self.pipeline is None:
            self.pipeline = CustomerDeployPipeline(require_cuda=self.require_cuda)
        return self.pipeline

    def _load_geometry(self, re_target: float, t_final_s: float, n_steps: int):
        if self.geom_mode == "Parametric":
            if self.param_geom is None:
                self._rebuild_parametric_geom(from_sliders=True)
            if self.param_geom is None:
                raise CustomerGeometryError("No parametric geometry available.")
            # Prefer edited walls if drag editor (or geom) exists
            override = geometry_to_params_override(self.param_geom)
            # If walls were never dragged and match parametric, still fine
            return build_parametric_customer_graph(
                re_target=re_target,
                t_final_s=t_final_s,
                n_steps=n_steps,
                params_override=override,
            )
        if not self.inbox_files:
            raise CustomerGeometryError(
                "No geometry selected. Browse or Open folder, then pick a file."
            )
        path = self.inbox_files[self.selected_inbox_idx]
        return load_customer_geometry(
            path, re_target=re_target, t_final_s=t_final_s, n_steps=n_steps,
        )

    def _on_run(self, _event: Any) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn_run.label.set_text("Running...")
        self.btn_run.ax.set_facecolor(C["accent_dim"])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        re_target = float(self.slider_re.val)
        hours = float(self.slider_hours.val)
        t_final_s = _hours_to_seconds(hours)
        n_steps = int(max(20, min(120, round(t_final_s / 135.0))))
        include_velocity = self.field_mode in ("clot_velocity", "scientific")

        def progress(msg: str) -> None:
            # Must stay on the main thread (signal/tqdm-safe).
            self._set_status(msg, tone="accent")
            self.fig.canvas.flush_events()

        try:
            # Quiet tqdm so libraries do not register signal handlers mid-run.
            os.environ["BIOCHEM_TQDM"] = "0"
            os.environ["BIOCHEM_QUIET"] = "1"
            progress(
                f"Building geometry (Re={re_target:.0f}, {hours:.1f} h, "
                f"{n_steps} steps)..."
            )
            data = self._load_geometry(re_target, t_final_s, n_steps)
            progress("Loading models (first run can take a minute)...")
            pipe = self._ensure_pipeline()
            self.traj = pipe.run(
                data,
                t_final_s=t_final_s,
                progress=progress,
                include_velocity=include_velocity,
            )
            self._disconnect_drag()
            self._science_rows = []
            self._science_csv_path = None
            if self.field_mode == "scientific":
                self._prepare_scientific_outputs()
            self._set_view_mode("results")
            n = max(self.traj.n_steps - 1, 1)
            self.slider_time.valmin = 0.0
            self.slider_time.valmax = float(n)
            self.slider_time.valstep = 1.0
            self.slider_time.ax.set_xlim(0.0, float(n))
            self.slider_time.set_val(0.0)
            if self.field_mode in ("clot", "scientific"):
                self.slider_time.set_active(True)
            else:
                self.slider_time.set_active(False)
            self._render_results()
            ms = (self.traj.elapsed_s / max(self.traj.n_steps, 1)) * 1000.0
            extra = ""
            if self.field_mode == "scientific" and self._science_csv_path is not None:
                extra = f" CSV ready: {self._science_csv_path.name}."
            self._set_status(
                f"Done in {self.traj.elapsed_s:.1f} s ({ms:.0f} ms/step).{extra}",
                tone="ok",
            )
        except CustomerGeometryError as exc:
            self._set_status(str(exc), tone="err")
        except Exception as exc:
            self._set_status(str(exc), tone="err")
            traceback.print_exc()
        finally:
            self._busy = False
            self.btn_run.label.set_text("Run prediction")
            _style_button(self.btn_run, primary=True)
            self.fig.canvas.draw_idle()

    def _prepare_scientific_outputs(self) -> None:
        assert self.traj is not None
        self._science_rows = trajectory_scientific_table(
            self.traj, seconds_per_ui_hour=_SECONDS_PER_UI_HOUR
        )
        out_dir = get_project_root() / "outputs" / "customer_predict"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"t{int(self.traj.t_sec[-1])}s"
        path = out_dir / f"scientific_metrics_{stamp}.csv"
        write_scientific_csv(path, self._science_rows)
        self._science_csv_path = path
        print(f"[save] Scientific metrics CSV: {path}", flush=True)
        self._open_science_timeseries_window()

    def _open_science_timeseries_window(self) -> None:
        """Popup figure with one panel per metric (not overlaid)."""
        self._close_science_fig()
        if not self._science_rows:
            return
        t_h = np.array([r["t_h"] for r in self._science_rows], dtype=np.float64)
        wall = np.array([r["wall_clot_pct"] for r in self._science_rows], dtype=np.float64)
        vessel = np.array([r["vessel_clot_pct"] for r in self._science_rows], dtype=np.float64)
        occ = np.array([r["max_occlusion_pct"] for r in self._science_rows], dtype=np.float64)
        clot_hop = np.array(
            [r.get("max_clot_lumen_hop", 0.0) for r in self._science_rows], dtype=np.float64
        )
        lumen_hop = np.array(
            [r.get("max_lumen_hop", 0.0) for r in self._science_rows], dtype=np.float64
        )

        fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True, facecolor="white")
        try:
            fig.canvas.manager.set_window_title("HemoGINO Scientific metrics")  # type: ignore[union-attr]
        except Exception:
            pass
        ax0, ax1, ax2 = axes
        ax0.plot(t_h, wall, color="#1f77b4", lw=2.0)
        ax0.set_ylabel("Wall coverage (%)")
        ax0.set_ylim(0.0, max(100.0, float(wall.max()) * 1.05 if wall.size else 100.0))
        ax0.grid(True, alpha=0.3)
        ax0.set_title("Wall coverage over time")

        ax1.plot(t_h, vessel, color="#2bb3a3", lw=2.0)
        ax1.set_ylabel("Vessel coverage (%)")
        ax1.set_ylim(0.0, max(100.0, float(vessel.max()) * 1.05 if vessel.size else 100.0))
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Vessel coverage over time")

        ax2.plot(t_h, occ, color="#d62728", lw=2.0, label="Max lumen-hop occlusion (%)")
        ax2.set_xlabel("Sim time (hrs)")
        ax2.set_ylabel("Occlusion (%)")
        ax2.set_ylim(0.0, max(100.0, float(occ.max()) * 1.05 if occ.size else 100.0))
        ax2.grid(True, alpha=0.3)
        if lumen_hop.size and float(lumen_hop.max()) > 0:
            ax2_r = ax2.twinx()
            ax2_r.plot(t_h, clot_hop, color="#9467bd", lw=1.5, ls="--", label="Max clot hop")
            ax2_r.set_ylabel("Max clot lumen hop")
            ax2_r.set_ylim(0.0, max(1.0, float(lumen_hop.max()) * 1.1))
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_r.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
        else:
            ax2.legend(loc="best", fontsize=8)
        ax2.set_title("Max occlusion (clot lumen-hop / max lumen hop)")
        fig.tight_layout()
        self._science_fig = fig
        fig.show()

    def _on_download(self, _event: Any) -> None:
        if not self._science_rows:
            self._set_status("No scientific results yet. Run in Scientific mode.", tone="warn")
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            # Fall back: open the auto-saved folder
            if self._science_csv_path is not None:
                _open_folder(self._science_csv_path.parent)
                self._set_status(f"Opened results folder ({exc}).", tone="warn")
            else:
                self._set_status(f"Download unavailable: {exc}", tone="err")
            return
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        default = (
            str(self._science_csv_path.name)
            if self._science_csv_path is not None
            else "scientific_metrics.csv"
        )
        path = filedialog.asksaveasfilename(
            title="Save scientific metrics CSV",
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return
        write_scientific_csv(path, self._science_rows)
        self._science_csv_path = Path(path)
        self._set_status(f"Saved {Path(path).name}", tone="ok")

    def _on_time(self, val: float) -> None:
        if self.traj is None or self.view_mode != "results":
            return
        if self.field_mode not in ("clot", "scientific"):
            return
        self._render_frame(int(val))

    def _clear_cbars(self) -> None:
        for cbar in self._cbars:
            try:
                cbar.remove()
            except Exception:
                pass
        self._cbars = []

    def _render_results(self) -> None:
        if self.traj is None:
            return
        if self.field_mode == "clot_velocity":
            self._render_velocity_bookends()
        else:
            # Clot + Scientific share the clot field + timeline.
            self._render_frame(int(self.slider_time.val))
            if self.field_mode == "scientific":
                self.btn_download.ax.set_visible(True)

    def _plot_field(
        self,
        ax: Any,
        values: np.ndarray,
        *,
        cmap: Any,
        vmin: float,
        vmax: float,
        title: str,
        cbar_label: str,
    ) -> None:
        """Node scatter on an opaque white viewport."""
        assert self.traj is not None
        pos = self.traj.pos
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        ax.clear()
        _style_viewport(ax)
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=vals,
            cmap=cmap,
            s=6.0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        _style_viewport(ax, title=title)
        cbar = self.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        _style_light_colorbar(cbar, label=cbar_label)
        self._cbars.append(cbar)

    def _plot_clot_field(self, ax: Any, phi: np.ndarray, *, title: str) -> None:
        """Two-layer clot viz: soft blue lumen, larger bright-red clot nodes on top."""
        assert self.traj is not None
        pos = self.traj.pos
        vals = np.asarray(phi, dtype=np.float64).reshape(-1)
        ax.clear()
        _style_viewport(ax)

        open_m = vals < _CLOT_PHI_THRESHOLD
        clot_m = ~open_m

        if open_m.any():
            ax.scatter(
                pos[open_m, 0],
                pos[open_m, 1],
                c=_CLOT_OPEN_COLOR,
                s=4.5,
                alpha=0.80,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

        # Continuous field underneath for colorbar mapping (low alpha).
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=vals,
            cmap=_CLOT_CMAP,
            s=3.0,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            alpha=0.15,
            rasterized=True,
            zorder=2,
        )

        if clot_m.any():
            # Size and opacity ramp with phi so mature clot reads clearly.
            clot_phi = np.clip(vals[clot_m], 0.0, 1.0)
            sizes = 10.0 + 24.0 * np.power(clot_phi, 1.25)
            ax.scatter(
                pos[clot_m, 0],
                pos[clot_m, 1],
                c=clot_phi,
                cmap=_CLOT_CMAP,
                s=sizes,
                vmin=0.0,
                vmax=1.0,
                linewidths=0.15,
                edgecolors="#7f0000",
                alpha=0.95,
                rasterized=True,
                zorder=5,
            )

        _style_viewport(ax, title=title)
        cbar = self.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        _style_light_colorbar(cbar, label="Clot phi")
        self._cbars.append(cbar)

    def _render_frame(self, index: int) -> None:
        if self.traj is None or self.traj.n_steps < 1:
            return
        self._apply_results_layout()
        fr = self.traj.frame(index)
        t_sec = float(fr["t_sec"])
        hours = t_sec / _SECONDS_PER_UI_HOUR
        self.time_readout.set_text(f"t = {t_sec:.0f} s  ({hours:.2f} hrs)")
        if self.field_mode == "scientific" and self._science_rows:
            last = self._science_rows[-1]
            self.time_readout.set_text(
                f"t = {t_sec:.0f} s ({hours:.2f} hrs)  |  "
                f"final wall={last['wall_clot_pct']:.1f}%  "
                f"vessel={last['vessel_clot_pct']:.1f}%  "
                f"occ={last['max_occlusion_pct']:.1f}%"
            )

        self._clear_cbars()
        phi = np.asarray(fr["phi"], dtype=np.float64)
        self._plot_clot_field(
            self.ax_right,
            phi,
            title=f"Clot fraction  |  {hours:.2f} hrs",
        )
        self.fig.canvas.draw_idle()

    def _render_velocity_bookends(self) -> None:
        if self.traj is None or self.traj.n_steps < 1:
            return
        self._apply_results_layout()
        self._clear_cbars()

        i0, i1 = 0, self.traj.n_steps - 1
        fr0 = self.traj.frame(i0)
        fr1 = self.traj.frame(i1)
        vel0 = np.asarray(fr0["vel_mag"], dtype=np.float64)
        vel1 = np.asarray(fr1["vel_mag"], dtype=np.float64)
        vmax_u = max(
            1e-6,
            float(np.percentile(np.concatenate([vel0, vel1]), 99)) if vel0.size else 1.5,
        )
        h0 = float(fr0["t_sec"]) / _SECONDS_PER_UI_HOUR
        h1 = float(fr1["t_sec"]) / _SECONDS_PER_UI_HOUR
        self.time_readout.set_text(f"Velocity  |  t = {h0:.2f} h  ->  {h1:.2f} h")
        self._plot_field(
            self.ax_left,
            vel0,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_u,
            title=f"Velocity |U|  |  first ({h0:.2f} h)",
            cbar_label="|U| (ND)",
        )
        self._plot_field(
            self.ax_right,
            vel1,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_u,
            title=f"Velocity |U|  |  final ({h1:.2f} h)",
            cbar_label="|U| (ND)",
        )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def _log_startup_device(*, require_cuda: bool) -> None:
    if not require_cuda:
        print("[i] device: cpu", flush=True)
        return
    from src.core_physics.t0_device import require_cuda_device

    import torch

    dev = require_cuda_device()
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    print(f"[i] device: cuda ({name})", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HemoGINO customer predict app")
    ap.add_argument("--cpu", action="store_true", help="Allow CPU (slow; CUDA recommended)")
    args = ap.parse_args(argv)
    require_cuda = not args.cpu
    inbox = ensure_inbox()
    print("[i] HemoRGP Predict", flush=True)
    print(f"[i] Geometries folder: {inbox}", flush=True)
    _log_startup_device(require_cuda=require_cuda)
    app = PredictApp(require_cuda=require_cuda)
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
