"""HemoGINO Customer Predict App (matplotlib desktop).

Radiology-inspired dark workspace:
  - left control rail
  - right graphics: geometry preview (editable in Parametric mode) until Run finishes,
    then dual ML field viewports + timeline scrubber

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
from src.utils.paths import get_project_root
from src.utils.plot_kinematics_fields import plot_wall_outline
from src.utils.vessel_drag_editor import WallControlPointEditor

C = {
    "bg": "#0f1419",
    "panel": "#1a2332",
    "panel2": "#243044",
    "border": "#2f3f56",
    "text": "#e8eef6",
    "muted": "#8fa3b8",
    "accent": "#2bb3a3",
    "accent_dim": "#1a7a70",
    "warn": "#e8a838",
    "err": "#e85d5d",
    "ok": "#3ecf8e",
    "slider": "#2bb3a3",
    "viewport": "#121820",
    "wall": "#5ec8ff",
    "fluid": "#3a4a5c",
    "inlet": "#3ecf8e",
    "outlet": "#e8a838",
}

_DEMO_PT = (
    get_project_root()
    / "data"
    / "phase_comparison_test"
    / "graphs_biochem"
    / "vessel_0.pt"
)

ViewMode = Literal["preview", "results"]


def _hours_to_seconds(hours: float) -> float:
    return max(float(hours), 0.1) * 3600.0


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
    face = C["accent"] if primary else C["panel2"]
    btn.ax.set_facecolor(face)
    btn.color = face
    btn.hovercolor = C["accent"] if primary else "#33475f"
    btn.label.set_color(C["bg"] if primary else C["text"])
    btn.label.set_fontsize(10)
    btn.label.set_fontweight("bold" if primary else "normal")
    for spine in btn.ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(1.0)


def _style_slider(slider: Slider) -> None:
    slider.ax.set_facecolor(C["panel2"])
    slider.label.set_color(C["muted"])
    slider.valtext.set_color(C["text"])
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


class PredictApp:
    def __init__(self, *, require_cuda: bool = True) -> None:
        self.require_cuda = require_cuda
        self.pipeline: CustomerDeployPipeline | None = None
        self.traj: CustomerTrajectory | None = None
        self.view_mode: ViewMode = "preview"
        self.inbox_files: list[Path] = []
        self.selected_inbox_idx = 0
        self.geom_mode = "Inbox"
        self.field_mode = "vel + clot"  # or "just clot"
        self.status = "Ready -- preview shows the loaded geometry. Run prediction for ML fields."
        self._busy = False
        self._cbars: list[Any] = []

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
            self.fig.canvas.manager.set_window_title("HemoGINO Predict")  # type: ignore[union-attr]
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

    def _build_layout(self) -> None:
        self.ax_header = self._add_panel([0.015, 0.915, 0.97, 0.07])
        self.ax_header.text(
            0.02, 0.62, "HemoGINO", transform=self.ax_header.transAxes,
            fontsize=16, fontweight="bold", color=C["accent"], va="center",
        )
        self.ax_header.text(
            0.14, 0.62, "Predict", transform=self.ax_header.transAxes,
            fontsize=16, fontweight="bold", color=C["text"], va="center",
        )
        self.ax_header.text(
            0.02, 0.22,
            "Vessel clot forecast  ·  preview geometry, then run ML prediction",
            transform=self.ax_header.transAxes, fontsize=8, color=C["muted"], va="center",
        )
        self.status_text = self.ax_header.text(
            0.98, 0.5, self.status, transform=self.ax_header.transAxes,
            fontsize=9, color=C["muted"], ha="right", va="center",
        )

        self.ax_rail = self._add_panel([0.015, 0.08, 0.26, 0.82])
        self.ax_rail.text(
            0.06, 0.97, "GEOMETRY", transform=self.ax_rail.transAxes,
            fontsize=8, fontweight="bold", color=C["accent"], va="top",
        )

        ax_mode = self.fig.add_axes([0.04, 0.80, 0.20, 0.07])
        ax_mode.set_facecolor(C["panel"])
        self.radio_mode = RadioButtons(ax_mode, ("Inbox", "Parametric"), active=0)
        _style_radio(self.radio_mode)
        self.radio_mode.on_clicked(self._on_mode)

        self.inbox_label = self.fig.text(
            0.04, 0.785, self._inbox_label_text(), fontsize=8, color=C["text"], va="top",
        )

        ax_prev = self.fig.add_axes([0.04, 0.735, 0.055, 0.028])
        ax_next = self.fig.add_axes([0.10, 0.735, 0.055, 0.028])
        ax_browse = self.fig.add_axes([0.16, 0.735, 0.055, 0.028])
        ax_folder = self.fig.add_axes([0.04, 0.695, 0.10, 0.028])
        ax_refresh = self.fig.add_axes([0.15, 0.695, 0.065, 0.028])
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_next = Button(ax_next, "Next")
        self.btn_browse = Button(ax_browse, "Browse")
        self.btn_folder = Button(ax_folder, "Open folder")
        self.btn_refresh = Button(ax_refresh, "Refresh")
        for b in (self.btn_prev, self.btn_next, self.btn_browse, self.btn_folder, self.btn_refresh):
            _style_button(b)
        self.btn_prev.on_clicked(lambda _e: self._cycle_inbox(-1))
        self.btn_next.on_clicked(lambda _e: self._cycle_inbox(1))
        self.btn_browse.on_clicked(self._on_browse)
        self.btn_folder.on_clicked(lambda _e: self._on_open_folder())
        self.btn_refresh.on_clicked(lambda _e: self._on_refresh())

        self.hint_text = self.fig.text(
            0.04, 0.675,
            "Browse starts in Geometries folder.\nOpen folder to drag files in.",
            fontsize=7.5, color=C["muted"], va="top",
        )

        # Parametric sliders (shown only in Parametric mode)
        self.ax_rail.text(
            0.06, 0.58, "SHAPE", transform=self.ax_rail.transAxes,
            fontsize=8, fontweight="bold", color=C["accent"], va="top",
        )
        ax_w = self.fig.add_axes([0.05, 0.52, 0.20, 0.025])
        ax_b = self.fig.add_axes([0.05, 0.475, 0.20, 0.025])
        ax_a = self.fig.add_axes([0.05, 0.43, 0.20, 0.025])
        self._param_axes = [ax_w, ax_b, ax_a]
        self._param_sliders["width"] = Slider(
            ax_w, "Width", 0.004, 0.012, valinit=0.008, valstep=0.0005, color=C["slider"]
        )
        self._param_sliders["bend"] = Slider(
            ax_b, "Bend deg", 0.0, 90.0, valinit=20.0, valstep=1.0, color=C["slider"]
        )
        self._param_sliders["amp"] = Slider(
            ax_a, "S amp", 0.0, 0.012, valinit=0.0, valstep=0.0005, color=C["slider"]
        )
        for s in self._param_sliders.values():
            _style_slider(s)
            s.on_changed(self._on_param_slider)

        self.param_hint = self.fig.text(
            0.04, 0.405,
            "Drag teal/red handles on the preview.\nRight-click pins a handle (gold).",
            fontsize=7.5, color=C["muted"], va="top",
        )

        self.ax_rail.text(
            0.06, 0.34, "CONDITIONS", transform=self.ax_rail.transAxes,
            fontsize=8, fontweight="bold", color=C["accent"], va="top",
        )
        ax_re = self.fig.add_axes([0.05, 0.285, 0.20, 0.025])
        self.slider_re = Slider(
            ax_re, "Inlet Re", 100.0, 900.0, valinit=DEFAULT_RE, valstep=10.0, color=C["slider"]
        )
        _style_slider(self.slider_re)
        ax_h = self.fig.add_axes([0.05, 0.24, 0.20, 0.025])
        self.slider_hours = Slider(
            ax_h, "Horizon (h)", 1.0, 50.0, valinit=25.0, valstep=1.0, color=C["slider"]
        )
        _style_slider(self.slider_hours)

        self.ax_rail.text(
            0.06, 0.205, "RUN MODE", transform=self.ax_rail.transAxes,
            fontsize=8, fontweight="bold", color=C["accent"], va="top",
        )
        ax_field = self.fig.add_axes([0.04, 0.125, 0.20, 0.065])
        ax_field.set_facecolor(C["panel"])
        self.radio_field = RadioButtons(
            ax_field, ("vel + clot", "just clot"), active=0
        )
        _style_radio(self.radio_field)
        self.radio_field.on_clicked(self._on_field)

        ax_run = self.fig.add_axes([0.04, 0.09, 0.20, 0.04])
        self.btn_run = Button(ax_run, "Run prediction")
        _style_button(self.btn_run, primary=True)
        self.btn_run.on_clicked(self._on_run)

        # Preview (full graphics area) + results (dual)
        self.ax_preview = self.fig.add_axes([0.30, 0.22, 0.685, 0.66])
        self.ax_left = self.fig.add_axes([0.30, 0.22, 0.33, 0.66])
        self.ax_right = self.fig.add_axes([0.655, 0.22, 0.33, 0.66])
        for ax in (self.ax_preview, self.ax_left, self.ax_right):
            ax.set_facecolor(C["viewport"])
            ax.set_aspect("equal")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(C["border"])

        self.ax_time_panel = self._add_panel([0.30, 0.08, 0.685, 0.10])
        self.ax_time_panel.text(
            0.02, 0.72, "TIMELINE", transform=self.ax_time_panel.transAxes,
            fontsize=8, fontweight="bold", color=C["accent"], va="center",
        )
        ax_t = self.fig.add_axes([0.36, 0.105, 0.58, 0.035])
        self.slider_time = Slider(
            ax_t, "t", 0.0, 1.0, valinit=0.0, valstep=1.0, color=C["slider"]
        )
        _style_slider(self.slider_time)
        self.slider_time.on_changed(self._on_time)
        self.slider_time.set_active(False)
        self.time_readout = self.ax_time_panel.text(
            0.98, 0.72, "preview mode", transform=self.ax_time_panel.transAxes,
            fontsize=9, color=C["text"], ha="right", va="center",
        )

        self._update_param_widgets_visibility()

    # --- view mode --------------------------------------------------------------

    def _set_view_mode(self, mode: ViewMode) -> None:
        self.view_mode = mode
        show_prev = mode == "preview"
        self.ax_preview.set_visible(show_prev)
        if show_prev:
            self.ax_left.set_visible(False)
            self.ax_right.set_visible(False)
            self.time_readout.set_text("preview mode")
            self.slider_time.set_active(False)
        else:
            clot_only = self.field_mode == "just clot"
            # just clot: full-width right panel; vel+clot: dual
            self.ax_left.set_visible(not clot_only)
            self.ax_right.set_visible(True)
            if clot_only:
                self.ax_right.set_position([0.30, 0.22, 0.685, 0.66])
            else:
                self.ax_left.set_position([0.30, 0.22, 0.33, 0.66])
                self.ax_right.set_position([0.655, 0.22, 0.33, 0.66])
        self.fig.canvas.draw_idle()

    def _disconnect_drag(self) -> None:
        if self.drag_editor is not None:
            self.drag_editor.disconnect()
            self.drag_editor = None

    def _update_param_widgets_visibility(self) -> None:
        show = self.geom_mode == "Parametric"
        for ax in self._param_axes:
            ax.set_visible(show)
        self.param_hint.set_visible(show)

    # --- inbox ------------------------------------------------------------------

    def _refresh_inbox(self) -> None:
        self.inbox_files = list_inbox()
        if self.selected_inbox_idx >= len(self.inbox_files):
            self.selected_inbox_idx = max(0, len(self.inbox_files) - 1)

    def _inbox_label_text(self) -> str:
        if not self.inbox_files:
            return "No files yet  ·  Browse or Open folder"
        p = self.inbox_files[self.selected_inbox_idx]
        return f"{self.selected_inbox_idx + 1}/{len(self.inbox_files)}  {p.name}"

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
        self._set_view_mode("preview")

    def _on_mode(self, label: str) -> None:
        self.geom_mode = label
        self._update_param_widgets_visibility()
        self._invalidate_results()
        self._refresh_preview()
        if label == "Parametric":
            self._set_status(
                "Parametric: drag wall handles on the preview, then Run prediction.",
                tone="accent",
            )
        else:
            self._set_status("Inbox: preview shows the selected file. Run when ready.")

    def _on_field(self, label: str) -> None:
        self.field_mode = "just clot" if "just" in label.lower() else "vel + clot"
        if self.view_mode == "results" and self.traj is not None:
            self._render_frame(int(self.slider_time.val))
        else:
            tip = "faster (skips velocity corrector)" if self.field_mode == "just clot" else "velocity + clot"
            self._set_status(f"Run mode: {self.field_mode} ({tip}).", tone="accent")

    def _cycle_inbox(self, delta: int) -> None:
        if self.geom_mode != "Inbox":
            self._set_status("Switch to Inbox to change files.", tone="warn")
            return
        if not self.inbox_files:
            self._set_status("Inbox empty -- Browse or Open folder.", tone="warn")
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
        self._set_status(f"Geometries refreshed: {len(self.inbox_files)} file(s).", tone="ok")

    def _on_open_folder(self) -> None:
        inbox = ensure_inbox()
        _open_folder(inbox)
        self._set_status(f"Opened folder: {inbox}", tone="accent")

    def _on_browse(self, _event: Any) -> None:
        inbox = ensure_inbox()
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            self._set_status(f"Browse unavailable ({exc}). Use Open folder.", tone="err")
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
                ("HemoGINO graph (.pt)", "*.pt"),
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
            self._set_status(f"Loaded {dest.name}. Preview updated.", tone="ok")
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
            if abs(bend) > 1e-6:
                overrides["curve_type"] = "arc"
            elif amp > 0:
                overrides["curve_type"] = "sine"
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
        self.ax_preview.set_facecolor(C["viewport"])
        if self.param_geom is None:
            self.ax_preview.text(
                0.5, 0.5, "No parametric geometry", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["muted"],
            )
            self.ax_preview.axis("off")
            self.fig.canvas.draw_idle()
            return

        def on_change(geom: VesselGeometry) -> None:
            self.param_geom = geom
            self._invalidate_results()
            if self.drag_editor and self.drag_editor.last_error:
                self._set_status(self.drag_editor.last_error, tone="warn")
            else:
                self._set_status("Walls edited -- Run prediction to remesh and forecast.", tone="accent")

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
            )
        else:
            plot_wall_outline(
                self.ax_preview,
                self.param_geom.top_coords,
                self.param_geom.bot_coords,
            )
        self.ax_preview.set_title(
            "Geometry preview  ·  drag handles to reshape",
            color=C["text"], fontsize=11, pad=8,
        )
        self.ax_preview.set_aspect("equal")
        self.ax_preview.axis("off")
        for spine in self.ax_preview.spines.values():
            spine.set_visible(True)
            spine.set_color(C["border"])
        self.fig.canvas.draw_idle()

    def _draw_inbox_preview(self) -> None:
        self._disconnect_drag()
        self.ax_preview.clear()
        self.ax_preview.set_facecolor(C["viewport"])
        if not self.inbox_files:
            self.ax_preview.text(
                0.5, 0.52, "No geometry loaded", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["muted"], fontsize=13, fontweight="bold",
            )
            self.ax_preview.text(
                0.5, 0.42, "Browse or Open folder, then select a file",
                transform=self.ax_preview.transAxes, ha="center", va="center",
                color=C["border"], fontsize=9,
            )
            self.ax_preview.axis("off")
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
                    pos[fluid, 0], pos[fluid, 1], c=C["fluid"], s=2.5, alpha=0.85, rasterized=True,
                )
            if wall.any():
                self.ax_preview.scatter(
                    pos[wall, 0], pos[wall, 1], c=C["wall"], s=4.0, label="wall", rasterized=True,
                )
            if inlet.any():
                self.ax_preview.scatter(
                    pos[inlet, 0], pos[inlet, 1], c=C["inlet"], s=12, label="inlet", zorder=3,
                )
            if outlet.any():
                self.ax_preview.scatter(
                    pos[outlet, 0], pos[outlet, 1], c=C["outlet"], s=12, label="outlet", zorder=3,
                )
            self.ax_preview.set_aspect("equal")
            self.ax_preview.axis("off")
            self.ax_preview.set_title(
                f"Geometry preview  ·  {path.name}",
                color=C["text"], fontsize=11, pad=8,
            )
            leg = self.ax_preview.legend(
                loc="upper right", fontsize=8, frameon=True,
                facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"],
            )
            _ = leg
            self._set_status(f"Previewing {path.name} ({pos.shape[0]} nodes).", tone="ok")
        except Exception as exc:
            self.ax_preview.text(
                0.5, 0.5, f"Preview failed:\n{exc}", transform=self.ax_preview.transAxes,
                ha="center", va="center", color=C["err"], fontsize=9, wrap=True,
            )
            self.ax_preview.axis("off")
            self._set_status(str(exc), tone="err")
        for spine in self.ax_preview.spines.values():
            spine.set_visible(True)
            spine.set_color(C["border"])
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
        include_velocity = self.field_mode == "vel + clot"

        def progress(msg: str) -> None:
            # Must stay on the main thread (signal/tqdm-safe).
            self._set_status(msg, tone="accent")
            self.fig.canvas.flush_events()

        try:
            # Quiet tqdm so libraries do not register signal handlers mid-run.
            os.environ["BIOCHEM_TQDM"] = "0"
            os.environ["BIOCHEM_QUIET"] = "1"
            progress(
                f"Building geometry (Re={re_target:.0f}, {hours:.0f}h, {n_steps} steps"
                f", mode={self.field_mode})..."
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
            self._set_view_mode("results")
            n = max(self.traj.n_steps - 1, 1)
            self.slider_time.valmin = 0.0
            self.slider_time.valmax = float(n)
            self.slider_time.valstep = 1.0
            self.slider_time.ax.set_xlim(0.0, float(n))
            self.slider_time.set_val(0.0)
            self.slider_time.set_active(True)
            self._render_frame(0)
            ms = (self.traj.elapsed_s / max(self.traj.n_steps, 1)) * 1000.0
            self._set_status(
                f"Done in {self.traj.elapsed_s:.1f}s ({ms:.0f} ms/step). Scrub timeline.",
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

    def _on_time(self, val: float) -> None:
        if self.traj is None or self.view_mode != "results":
            return
        self._render_frame(int(val))

    def _clear_cbars(self) -> None:
        for cbar in self._cbars:
            try:
                cbar.remove()
            except Exception:
                pass
        self._cbars = []

    def _render_frame(self, index: int) -> None:
        if self.traj is None or self.traj.n_steps < 1:
            return
        # Keep layout in sync with run mode
        clot_only = self.field_mode == "just clot" or not bool(
            (self.traj.meta or {}).get("include_velocity", True)
        )
        self.ax_left.set_visible(not clot_only)
        self.ax_right.set_visible(True)
        if clot_only:
            self.ax_right.set_position([0.30, 0.22, 0.685, 0.66])
        else:
            self.ax_left.set_position([0.30, 0.22, 0.33, 0.66])
            self.ax_right.set_position([0.655, 0.22, 0.33, 0.66])

        fr = self.traj.frame(index)
        pos = self.traj.pos
        t_sec = float(fr["t_sec"])
        hours = t_sec / 3600.0
        self.time_readout.set_text(f"t = {t_sec:.0f} s   ({hours:.2f} h)")

        self._clear_cbars()
        self.ax_left.clear()
        self.ax_right.clear()

        phi = np.asarray(fr["phi"], dtype=np.float64)
        vmax_phi = max(1.0, float(np.percentile(phi, 99.5)) if phi.size else 1.0)

        if not clot_only:
            vel = np.asarray(fr["vel_mag"], dtype=np.float64)
            vmax_u = max(1.0, float(np.percentile(vel, 99))) if vel.size else 1.5
            sc0 = self.ax_left.scatter(
                pos[:, 0], pos[:, 1], c=vel, cmap="coolwarm", s=3.0,
                vmin=0.0, vmax=vmax_u, rasterized=True,
            )
            self.ax_left.set_aspect("equal")
            self.ax_left.axis("off")
            self.ax_left.set_facecolor(C["viewport"])
            self.ax_left.set_title(
                f"Velocity  |U|   ·   {hours:.2f} h", color=C["text"], fontsize=11, pad=8,
            )
            c0 = self.fig.colorbar(sc0, ax=self.ax_left, fraction=0.046, pad=0.02)
            c0.set_label("|U| ND", color=C["muted"], fontsize=8)
            c0.ax.yaxis.set_tick_params(color=C["muted"], labelcolor=C["muted"])
            c0.outline.set_edgecolor(C["border"])  # type: ignore[attr-defined]
            self._cbars.append(c0)
            for spine in self.ax_left.spines.values():
                spine.set_visible(True)
                spine.set_color(C["border"])

        sc1 = self.ax_right.scatter(
            pos[:, 0], pos[:, 1], c=phi, cmap="magma", s=3.0,
            vmin=0.0, vmax=vmax_phi, rasterized=True,
        )
        self.ax_right.set_aspect("equal")
        self.ax_right.axis("off")
        self.ax_right.set_facecolor(C["viewport"])
        self.ax_right.set_title(
            f"Clot phi   ·   {hours:.2f} h", color=C["text"], fontsize=11, pad=8,
        )
        c1 = self.fig.colorbar(sc1, ax=self.ax_right, fraction=0.046, pad=0.02)
        c1.set_label("clot phi", color=C["muted"], fontsize=8)
        c1.ax.yaxis.set_tick_params(color=C["muted"], labelcolor=C["muted"])
        c1.outline.set_edgecolor(C["border"])  # type: ignore[attr-defined]
        self._cbars.append(c1)
        for spine in self.ax_right.spines.values():
            spine.set_visible(True)
            spine.set_color(C["border"])
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HemoGINO customer predict app")
    ap.add_argument("--cpu", action="store_true", help="Allow CPU (slow; CUDA recommended)")
    args = ap.parse_args(argv)
    inbox = ensure_inbox()
    print("[i] HemoGINO Predict", flush=True)
    print(f"[i] Geometries folder: {inbox}", flush=True)
    app = PredictApp(require_cuda=not args.cpu)
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
