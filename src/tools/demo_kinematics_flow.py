"""
Interactive kinematics flow demo: parametric vessel geometry + GINO-DEQ inference.



Requires Gmsh on PATH (same as datagen). Checkpoint rheology must match ``--rheology``.
First ``Update flow`` may take several seconds on CPU.



**Edit walls workflow:** click **Edit walls** to drag interior wall station handles
(top=blue circles, bottom=red squares). Inlet/outlet stations (i=0, i=n-1) are pinned.
Drag updates the polyline preview only; click **Update flow** to mesh, infer, and refresh
|U| and p panels. Validation failures revert the last drag and appear in the status line.
Click **Use sliders** to return to parametric mode (freeform edits are discarded unless
you mesh first). **Randomize** in edit mode switches back to parametric and clears edits.



Run::



    python -m src.tools.demo_kinematics_flow
    python -m src.bin.main inspect flow -- --checkpoint outputs/kinematics/kinematics_best.pth --rheology carreau
"""



from __future__ import annotations



import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple



import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider
from torch_geometric.data import Data



from src.config import PhysicsConfig, VesselConfig
from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    build_vessel_mesh,
    make_vessel_params,
    recompute_pathology_offsets,
)
from src.data_gen.lib.vessel_geometry import (
    VesselGeometry,
    compute_geometry_from_params,
    default_max_wall_displacement_m,
    geometry_to_params_override,
    snapshot_walls_from_params,
    subsample_handle_indices,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root
from src.utils.plot_kinematics_fields import plot_kinematics_speed_pressure, plot_wall_outline
from src.utils.vessel_drag_editor import WallControlPointEditor





class GeometryBuildError(RuntimeError):
    """Raised when Gmsh mesh generation fails."""





GeometryMode = Literal["parametric", "edited_walls"]





@dataclass

class DemoState:
    params: Dict[str, Any]
    cfg_dict: Dict[str, Any]
    phys_cfg: PhysicsConfig
    model: Any
    device: torch.device
    geometry_mode: GeometryMode = "parametric"
    geom: VesselGeometry | None = None
    baseline_geom: VesselGeometry | None = None
    drag_editor: WallControlPointEditor | None = None
    level: int = 0
    mesh_coarse_factor: float = 1.0
    pathology_strength: float = 0.5
    pathology_loc: float = 0.5
    last_data: Data | None = None
    last_pred: torch.Tensor | None = None
    last_meta: Dict[str, Any] | None = None
    last_mesh: meshio.Mesh | None = None
    last_error: str | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    vessel_cfg: VesselConfig = field(default_factory=lambda: VesselConfig(phase="kinematics"))





def _default_params(level: int, cfg: VesselConfig, rng: np.random.Generator) -> Dict[str, Any]:
    return make_vessel_params(idx=0, level=level, cfg=cfg, rng=rng)





def _sync_derived_params(params: Dict[str, Any], cfg: VesselConfig) -> Dict[str, Any]:
    out = dict(params)
    curve_type = str(out.get("curve_type", "straight"))
    v_type = str(out.get("v_type", "straight"))
    width = float(out.get("width", cfg.width_min))
    L = cfg.base_length



    if curve_type == "straight":
        out["angle_span"] = 0.0
        out["amplitude"] = 0.0
    elif curve_type in ("arc", "hook"):
        out["amplitude"] = 0.0
        offsets = np.array(out.get("offsets", []), dtype=float)
        max_half_width = (width / 2.0) + max(0.0, float(np.max(offsets) if offsets.size else 0.0)) + (0.08 * width)
        min_safe_radius = 1.6 * max_half_width
        max_safe_angle_span = L / min_safe_radius
        out["angle_span"] = min(float(out.get("angle_span", 0.0)), max_safe_angle_span)
    else:
        out["angle_span"] = 0.0



    if v_type == "straight":
        out["path_loc"] = 2
        out["offsets"] = [0.0] * cfg.num_ctrl_pts



    return out





def _build_mesh_params(state: DemoState, params: Dict[str, Any]) -> Dict[str, Any]:
    if state.geometry_mode == "edited_walls" and state.geom is not None:
        return geometry_to_params_override(state.geom)
    return params





def _build_mesh_files(params: Dict[str, Any], cfg_dict: Dict[str, Any], work_dir: Path) -> Tuple[Path, Path]:
    idx, success, error_msg = build_vessel_mesh(params, cfg_dict, work_dir)
    if not success:
        raise GeometryBuildError(error_msg or "mesh build failed")
    msh_path = work_dir / f"vessel_{idx}.msh"
    json_path = work_dir / f"vessel_{idx}.json"
    if not msh_path.exists() or not json_path.exists():
        raise GeometryBuildError(f"missing mesh sidecar after build (idx={idx})")
    return msh_path, json_path





def _mesh_and_graph(
    params: Dict[str, Any],
    cfg_dict: Dict[str, Any],
    builder: MeshToGraph,
) -> Tuple[Data, Dict[str, Any], meshio.Mesh]:
    with tempfile.TemporaryDirectory(prefix="kin_demo_") as tmp:
        work = Path(tmp)
        msh_path, json_path = _build_mesh_files(params, cfg_dict, work)
        mesh = meshio.read(msh_path)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        data = builder.process_mesh(mesh, meta, stem="demo")
        if data is None:
            raise GeometryBuildError("process_mesh returned None (no triangles or wall nodes)")
        return data, meta, mesh





def _predict(model, data: Data, device: torch.device) -> torch.Tensor:
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        return predict_kinematics(model, data)





def _plot_geometry_mesh(ax, mesh: meshio.Mesh | None, meta: Dict[str, Any] | None) -> None:
    if mesh is None:
        return
    nodes = mesh.points[:, :2]
    tris = mesh.cells_dict.get("triangle")
    if tris is None:
        return
    poly = PolyCollection(
        nodes[tris], edgecolors="none", facecolors="lightsteelblue", alpha=0.25, linewidths=0.0
    )
    ax.add_collection(poly)





def _geometry_panel_title(state: DemoState) -> str:
    if state.geom is None:
        return "Lumen geometry"
    title = f"Lumen walls | mode={state.geometry_mode}"
    if state.geometry_mode == "edited_walls":
        title += " | drag blue/red; R-click fix (gold)"
    if state.geom.d_inlet:
        title += f" | d_inlet={state.geom.d_inlet * 100:.2f} cm"
    return title


def _plot_geometry_panel(
    ax,
    state: DemoState,
    *,
    show_mesh: bool = False,
    show_handles: bool = False,
) -> None:
    """Redraw geometry ax (no drag). Do not call while ``drag_editor`` owns the ax."""
    ax.clear()
    if state.geom is not None:
        hi = subsample_handle_indices(state.geom.n) if show_handles else None
        plot_wall_outline(
            ax,
            state.geom.top_coords,
            state.geom.bot_coords,
            highlight_handles=hi,
        )
        ax.set_title(_geometry_panel_title(state), fontsize=10)
    else:
        ax.text(0.5, 0.5, "No geometry", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    if show_mesh and state.last_mesh is not None:
        _plot_geometry_mesh(ax, state.last_mesh, state.last_meta)





def _pos_si_from_data(data: Data) -> np.ndarray:
    d_bar = float(data.d_bar.item()) if hasattr(data, "d_bar") else 1.0
    pos_nd = data.x[:, :2].detach().cpu().numpy()
    return pos_nd * d_bar





def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)





def _cfg_dict_from_generator(gen: VesselGenerator, *, mesh_coarse_factor: float = 1.0) -> Dict[str, Any]:
    cfg = dict(gen._cfg_dict())
    cfg["unit"] = "m"
    if mesh_coarse_factor != 1.0:
        cfg["mesh_lc"] = float(cfg["mesh_lc"]) * float(mesh_coarse_factor)
    return cfg





def _params_from_widgets(
    state: DemoState,
    sliders: Dict[str, Slider],
    level_radio: RadioButtons,
) -> Dict[str, Any]:
    if state.geometry_mode == "edited_walls":
        state.mesh_coarse_factor = float(sliders["mesh_coarse"].val)
        return _build_mesh_params(state, state.params)

    params = dict(state.params)
    params["width"] = float(sliders["width"].val)
    params["angle_span"] = math.radians(float(sliders["angle_deg"].val))
    params["amplitude"] = float(sliders["amplitude"].val)
    params["bend_sign"] = float(sliders["bend_sign"].val)
    params["level"] = int(level_radio.value_selected.replace("L", ""))
    state.level = params["level"]
    state.mesh_coarse_factor = float(sliders["mesh_coarse"].val)
    state.pathology_strength = float(sliders["path_strength"].val)
    state.pathology_loc = float(sliders["path_loc"].val)
    params = recompute_pathology_offsets(
        params,
        state.vessel_cfg,
        state.rng,
        strength=state.pathology_strength,
        path_loc_frac=state.pathology_loc,
    )
    return _sync_derived_params(params, state.vessel_cfg)





def _sync_sliders_from_params(
    params: Dict[str, Any],
    sliders: Dict[str, Slider],
    level_radio: RadioButtons,
) -> None:
    sliders["width"].set_val(float(params.get("width", sliders["width"].val)))
    sliders["angle_deg"].set_val(math.degrees(float(params.get("angle_span", 0.0))))
    sliders["amplitude"].set_val(float(params.get("amplitude", 0.0)))
    sliders["bend_sign"].set_val(float(params.get("bend_sign", 1.0)))
    level = int(params.get("level", 0))
    level_label = f"L{level}"
    if level_label in level_radio.labels:
        level_radio.set_active(level_radio.labels.index(level_label))





def _set_parametric_widgets_active(sliders, level_radio, *, active: bool) -> None:
    alpha = 1.0 if active else 0.35
    for key, w in sliders.items():
        if key == "mesh_coarse":
            continue
        w.ax.set_alpha(alpha if active else 0.35)
    level_radio.ax.set_alpha(alpha)





def _update_status_text(
    status_ax,
    state: DemoState,
    *,
    n_nodes: int | None = None,
    infer_s: float | None = None,
) -> None:
    status_ax.clear()
    status_ax.axis("off")
    parts: List[str] = [f"mode={state.geometry_mode}"]
    if state.geom is not None:
        n_handles = len(subsample_handle_indices(state.geom.n))
        parts.append(f"handles={n_handles}")
        if state.drag_editor is not None:
            parts.append(f"fixed={state.drag_editor.n_fixed_handles}")
            parts.append("R-click toggle fix")
        parts.append(f"d_bar={state.geom.d_bar:.4g} m")
        parts.append(f"d_inlet={state.geom.d_inlet:.4g} m")
    elif state.last_meta:
        parts.append(f"d_bar={float(state.last_meta.get('d_bar', 0)):.4g} m")
    if n_nodes is not None:
        parts.append(f"N={n_nodes}")
    if infer_s is not None:
        parts.append(f"infer={infer_s:.2f}s")
    if state.drag_editor and state.drag_editor.last_error:
        parts.append(f"[ERR] {state.drag_editor.last_error}")
    elif state.last_error:
        parts.append(f"[ERR] {state.last_error}")
    if not parts:
        parts = ["Click Update flow"]
    elif state.geometry_mode == "parametric":
        parts.append("[i] Edit walls to drag handles")
    status_ax.text(0.01, 0.5, " | ".join(parts), va="center", fontsize=9)





def run_gui(state: DemoState, *, run_on_start: bool = False) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    ax_geom = fig.add_subplot(gs[0, :])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_p = fig.add_subplot(gs[1, 1])
    field_axes = [ax_speed, ax_p]



    status_ax = fig.add_axes([0.08, 0.01, 0.84, 0.03])



    cfg = state.vessel_cfg
    w_lo, w_hi = cfg.width_min, cfg.width_max
    sliders: Dict[str, Slider] = {}
    sliders["width"] = Slider(
        fig.add_axes([0.12, 0.16, 0.35, 0.02]),
        "width [m]",
        valmin=w_lo,
        valmax=w_hi,
        valinit=float(state.params["width"]),
    )
    sliders["angle_deg"] = Slider(
        fig.add_axes([0.12, 0.13, 0.35, 0.02]),
        "bend [deg]",
        valmin=0.0,
        valmax=125.0,
        valinit=math.degrees(float(state.params.get("angle_span", 0.0))),
    )
    sliders["amplitude"] = Slider(
        fig.add_axes([0.12, 0.10, 0.35, 0.02]),
        "S amp [m]",
        valmin=0.0,
        valmax=0.015,
        valinit=float(state.params.get("amplitude", 0.0)),
    )
    sliders["bend_sign"] = Slider(
        fig.add_axes([0.12, 0.07, 0.35, 0.02]),
        "bend sign",
        valmin=-1.0,
        valmax=1.0,
        valinit=float(state.params.get("bend_sign", 1.0)),
    )
    sliders["path_strength"] = Slider(
        fig.add_axes([0.55, 0.13, 0.35, 0.02]),
        "path strength",
        valmin=0.0,
        valmax=1.0,
        valinit=state.pathology_strength,
    )
    sliders["path_loc"] = Slider(
        fig.add_axes([0.55, 0.10, 0.35, 0.02]),
        "path loc",
        valmin=0.0,
        valmax=1.0,
        valinit=state.pathology_loc,
    )
    sliders["mesh_coarse"] = Slider(
        fig.add_axes([0.55, 0.07, 0.35, 0.02]),
        "mesh coarse x",
        valmin=1.0,
        valmax=3.0,
        valinit=1.0,
    )



    ax_level = fig.add_axes([0.55, 0.22, 0.06, 0.09])
    level_radio = RadioButtons(ax_level, ("L0", "L1", "L2"), active=int(state.params.get("level", 0)))

    ax_update = fig.add_axes([0.12, 0.03, 0.11, 0.035])
    ax_edit = fig.add_axes([0.24, 0.03, 0.11, 0.035])
    ax_sliders = fig.add_axes([0.36, 0.03, 0.11, 0.035])
    ax_rand = fig.add_axes([0.48, 0.03, 0.10, 0.035])
    ax_reset = fig.add_axes([0.60, 0.03, 0.08, 0.035])
    btn_update = Button(ax_update, "Update flow")
    btn_edit = Button(ax_edit, "Edit walls")
    btn_sliders = Button(ax_sliders, "Use sliders")
    btn_rand = Button(ax_rand, "Randomize")
    btn_reset = Button(ax_reset, "Reset")
    btn_sliders.ax.set_visible(False)



    initial_params = dict(state.params)
    builder = MeshToGraph(phase="kinematics", rheology=state.phys_cfg.viscosity_model)
    busy = {"flag": False}



    def refresh_fields() -> None:
        if state.last_data is None or state.last_pred is None:
            for ax in field_axes:
                ax.clear()
                ax.axis("off")
            return
        pos = _pos_si_from_data(state.last_data)
        pred_np = state.last_pred.detach().cpu().numpy()
        u_ref = float(state.last_data.u_ref.item())
        p_ref = float(state.phys_cfg.get_p_ref(u_ref))
        for ax in field_axes:
            ax.clear()
        plot_kinematics_speed_pressure(
            fig,
            field_axes,
            pos,
            pred_np,
            si_scale=(u_ref, p_ref),
            show_si=False,
        )



    def on_geom_drag(geom: VesselGeometry) -> None:
        state.geom = geom
        state.last_error = None
        ax_geom.set_title(_geometry_panel_title(state), fontsize=10)
        _update_status_text(status_ax, state)
        fig.canvas.draw_idle()



    def enter_edit_mode(_event=None) -> None:
        try:
            params = _sync_derived_params(state.params, state.vessel_cfg)
            state.geom = compute_geometry_from_params(params, state.cfg_dict)
            state.baseline_geom = compute_geometry_from_params(params, state.cfg_dict)
        except Exception as exc:
            state.last_error = str(exc)
            _update_status_text(status_ax, state)
            fig.canvas.draw_idle()
            return
        state.geometry_mode = "edited_walls"
        if state.drag_editor is not None:
            state.drag_editor.disconnect()
        ref_w = float(state.params.get("width", state.vessel_cfg.width_min))
        state.drag_editor = WallControlPointEditor(
            fig,
            ax_geom,
            state.geom,
            cfg_dict=state.cfg_dict,
            on_change=on_geom_drag,
            baseline_top=state.baseline_geom.top_coords,
            baseline_bot=state.baseline_geom.bot_coords,
            max_wall_displacement_m=default_max_wall_displacement_m(ref_w, state.cfg_dict),
            drag_sigma_stations=8.5,
        )
        ax_geom.set_title(_geometry_panel_title(state), fontsize=10)
        _set_parametric_widgets_active(sliders, level_radio, active=False)
        btn_edit.ax.set_visible(False)
        btn_sliders.ax.set_visible(True)
        _update_status_text(status_ax, state)
        fig.canvas.draw()



    def leave_edit_mode(_event=None) -> None:
        if state.drag_editor is not None:
            state.drag_editor.disconnect()
            state.drag_editor = None
        state.geometry_mode = "parametric"
        state.geom = None
        state.baseline_geom = None
        _set_parametric_widgets_active(sliders, level_radio, active=True)
        btn_edit.ax.set_visible(True)
        btn_sliders.ax.set_visible(False)
        _plot_geometry_panel(ax_geom, state, show_mesh=state.last_mesh is not None)
        _update_status_text(status_ax, state)
        fig.canvas.draw_idle()



    def do_update(_event=None) -> None:
        if busy["flag"]:
            return
        busy["flag"] = True
        btn_update.label.set_text("Running...")
        fig.canvas.draw_idle()
        t0 = time.perf_counter()
        try:
            mesh_params = _params_from_widgets(state, sliders, level_radio)
            if state.geometry_mode == "parametric":
                state.params = mesh_params
            gen = VesselGenerator(phase="kinematics")
            state.cfg_dict = _cfg_dict_from_generator(gen, mesh_coarse_factor=state.mesh_coarse_factor)
            data, meta, mesh = _mesh_and_graph(mesh_params, state.cfg_dict, builder)
            pred = _predict(state.model, data, state.device)
            state.last_data = data
            state.last_pred = pred
            state.last_meta = meta
            state.last_mesh = mesh
            state.last_error = None
            if state.geometry_mode == "parametric":
                state.geom = compute_geometry_from_params(mesh_params, state.cfg_dict)
            if state.drag_editor is not None and state.geom is not None:
                state.drag_editor.set_geometry(state.geom)
                ax_geom.set_title(_geometry_panel_title(state), fontsize=10)
                if state.last_mesh is not None:
                    _plot_geometry_mesh(ax_geom, state.last_mesh, state.last_meta)
            else:
                _plot_geometry_panel(ax_geom, state, show_mesh=True)
            refresh_fields()
            infer_s = time.perf_counter() - t0
            _update_status_text(status_ax, state, n_nodes=data.num_nodes, infer_s=infer_s)
        except Exception as exc:
            state.last_error = str(exc)
            _update_status_text(status_ax, state, n_nodes=getattr(state.last_data, "num_nodes", None))
        finally:
            busy["flag"] = False
            btn_update.label.set_text("Update flow")
            fig.canvas.draw_idle()



    def do_randomize(_event=None) -> None:
        if state.geometry_mode == "edited_walls":
            leave_edit_mode()
        level = int(level_radio.value_selected.replace("L", ""))
        state.params = make_vessel_params(level=level, cfg=state.vessel_cfg, rng=state.rng)
        _sync_sliders_from_params(state.params, sliders, level_radio)
        state.geom = None
        state.last_error = None
        _plot_geometry_panel(ax_geom, state, show_mesh=False)
        _update_status_text(status_ax, state)
        fig.canvas.draw_idle()

    def on_level_change(label: str) -> None:
        if state.geometry_mode == "edited_walls":
            leave_edit_mode()
        level = int(str(label).replace("L", ""))
        state.params = make_vessel_params(level=level, cfg=state.vessel_cfg, rng=state.rng)
        _sync_sliders_from_params(state.params, sliders, level_radio)
        do_update()

    def do_reset(_event=None) -> None:
        if state.geometry_mode == "edited_walls":
            leave_edit_mode()
        state.params = dict(initial_params)
        _sync_sliders_from_params(state.params, sliders, level_radio)
        state.geom = None
        state.last_error = None
        _plot_geometry_panel(ax_geom, state, show_mesh=False)
        _update_status_text(status_ax, state)
        fig.canvas.draw_idle()



    btn_update.on_clicked(do_update)
    btn_edit.on_clicked(enter_edit_mode)
    btn_sliders.on_clicked(leave_edit_mode)
    btn_rand.on_clicked(do_randomize)
    btn_reset.on_clicked(do_reset)
    level_radio.on_clicked(on_level_change)



    try:
        state.geom = compute_geometry_from_params(
            _sync_derived_params(state.params, state.vessel_cfg), state.cfg_dict
        )
    except Exception:
        state.geom = None
    _plot_geometry_panel(ax_geom, state, show_mesh=False)
    _update_status_text(status_ax, state)



    if run_on_start:
        do_update()



    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.20)
    plt.show()





def run_no_gui(
    state: DemoState,
    out_path: Path,
    *,
    max_iters: int,
) -> None:
    """One-shot mesh -> graph -> infer -> PNG (smoke/CI)."""
    matplotlib_backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        if state.geometry_mode == "edited_walls" and state.geom is not None:
            mesh_params = geometry_to_params_override(state.geom)
        else:
            mesh_params = _sync_derived_params(state.params, state.vessel_cfg)
        gen = VesselGenerator(phase="kinematics")
        state.cfg_dict = _cfg_dict_from_generator(gen, mesh_coarse_factor=state.mesh_coarse_factor)
        builder = MeshToGraph(phase="kinematics", rheology=state.phys_cfg.viscosity_model)
        data, meta, mesh = _mesh_and_graph(mesh_params, state.cfg_dict, builder)
        pred = _predict(state.model, data, state.device)



        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0])
        ax_geom = fig.add_subplot(gs[0, :])
        field_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        if state.geom is not None:
            plot_wall_outline(ax_geom, state.geom.top_coords, state.geom.bot_coords)
        _plot_geometry_mesh(ax_geom, mesh, meta)
        pos = _pos_si_from_data(data)
        plot_kinematics_speed_pressure(fig, field_axes, pos, pred.detach().cpu().numpy())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path} (N={data.num_nodes}, d_bar={meta.get('d_bar')}, mode={state.geometry_mode})")
    finally:
        plt.switch_backend(matplotlib_backend)





def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parametric kinematics flow demo (matplotlib GUI).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Kinematics .pth (default: search outputs/kinematics/)")
    parser.add_argument("--rheology", choices=("newtonian", "carreau"), default="carreau")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-iters", type=int, default=25, help="DEQ iterations (12 for faster preview)")
    parser.add_argument("--mesh-coarse", action="store_true", help="Coarsen mesh (2x mesh_lc) for speed")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for Randomize")
    parser.add_argument("--run-on-start", action="store_true", help="Auto-run Update flow once at startup")
    parser.add_argument(
        "--geometry-mode",
        choices=("parametric", "edited_walls"),
        default="parametric",
        help="Geometry source (edited_walls for CI with pre-built geom on state)",
    )
    parser.add_argument("--load-geometry", type=Path, default=None, help="Optional edited wall JSON (top_coords/bot_coords)")
    parser.add_argument("--no-gui", action="store_true", help="One-shot infer + save PNG under outputs/reports/figures/kinematics/")
    return parser





def _load_geometry_file(path: Path, idx: int = 0) -> VesselGeometry:
    from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"--load-geometry file not found: {path}\n"
            "Export top_coords/bot_coords JSON from a prior edit, or omit --load-geometry "
            "and use --geometry-mode edited_walls to snapshot parametric walls."
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    top = np.asarray(payload["top_coords"], dtype=float)
    bot = np.asarray(payload["bot_coords"], dtype=float)
    return compute_geometry_from_walls(top, bot, idx=idx, unit=str(payload.get("unit", "m")), params=payload)





def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ckpt = resolve_kinematics_checkpoint(args.checkpoint)
    device = _resolve_device(args.device)
    phys_cfg = PhysicsConfig(phase="kinematics", rheology=args.rheology)
    model = load_kinematics_predictor(ckpt, device, phys_cfg=phys_cfg, max_iters=args.max_iters)



    vessel_cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(args.seed)
    params = _default_params(level=0, cfg=vessel_cfg, rng=rng)
    gen = VesselGenerator(phase="kinematics")
    coarse = 2.0 if args.mesh_coarse else 1.0
    cfg_dict = _cfg_dict_from_generator(gen, mesh_coarse_factor=coarse)



    geom: VesselGeometry | None = None
    geometry_mode: GeometryMode = args.geometry_mode
    if args.load_geometry is not None:
        geom = _load_geometry_file(args.load_geometry)
        geometry_mode = "edited_walls"
    elif geometry_mode == "edited_walls":
        params_sync = _sync_derived_params(params, vessel_cfg)
        geom = compute_geometry_from_params(params_sync, cfg_dict)
        top, bot = snapshot_walls_from_params(params_sync, cfg_dict)
        from src.data_gen.lib.vessel_geometry import compute_geometry_from_walls



        geom = compute_geometry_from_walls(
            top, bot, idx=0, unit="m", params=params_sync, base_length=vessel_cfg.base_length
        )



    state = DemoState(
        params=params,
        cfg_dict=cfg_dict,
        phys_cfg=phys_cfg,
        model=model,
        device=device,
        geometry_mode=geometry_mode,
        geom=geom,
        mesh_coarse_factor=coarse,
        rng=rng,
        vessel_cfg=vessel_cfg,
    )



    if args.no_gui:
        stamp = time.strftime("%Y%m%dT%H%M%SZ")
        out = get_project_root() / "outputs" / "reports" / "figures" / "kinematics" / f"demo_{stamp}.png"
        run_no_gui(state, out, max_iters=args.max_iters)
    else:
        run_gui(state, run_on_start=args.run_on_start)





if __name__ == "__main__":
    main()


