"""
Interactive inspection of processed graph ``.pt`` samples (Kinematics / Kine phase).

Run as a script (not via pytest)::

    python -m src.tools.inspect_graph_sample --inspect-sample --phase kinematics

Lists graph/COMSOL overlap and visualizes features, labels, WLS condition numbers, and BC masks.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, RadioButtons, TextBox
from torch_geometric.utils import degree

from src.config import NodeFeat, PhysicsConfig, VesselConfig
from src.core_physics.physics_kernels import scatter_add


def _resolve_kinematics_subdir(base_dir: Path, requested: str | None) -> Path:
    """Resolve kinematics rheology folder consistently with inspect_kinematics_data."""
    if requested:
        chosen = requested.strip().lower()
        if chosen not in {"carreau", "newtonian"}:
            raise ValueError(f"Invalid rheology '{requested}'. Use 'carreau' or 'newtonian'.")
        out = base_dir / chosen
        if not out.is_dir():
            raise FileNotFoundError(f"Requested rheology folder not found: {out}")
        return out

    # Flat legacy layout
    if list(base_dir.glob("vessel_*.*")):
        return base_dir

    available = []
    for name in ("carreau", "newtonian"):
        d = base_dir / name
        if d.is_dir() and list(d.glob("vessel_*.*")):
            available.append(d)
    if not available:
        return base_dir
    return next((d for d in available if d.name == "carreau"), available[0])


def _default_graph_dir_for_phase(phase: str) -> Path:
    d = Path(VesselConfig(phase=phase).graph_output_dir)
    return d


def _default_cfd_dir_for_phase(phase: str) -> Path:
    d = Path(VesselConfig(phase=phase).output_dir)
    return d


def analyze_geometric_quality(data):
    """Analyzes mesh quality via WLS condition numbers (2nd-order 5x5 matrix)."""
    row, col = data.edge_index
    num_nodes = data.num_nodes
    d = degree(row, num_nodes, dtype=torch.long)

    pos_diff = data.x[col, :2] - data.x[row, :2]
    dist = torch.norm(pos_diff, dim=1)

    ones = torch.ones_like(dist)
    count = scatter_add(ones, row, dim=0, dim_size=num_nodes)
    sum_dist = scatter_add(dist, row, dim=0, dim_size=num_nodes)
    avg_edge_len = (sum_dist / (count + 1e-6))[row]

    pos_diff_norm = pos_diff / (avg_edge_len.unsqueeze(1) + 1e-8)
    dx, dy = pos_diff_norm[:, 0], pos_diff_norm[:, 1]
    W = 1.0 / (dx**2 + dy**2 + 1e-8)

    V = torch.stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], dim=1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V.unsqueeze(2), V.unsqueeze(1))
    M = scatter_add(M_e.view(-1, 25), row, dim=0, dim_size=num_nodes).view(num_nodes, 5, 5)

    try:
        eigenvalues = torch.linalg.eigvalsh(M)
        cond_numbers = eigenvalues[:, -1] / (torch.abs(eigenvalues[:, 0]) + 1e-12)
        return cond_numbers.cpu().numpy(), (d >= 5).cpu().numpy()
    except RuntimeError:
        return np.zeros(num_nodes), np.zeros(num_nodes, dtype=bool)


def plot_field(ax, pos, values, title, cmap="viridis", colorbar=True, **kwargs):
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap=cmap, s=2, **kwargs)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")

    if colorbar:
        plt.colorbar(sc, ax=ax)

    return sc


def _collect_vessel_files(base_dir, suffix):
    """Collect vessel files keyed by vessel index (stable direct-dir first)."""
    base = Path(base_dir)
    by_idx = {}
    direct = sorted(base.glob(f"vessel_*{suffix}"))
    source = direct if direct else sorted(base.rglob(f"vessel_*{suffix}"))
    for path in source:
        idx = _extract_vessel_idx(path)
        if idx is None or idx in by_idx:
            continue
        by_idx[idx] = path
    return by_idx


def inspect_sample(
    filename=None,
    phase="kinematics",
    proc_dir=None,
    cfd_dir=None,
    phase_options=None,
    restrict_to_overlap=True,
    rheology=None,
    allow_stale_pairs=False,
):
    if phase_options is None:
        phase_options = ["kinematics"]

    requested_vessel_idx = _extract_vessel_idx(Path(filename)) if filename is not None else None
    state = {
        "phase": phase,
        "vessel_idx": requested_vessel_idx,
        "overlap_indices": [],
        "key_cid": None,
        "widgets": [],
    }
    fig = plt.figure(figsize=(20, 12))

    def _graph_dir_for_current_phase():
        if proc_dir is not None:
            base = Path(proc_dir)
        else:
            base = _default_graph_dir_for_phase(state["phase"])
        if state["phase"] == "kinematics":
            return _resolve_kinematics_subdir(base, rheology)
        return base

    def _cfd_dir_for_current_phase():
        if cfd_dir is not None:
            base = Path(cfd_dir)
        else:
            base = _default_cfd_dir_for_phase(state["phase"])
        if state["phase"] == "kinematics":
            return _resolve_kinematics_subdir(base, rheology)
        return base

    def _sync_overlap_for_current_phase():
        graph_dir = _graph_dir_for_current_phase()
        cfd_dir_local = _cfd_dir_for_current_phase()
        graph_files_by_idx = _collect_vessel_files(graph_dir, ".pt")
        if not allow_stale_pairs and restrict_to_overlap:
            bad = _find_incompatible_graph_cfd_pairs(graph_dir, cfd_dir_local)
            if bad:
                preview = ", ".join(str(v) for v in bad[:20])
                if len(bad) > 20:
                    preview = f"{preview}, ..."
                raise RuntimeError(
                    "Refusing to inspect with stale graph/CFD pairings. "
                    f"Found {len(bad)} incompatible same-ID pairs in current view: [{preview}]. "
                    "Regenerate anchors/graphs, or rerun with --allow-stale-pairs."
                )
        if restrict_to_overlap:
            state["overlap_indices"] = list_indices_with_valid_comsol_results(
                phase=state["phase"],
                proc_dir=graph_dir,
                cfd_dir=cfd_dir_local,
                emit=False,
            )
        else:
            state["overlap_indices"] = sorted(graph_files_by_idx)
        state["graph_files_by_idx"] = graph_files_by_idx
        if len(state["overlap_indices"]) == 0:
            state["vessel_idx"] = None
        elif state["vessel_idx"] not in set(state["overlap_indices"]):
            state["vessel_idx"] = state["overlap_indices"][0]

    def _render_current():
        _sync_overlap_for_current_phase()
        fig.clf()
        state["widgets"] = []
        fig.subplots_adjust(bottom=0.18, left=0.03, right=0.98, top=0.95, wspace=0.15, hspace=0.18)
        overlap_sorted = sorted(state["overlap_indices"])

        def _set_phase_and_redraw(new_phase):
            if new_phase == state["phase"]:
                return
            state["phase"] = new_phase
            state["vessel_idx"] = None
            _render_current()

        def _set_vessel_and_redraw(vessel_idx):
            if vessel_idx not in set(overlap_sorted):
                print(f"Vessel {vessel_idx} is not in overlap set for {state['phase']}.")
                return
            state["vessel_idx"] = vessel_idx
            _render_current()

        phase_ax = fig.add_axes([0.01, 0.02, 0.15, 0.14])
        phase_labels = [t.upper() for t in phase_options]
        active_phase_idx = phase_options.index(state["phase"]) if state["phase"] in phase_options else 0
        phase_radio = RadioButtons(phase_ax, phase_labels, active=active_phase_idx)

        def _on_phase_change(label):
            _set_phase_and_redraw(label.lower())

        phase_radio.on_clicked(_on_phase_change)
        state["widgets"].append(phase_radio)

        if len(overlap_sorted) == 0:
            msg_ax = fig.add_subplot(111)
            msg_ax.axis("off")
            msg_ax.text(
                0.5,
                0.5,
                (
                    f"No valid graph+COMSOL overlap for {state['phase'].upper()}"
                    if restrict_to_overlap
                    else f"No graph files found for {state['phase'].upper()}"
                ),
                ha="center",
                va="center",
                fontsize=16,
                color="crimson",
            )
            fig.text(0.20, 0.02, "Switch phase with the radio toggles on the left.", fontsize=10)
            fig.canvas.draw_idle()
            return

        filename_local = f"vessel_{state['vessel_idx']}.pt"
        data_path = state.get("graph_files_by_idx", {}).get(state["vessel_idx"])
        if data_path is None:
            graph_dir = _graph_dir_for_current_phase()
            data_path = graph_dir / filename_local
        if not data_path.exists():
            print(f"File {filename_local} not found in {_graph_dir_for_current_phase()}")
            fig.canvas.draw_idle()
            return

        phys_cfg = PhysicsConfig(phase=state["phase"])
        print(f"\n{'=' * 60}\n INSPECTING: {data_path.name} | PHASE: {state['phase'].upper()}\n{'=' * 60}")
        data = torch.load(data_path, weights_only=False)

        print("\n Architecture & Invariants")
        expected_channels = NodeFeat.WIDTH_D2.stop
        if data.x.shape[1] != expected_channels:
            print(f" ❌ FAIL: Feature mismatch! Expected {expected_channels}, got {data.x.shape}.")
        else:
            print(f" ✅ PASS: Features aligned ({expected_channels} channels).")

        print("\n Boundary & Physics Sanity")
        if data.y is not None:
            wall_vel = torch.norm(data.y[data.mask_wall, :2], dim=1).max().item()
            status = "✅ PASS" if wall_vel < 1e-3 else "❌ FAIL"
            print(f" {status}: No-slip condition (Max Wall Vel: {wall_vel:.2e})")

        cond_nums, mask_valid = analyze_geometric_quality(data)
        print("\n Mesh Stability (WLS Condition Numbers)")
        print(f" -> Mean: {np.mean(cond_nums[mask_valid]):.2e} | Max: {np.max(cond_nums[mask_valid]):.2e}")
        print("\nRendering visualization...")

        pos = data.x[:, :2].cpu().numpy()
        vel_mag_gt = torch.norm(data.y[:, 0:2], dim=1).cpu().numpy()
        vel_mag_prior = torch.norm(data.x[:, 11:13], dim=1).cpu().numpy()

        axes = fig.subplots(3, 4).flatten()
        plots = [
            ("Input: ND-SDF", data.x[:, 2], "viridis"),
            ("Mesh: Log10(WLS Cond)", np.log10(cond_nums + 1), "magma"),
            ("Input: Wall Normals", None, None),
            ("Input: Boundary Masks", None, None),
            ("GT: Velocity Magnitude", vel_mag_gt, "jet"),
            ("GT: Pressure", data.y[:, 2], "coolwarm"),
            ("GT: ND-Viscosity", data.y[:, 3], "plasma"),
            ("GT: Wall Shear Stress", data.y[:, 4], "inferno"),
            ("Prior: Velocity Magnitude", vel_mag_prior, "jet"),
            ("Prior: Viscosity", data.x[:, 13], "plasma"),
            ("Prior: WSS", data.x[:, 14], "inferno"),
        ]

        for i, (title, values, cmap) in enumerate(plots):
            ax = axes[i]
            if values is not None:
                if torch.is_tensor(values):
                    values = values.cpu().numpy()
                plot_field(ax, pos, values, title, cmap=cmap)
            else:
                if title == "Input: Wall Normals":
                    mask_w = data.mask_wall.cpu().numpy()
                    ax.scatter(pos[:, 0], pos[:, 1], color="lightgray", s=1, alpha=0.1)
                    ax.quiver(
                        pos[mask_w, 0],
                        pos[mask_w, 1],
                        data.x[mask_w, 4],
                        data.x[mask_w, 5],
                        color="red",
                        scale=30,
                    )
                    ax.set_title(title)
                    ax.set_aspect("equal")
                    ax.axis("off")
                elif title == "Input: Boundary Masks":
                    ax.scatter(pos[data.mask_inlet, 0], pos[data.mask_inlet, 1], c="green", s=5, label="Inlet")
                    ax.scatter(pos[data.mask_outlet, 0], pos[data.mask_outlet, 1], c="blue", s=5, label="Outlet")
                    ax.scatter(pos[data.mask_wall, 0], pos[data.mask_wall, 1], c="black", s=2, label="Wall")
                    ax.legend(loc="upper right", fontsize="x-small")
                    ax.set_title(title)
                    ax.set_aspect("equal")
                    ax.axis("off")

        meta_ax = axes[-1]
        meta_ax.axis("off")
        meta_ax.text(
            0,
            0.5,
            f"Phase: {state['phase']}\nFile: {data_path}\nRe: {phys_cfg.re_target}\nModel: {phys_cfg.viscosity_model}\n"
            f"{'Overlap' if restrict_to_overlap else 'Available'} count: {len(overlap_sorted)}",
            fontsize=10,
        )

        current_pos = overlap_sorted.index(state["vessel_idx"])
        prev_ax = fig.add_axes([0.72, 0.03, 0.08, 0.05])
        next_ax = fig.add_axes([0.81, 0.03, 0.08, 0.05])
        jump_ax = fig.add_axes([0.56, 0.03, 0.14, 0.05])
        prev_btn = Button(prev_ax, "Prev")
        next_btn = Button(next_ax, "Next")
        jump_box = TextBox(jump_ax, "Vessel ID ", initial=str(state["vessel_idx"]))

        def _go_prev(_event):
            target = overlap_sorted[(current_pos - 1) % len(overlap_sorted)]
            _set_vessel_and_redraw(target)

        def _go_next(_event):
            target = overlap_sorted[(current_pos + 1) % len(overlap_sorted)]
            _set_vessel_and_redraw(target)

        def _go_jump(text):
            text = text.strip()
            if not text:
                return
            try:
                target = int(text)
            except ValueError:
                print("Invalid vessel ID. Enter an integer (example: 73).")
                return
            _set_vessel_and_redraw(target)

        def _on_key(event):
            if event.key == "left":
                _go_prev(None)
            elif event.key == "right":
                _go_next(None)

        prev_btn.on_clicked(_go_prev)
        next_btn.on_clicked(_go_next)
        jump_box.on_submit(_go_jump)
        state["widgets"].extend([prev_btn, next_btn, jump_box])
        if state["key_cid"] is not None:
            fig.canvas.mpl_disconnect(state["key_cid"])
        state["key_cid"] = fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.text(0.20, 0.02, "Phase toggle (left), vessel Prev/Next, Left/Right keys, or type vessel ID.", fontsize=10)

        fig.canvas.draw_idle()

    _render_current()
    plt.show()


def _pick_filename_interactively(data_dir):
    files = sorted([p.name for p in Path(data_dir).glob("*.pt")])
    if len(files) == 0:
        print(f"No .pt files found in {data_dir}")
        return None

    print("\nAvailable graph files:")
    for idx, name in enumerate(files):
        print(f"  [ {idx} ] {name}")

    while True:
        user_input = input(f"\nSelect index [0-{len(files) - 1}] or q to quit: ").strip()
        if user_input.lower() in ["q", "quit", "exit"]:
            return None

        try:
            idx = int(user_input)
            if 0 <= idx < len(files):
                return files[idx]
            print(f"Invalid selection. Enter a value in [ 0, {len(files) - 1} ].")
        except ValueError:
            print("Invalid input. Enter an integer index.")


def _extract_vessel_idx(path_obj):
    match = re.match(r"vessel_(\d+)$", path_obj.stem)
    return int(match.group(1)) if match else None


def _is_valid_comsol_result(npz_path):
    required_keys = ("u", "v", "p", "mu")
    try:
        with np.load(npz_path) as data_npz:
            for key in required_keys:
                if key not in data_npz:
                    return False
                arr = np.asarray(data_npz[key])
                if arr.size == 0 or not np.isfinite(arr).all():
                    return False

            if np.max(np.abs(data_npz["u"])) < 1e-10 and np.max(np.abs(data_npz["v"])) < 1e-10:
                return False
            if np.std(data_npz["p"]) < 1e-12:
                return False
    except Exception:
        return False

    return True


def _is_graph_cfd_node_compatible(pt_path, npz_path):
    """Basic compatibility gate: graph node count must match same-ID CFD node count."""
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        graph_nodes = int(data.num_nodes)
        with np.load(npz_path) as data_npz:
            cfd_nodes = int(np.asarray(data_npz["x"]).size)
    except Exception:
        return False
    return graph_nodes == cfd_nodes


def _find_incompatible_graph_cfd_pairs(graph_dir, comsol_dir):
    """Return vessel IDs whose graph/CFD pair exists but fails compatibility."""
    graph_files = _collect_vessel_files(graph_dir, ".pt")
    cfd_files = _collect_vessel_files(comsol_dir, ".npz")
    out = []
    for idx in sorted(set(graph_files) & set(cfd_files)):
        if not _is_valid_comsol_result(cfd_files[idx]):
            continue
        if not _is_graph_cfd_node_compatible(graph_files[idx], cfd_files[idx]):
            out.append(idx)
    return out


def list_indices_with_valid_comsol_results(
    phase="kinematics",
    proc_dir=None,
    cfd_dir=None,
    verbose=False,
    print_indices=False,
    emit=True,
    require_graph_cfd_compat=True,
    fail_on_incompat=False,
):
    graph_dir = Path(proc_dir) if proc_dir is not None else _default_graph_dir_for_phase(phase)
    comsol_dir = Path(cfd_dir) if cfd_dir is not None else _default_cfd_dir_for_phase(phase)

    if not graph_dir.exists():
        if emit:
            print(f"Graph directory not found: {graph_dir}")
        return []
    if not comsol_dir.exists():
        if emit:
            print(f"COMSOL results directory not found: {comsol_dir}")
        return []

    graph_indices = set(_collect_vessel_files(graph_dir, ".pt").keys())

    valid_cfd_indices = set()
    invalid_cfd_indices = set()
    incompatible_pairs = set()
    graph_files = _collect_vessel_files(graph_dir, ".pt")
    for idx, npz_path in _collect_vessel_files(comsol_dir, ".npz").items():
        if _is_valid_comsol_result(npz_path):
            if require_graph_cfd_compat and idx in graph_files:
                if not _is_graph_cfd_node_compatible(graph_files[idx], npz_path):
                    incompatible_pairs.add(idx)
                    continue
            valid_cfd_indices.add(idx)
        else:
            invalid_cfd_indices.add(idx)

    overlap_indices = sorted(graph_indices & valid_cfd_indices)

    if fail_on_incompat and incompatible_pairs:
        preview = ", ".join(str(v) for v in sorted(incompatible_pairs)[:20])
        if len(incompatible_pairs) > 20:
            preview = f"{preview}, ..."
        raise RuntimeError(
            "Detected stale graph/CFD pairings (same vessel ID but incompatible nodes). "
            f"Count={len(incompatible_pairs)}; sample IDs: [{preview}]. "
            "Regenerate anchor CFD + graphs, or rerun with --allow-stale-pairs."
        )

    if verbose and emit:
        print(f"\n{'=' * 60}")
        print(f" VALID GRAPH + COMSOL OVERLAP ({phase.upper()})")
        print(f"{'=' * 60}")
        print(f"Graph dir: {graph_dir}")
        print(f"COMSOL dir: {comsol_dir}")
        print(f"Graph samples found: {len(graph_indices)}")
        print(f"Valid COMSOL samples found: {len(valid_cfd_indices)}")
        if invalid_cfd_indices:
            print(f"Invalid COMSOL samples skipped: {len(invalid_cfd_indices)}")
        if incompatible_pairs:
            print(f"Graph/CFD node-mismatch pairs skipped: {len(incompatible_pairs)}")
        print(f"Overlap samples: {len(overlap_indices)}")
        print(f"Overlap indices: {overlap_indices}")
    elif emit:
        print(len(overlap_indices))
        if print_indices:
            print(", ".join(str(v) for v in overlap_indices))

    return overlap_indices


def _normalize_phase_value(raw_phase):
    if raw_phase is None:
        return None
    normalized = str(raw_phase).strip().lower()
    aliases = {
        "1": "kinematics",
        "kinematics": "kinematics",
    }
    return aliases.get(normalized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Kine phase (Kinematics/2) graph samples")
    parser.add_argument("--filename", type=str, default=None, help="Graph filename (e.g. vessel_73.pt).")
    parser.add_argument("--phase", type=str, default=None, help="Phase for default directories")
    parser.add_argument("--proc-dir", type=str, default=None, help="Directory with graph .pt files")
    parser.add_argument("--cfd-dir", type=str, default=None, help="Directory with COMSOL .npz CFD files")
    parser.add_argument(
        "--rheology",
        type=str,
        default=None,
        choices=("newtonian", "carreau"),
        help="Kinematics rheology folder to inspect (default: auto, prefers carreau).",
    )
    parser.add_argument(
        "--list-cfd-overlap",
        action="store_true",
        help="Print overlap count for graphs with valid COMSOL CFD outputs",
    )
    parser.add_argument("--inspect-sample", action="store_true", help="Interactive inspection / picker")
    parser.add_argument("--verbose-overlap", action="store_true", help="Full overlap diagnostics")
    parser.add_argument(
        "--allow-stale-pairs",
        action="store_true",
        help="Allow graph/CFD same-ID node mismatches (default: fail loudly).",
    )
    args = parser.parse_args()

    if args.phase is not None:
        selected_phase = _normalize_phase_value(args.phase)
        if selected_phase is None:
            print("Invalid --phase value. Use: kinematics or kinematics (also accepts 1/2).")
            raise SystemExit(1)
    else:
        selected_phase = "kinematics"

    inspect_mode = args.inspect_sample or (args.filename is not None)
    overlap_mode = args.list_cfd_overlap

    if overlap_mode:
        overlap_proc_dir = Path(args.proc_dir) if args.proc_dir is not None else _default_graph_dir_for_phase(selected_phase)
        overlap_cfd_dir = Path(args.cfd_dir) if args.cfd_dir is not None else _default_cfd_dir_for_phase(selected_phase)
        if selected_phase == "kinematics":
            overlap_proc_dir = _resolve_kinematics_subdir(overlap_proc_dir, args.rheology)
            overlap_cfd_dir = _resolve_kinematics_subdir(overlap_cfd_dir, args.rheology)
        list_indices_with_valid_comsol_results(
            phase=selected_phase,
            proc_dir=overlap_proc_dir,
            cfd_dir=overlap_cfd_dir,
            verbose=args.verbose_overlap,
            fail_on_incompat=not args.allow_stale_pairs,
        )
        raise SystemExit(0)

    if inspect_mode and args.filename is None:
        default_dir = Path(args.proc_dir) if args.proc_dir is not None else _default_graph_dir_for_phase(selected_phase)
        selected_filename = _pick_filename_interactively(default_dir)
        if selected_filename is None:
            print("Exiting without action.")
            raise SystemExit(0)
    else:
        selected_filename = args.filename

    inspect_sample(
        filename=selected_filename,
        phase=selected_phase,
        proc_dir=args.proc_dir,
        cfd_dir=args.cfd_dir,
        restrict_to_overlap=not inspect_mode,
        rheology=args.rheology,
        allow_stale_pairs=args.allow_stale_pairs,
    )
