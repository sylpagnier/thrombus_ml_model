"""
Biochem export / graph inspector: boundary checks, unit audits, summaries, and matplotlib views.

**Default:** one anchor stem at a time — **brief** availability line, qualitative text (boundaries, unit audit, graph
summary) for **that** stem only, then **one** matplotlib window (domain time-slider if multi-``@ t=``, else a
single domain 2×2; **graph-only** stems use graph time-slider or steady 2×2). Use Prev/Next controls (or
left/right keys) to cycle stems manually; ``r`` still jumps to random. For the **full** stems/times table use
``--summary``.

Examples:
    python -m src.tools.inspect_biochem_data --phase biochem_anchors
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --summary
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --stem patient001
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --stem vessel_001 --unit-audit
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --stem vessel_001 --graph-summary
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --stem vessel_001 --plot-domain
    python -m src.tools.inspect_biochem_data --phase biochem_mix --plot-domain-interactive
    python -m src.tools.inspect_biochem_data --phase biochem_anchors --stem vessel_001 --plot-graph
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from matplotlib.widgets import Button, Slider
from scipy.spatial import cKDTree
from src.config import BiochemConfig, VesselConfig


FIELD_COLUMNS = [
    "x",
    "y",
    "u",
    "v",
    "p",
    "mu_eff",
    "rp",
    "ap",
    "apr",
    "aps",
    "PT",
    "th",
    "at",
    "fg",
    "fi",
    "M",
    "Mas",
    "Mat",
]

_PHASE_CHOICES = ("biochem", "biochem_anchors", "biochem_mix", "biochem_patients")
U_CMAP = "jet"
MU_CMAP = "viridis"


def _robust_vmin_vmax(values: np.ndarray, *, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """Percentile-based color limits to keep fields readable under outliers."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def _resolve_export_dir(phase: str) -> Path:
    cfg = VesselConfig(phase=phase)
    return Path(cfg.output_dir)


def _resolve_graph_dir(phase: str) -> Path:
    cfg = VesselConfig(phase=phase)
    return Path(cfg.graph_output_dir)


def _domain_txt_stems(export_dir: Path) -> list[str]:
    stems = []
    for p in sorted(export_dir.glob("*.txt")):
        if p.stem.endswith(("_inlet", "_outlet", "_wall")):
            continue
        stems.append(p.stem)
    return stems


def _parse_times_from_header(domain_file: Path) -> list[float]:
    times: list[float] = []
    with open(domain_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("% x") and "@ t=" in line:
                for match in re.finditer(r"t=([0-9.]+)", line):
                    t_val = float(match.group(1))
                    if t_val not in times:
                        times.append(t_val)
                break
    return times


VARS_PER_COMSOL_STEP = 18  # Wide COMSOL export: x,y + 16 fields per time column-group


def _load_first_block(domain_file: Path, sample_rows: int = 100000) -> pd.DataFrame:
    df_full = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None, nrows=sample_rows)
    if df_full.shape[1] < 20:
        raise ValueError(f"Unexpected format in {domain_file.name}: got {df_full.shape[1]} columns (need >=20).")
    df = df_full.iloc[:, 2:20].copy()
    df.columns = FIELD_COLUMNS
    return df


def _load_comsol_trajectory(domain_file: Path) -> tuple[list[float], dict[float, pd.DataFrame]]:
    """Parse wide-format Phase3 COMSOL domain ``.txt`` (same layout as ``extract_biochem_comsol_data``)."""
    with open(domain_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    header_line = ""
    for line in lines:
        if line.startswith("% x") and "@ t=" in line:
            header_line = line
            break

    df_full = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None)
    ncol = int(df_full.shape[1])

    if not header_line or ncol < 2 + VARS_PER_COMSOL_STEP:
        df0 = df_full.iloc[:, 2:20].copy()
        df0.columns = FIELD_COLUMNS
        return [0.0], {0.0: df0}

    times: list[float] = []
    for m in re.finditer(r"t=([0-9.]+)", header_line):
        t_val = float(m.group(1))
        if t_val not in times:
            times.append(t_val)

    if not times:
        df0 = df_full.iloc[:, 2:20].copy()
        df0.columns = FIELD_COLUMNS
        return [0.0], {0.0: df0}

    time_blocks: dict[float, pd.DataFrame] = {}
    for i, t_val in enumerate(times):
        start_col = 2 + i * VARS_PER_COMSOL_STEP
        end_col = start_col + VARS_PER_COMSOL_STEP
        if end_col > ncol:
            raise ValueError(
                f"{domain_file.name}: header lists {len(times)} times but columns only reach {ncol} "
                f"(need up to {end_col} for step {i})."
            )
        df_step = df_full.iloc[:, start_col:end_col].copy()
        df_step.columns = FIELD_COLUMNS
        time_blocks[t_val] = df_step

    return times, time_blocks


def _subsample_node_idx(num_nodes: int, max_points: int) -> np.ndarray:
    if num_nodes <= max_points:
        return np.arange(num_nodes, dtype=np.int64)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(num_nodes, size=max_points, replace=False))


def _list_graph_stems(graph_dir: Path) -> list[str]:
    if not graph_dir.exists():
        return []
    return sorted(p.stem for p in graph_dir.glob("*.pt"))


def _stem_candidates_union(export_dir: Path, graph_dir: Path) -> list[str]:
    return sorted(set(_domain_txt_stems(export_dir)) | set(_list_graph_stems(graph_dir)))


def _attach_stem_navigation_controls(
    fig: plt.Figure,
    *,
    current_stem: str,
    all_stems: list[str],
    next_holder: dict[str, str | None],
    enable: bool,
) -> None:
    """Attach deterministic Prev/Next + optional random stem controls."""
    if not enable:
        return

    if len(all_stems) == 0:
        return

    try:
        i_cur = all_stems.index(current_stem)
    except ValueError:
        i_cur = 0

    def _set_stem(target: str) -> None:
        print(f"\nSwitching to stem: {target}")
        next_holder["value"] = target
        plt.close(fig)

    def _prev(_event=None) -> None:
        if len(all_stems) <= 1:
            print("Only one stem available; cannot move.")
            return
        _set_stem(all_stems[(i_cur - 1) % len(all_stems)])

    def _next(_event=None) -> None:
        if len(all_stems) <= 1:
            print("Only one stem available; cannot move.")
            return
        _set_stem(all_stems[(i_cur + 1) % len(all_stems)])

    def _random(_event=None) -> None:
        candidates = [s for s in all_stems if s != current_stem]
        if not candidates:
            print("Only one stem available; cannot pick random.")
            return
        _set_stem(random.choice(candidates))

    prev_ax = fig.add_axes([0.58, 0.02, 0.12, 0.05])
    next_ax = fig.add_axes([0.71, 0.02, 0.12, 0.05])
    rand_ax = fig.add_axes([0.84, 0.02, 0.14, 0.05])
    Button(prev_ax, "Prev stem").on_clicked(_prev)
    Button(next_ax, "Next stem").on_clicked(_next)
    Button(rand_ax, "Random").on_clicked(_random)

    def _on_key(event) -> None:
        k = getattr(event, "key", None)
        if k in ("left", "p"):
            _prev()
        elif k in ("right", "n"):
            _next()
        elif k == "r":
            _random()

    fig.canvas.mpl_connect("key_press_event", _on_key)


def plot_domain_trajectory_slider(
    stem: str,
    export_dir: Path,
    *,
    max_points: int = 65_000,
    regen_stems: list[str] | None = None,
    next_holder: dict[str, str | None] | None = None,
    current_stem_for_regen: str | None = None,
    enable_regenerate: bool = True,
) -> None:
    """Scroll through COMSOL domain snapshots in time (wide export), fixed color scales across times."""
    domain_file = export_dir / f"{stem}.txt"
    _times_hdr, blocks = _load_comsol_trajectory(domain_file)
    times_sorted = sorted(blocks.keys())
    df0 = blocks[times_sorted[0]]
    n = len(df0)
    idx = _subsample_node_idx(n, max_points)
    x_s = df0["x"].to_numpy(dtype=np.float64)[idx]
    y_s = df0["y"].to_numpy(dtype=np.float64)[idx]
    tree = cKDTree(df0[["x", "y"]].values)
    t_axis = np.array(times_sorted, dtype=np.float64)
    t_count = len(times_sorted)
    vel = np.zeros((t_count, idx.size), dtype=np.float64)
    p_all = np.zeros((t_count, idx.size), dtype=np.float64)
    th_all = np.zeros((t_count, idx.size), dtype=np.float64)
    mu_all = np.zeros((t_count, idx.size), dtype=np.float64)
    for ti, tv in enumerate(times_sorted):
        dfi = blocks[tv].iloc[idx].reset_index(drop=True)
        u = dfi["u"].to_numpy(dtype=np.float64)
        v = dfi["v"].to_numpy(dtype=np.float64)
        vel[ti] = np.sqrt(u**2 + v**2)
        p_all[ti] = dfi["p"].to_numpy(dtype=np.float64)
        th_all[ti] = dfi["th"].to_numpy(dtype=np.float64)
        mu_all[ti] = dfi["mu_eff"].to_numpy(dtype=np.float64)

    vel_vmin, vel_vmax = float(vel.min()), float(vel.max())
    p_vmin, p_vmax = float(p_all.min()), float(p_all.max())
    th_vmin, th_vmax = float(th_all.min()), float(th_all.max())
    mu_vmin, mu_vmax = _robust_vmin_vmax(mu_all)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    bottom = 0.22 if (regen_stems and next_holder is not None and enable_regenerate) else 0.12
    plt.subplots_adjust(bottom=bottom, hspace=0.25)
    ti0 = 0
    fig.suptitle(f"Phase3 domain export — {stem}  (t={t_axis[ti0]:.6g})", fontsize=14, fontweight="bold")

    sc0 = axs[0, 0].scatter(x_s, y_s, c=vel[ti0], cmap=U_CMAP, s=2, vmin=vel_vmin, vmax=vel_vmax, rasterized=True)
    fig.colorbar(sc0, ax=axs[0, 0], label="|U|")
    axs[0, 0].set_title("|U|")
    sc1 = axs[0, 1].scatter(x_s, y_s, c=p_all[ti0], cmap="coolwarm", s=2, vmin=p_vmin, vmax=p_vmax, rasterized=True)
    fig.colorbar(sc1, ax=axs[0, 1], label="p")
    axs[0, 1].set_title("Pressure")
    sc2 = axs[1, 0].scatter(x_s, y_s, c=th_all[ti0], cmap="inferno", s=2, vmin=th_vmin, vmax=th_vmax, rasterized=True)
    fig.colorbar(sc2, ax=axs[1, 0], label="th")
    axs[1, 0].set_title("th")
    sc3 = axs[1, 1].scatter(x_s, y_s, c=mu_all[ti0], cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(sc3, ax=axs[1, 1], label="mu_eff")
    axs[1, 1].set_title("mu_eff")

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    if t_count > 1:
        s_y = 0.09 if (regen_stems and next_holder is not None and enable_regenerate) else 0.02
        ax_slider = plt.axes((0.2, s_y, 0.52, 0.03))
        slider = Slider(
            ax_slider,
            "Time index",
            valmin=0,
            valmax=t_count - 1,
            valinit=0,
            valstep=1,
            color="teal",
        )

        def _upd(_val: float) -> None:
            k = int(slider.val)
            sc0.set_array(vel[k])
            sc1.set_array(p_all[k])
            sc2.set_array(th_all[k])
            sc3.set_array(mu_all[k])
            fig.suptitle(f"Phase3 domain export — {stem}  (t={t_axis[k]:.6g})", fontsize=14, fontweight="bold")
            fig.canvas.draw_idle()

        slider.on_changed(_upd)

    if regen_stems is not None and next_holder is not None and current_stem_for_regen is not None:
        _attach_stem_navigation_controls(
            fig,
            current_stem=current_stem_for_regen,
            all_stems=regen_stems,
            next_holder=next_holder,
            enable=enable_regenerate,
        )

    plt.show()


def plot_graph_trajectory_slider(
    data,
    stem: str,
    *,
    max_points: int = 65_000,
    regen_stems: list[str] | None = None,
    next_holder: dict[str, str | None] | None = None,
    current_stem_for_regen: str | None = None,
    enable_regenerate: bool = True,
) -> None:
    """Time slider for transient graph labels ``y`` shaped ``[T, N, C]`` (Phase3 trajectories)."""
    y = data.y
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.float().cpu().numpy()
    if y.ndim != 3 or y.shape[0] < 2:
        return

    t_tensor = getattr(data, "t", None)
    if t_tensor is None:
        t_axis = np.arange(y.shape[0], dtype=np.float64)
    else:
        t_axis = t_tensor.detach().cpu().numpy().reshape(-1)

    t_count, n, c = y.shape
    pos = data.x[:, :2].detach().cpu().numpy()
    idx = _subsample_node_idx(n, max_points)
    x_s, y_coord = pos[idx, 0], pos[idx, 1]

    u = y[:, idx, 0]
    v = y[:, idx, 1]
    vel = np.sqrt(u**2 + v**2)
    p_all = y[:, idx, 2]
    th_ch = min(11, c - 1)
    th_all = y[:, idx, th_ch]
    mu_ch = y[:, idx, 3] if c > 3 else np.zeros((t_count, idx.size), dtype=np.float64)

    vel_vmin, vel_vmax = float(vel.min()), float(vel.max())
    p_vmin, p_vmax = float(p_all.min()), float(p_all.max())
    th_vmin, th_vmax = float(th_all.min()), float(th_all.max())
    mu_vmin, mu_vmax = _robust_vmin_vmax(mu_ch)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    bottom = 0.22 if (regen_stems and next_holder is not None and enable_regenerate) else 0.12
    plt.subplots_adjust(bottom=bottom, hspace=0.25)
    ti0 = 0
    fig.suptitle(
        f"Phase3 graph labels — {stem}  (t={float(t_axis[ti0]):.6g})",
        fontsize=14,
        fontweight="bold",
    )

    sc0 = axs[0, 0].scatter(x_s, y_coord, c=vel[ti0], cmap=U_CMAP, s=2, vmin=vel_vmin, vmax=vel_vmax, rasterized=True)
    fig.colorbar(sc0, ax=axs[0, 0], label="|U|")
    axs[0, 0].set_title("|U| (y)")
    sc1 = axs[0, 1].scatter(x_s, y_coord, c=p_all[ti0], cmap="coolwarm", s=2, vmin=p_vmin, vmax=p_vmax, rasterized=True)
    fig.colorbar(sc1, ax=axs[0, 1], label="p")
    axs[0, 1].set_title("p (y)")
    sc2 = axs[1, 0].scatter(x_s, y_coord, c=th_all[ti0], cmap="inferno", s=2, vmin=th_vmin, vmax=th_vmax, rasterized=True)
    fig.colorbar(sc2, ax=axs[1, 0], label=f"ch{th_ch}")
    axs[1, 0].set_title("th / channel")
    sc3 = axs[1, 1].scatter(x_s, y_coord, c=mu_ch[ti0], cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(sc3, ax=axs[1, 1], label="ch3")
    axs[1, 1].set_title("mu / channel 3")

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    s_y = 0.09 if (regen_stems and next_holder is not None and enable_regenerate) else 0.02
    ax_slider = plt.axes((0.2, s_y, 0.52, 0.03))
    slider = Slider(
        ax_slider,
        "Time index",
        valmin=0,
        valmax=t_count - 1,
        valinit=0,
        valstep=1,
        color="teal",
    )

    def _upd(_val: float) -> None:
        k = int(slider.val)
        sc0.set_array(vel[k])
        sc1.set_array(p_all[k])
        sc2.set_array(th_all[k])
        sc3.set_array(mu_ch[k])
        tt = float(t_axis[k]) if k < len(t_axis) else float(k)
        fig.suptitle(f"Phase3 graph labels — {stem}  (t={tt:.6g})", fontsize=14, fontweight="bold")
        fig.canvas.draw_idle()

    slider.on_changed(_upd)

    if regen_stems is not None and next_holder is not None and current_stem_for_regen is not None:
        _attach_stem_navigation_controls(
            fig,
            current_stem=current_stem_for_regen,
            all_stems=regen_stems,
            next_holder=next_holder,
            enable=enable_regenerate,
        )

    plt.show()


def _plot_domain_single_time_dashboard(
    stem: str,
    export_dir: Path,
    sample_rows: int,
    *,
    regen_stems: list[str],
    next_holder: dict[str, str | None],
    current_stem_for_regen: str,
    enable_regenerate: bool,
) -> None:
    """One COMSOL time slice (narrow / first block), 2×2 + optional stem navigation."""
    domain_file = export_dir / f"{stem}.txt"
    df = _load_first_block(domain_file, sample_rows=sample_rows)
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    u = df["u"].to_numpy(dtype=np.float64)
    v = df["v"].to_numpy(dtype=np.float64)
    p = df["p"].to_numpy(dtype=np.float64)
    vel = np.sqrt(u**2 + v**2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.18, hspace=0.25)
    fig.suptitle(f"Phase3 domain (single time) — {stem}", fontsize=14, fontweight="bold")
    ax = axes.flatten()
    s0 = ax[0].scatter(x, y, c=vel, cmap=U_CMAP, s=2, rasterized=True)
    fig.colorbar(s0, ax=ax[0], label="|U|")
    ax[0].set_title("|U|")
    s1 = ax[1].scatter(x, y, c=p, cmap="coolwarm", s=2, rasterized=True)
    fig.colorbar(s1, ax=ax[1], label="p")
    ax[1].set_title("Pressure")
    s2 = ax[2].scatter(x, y, c=df["th"].to_numpy(dtype=np.float64), cmap="inferno", s=2, rasterized=True)
    fig.colorbar(s2, ax=ax[2], label="th")
    ax[2].set_title("th")
    mu_vals = df["mu_eff"].to_numpy(dtype=np.float64)
    mu_vmin, mu_vmax = _robust_vmin_vmax(mu_vals)
    s3 = ax[3].scatter(x, y, c=mu_vals, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(s3, ax=ax[3], label="mu_eff")
    ax[3].set_title("mu_eff (left/right: stem, r: random)")
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    _attach_stem_navigation_controls(
        fig,
        current_stem=current_stem_for_regen,
        all_stems=regen_stems,
        next_holder=next_holder,
        enable=enable_regenerate,
    )
    plt.show()


def _plot_graph_steady_dashboard_with_regen(
    data,
    stem: str,
    *,
    regen_stems: list[str],
    next_holder: dict[str, str | None],
    current_stem_for_regen: str,
    enable_regenerate: bool,
) -> None:
    pos = data.x[:, :2].detach().cpu().numpy()
    y_s = data.y
    if not torch.is_tensor(y_s):
        y_s = torch.as_tensor(y_s)
    y_s = y_s.float().cpu().numpy()
    if y_s.ndim != 2:
        return
    u, v = y_s[:, 0], y_s[:, 1]
    pr = y_s[:, 2] if y_s.shape[1] > 2 else np.zeros_like(u)
    vel = np.sqrt(u**2 + v**2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.18, hspace=0.25)
    fig.suptitle(f"Phase3 graph (steady) — {stem}", fontsize=14, fontweight="bold")
    ax = axes.flatten()
    s0 = ax[0].scatter(pos[:, 0], pos[:, 1], c=vel, cmap=U_CMAP, s=2, rasterized=True)
    fig.colorbar(s0, ax=ax[0], label="|U|")
    ax[0].set_title("|U|")
    s1 = ax[1].scatter(pos[:, 0], pos[:, 1], c=pr, cmap="coolwarm", s=2, rasterized=True)
    fig.colorbar(s1, ax=ax[1], label="p")
    ax[1].set_title("p")
    th_ch = min(11, y_s.shape[1] - 1)
    s2 = ax[2].scatter(pos[:, 0], pos[:, 1], c=y_s[:, th_ch], cmap="inferno", s=2, rasterized=True)
    fig.colorbar(s2, ax=ax[2], label=f"ch{th_ch}")
    ax[2].set_title("th / channel")
    mu_ch = y_s[:, 3] if y_s.shape[1] > 3 else np.zeros_like(u)
    mu_vmin, mu_vmax = _robust_vmin_vmax(mu_ch)
    s3 = ax[3].scatter(pos[:, 0], pos[:, 1], c=mu_ch, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(s3, ax=ax[3])
    ax[3].set_title("μ / ch3")
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    _attach_stem_navigation_controls(
        fig,
        current_stem=current_stem_for_regen,
        all_stems=regen_stems,
        next_holder=next_holder,
        enable=enable_regenerate,
    )
    plt.show()


def run_biochem_default_inspector(
    export_dir: Path,
    graph_dir: Path,
    *,
    start_stem: str | None,
    sample_rows: int,
    enable_regenerate: bool = True,
) -> None:
    """One anchor stem per figure; qualitative lines + matplotlib with stem navigation controls."""
    all_stems = _stem_candidates_union(export_dir, graph_dir)
    if not all_stems:
        print("No domain *.txt or graph *.pt found.")
        return

    n_show = min(5, len(all_stems))
    preview = ", ".join(all_stems[:n_show]) + (" …" if len(all_stems) > n_show else "")
    print(f"\n{len(all_stems)} stem(s) available [{preview}]. One window at a time.\n")

    current = start_stem if (start_stem in all_stems) else random.choice(all_stems)
    enable_btn = enable_regenerate

    while True:
        next_holder: dict[str, str | None] = {"value": None}
        print(f"--- Stem: {current} ---")
        domain_path = export_dir / f"{current}.txt"
        graph_path = graph_dir / f"{current}.pt"

        if domain_path.exists():
            try:
                inspect_boundaries(current, export_dir)
            except Exception as exc:
                print(f"(boundaries skipped: {exc})")
            try:
                audit_units(current, export_dir, sample_rows=max(1000, sample_rows))
            except Exception as exc:
                print(f"(unit audit skipped: {exc})")
        if graph_path.exists():
            try:
                summarize_graph(current, graph_dir)
            except Exception as exc:
                print(f"(graph summary skipped: {exc})")

        if domain_path.exists():
            try:
                _t, blocks = _load_comsol_trajectory(domain_path)
                if len(blocks) > 1:
                    plot_domain_trajectory_slider(
                        current,
                        export_dir,
                        regen_stems=all_stems,
                        next_holder=next_holder,
                        current_stem_for_regen=current,
                        enable_regenerate=enable_btn,
                    )
                else:
                    _plot_domain_single_time_dashboard(
                        current,
                        export_dir,
                        sample_rows,
                        regen_stems=all_stems,
                        next_holder=next_holder,
                        current_stem_for_regen=current,
                        enable_regenerate=enable_btn,
                    )
            except Exception as exc:
                print(f"Domain plot failed ({exc}).")
                return

        elif graph_path.exists():
            try:
                data = torch.load(graph_path, map_location="cpu", weights_only=False)
                y = getattr(data, "y", None)
                if y is not None and torch.is_tensor(y) and y.dim() == 3 and y.shape[0] > 1:
                    plot_graph_trajectory_slider(
                        data,
                        current,
                        regen_stems=all_stems,
                        next_holder=next_holder,
                        current_stem_for_regen=current,
                        enable_regenerate=enable_btn,
                    )
                elif y is not None and torch.is_tensor(y) and y.dim() == 2:
                    _plot_graph_steady_dashboard_with_regen(
                        data,
                        current,
                        regen_stems=all_stems,
                        next_holder=next_holder,
                        current_stem_for_regen=current,
                        enable_regenerate=enable_btn,
                    )
                else:
                    print("(Graph y not 2D/3D — nothing to plot.)")
            except Exception as exc:
                print(f"Graph plot failed ({exc}).")
                return
        else:
            print(f"No domain or graph file for stem {current!r}.")
            return

        if next_holder["value"] is None:
            break
        current = next_holder["value"]


def _boundary_mask(boundary_file: Path, tree: cKDTree, num_nodes: int, tolerance: float = 1e-6) -> np.ndarray:
    mask = np.zeros(num_nodes, dtype=bool)
    if not boundary_file.exists():
        return mask
    bdf = pd.read_csv(boundary_file, comment="%", sep=r"\s+", header=None)
    coords = np.unique(bdf.iloc[:, -2:].values, axis=0)
    dist, idx = tree.query(coords)
    valid = idx[dist < tolerance]
    mask[valid] = True
    return mask


def inspect_boundaries(stem: str, export_dir: Path) -> None:
    domain_file = export_dir / f"{stem}.txt"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain export: {domain_file}")

    df = _load_first_block(domain_file)
    tree = cKDTree(df[["x", "y"]].values)
    n = len(df)

    m_in = _boundary_mask(export_dir / f"{stem}_inlet.txt", tree, n)
    m_out = _boundary_mask(export_dir / f"{stem}_outlet.txt", tree, n)
    m_wall = _boundary_mask(export_dir / f"{stem}_wall.txt", tree, n)
    m_int = ~(m_in | m_out | m_wall)

    print(f"\n=== Boundary summary: {stem} ===")
    print(f"Nodes total     : {n}")
    print(f"Inlet nodes     : {int(m_in.sum())}")
    print(f"Outlet nodes    : {int(m_out.sum())}")
    print(f"Wall nodes      : {int(m_wall.sum())}")
    print(f"Interior nodes  : {int(m_int.sum())}")


def audit_units(stem: str, export_dir: Path, sample_rows: int = 50000) -> None:
    domain_file = export_dir / f"{stem}.txt"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain export: {domain_file}")

    # FIX 1: Load the full trajectory and extract the FINAL timestep, not t=0
    times, blocks = _load_comsol_trajectory(domain_file)
    if not times:
        print(f"Failed to load time trajectories for {stem}.")
        return

    final_t = sorted(blocks.keys())[-1]
    df = blocks[final_t]

    bio = BiochemConfig(phase="biochem")

    expected_cgs = {
        "rp": bio.c_RP0 / 1e6,
        "ap": (0.05 * bio.c_RP0) / 1e6,
        "apr": bio.APRcrit * 1e3,
        "aps": bio.APScrit * 1e3,
        "PT": bio.c_pT0 * 1e3,
        "th": bio.Tcrit * 1e3,
        "at": bio.cAT0 * 1e3,
        "fg": bio.c_Fg0 * 1e3,
        "fi": bio.c_Fg0 * 1e3,
        "M": bio.Minf / 1e4,
        "Mas": bio.Minf / 1e4,
        "Mat": bio.Minf / 1e4,
    }

    species_cols = ["rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"]

    print(f"\n=== Phase3 unit audit: {stem} (Evaluated at final t={final_t}) ===")

    # FIX 2: Swap p95+ for Max Val to capture highly localized boundary physics
    print(f"{'col':<5} {'Max Val':>12} {'ref(CGS)':>12} {'ratio':>10}  likely family")
    for col in species_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]

        # Track the absolute maximum concentration in the domain
        max_val = float(np.nanmax(vals)) if vals.size > 0 else 0.0

        ref = max(float(expected_cgs[col]), 1e-18)
        ratio = max_val / ref if max_val > 0 else 0.0

        if col in ("rp", "ap"):
            family = "plt/ml-ish" if 0.1 <= ratio <= 10 else "check"
        elif col in ("M", "Mas", "Mat"):
            family = "plt/cm^2-ish" if 0.1 <= ratio <= 10 else "check"
        else:
            family = "uM-ish" if 0.1 <= ratio <= 10 else "check"
        print(f"{col:<5} {max_val:12.4g} {ref:12.4g} {ratio:10.3g}  {family}")

    print("Hint: for phase-3 solutes in uM, conversion to SI is uM * 1e-3 -> mol/m³.")


def summarize_graph(stem: str, graph_dir: Path) -> None:
    graph_file = graph_dir / f"{stem}.pt"
    if not graph_file.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_file}")
    data = torch.load(graph_file, map_location="cpu", weights_only=False)
    print(f"\n=== Graph summary: {stem} ===")
    print(f"x shape              : {tuple(data.x.shape)}")
    print(f"y shape              : {tuple(data.y.shape) if hasattr(data, 'y') else '<missing>'}")
    print(f"num_nodes            : {int(data.num_nodes)}")
    print(f"num_edges            : {int(data.edge_index.shape[1])}")
    print(f"inlet/outlet/wall    : {int(data.mask_inlet.sum())}/{int(data.mask_outlet.sum())}/{int(data.mask_wall.sum())}")
    if hasattr(data, "t"):
        t = data.t.detach().cpu().numpy()
        if t.size > 1:
            print(f"time range           : {float(t.min()):.6g} -> {float(t.max()):.6g} (dt~{float(np.median(np.diff(t))):.6g})")
    if hasattr(data, "re_actual"):
        re_val = float(data.re_actual.mean().item()) if torch.is_tensor(data.re_actual) else float(data.re_actual)
        print(f"re_actual            : {re_val:.4g}")


def print_summary_table(phase: str, export_dir: Path, graph_dir: Path) -> None:
    stems = _domain_txt_stems(export_dir)
    if not stems:
        print(f"No biochem export stems found in {export_dir}")
        return

    print(f"\n=== Phase3 summary ({phase}) ===")
    print(f"{'stem':<24} {'times':>6} {'graph?':>8}")
    for stem in stems:
        times = _parse_times_from_header(export_dir / f"{stem}.txt")
        g_exists = (graph_dir / f"{stem}.pt").exists()
        print(f"{stem:<24} {len(times):>6} {str(g_exists):>8}")


def plot_domain_static(stem: str, export_dir: Path, *, sample_rows: int = 80_000) -> None:
    """Scatter / quiver for the first loaded block of ``{stem}.txt`` (same sampling as audits)."""
    domain_file = export_dir / f"{stem}.txt"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain export: {domain_file}")
    df = _load_first_block(domain_file, sample_rows=sample_rows)
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    u = df["u"].to_numpy(dtype=np.float64)
    v = df["v"].to_numpy(dtype=np.float64)
    p = df["p"].to_numpy(dtype=np.float64)
    mu = df["mu_eff"].to_numpy(dtype=np.float64)
    vel = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ax = axes.flatten()

    s0 = ax[0].scatter(x, y, c=vel, cmap=U_CMAP, s=2, rasterized=True)
    fig.colorbar(s0, ax=ax[0], label="|U|")
    ax[0].set_title("|U|")
    s1 = ax[1].scatter(x, y, c=p, cmap="coolwarm", s=2, rasterized=True)
    fig.colorbar(s1, ax=ax[1], label="p")
    ax[1].set_title("Pressure")
    mu_vmin, mu_vmax = _robust_vmin_vmax(mu)
    s2 = ax[2].scatter(x, y, c=mu, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(s2, ax=ax[2], label="mu_eff")
    ax[2].set_title("mu_eff")

    s3 = ax[3].scatter(x, y, c=df["th"].to_numpy(dtype=np.float64), cmap="inferno", s=2, rasterized=True)
    fig.colorbar(s3, ax=ax[3], label="th")
    ax[3].set_title("th (sample)")

    s4 = ax[4].scatter(x, y, c=df["rp"].to_numpy(dtype=np.float64), cmap="cividis", s=2, rasterized=True)
    fig.colorbar(s4, ax=ax[4], label="rp")
    ax[4].set_title("rp (sample)")

    scat_mu = ax[5].scatter(x, y, c=mu, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
    fig.colorbar(scat_mu, ax=ax[5], label="mu_eff")
    ax[5].set_title("mu_eff")

    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    fig.suptitle(f"Phase3 domain export — {stem} (first block, up to {sample_rows} rows)")
    plt.tight_layout()
    plt.show()


def _y_slice_for_plot(data: Data) -> np.ndarray:
    """Return a [N, C] numpy snapshot from ``data.y`` (steady or transient)."""
    y = data.y
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.float().cpu()
    if y.dim() == 3:
        # [T, N, C] transient stores
        y = y[-1]
    return y.numpy()


def plot_graph_static(stem: str, graph_dir: Path) -> None:
    """Node scatter: kinematics + dynamic channels from a processed ``.pt`` graph."""
    graph_file = graph_dir / f"{stem}.pt"
    if not graph_file.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_file}")
    data = torch.load(graph_file, map_location="cpu", weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    y_s = _y_slice_for_plot(data)
    u = y_s[:, 0]
    v = y_s[:, 1]
    pr = y_s[:, 2] if y_s.shape[1] > 2 else np.zeros_like(u)
    vel = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes.flatten()

    s0 = ax[0].scatter(pos[:, 0], pos[:, 1], c=vel, cmap=U_CMAP, s=2, rasterized=True)
    fig.colorbar(s0, ax=ax[0], label="|U| (labels)")
    ax[0].set_title("|U| from y")
    s1 = ax[1].scatter(pos[:, 0], pos[:, 1], c=pr, cmap="coolwarm", s=2, rasterized=True)
    fig.colorbar(s1, ax=ax[1], label="p")
    ax[1].set_title("Pressure from y")
    th_ch = min(11, y_s.shape[1] - 1)
    s2 = ax[2].scatter(pos[:, 0], pos[:, 1], c=y_s[:, th_ch], cmap="inferno", s=2, rasterized=True)
    fig.colorbar(s2, ax=ax[2], label=f"ch{th_ch}")
    ax[2].set_title("th / channel")
    mu_ch = y_s[:, 3] if y_s.shape[1] > 3 else np.zeros_like(u)
    s3 = ax[3].scatter(pos[:, 0], pos[:, 1], c=mu_ch, cmap=MU_CMAP, s=2, rasterized=True)
    fig.colorbar(s3, ax=ax[3], label="mu / extra")
    ax[3].set_title("Channel 3+ (mu_nd or extra)")
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    fig.suptitle(f"Phase3 graph — {stem}")
    plt.tight_layout()
    plt.show()


def inspect_domain_interactive(*, export_dir: Path, start_stem: str | None, sample_rows: int) -> None:
    stems = _domain_txt_stems(export_dir)
    if not stems:
        print(f"No domain stems in {export_dir}")
        return
    current = start_stem if start_stem in stems else random.choice(stems)

    while True:
        next_holder: dict[str, str | None] = {"value": None}
        print(f"\nPlotting domain stem: {current}")
        domain_file = export_dir / f"{current}.txt"
        df = _load_first_block(domain_file, sample_rows=sample_rows)
        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)
        u = df["u"].to_numpy(dtype=np.float64)
        v = df["v"].to_numpy(dtype=np.float64)
        p = df["p"].to_numpy(dtype=np.float64)
        vel = np.sqrt(u**2 + v**2)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes.flatten()
        s0 = ax[0].scatter(x, y, c=vel, cmap=U_CMAP, s=2, rasterized=True)
        fig.colorbar(s0, ax=ax[0], label="|U|")
        ax[0].set_title(f"|U| — {current}")
        s1 = ax[1].scatter(x, y, c=p, cmap="coolwarm", s=2, rasterized=True)
        fig.colorbar(s1, ax=ax[1], label="p")
        ax[1].set_title("Pressure")
        s2 = ax[2].scatter(x, y, c=df["th"].to_numpy(dtype=np.float64), cmap="inferno", s=2, rasterized=True)
        fig.colorbar(s2, ax=ax[2], label="th")
        ax[2].set_title("th")
        mu_vals = df["mu_eff"].to_numpy(dtype=np.float64)
        mu_vmin, mu_vmax = _robust_vmin_vmax(mu_vals)
        s3 = ax[3].scatter(x, y, c=mu_vals, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
        fig.colorbar(s3, ax=ax[3], label="mu_eff")
        ax[3].set_title("mu_eff (left/right: stem, r: random)")
        for a in ax:
            a.set_aspect("equal")
            a.axis("off")
        plt.tight_layout()

        def _set_stem(target: str) -> None:
            print(f"\nSwitching domain view: {target}")
            next_holder["value"] = target
            plt.close(fig)

        def _prev() -> None:
            if len(stems) <= 1:
                print("Only one stem; cannot move.")
                return
            i_cur = stems.index(current)
            _set_stem(stems[(i_cur - 1) % len(stems)])

        def _next() -> None:
            if len(stems) <= 1:
                print("Only one stem; cannot move.")
                return
            i_cur = stems.index(current)
            _set_stem(stems[(i_cur + 1) % len(stems)])

        def _regen() -> None:
            candidates = [s for s in stems if s != current]
            if len(candidates) == 0:
                print("Only one stem; cannot pick random.")
                return
            _set_stem(random.choice(candidates))

        prev_ax = fig.add_axes([0.60, 0.02, 0.12, 0.06])
        next_ax = fig.add_axes([0.73, 0.02, 0.12, 0.06])
        rand_ax = fig.add_axes([0.86, 0.02, 0.12, 0.06])
        Button(prev_ax, "Prev").on_clicked(lambda _e: _prev())
        Button(next_ax, "Next").on_clicked(lambda _e: _next())
        Button(rand_ax, "Random").on_clicked(lambda _e: _regen())
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: (
                _prev() if getattr(e, "key", None) in ("left", "p")
                else _next() if getattr(e, "key", None) in ("right", "n")
                else _regen() if getattr(e, "key", None) == "r"
                else None
            ),
        )
        plt.show()
        if next_holder["value"] is None:
            break
        current = next_holder["value"]


def inspect_graph_interactive(*, graph_dir: Path, start_stem: str | None) -> None:
    stems = _list_graph_stems(graph_dir)
    if not stems:
        print(f"No *.pt graphs in {graph_dir}")
        return
    current = start_stem if start_stem in stems else random.choice(stems)

    while True:
        next_holder: dict[str, str | None] = {"value": None}
        print(f"\nPlotting graph stem: {current}")
        data = torch.load(graph_dir / f"{current}.pt", map_location="cpu", weights_only=False)
        pos = data.x[:, :2].detach().cpu().numpy()
        y_s = _y_slice_for_plot(data)
        u, v = y_s[:, 0], y_s[:, 1]
        vel = np.sqrt(u**2 + v**2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        s0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=vel, cmap=U_CMAP, s=2, rasterized=True)
        fig.colorbar(s0, ax=axes[0], label="|U|")
        axes[0].set_title(f"|U| — {current}")
        mu_ch = y_s[:, 3] if y_s.shape[1] > 3 else np.zeros_like(u)
        mu_vmin, mu_vmax = _robust_vmin_vmax(mu_ch)
        s1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=mu_ch, cmap=MU_CMAP, s=2, vmin=mu_vmin, vmax=mu_vmax, rasterized=True)
        fig.colorbar(s1, ax=axes[1], label="ch3")
        axes[1].set_title("mu / channel 3 (left/right: stem, r: random)")
        for a in axes:
            a.set_aspect("equal")
            a.axis("off")
        plt.tight_layout()
        def _set_stem(target: str) -> None:
            print(f"\nSwitching graph view: {target}")
            next_holder["value"] = target
            plt.close(fig)

        def _prev() -> None:
            if len(stems) <= 1:
                print("Only one graph; cannot move.")
                return
            i_cur = stems.index(current)
            _set_stem(stems[(i_cur - 1) % len(stems)])

        def _next() -> None:
            if len(stems) <= 1:
                print("Only one graph; cannot move.")
                return
            i_cur = stems.index(current)
            _set_stem(stems[(i_cur + 1) % len(stems)])

        def _regen() -> None:
            candidates = [s for s in stems if s != current]
            if len(candidates) == 0:
                print("Only one graph; cannot pick random.")
                return
            _set_stem(random.choice(candidates))

        prev_ax = fig.add_axes([0.60, 0.02, 0.12, 0.08])
        next_ax = fig.add_axes([0.73, 0.02, 0.12, 0.08])
        rand_ax = fig.add_axes([0.86, 0.02, 0.12, 0.08])
        Button(prev_ax, "Prev").on_clicked(lambda _e: _prev())
        Button(next_ax, "Next").on_clicked(lambda _e: _next())
        Button(rand_ax, "Random").on_clicked(lambda _e: _regen())
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: (
                _prev() if getattr(e, "key", None) in ("left", "p")
                else _next() if getattr(e, "key", None) in ("right", "n")
                else _regen() if getattr(e, "key", None) == "r"
                else None
            ),
        )
        plt.show()
        if next_holder["value"] is None:
            break
        current = next_holder["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect biochem exports and processed graphs.")
    parser.add_argument("--phase", type=str, default="biochem_anchors", choices=list(_PHASE_CHOICES))
    parser.add_argument("--stem", type=str, default=None, help="Stem name without extension (e.g. vessel_001).")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the stems/times/graph table only, then exit (no matplotlib).",
    )
    parser.add_argument("--boundaries", action="store_true", help="Print boundary node counts for the selected stem.")
    parser.add_argument("--unit-audit", action="store_true", help="Run unit-magnitude audit for the selected stem.")
    parser.add_argument("--graph-summary", action="store_true", help="Print processed .pt summary for the selected stem.")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Rows to sample for unit audit and domain plots.")
    parser.add_argument(
        "--plot-domain",
        action="store_true",
        help="Matplotlib scatter/quiver snapshot of the domain export (first CSV block).",
    )
    parser.add_argument(
        "--plot-domain-interactive",
        action="store_true",
        help="Interactive domain view with Prev/Next stem controls (left/right keys); 'r' for random.",
    )
    parser.add_argument(
        "--plot-graph",
        action="store_true",
        help="Static plot of a processed graph .pt (labels + channel views).",
    )
    parser.add_argument(
        "--plot-graph-interactive",
        action="store_true",
        help="Interactive graph view with Prev/Next stem controls (left/right keys); 'r' for random.",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Default mode: disable stem navigation controls/hotkeys (default is on).",
    )
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.phase)
    graph_dir = _resolve_graph_dir(args.phase)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")

    plot_flags = (
        args.boundaries,
        args.unit_audit,
        args.graph_summary,
        args.plot_domain,
        args.plot_domain_interactive,
        args.plot_graph,
        args.plot_graph_interactive,
    )
    any_explicit = any(plot_flags)

    if args.summary:
        print_summary_table(args.phase, export_dir, graph_dir)
        return

    if not any_explicit:
        run_biochem_default_inspector(
            export_dir,
            graph_dir,
            start_stem=args.stem,
            sample_rows=max(1000, args.sample_rows),
            enable_regenerate=not args.no_regenerate,
        )
        return

    stem = args.stem
    needs_auto_stem = stem is None and any(plot_flags) and not (
        args.plot_domain_interactive or args.plot_graph_interactive
    )
    if needs_auto_stem:
        domain_stems = _domain_txt_stems(export_dir)
        graph_stems = _list_graph_stems(graph_dir)
        needs_domain = args.boundaries or args.unit_audit or args.plot_domain
        needs_graph_file = args.plot_graph or args.graph_summary
        if needs_graph_file:
            if not graph_stems:
                raise FileNotFoundError(f"No graph *.pt files in {graph_dir}")
            stem = graph_stems[0]
        elif needs_domain:
            if not domain_stems:
                raise FileNotFoundError(f"No domain *.txt stems in {export_dir}")
            stem = domain_stems[0]
        elif domain_stems:
            stem = domain_stems[0]
        elif graph_stems:
            stem = graph_stems[0]
        else:
            raise FileNotFoundError(f"No domain txt or graph stems under {export_dir} / {graph_dir}")

    if args.boundaries and stem:
        inspect_boundaries(stem, export_dir)
    if args.unit_audit and stem:
        audit_units(stem, export_dir, sample_rows=max(1000, args.sample_rows))
    if args.graph_summary and stem:
        summarize_graph(stem, graph_dir)

    if args.plot_domain and stem:
        plot_domain_static(stem, export_dir, sample_rows=max(1000, args.sample_rows))

    if args.plot_graph and stem:
        plot_graph_static(stem, graph_dir)

    if args.plot_domain_interactive:
        inspect_domain_interactive(
            export_dir=export_dir,
            start_stem=args.stem,
            sample_rows=max(1000, args.sample_rows),
        )

    if args.plot_graph_interactive:
        inspect_graph_interactive(graph_dir=graph_dir, start_stem=args.stem)


if __name__ == "__main__":
    main()
