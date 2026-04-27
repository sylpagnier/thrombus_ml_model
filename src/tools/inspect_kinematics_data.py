"""
Kinematics anchor inspector: COMSOL `vessel_*.npz` (newtonian/carreau).

**Default behavior** (no extra flags): runs on unified ``kinematics`` data. Then full-directory **health scan**
(quality flags printed to console) and **interactive** matplotlib (random ``vessel_*.npz`` or ``--sample-idx``;
Regenerate button / ``r`` key).

When a matching processed graph ``vessel_<idx>.pt`` exists, the interactive plot shows **network targets (y)**
and **mesh-based priors (x)** in non-dimensional form; otherwise it falls back to raw SI fields from the ``.npz``.

Examples:
    python -m src.tools.inspect_kinematics_data
    python src/tools/inspect_kinematics_data.py
    python -m src.tools.inspect_kinematics_data --phase kinematics
    python -m src.tools.inspect_kinematics_data --phase kinematics --rheology newtonian
    python -m src.tools.inspect_kinematics_data --phase kinematics --rheology carreau --sample-idx 10
    python -m src.tools.inspect_kinematics_data --phase kinematics --summary
    python -m src.tools.inspect_kinematics_data --phase kinematics --scan-only
    python -m src.tools.inspect_kinematics_data --phase kinematics --skip-health-scan
    python -m src.tools.inspect_kinematics_data --phase kinematics --plot-static --sample-idx 0
    python -m src.tools.inspect_kinematics_data --inspect-template-tags
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as ``python src/tools/inspect_kinematics_data.py`` (IDE / full path): put repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "src").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

from src.config import NodeFeat, PhysicsConfig, PredChannels, VesselConfig
from src.utils.paths import get_project_root


def _format_n_subdir(n_value: float) -> str:
    return f"n_{float(n_value):.3f}"


def _resolve_kinematics_n_subdir(base_dir: Path, requested: str | None, *, kind: str) -> str:
    """Resolve kinematics subdir (newtonian/carreau, legacy n_*, or auto)."""
    legacy_aliases = {"newtonian": "newtonian", "carreau": "carreau"}
    if requested:
        raw = requested.strip()
        low = raw.lower()
        if low in legacy_aliases:
            chosen = legacy_aliases[low]
            chosen_dir = base_dir / chosen
            if not chosen_dir.is_dir():
                raise FileNotFoundError(
                    f"Requested --rheology '{requested}' not found under {base_dir} for {kind}."
                )
            return chosen
        chosen = raw if raw.startswith("n_") else _format_n_subdir(float(raw))
        chosen_dir = base_dir / chosen
        if not chosen_dir.is_dir():
            raise FileNotFoundError(
                f"Requested subdir '{requested}' not found under {base_dir} for {kind}."
            )
        return chosen

    # First, support flat layouts where vessel_*.npz / vessel_*.pt are in the root.
    if list(base_dir.glob("vessel_*.*")):
        return ""

    # Current layout keeps per-rheology folders under kinematics.
    available = []
    for name in ("carreau", "newtonian"):
        d = base_dir / name
        if d.is_dir() and list(d.glob("vessel_*.*")):
            available.append(name)
    if available:
        chosen = "carreau" if "carreau" in available else available[0]
        return chosen

    return ""


def _resolve_anchor_dir(phase: str, n_subdir: str | None = None) -> Path:
    root = get_project_root()
    cfg = VesselConfig(phase=phase)
    p = Path(cfg.output_dir)
    base = p if p.is_absolute() else root / p
    if phase == "kinematics":
        chosen = _resolve_kinematics_n_subdir(base, n_subdir, kind="anchor labels")
        return base / chosen
    return base


def _iter_anchor_files(anchor_dir: Path):
    return sorted(anchor_dir.glob("vessel_*.npz"))


def _extract_idx(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


_GRAPH_ANCHOR_CACHE: dict[tuple[str, str | None, int], bool] = {}


def _has_labeled_anchor_graph(phase: str, sample_idx: int, n_subdir: str | None = None) -> bool:
    """True when ``vessel_<idx>.pt`` exists and marks ``is_anchor=True``."""
    cache_key = (phase, n_subdir, int(sample_idx))
    if cache_key in _GRAPH_ANCHOR_CACHE:
        return _GRAPH_ANCHOR_CACHE[cache_key]

    pt_path = _resolve_graph_pt_path(phase, int(sample_idx), n_subdir=n_subdir) / f"vessel_{sample_idx}.pt"
    if not pt_path.exists():
        _GRAPH_ANCHOR_CACHE[cache_key] = False
        return False
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        _GRAPH_ANCHOR_CACHE[cache_key] = False
        return False
    is_anchor_attr = getattr(data, "is_anchor", None)
    if is_anchor_attr is None:
        _GRAPH_ANCHOR_CACHE[cache_key] = False
        return False
    try:
        result = bool(np.asarray(is_anchor_attr).reshape(-1)[0])
    except Exception:
        result = False
    _GRAPH_ANCHOR_CACHE[cache_key] = result
    return result


def _list_sample_indices(
    phase: str,
    n_subdir: str | None = None,
    *,
    require_labeled_graph: bool = False,
) -> list[int]:
    out: list[int] = []
    for f in _iter_anchor_files(_resolve_anchor_dir(phase, n_subdir=n_subdir)):
        idx = _extract_idx(f)
        if idx is not None and (not require_labeled_graph or _has_labeled_anchor_graph(phase, idx, n_subdir=n_subdir)):
            out.append(idx)
    return out


def _compute_metrics_from_npz(data) -> dict:
    """Health metrics for a loaded ``np.load`` handle (``NpzFile`` or mapping)."""
    keys = list(data.keys())
    if "x" not in keys or "y" not in keys:
        return {"ok": False, "reason": "missing_xy"}
    if "u" not in keys or "v" not in keys or "p" not in keys:
        return {"ok": False, "reason": "missing_uvp"}

    x = np.asarray(data["x"]).reshape(-1)
    u = np.asarray(data["u"]).reshape(-1)
    v = np.asarray(data["v"]).reshape(-1)
    p = np.asarray(data["p"]).reshape(-1)
    vel_mag = np.sqrt(u**2 + v**2)

    has_mu = "mu" in keys
    mu = np.asarray(data["mu"]).reshape(-1) if has_mu else None

    total_nodes = len(x)
    nan_count = int(np.isnan(u).sum() + np.isnan(v).sum() + np.isnan(p).sum())
    if has_mu:
        nan_count += int(np.isnan(mu).sum())
    denom = max(total_nodes * (4 if has_mu else 3), 1)
    nan_ratio = nan_count / denom

    p_std = float(np.nanstd(p)) if p.size else 0.0
    u_abs_max = float(np.nanmax(np.abs(u))) if u.size else 0.0

    flags: list[str] = []
    if nan_ratio > 0.0:
        flags.append("has_nans")
    if p_std < 1e-6:
        flags.append("flat_pressure")
    if u_abs_max < 1e-5:
        flags.append("low_velocity")
    if has_mu and (float(np.nanmin(mu)) <= 0.0 or float(np.nanmax(mu)) > 20.0):
        flags.append("mu_outlier")

    return {
        "ok": True,
        "nodes": total_nodes,
        "vel_min": float(np.nanmin(vel_mag)),
        "vel_max": float(np.nanmax(vel_mag)),
        "vel_mean": float(np.nanmean(vel_mag)),
        "p_min": float(np.nanmin(p)),
        "p_max": float(np.nanmax(p)),
        "p_std": p_std,
        "u_abs_max": u_abs_max,
        "has_mu": has_mu,
        "mu_min": (float(np.nanmin(mu)) if has_mu else None),
        "mu_max": (float(np.nanmax(mu)) if has_mu else None),
        "nan_ratio": nan_ratio,
        "quality_flags": flags,
    }


def summary(phase: str, n_subdir: str | None = None) -> None:
    anchor_dir = _resolve_anchor_dir(phase, n_subdir=n_subdir)
    files = list(_iter_anchor_files(anchor_dir))
    if not files:
        print(f"No vessel_*.npz files found in {anchor_dir}")
        return

    rows: list[dict] = []
    for f in files:
        idx = _extract_idx(f)
        try:
            with np.load(f) as npz:
                m = _compute_metrics_from_npz(npz)
        except Exception as exc:
            m = {"ok": False, "reason": f"load_error:{exc}"}
        m["sample_idx"] = idx
        rows.append(m)

    valid = [r for r in rows if r.get("ok")]
    print(f"\n=== Kinematics anchor summary ({phase}) ===")
    print(f"anchor dir      : {anchor_dir}")
    print(f"files total     : {len(rows)}")
    print(f"files valid     : {len(valid)}")
    print(f"files invalid   : {len(rows) - len(valid)}")
    if valid:
        print(f"p_std median    : {np.median([r['p_std'] for r in valid]):.3e}")
        print(f"vel_max median  : {np.median([r['vel_max'] for r in valid]):.3e}")
        print(f"nan_ratio max   : {np.max([r['nan_ratio'] for r in valid]):.3e}")


def health_scan_anchors(phase: str, n_subdir: str | None = None) -> list[dict]:
    """Full-directory scan with quality flags printed to console."""
    data_dir = _resolve_anchor_dir(phase, n_subdir=n_subdir)
    files = sorted(data_dir.glob("vessel_*.npz"))
    if not files:
        print(f"No vessel_*.npz files found in {data_dir}")
        return []

    print(f"\nScanning {len(files)} anchors in {data_dir} ...")
    rows: list[dict] = []
    for f in files:
        try:
            sample_idx = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            d = np.load(f)
            m = _compute_metrics_from_npz(d)
            d.close()
        except Exception as e:
            m = {"ok": False, "reason": f"load_error:{e}"}
        m["sample_idx"] = sample_idx
        rows.append(m)

    valid = [r for r in rows if r.get("ok")]
    invalid = [r for r in rows if not r.get("ok")]
    flagged = [r for r in valid if len(r.get("quality_flags", [])) > 0]

    print("\n--- Anchor health scan ---")
    print(f"Total files: {len(rows)}")
    print(f"Valid files: {len(valid)}")
    print(f"Invalid files: {len(invalid)}")
    print(f"Flagged quality files: {len(flagged)}")

    if valid:
        pstd = np.array([r["p_std"] for r in valid], dtype=float)
        uabs = np.array([r["u_abs_max"] for r in valid], dtype=float)
        nanr = np.array([r["nan_ratio"] for r in valid], dtype=float)
        print(
            f"p_std median={np.median(pstd):.3e} | "
            f"u_abs_max median={np.median(uabs):.3e} | "
            f"nan_ratio max={np.max(nanr):.3e}"
        )

    if flagged:
        print("\nTop flagged anchors:")
        ranked = sorted(flagged, key=lambda x: (len(x["quality_flags"]), x["nan_ratio"]), reverse=True)
        for r in ranked[:20]:
            print(
                f"  vessel_{r['sample_idx']}: flags={r['quality_flags']} "
                f"p_std={r['p_std']:.2e} u_abs_max={r['u_abs_max']:.2e} nan_ratio={r['nan_ratio']:.2e}"
            )

    return rows


def _resolve_phase_from_cli(cli_phase: str | None) -> str:
    """Resolve CLI phase; keep legacy phase aliases for compatibility."""
    raw = (cli_phase or "kinematics").strip().lower()
    legacy_aliases = {"1", "2", "phase1", "phase2"}
    if raw in legacy_aliases:
        print("Note: legacy phase selector is deprecated; using unified 'kinematics' dataset.")
        phase = "kinematics"
    else:
        phase = raw

    if phase != "kinematics":
        raise ValueError(f"Unsupported phase '{cli_phase}'. Use --phase kinematics.")

    anchor_dir = _resolve_anchor_dir(phase)
    n_npz = len(list(_iter_anchor_files(anchor_dir)))
    print("\n--- Kinematics anchor inspection ---")
    print(f"  Anchor NPZ directory: {anchor_dir}")
    print(f"  vessel_*.npz files found: {n_npz}")
    print()
    return phase


def _resolve_graph_pt_path(phase: str, sample_idx: int, n_subdir: str | None = None) -> Path:
    root = get_project_root()
    cfg = VesselConfig(phase=phase)
    proc = Path(cfg.graph_output_dir)
    base = proc if proc.is_absolute() else root / proc
    if phase == "kinematics":
        chosen = _resolve_kinematics_n_subdir(base, n_subdir, kind="graph outputs")
        return base / chosen
    return base


def _try_load_graph_data(
    phase: str,
    sample_idx: int,
    n_subdir: str | None = None,
    *,
    expected_nodes: int | None = None,
) -> dict | None:
    """Load mesh priors/labels from ``vessel_<idx>.pt`` when graph is a labeled anchor."""
    pt_path = _resolve_graph_pt_path(phase, sample_idx, n_subdir=n_subdir) / f"vessel_{sample_idx}.pt"
    if not pt_path.exists():
        return None
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    is_anchor_attr = getattr(data, "is_anchor", None)
    if is_anchor_attr is None:
        return None
    try:
        is_anchor = bool(np.asarray(is_anchor_attr).reshape(-1)[0])
    except Exception:
        return None
    if not is_anchor:
        return None
    if (
        not hasattr(data, "x")
        or not hasattr(data, "y")
        or data.x.shape[1] < NodeFeat.WSS_PRIOR.stop
        or data.y is None
        or data.y.shape[1] < 3
    ):
        return None

    d_bar = float(data.d_bar.item()) if getattr(data, "d_bar", None) is not None else 1.0
    u_ref_t = getattr(data, "u_ref", None)
    u_ref = float(np.asarray(u_ref_t).reshape(-1)[0]) if u_ref_t is not None else None

    # 1. Physical Coordinates (for plotting)
    xy_nd = data.x[:, NodeFeat.XY].cpu().numpy()
    x = xy_nd[:, 0] * d_bar
    y = xy_nd[:, 1] * d_bar

    # 2. Graph Priors (x features)
    uv_p = data.x[:, NodeFeat.UV_PRIOR].cpu().numpy()
    u_p, v_p = uv_p[:, 0], uv_p[:, 1]
    vel_p = np.sqrt(u_p**2 + v_p**2)
    mu_p = data.x[:, NodeFeat.MU_PRIOR].cpu().numpy().reshape(-1)
    wss_p = data.x[:, NodeFeat.WSS_PRIOR].cpu().numpy().reshape(-1)

    # 3. Graph Labels (y targets) -> scaled data the network trains on
    y_tensor = data.y.cpu().numpy()
    u_label = y_tensor[:, PredChannels.U]
    v_label = y_tensor[:, PredChannels.V]
    vel_label = np.sqrt(u_label**2 + v_label**2)
    p_label = y_tensor[:, PredChannels.P]

    has_mu = y_tensor.shape[1] > PredChannels.MU_EFF_ND
    mu_label = y_tensor[:, PredChannels.MU_EFF_ND] if has_mu else None

    if expected_nodes is not None and int(y_tensor.shape[0]) != int(expected_nodes):
        return None

    return {
        "x": x,
        "y": y,
        "u_prior": u_p,
        "v_prior": v_p,
        "vel_prior": vel_p,
        "mu_prior": mu_p,
        "wss_prior": wss_p,
        "u_label": u_label,
        "v_label": v_label,
        "vel_label": vel_label,
        "p_label": p_label,
        "mu_label": mu_label,
        "path": pt_path,
        "d_bar": d_bar,
        "u_ref": u_ref,
    }


def _resolve_u_ref_anchor(
    phase: str,
    *,
    graph_data: dict | None,
    d_bar_npz: float | None,
) -> float | None:
    """Reference velocity [m/s] for ``|U|_nd = |U|_SI / u_ref`` (same as ``mesh_to_graph``)."""
    if graph_data is not None and graph_data.get("u_ref") is not None:
        return float(graph_data["u_ref"])
    phys = PhysicsConfig(phase=phase)
    db = None
    if graph_data is not None and graph_data.get("d_bar") is not None:
        db = float(graph_data["d_bar"])
    elif d_bar_npz is not None:
        db = float(d_bar_npz)
    if db is None or db <= 0.0:
        return None
    return float(phys.get_u_ref(db))


def _load_anchor_npz(sample_idx: int, phase: str, n_subdir: str | None = None):
    data_dir = _resolve_anchor_dir(phase, n_subdir=n_subdir)
    file_path = data_dir / f"vessel_{sample_idx}.npz"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None, None
    try:
        data = np.load(file_path)
        return data, file_path
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return None, None


def plot_sample_static(phase: str, sample_idx: int, n_subdir: str | None = None) -> None:
    """Single-window scatter / quiver (no regenerate loop)."""
    anchor_dir = _resolve_anchor_dir(phase, n_subdir=n_subdir)
    file_path = anchor_dir / f"vessel_{sample_idx}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Sample not found: {file_path}")
    if not _has_labeled_anchor_graph(phase, sample_idx, n_subdir=n_subdir):
        raise FileNotFoundError(
            f"Sample vessel_{sample_idx} is not available as a labeled anchor graph "
            f"under {_resolve_graph_pt_path(phase, sample_idx, n_subdir=n_subdir)}."
        )
    with np.load(file_path) as npz:
        x = np.asarray(npz["x"]).reshape(-1)
        y = np.asarray(npz["y"]).reshape(-1)
        u = np.asarray(npz["u"]).reshape(-1)
        v = np.asarray(npz["v"]).reshape(-1)
        p = np.asarray(npz["p"]).reshape(-1)
        vel = np.sqrt(u**2 + v**2)
        mu = np.asarray(npz["mu"]).reshape(-1) if "mu" in npz else None
        d_bar_npz = float(np.asarray(npz["d_bar"]).reshape(-1)[0]) if "d_bar" in npz.files else None
    gd = _try_load_graph_data(phase, sample_idx, n_subdir=n_subdir, expected_nodes=len(x))

    if gd is not None:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        u_ref = _resolve_u_ref_anchor(phase, graph_data=gd, d_bar_npz=d_bar_npz)
        if u_ref is not None and u_ref > 0.0:
            vel_plot = vel / u_ref
            vp = gd["vel_prior"]
            vmin_u = min(float(np.nanmin(vel_plot)), float(np.nanmin(vp)))
            vmax_u = max(float(np.nanmax(vel_plot)), float(np.nanmax(vp)))
            if not np.isfinite(vmin_u) or not np.isfinite(vmax_u) or vmax_u <= vmin_u:
                vmin_u, vmax_u = 0.0, 1.0
        else:
            vel_plot = vel
            vmin_u, vmax_u = None, None
        kw_u = {"cmap": "jet", "s": 2}
        if vmin_u is not None:
            kw_u["vmin"], kw_u["vmax"] = vmin_u, vmax_u
        s0 = axes[0, 0].scatter(x, y, c=vel_plot, **kw_u)
        u_label = "|U|/u_ref (COMSOL, ND)" if u_ref else "|U| (COMSOL)"
        fig.colorbar(s0, ax=axes[0, 0], label=u_label)
        axes[0, 0].set_title("COMSOL |U|")
        if mu is not None:
            s2 = axes[0, 1].scatter(x, y, c=mu, cmap="magma", s=2)
            fig.colorbar(s2, ax=axes[0, 1], label="mu")
            axes[0, 1].set_title("COMSOL viscosity")
        else:
            axes[0, 1].axis("off")

        s1 = axes[0, 2].scatter(x, y, c=p, cmap="coolwarm", s=2)
        fig.colorbar(s1, ax=axes[0, 2], label="p")
        axes[0, 2].set_title("COMSOL pressure")

        px, py = gd["x"], gd["y"]
        kw_p = {"cmap": "jet", "s": 2}
        if vmin_u is not None:
            kw_p["vmin"], kw_p["vmax"] = vmin_u, vmax_u
        sp0 = axes[1, 0].scatter(px, py, c=gd["vel_prior"], **kw_p)
        fig.colorbar(sp0, ax=axes[1, 0], label="|U_prior| (ND)")
        axes[1, 0].set_title("Prior |U| (mesh)")
        sp1 = axes[1, 1].scatter(px, py, c=gd["mu_prior"], cmap="plasma", s=2)
        fig.colorbar(sp1, ax=axes[1, 1], label="mu_prior")
        axes[1, 1].set_title("Prior viscosity")
        sp2 = axes[1, 2].scatter(px, py, c=gd["wss_prior"], cmap="inferno", s=2)
        fig.colorbar(sp2, ax=axes[1, 2], label="wss_prior")
        axes[1, 2].set_title("Prior WSS (wall)")

        k = 20 if len(x) > 1000 else 1
        kp = 20 if len(px) > 1000 else 1
        axes[2, 0].quiver(x[::k], y[::k], u[::k], v[::k], color="black")
        axes[2, 0].set_title("COMSOL velocity vectors")
        axes[2, 1].quiver(px[::kp], py[::kp], gd["u_prior"][::kp], gd["v_prior"][::kp], color="darkgreen")
        axes[2, 1].set_title("Prior velocity vectors")
        axes[2, 2].axis("off")
        axes[2, 2].text(
            0.5,
            0.5,
            f"Priors from\n{gd['path'].name}\n(mesh discretization;\nCOMSOL grid above)",
            ha="center",
            va="center",
            fontsize=10,
            transform=axes[2, 2].transAxes,
        )
        for row in axes:
            for a in row:
                a.set_aspect("equal")
                a.axis("off")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes.flatten()
        s0 = ax[0].scatter(x, y, c=vel, cmap="viridis", s=2)
        fig.colorbar(s0, ax=ax[0], label="|U|")
        ax[0].set_title("Velocity magnitude")
        s1 = ax[1].scatter(x, y, c=p, cmap="coolwarm", s=2)
        fig.colorbar(s1, ax=ax[1], label="p")
        ax[1].set_title("Pressure")
        if mu is not None:
            s2 = ax[2].scatter(x, y, c=mu, cmap="magma", s=2)
            fig.colorbar(s2, ax=ax[2], label="mu")
            ax[2].set_title("Viscosity")
        else:
            ax[2].axis("off")
        k = 20 if len(x) > 1000 else 1
        ax[3].quiver(x[::k], y[::k], u[::k], v[::k], color="black")
        ax[3].set_title("Velocity vectors")
        for a in ax:
            a.set_aspect("equal")
            a.axis("off")

    fig.suptitle(f"{phase} sample vessel_{sample_idx}")
    plt.tight_layout()
    plt.show()


def inspect_anchor_interactive(
    *, sample_idx: int, phase: str, n_subdir: str | None = None, enable_regenerate: bool = True
) -> None:
    """Interactive 2x2 view with optional random-resample button and ``r`` hotkey."""
    current_idx = int(sample_idx)
    current_subdir = n_subdir
    toggle_options: list[str] = []
    if phase == "kinematics":
        for candidate in ("newtonian", "carreau"):
            if _list_sample_indices(phase, n_subdir=candidate, require_labeled_graph=True):
                toggle_options.append(candidate)

    while True:
        all_indices = _list_sample_indices(phase, n_subdir=current_subdir, require_labeled_graph=True)
        if not all_indices:
            print(
                f"No labeled anchor samples found in {_resolve_anchor_dir(phase, n_subdir=current_subdir)} "
                f"with matching anchor graphs."
            )
            return
        if all_indices and current_idx not in all_indices:
            current_idx = int(random.choice(all_indices))

        data, file_path = _load_anchor_npz(sample_idx=current_idx, phase=phase, n_subdir=current_subdir)
        if data is None:
            return

        next_idx_holder: dict[str, int | None] = {"value": None}
        next_subdir_holder: dict[str, str | None] = {"value": None}
        active_dir_name = _resolve_anchor_dir(phase, n_subdir=current_subdir).name

        print(f"\nLoading: {file_path.name}")
        if active_dir_name in ("newtonian", "carreau"):
            print(f"Rheology: {active_dir_name}")
        try:
            keys = list(data.keys())
            print(f"Available Keys: {keys}")
            if "x" not in keys or "y" not in keys:
                print("Spatial coordinates (x, y) missing. Cannot plot spatial map.")
                return

            x = data["x"].flatten()
            y = data["y"].flatten()
            u = data["u"].flatten()
            v = data["v"].flatten()
            p = data["p"].flatten()
            vel_mag = np.sqrt(u**2 + v**2)

            has_mu = "mu" in keys
            mu = data["mu"].flatten() if has_mu else None

            graph_data = _try_load_graph_data(
                phase,
                current_idx,
                n_subdir=current_subdir,
                expected_nodes=len(x),
            )

            print(f"--- Data Summary (Sample {current_idx}) ---")
            if "d_bar" in keys:
                print(f"Mean Diameter (d_bar): {data['d_bar']:.4f} m")
            print(f"Nodes: {len(x)}")
            print(f"Velocity Range: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s")
            print(f"Pressure Range: {p.min():.4f} - {p.max():.4f} Pa")
            if has_mu:
                print(f"Viscosity Range: {mu.min():.6f} - {mu.max():.6f} Pa*s")

            if graph_data is not None:
                print(f"Loaded scaled labels (y) and priors (x) from graph: {graph_data['path']}")
                print(f"--- Graph Data Summary (Sample {current_idx}) ---")
                print(
                    f"Velocity Target Range (ND): {graph_data['vel_label'].min():.4f} - {graph_data['vel_label'].max():.4f}"
                )
                print(
                    f"Pressure Target Range (ND): {graph_data['p_label'].min():.4f} - {graph_data['p_label'].max():.4f}"
                )

                fig, axes = plt.subplots(3, 3, figsize=(14, 12))
                px, py = graph_data["x"], graph_data["y"]

                vl = graph_data["vel_label"]
                vp = graph_data["vel_prior"]
                vmin_u = min(float(np.nanmin(vl)), float(np.nanmin(vp)))
                vmax_u = max(float(np.nanmax(vl)), float(np.nanmax(vp)))
                if not np.isfinite(vmin_u) or not np.isfinite(vmax_u) or vmax_u <= vmin_u:
                    vmin_u, vmax_u = 0.0, 1.0
                u_kw = {"cmap": "jet", "s": 2, "vmin": vmin_u, "vmax": vmax_u}

                # ROW 1: Training labels (y)
                sc0 = axes[0, 0].scatter(px, py, c=vl, **u_kw)
                fig.colorbar(sc0, ax=axes[0, 0], label="|U| (ND Label)")
                axes[0, 0].set_title(f"Target |U| (Sample {current_idx})")
                axes[0, 0].set_aspect("equal")

                if graph_data["mu_label"] is not None:
                    sc2 = axes[0, 1].scatter(px, py, c=graph_data["mu_label"], cmap="magma", s=2)
                    fig.colorbar(sc2, ax=axes[0, 1], label=r"Viscosity $\mu$ (ND Label)")
                    axes[0, 1].set_title("Target Viscosity")
                    axes[0, 1].set_aspect("equal")
                else:
                    axes[0, 1].axis("off")

                sc1 = axes[0, 2].scatter(px, py, c=graph_data["p_label"], cmap="plasma", s=2)
                fig.colorbar(sc1, ax=axes[0, 2], label="Pressure (ND Label)")
                axes[0, 2].set_title("Target Pressure")
                axes[0, 2].set_aspect("equal")

                # ROW 2: Network priors (x)
                sp0 = axes[1, 0].scatter(px, py, c=vp, **u_kw)
                fig.colorbar(sp0, ax=axes[1, 0], label="|U_prior| (ND Feature)")
                axes[1, 0].set_title("Prior |U|")
                axes[1, 0].set_aspect("equal")

                sp1 = axes[1, 1].scatter(px, py, c=graph_data["mu_prior"], cmap="plasma", s=2)
                fig.colorbar(sp1, ax=axes[1, 1], label="mu_prior (ND Feature)")
                axes[1, 1].set_title("Prior Viscosity")
                axes[1, 1].set_aspect("equal")

                sp2 = axes[1, 2].scatter(px, py, c=graph_data["wss_prior"], cmap="inferno", s=2)
                fig.colorbar(sp2, ax=axes[1, 2], label="wss_prior (ND Feature)")
                axes[1, 2].set_title("Prior WSS")
                axes[1, 2].set_aspect("equal")

                # ROW 3: Vector comparison
                kp = 20 if len(px) > 1000 else 1
                tscale = max(float(graph_data["vel_label"].max()) * 10.0, 1e-8)
                axes[2, 0].quiver(
                    px[::kp],
                    py[::kp],
                    graph_data["u_label"][::kp],
                    graph_data["v_label"][::kp],
                    color="white",
                    alpha=0.8,
                    scale=tscale,
                )
                axes[2, 0].set_facecolor("black")
                axes[2, 0].set_title("Target Vectors (press 'r')")
                axes[2, 0].set_aspect("equal")

                pscale = max(float(graph_data["vel_prior"].max()) * 10.0, 1e-8)
                axes[2, 1].quiver(
                    px[::kp],
                    py[::kp],
                    graph_data["u_prior"][::kp],
                    graph_data["v_prior"][::kp],
                    color="white",
                    alpha=0.8,
                    scale=pscale,
                )
                axes[2, 1].set_facecolor("black")
                axes[2, 1].set_title("Prior Vectors")
                axes[2, 1].set_aspect("equal")
                axes[2, 2].axis("off")
            else:
                print(
                    f"No compatible labeled anchor graph plotted (missing/unlabeled/node-count mismatch in "
                    f"{_resolve_graph_pt_path(phase, current_idx, n_subdir=current_subdir) / f'vessel_{current_idx}.pt'}). "
                    "Displaying raw unscaled SI COMSOL data."
                )
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                ax = axes.flatten()

                sc0 = ax[0].scatter(x, y, c=vel_mag, cmap="viridis", s=2)
                plt.colorbar(sc0, ax=ax[0], label="|U| (m/s)")
                ax[0].set_title(f"Velocity Magnitude (Sample {current_idx})")
                ax[0].set_aspect("equal")

                sc1 = ax[1].scatter(x, y, c=p, cmap="plasma", s=2)
                plt.colorbar(sc1, ax=ax[1], label="Relative Pressure (Pa)")
                ax[1].set_title("Relative Pressure Field")
                ax[1].set_aspect("equal")

                if has_mu:
                    sc2 = ax[2].scatter(x, y, c=mu, cmap="magma", s=2)
                    plt.colorbar(sc2, ax=ax[2], label=r"Viscosity $\mu$ (Pa*s)")
                    ax[2].set_title("Dynamic Viscosity Field")
                    ax[2].set_aspect("equal")
                else:
                    ax[2].axis("off")

                k = 20 if len(x) > 1000 else 1
                scale = max(float(vel_mag.max()) * 10.0, 1e-8)
                ax[3].quiver(x[::k], y[::k], u[::k], v[::k], color="white", alpha=0.8, scale=scale)
                ax[3].set_facecolor("black")
                ax[3].set_title("Velocity Vector Field (press 'r' or click Regenerate)")
                ax[3].set_aspect("equal")

            figure_title = f"{phase} sample vessel_{current_idx}"
            if active_dir_name in ("newtonian", "carreau"):
                figure_title = f"{figure_title} [{active_dir_name}]"
            fig.suptitle(figure_title)
            plt.tight_layout()
            if enable_regenerate and len(all_indices) > 1:

                def _pick_next_random() -> int | None:
                    candidates = [i for i in all_indices if i != current_idx]
                    if not candidates:
                        print("Only one sample available; cannot regenerate another random sample.")
                        return None
                    return random.choice(candidates)

                def _regenerate() -> None:
                    next_idx = _pick_next_random()
                    if next_idx is None:
                        return
                    print(f"\nRegenerating with random sample: vessel_{next_idx}.npz")
                    next_idx_holder["value"] = int(next_idx)
                    plt.close(fig)

                def _on_key(event):
                    if event.key == "r":
                        _regenerate()

                btn_ax = fig.add_axes([0.74, 0.02, 0.23, 0.05])
                regen_btn = Button(btn_ax, "Regenerate Random")
                regen_btn.on_clicked(lambda _event: _regenerate())
                fig.canvas.mpl_connect("key_press_event", _on_key)

            if len(toggle_options) > 1 and active_dir_name in toggle_options:
                target_subdir = toggle_options[1] if active_dir_name == toggle_options[0] else toggle_options[0]

                def _toggle_rheology() -> None:
                    target_indices = _list_sample_indices(
                        phase,
                        n_subdir=target_subdir,
                        require_labeled_graph=True,
                    )
                    if not target_indices:
                        print(f"No samples found in rheology '{target_subdir}'.")
                        return
                    next_subdir_holder["value"] = target_subdir
                    if current_idx in target_indices:
                        next_idx_holder["value"] = int(current_idx)
                    else:
                        next_idx_holder["value"] = int(random.choice(target_indices))
                    print(
                        f"\nSwitching rheology to {target_subdir}: "
                        f"vessel_{next_idx_holder['value']}.npz"
                    )
                    plt.close(fig)

                toggle_ax = fig.add_axes([0.50, 0.02, 0.22, 0.05])
                toggle_btn = Button(toggle_ax, f"Switch to {target_subdir}")
                toggle_btn.on_clicked(lambda _event: _toggle_rheology())

            plt.show()
        except Exception as e:
            print(f"Error inspecting data: {e}")
            return
        finally:
            data.close()

        if next_subdir_holder["value"] is not None:
            current_subdir = next_subdir_holder["value"]
        if next_idx_holder["value"] is None:
            break
        current_idx = int(next_idx_holder["value"])


def inspect_template_tags() -> None:
    cfg = VesselConfig(phase="kinematics")
    template = Path(cfg.template_path)
    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")
    try:
        import mph
    except Exception as exc:
        raise RuntimeError("`mph` is required for template inspection.") from exc

    print(f"Inspecting template: {template}")
    client = mph.start()
    model = client.load(str(template))
    try:
        comp_tags = model.java.component().tags()
        print("\n=== COMSOL TAG INSPECTION ===")
        for c_tag in comp_tags:
            comp = model.java.component(c_tag)
            print(f"\nComponent: {c_tag}")
            mesh_tags = comp.mesh().tags()
            print(f"  Meshes: {list(mesh_tags)}")
            phys_tags = comp.physics().tags()
            print(f"  Physics: {list(phys_tags)}")
            mat_tags = comp.material().tags()
            print(f"  Materials: {list(mat_tags)}")
    finally:
        try:
            model.remove()
        except Exception:
            pass
        try:
            client.clear()
            client.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect kinematics anchor data, health scan, plots, and template tags.")
    parser.add_argument(
        "--phase",
        type=str,
        default="kinematics",
        metavar="NAME",
        help=(
            "Dataset phase (default: kinematics). "
            "Legacy aliases 1/2/phase1/phase2 are accepted and mapped to kinematics."
        ),
    )
    parser.add_argument(
        "--rheology",
        type=str,
        default=None,
        choices=("newtonian", "carreau"),
        help=(
            "Kinematics rheology folder to inspect. Default: auto-select (prefers carreau when both exist)."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact statistics only (no full health scan, no plot). Exits after one table.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Full-directory health scan, then exit (no matplotlib window).",
    )
    parser.add_argument(
        "--skip-health-scan",
        action="store_true",
        help="Skip full-directory scan; open interactive plot only (after default, use with care).",
    )
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index (vessel_<idx>.npz) for plotting.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="No-op: interactive plotting is the default. Kept for compatibility with older invocations.",
    )
    parser.add_argument(
        "--plot-static",
        action="store_true",
        help="Single-window static plot instead of interactive; requires --sample-idx.",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Disable random-resample button / 'r' key on the interactive plot.",
    )
    parser.add_argument(
        "--inspect-template-tags",
        action="store_true",
        help="Run live COMSOL tag inspection for kinematics template (.mph + mph package required).",
    )
    parser.add_argument(
        "--n-subdir",
        type=str,
        default=None,
        help=(
            "Deprecated alias for kinematics subdir selection (legacy n_* folders). "
            "Prefer --rheology newtonian|carreau."
        ),
    )
    args = parser.parse_args()

    if args.inspect_template_tags:
        inspect_template_tags()
        return

    phase = _resolve_phase_from_cli(args.phase)
    selected_subdir = args.rheology if args.rheology is not None else args.n_subdir

    if args.scan_only:
        health_scan_anchors(phase, n_subdir=selected_subdir)
        return

    if args.summary:
        summary(phase, n_subdir=selected_subdir)
        return

    if not args.skip_health_scan:
        health_scan_anchors(phase, n_subdir=selected_subdir)

    if args.plot_static:
        if args.sample_idx is None:
            raise ValueError("--plot-static requires --sample-idx")
        plot_sample_static(phase, args.sample_idx, n_subdir=selected_subdir)
        return

    data_dir = _resolve_anchor_dir(phase, n_subdir=selected_subdir)
    if args.sample_idx is not None:
        if not _has_labeled_anchor_graph(phase, args.sample_idx, n_subdir=selected_subdir):
            raise FileNotFoundError(
                f"Sample vessel_{args.sample_idx} is not available as a labeled anchor graph "
                f"in {_resolve_graph_pt_path(phase, args.sample_idx, n_subdir=selected_subdir)}."
            )
        sample_idx = args.sample_idx
    else:
        indices = _list_sample_indices(phase, n_subdir=selected_subdir, require_labeled_graph=True)
        if not indices:
            print(f"No labeled anchor samples found in {data_dir}")
            return
        sample_idx = int(random.choice(indices))
        print(f"\nRandom sample selected for plotting: vessel_{sample_idx}.npz")

    inspect_anchor_interactive(
        sample_idx=sample_idx,
        phase=phase,
        n_subdir=selected_subdir,
        enable_regenerate=not args.no_regenerate,
    )


if __name__ == "__main__":
    main()
