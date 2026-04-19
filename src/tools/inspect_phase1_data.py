"""
Phase 1 anchor inspector: COMSOL `vessel_*.npz` (tier1/tier2).

**Default behavior** (no extra flags): same style as ``vessel_generator`` — if ``--tier`` is omitted, **Tier** is
chosen interactively (``1`` or ``2``). Non-interactive scripts should pass ``--tier 1`` or ``--tier 2``. Then
full-directory **health scan**
(quality flags + optional CSV) and **interactive** matplotlib (random ``vessel_*.npz`` or ``--sample-idx``;
Regenerate button / ``r`` key).

When a matching processed graph ``vessel_<idx>.pt`` exists, the interactive plot shows **network targets (y)**
and **mesh-based priors (x)** in non-dimensional form; otherwise it falls back to raw SI fields from the ``.npz``.

Examples:
    python -m src.tools.inspect_phase1_data
    python src/tools/inspect_phase1_data.py
    python -m src.tools.inspect_phase1_data --tier 1
    python -m src.tools.inspect_phase1_data --tier 1 --sample-idx 10
    python -m src.tools.inspect_phase1_data --tier 1 --summary
    python -m src.tools.inspect_phase1_data --tier 1 --scan-only
    python -m src.tools.inspect_phase1_data --tier 1 --skip-health-scan
    python -m src.tools.inspect_phase1_data --tier 1 --plot-static --sample-idx 0
    python -m src.tools.inspect_phase1_data --inspect-template-tags
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

# Allow running as ``python src/tools/inspect_phase1_data.py`` (IDE / full path): put repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "src").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

from src.config import NodeFeat, PhysicsConfig, PredChannels, VesselConfig
from src.utils.paths import get_project_root, reports_dir


def _resolve_anchor_dir(tier: str) -> Path:
    root = get_project_root()
    cfg = VesselConfig(tier=tier)
    p = Path(cfg.output_dir)
    return p if p.is_absolute() else root / p


def _iter_anchor_files(anchor_dir: Path):
    return sorted(anchor_dir.glob("vessel_*.npz"))


def _extract_idx(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def _list_sample_indices(tier: str) -> list[int]:
    out: list[int] = []
    for f in _iter_anchor_files(_resolve_anchor_dir(tier)):
        idx = _extract_idx(f)
        if idx is not None:
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


def summary(tier: str) -> None:
    anchor_dir = _resolve_anchor_dir(tier)
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
    print(f"\n=== Phase1 anchor summary ({tier}) ===")
    print(f"anchor dir      : {anchor_dir}")
    print(f"files total     : {len(rows)}")
    print(f"files valid     : {len(valid)}")
    print(f"files invalid   : {len(rows) - len(valid)}")
    if valid:
        print(f"p_std median    : {np.median([r['p_std'] for r in valid]):.3e}")
        print(f"vel_max median  : {np.median([r['vel_max'] for r in valid]):.3e}")
        print(f"nan_ratio max   : {np.max([r['nan_ratio'] for r in valid]):.3e}")


def health_scan_anchors(tier: str, *, export_csv: bool = True) -> list[dict]:
    """Full-directory scan with optional quality flags and ``outputs/reports/<tier>_anchor_health.csv``."""
    data_dir = _resolve_anchor_dir(tier)
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

    if export_csv and rows:
        out_path = reports_dir() / f"{tier}_anchor_health.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_idx",
                    "ok",
                    "reason",
                    "nodes",
                    "vel_min",
                    "vel_max",
                    "vel_mean",
                    "p_min",
                    "p_max",
                    "p_std",
                    "u_abs_max",
                    "has_mu",
                    "mu_min",
                    "mu_max",
                    "nan_ratio",
                    "quality_flags",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.get("sample_idx"),
                        r.get("ok"),
                        r.get("reason"),
                        r.get("nodes"),
                        r.get("vel_min"),
                        r.get("vel_max"),
                        r.get("vel_mean"),
                        r.get("p_min"),
                        r.get("p_max"),
                        r.get("p_std"),
                        r.get("u_abs_max"),
                        r.get("has_mu"),
                        r.get("mu_min"),
                        r.get("mu_max"),
                        r.get("nan_ratio"),
                        "|".join(r.get("quality_flags", [])),
                    ]
                )
        print(f"Wrote anchor health CSV: {out_path}")
    return rows


def _prompt_int_choice(label: str, allowed: Tuple[int, ...]) -> int:
    """Read an integer from stdin until it is one of ``allowed`` (same pattern as ``vessel_generator``)."""
    allowed_str = "/".join(str(x) for x in allowed)
    while True:
        raw = input(f"{label} ({allowed_str}): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Enter an integer: {allowed_str}")
            continue
        if v in allowed:
            return v
        print(f"  Must be one of: {allowed_str}")


def _resolve_tier_from_cli(cli_tier: int | None) -> str:
    """Use ``--tier`` (1 or 2) when set; else prompt like ``vessel_generator`` (no isatty shortcut)."""
    if cli_tier is not None:
        return f"tier{cli_tier}"
    tier_n = _prompt_int_choice("Tier", (1, 2))
    tier = f"tier{tier_n}"
    anchor_dir = _resolve_anchor_dir(tier)
    n_npz = len(list(_iter_anchor_files(anchor_dir)))
    print("\n--- Phase 1 anchor inspection ---")
    print(f"  Anchor NPZ directory: {anchor_dir}")
    print(f"  vessel_*.npz files found: {n_npz}")
    print()
    return tier


def _resolve_graph_pt_path(tier: str, sample_idx: int) -> Path:
    root = get_project_root()
    cfg = VesselConfig(tier=tier)
    proc = Path(cfg.graph_output_dir)
    return proc if proc.is_absolute() else root / proc


def _try_load_graph_data(tier: str, sample_idx: int) -> dict | None:
    """Load both mesh-based priors (x) and scaled training labels (y) from ``vessel_<idx>.pt``."""
    pt_path = _resolve_graph_pt_path(tier, sample_idx) / f"vessel_{sample_idx}.pt"
    if not pt_path.exists():
        return None
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
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
    tier: str,
    *,
    graph_data: dict | None,
    d_bar_npz: float | None,
) -> float | None:
    """Reference velocity [m/s] for ``|U|_nd = |U|_SI / u_ref`` (same as ``mesh_to_graph``)."""
    if graph_data is not None and graph_data.get("u_ref") is not None:
        return float(graph_data["u_ref"])
    phys = PhysicsConfig(tier=tier)
    db = None
    if graph_data is not None and graph_data.get("d_bar") is not None:
        db = float(graph_data["d_bar"])
    elif d_bar_npz is not None:
        db = float(d_bar_npz)
    if db is None or db <= 0.0:
        return None
    return float(phys.get_u_ref(db))


def _load_anchor_npz(sample_idx: int, tier: str):
    data_dir = _resolve_anchor_dir(tier)
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


def plot_sample_static(tier: str, sample_idx: int) -> None:
    """Single-window scatter / quiver (no regenerate loop)."""
    anchor_dir = _resolve_anchor_dir(tier)
    file_path = anchor_dir / f"vessel_{sample_idx}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Sample not found: {file_path}")
    gd = _try_load_graph_data(tier, sample_idx)
    with np.load(file_path) as npz:
        x = np.asarray(npz["x"]).reshape(-1)
        y = np.asarray(npz["y"]).reshape(-1)
        u = np.asarray(npz["u"]).reshape(-1)
        v = np.asarray(npz["v"]).reshape(-1)
        p = np.asarray(npz["p"]).reshape(-1)
        vel = np.sqrt(u**2 + v**2)
        mu = np.asarray(npz["mu"]).reshape(-1) if "mu" in npz else None
        d_bar_npz = float(np.asarray(npz["d_bar"]).reshape(-1)[0]) if "d_bar" in npz.files else None

    if gd is not None:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        u_ref = _resolve_u_ref_anchor(tier, graph_data=gd, d_bar_npz=d_bar_npz)
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
        s1 = axes[0, 1].scatter(x, y, c=p, cmap="coolwarm", s=2)
        fig.colorbar(s1, ax=axes[0, 1], label="p")
        axes[0, 1].set_title("COMSOL pressure")
        if mu is not None:
            s2 = axes[0, 2].scatter(x, y, c=mu, cmap="magma", s=2)
            fig.colorbar(s2, ax=axes[0, 2], label="mu")
            axes[0, 2].set_title("COMSOL viscosity")
        else:
            axes[0, 2].axis("off")

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

    fig.suptitle(f"{tier} sample vessel_{sample_idx}")
    plt.tight_layout()
    plt.show()


def inspect_anchor_interactive(*, sample_idx: int, tier: str, enable_regenerate: bool = True) -> None:
    """Interactive 2x2 view with optional random-resample button and ``r`` hotkey."""
    current_idx = int(sample_idx)
    all_indices = _list_sample_indices(tier) if enable_regenerate else []

    while True:
        data, file_path = _load_anchor_npz(sample_idx=current_idx, tier=tier)
        if data is None:
            return

        next_idx_holder: dict[str, int | None] = {"value": None}

        print(f"\nLoading: {file_path.name}")
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

            graph_data = _try_load_graph_data(tier, current_idx)

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

                sc1 = axes[0, 1].scatter(px, py, c=graph_data["p_label"], cmap="plasma", s=2)
                fig.colorbar(sc1, ax=axes[0, 1], label="Pressure (ND Label)")
                axes[0, 1].set_title("Target Pressure")
                axes[0, 1].set_aspect("equal")

                if graph_data["mu_label"] is not None:
                    sc2 = axes[0, 2].scatter(px, py, c=graph_data["mu_label"], cmap="magma", s=2)
                    fig.colorbar(sc2, ax=axes[0, 2], label=r"Viscosity $\mu$ (ND Label)")
                    axes[0, 2].set_title("Target Viscosity")
                    axes[0, 2].set_aspect("equal")
                else:
                    axes[0, 2].axis("off")

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
                    f"No graph plotted (missing {_resolve_graph_pt_path(tier, current_idx) / f'vessel_{current_idx}.pt'}). "
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

            plt.show()
        except Exception as e:
            print(f"Error inspecting data: {e}")
            return
        finally:
            data.close()

        if next_idx_holder["value"] is None:
            break
        current_idx = int(next_idx_holder["value"])


def inspect_template_tags() -> None:
    cfg = VesselConfig(tier="tier1")
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
    parser = argparse.ArgumentParser(description="Inspect Phase1 anchor data, health CSV, plots, and template tags.")
    parser.add_argument(
        "--tier",
        type=int,
        choices=(1, 2),
        default=None,
        metavar="N",
        help="Phase-1 anchor tier: 1=tier1, 2=tier2. Omit for an interactive prompt (see vessel_generator).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact statistics only (no full health scan, no plot). Exits after one table.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Full-directory health scan + optional CSV, then exit (no matplotlib window).",
    )
    parser.add_argument(
        "--skip-health-scan",
        action="store_true",
        help="Skip full-directory scan; open interactive plot only (after default, use with care).",
    )
    parser.add_argument("--no-export-csv", action="store_true", help="Disable CSV export during health scan.")
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
        help="Run live COMSOL tag inspection for phase1 template (.mph + mph package required).",
    )
    args = parser.parse_args()

    if args.inspect_template_tags:
        inspect_template_tags()
        return

    tier = _resolve_tier_from_cli(args.tier)

    if args.scan_only:
        health_scan_anchors(tier, export_csv=(not args.no_export_csv))
        return

    if args.summary:
        summary(tier)
        return

    if not args.skip_health_scan:
        health_scan_anchors(tier, export_csv=(not args.no_export_csv))

    if args.plot_static:
        if args.sample_idx is None:
            raise ValueError("--plot-static requires --sample-idx")
        plot_sample_static(tier, args.sample_idx)
        return

    data_dir = _resolve_anchor_dir(tier)
    if args.sample_idx is not None:
        sample_idx = args.sample_idx
    else:
        files = sorted(data_dir.glob("vessel_*.npz"))
        if not files:
            print(f"No vessel_*.npz files found in {data_dir}")
            return
        sample_idx = int(random.choice(files).stem.split("_")[-1])
        print(f"\nRandom sample selected for plotting: vessel_{sample_idx}.npz")

    inspect_anchor_interactive(
        sample_idx=sample_idx,
        tier=tier,
        enable_regenerate=not args.no_regenerate,
    )


if __name__ == "__main__":
    main()
