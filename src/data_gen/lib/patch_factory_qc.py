"""patch_factory_qc.py
-------------------
Quality control for the Patch Factory dataset (``patch_{i}.npz`` + ``patch_{i}.json``).

The dataset is only useful if the residual ``dU = U - shear_rate*y`` is the clot's signal
and *nothing else*. QC is grouped into three buckets:

  (a) Baseline purity  -- away from the clot, ``du`` / ``dv`` must be ~0. This is the whole
      reason we moved to a structured grid + prescribed-freestream top. A nonzero far-field
      residual means mesh noise or a baseline-BC error is leaking into the label.
  (b) Physics/conservation -- no-slip bottom wall (u,v ~ 0 at y=0), inlet pins the baseline
      (du ~ 0 at x=0), incompressible mass balance (inflow == outflow), top magnitude
      ``max u ~ shear_rate*H``, finite fields.
  (c) Perturbation sanity -- inside/over the high-viscosity clot the flow is *slowed*
      (mean du < 0), the perturbation has a healthy signal-to-noise ratio over the far field,
      and the viscosity field peaks near ``clot_mu``.

All residuals are normalized by the characteristic velocity ``U_ref = shear_rate * height``.

CLI:
    python -m src.data_gen.lib.patch_factory_qc --dir data/processed/cfd_results_patch_factory --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.paths import data_root, reports_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class QCThresholds:
    """Relative thresholds (vs U_ref) above which a sample is flagged."""
    baseline_rms: float = 1e-2      # far-field |du|/U_ref RMS
    wall_noslip: float = 1e-2       # |u| at bottom wall / U_ref
    inlet_du: float = 2e-2          # |du| at inlet / U_ref
    top_rms: float = 2e-2           # far-field decay at the lid / U_ref
    mass_rel_err: float = 1e-2      # |flux_out - flux_in| / |flux_in|
    magnitude_rel_err: float = 5e-2 # |max(u) - shear_rate*H| / (shear_rate*H)
    min_snr: float = 5.0            # clot peak |du| / far-field RMS du
    min_clot_dmu: float = 0.5       # max(mu) / clot_mu (viscosity actually rose)


# np.trapz was renamed to np.trapezoid in NumPy 2.0 (and removed from some 2.x builds).
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _reshape(arr: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(ny, nx)


def load_patch(npz_path: Path) -> Dict[str, Any]:
    """Load one patch npz into flat arrays + scalar metadata."""
    z = np.load(npz_path)
    d: Dict[str, Any] = {k: z[k] for k in z.files}
    d["path"] = Path(npz_path)
    d["idx"] = int(d.get("config_id", -1))
    for k in (
        "length", "height", "grid_spacing", "shear_rate", "clot_x_center",
        "clot_width", "clot_height", "clot_mu_peak",
    ):
        if k in d:
            d[k] = float(np.asarray(d[k]).ravel()[0])
    d["grid_nx"] = int(np.asarray(d["grid_nx"]).ravel()[0])
    d["grid_ny"] = int(np.asarray(d["grid_ny"]).ravel()[0])
    d["clot_shape"] = str(np.asarray(d.get("clot_shape", "")).ravel()[0])
    d["is_dry_run"] = bool(np.asarray(d.get("dry_run", False)).ravel()[0]) if "dry_run" in d else False
    return d


def _flux(u: np.ndarray, y: np.ndarray, col_mask: np.ndarray) -> float:
    """Volumetric inflow per unit depth across a column: integral of u dy."""
    yy = y[col_mask]
    uu = u[col_mask]
    order = np.argsort(yy)
    return float(_trapezoid(uu[order], yy[order]))


def compute_sample_qc(d: Dict[str, Any], thr: Optional[QCThresholds] = None) -> Dict[str, Any]:
    """Per-sample physics + baseline + perturbation diagnostics."""
    thr = thr or QCThresholds()
    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.float64)
    u = np.asarray(d["u"], dtype=np.float64)
    v = np.asarray(d["v"], dtype=np.float64)
    du = np.asarray(d.get("du", u - d["shear_rate"] * y), dtype=np.float64)
    dv = np.asarray(d.get("dv", v), dtype=np.float64)
    mu = np.asarray(d["mu"], dtype=np.float64) if "mu" in d else None

    H = float(d["height"])
    L = float(d["length"])
    gdot = float(d["shear_rate"])
    cx = float(d["clot_x_center"])
    cw = float(d["clot_width"])
    ch = float(d["clot_height"])
    u_ref = max(gdot * H, 1e-12)

    eps_x = 0.51 * float(d.get("grid_spacing", (x.max() - x.min()) / max(d["grid_nx"] - 1, 1)))
    eps_y = eps_x

    # --- region masks ---
    wall_mask = y <= (y.min() + eps_y)
    top_mask = y >= (y.max() - eps_y)
    inlet_mask = x <= (x.min() + eps_x)
    outlet_mask = x >= (x.max() - eps_x)
    # Upstream "fully developed" strip: flow here has not seen the clot.
    far_mask = x <= (cx - 2.0 * cw)
    if far_mask.sum() < 16:  # clot near inlet: fall back to a top-corner far field
        far_mask = top_mask & (x <= (cx - cw))
    if far_mask.sum() < 16:
        far_mask = top_mask
    # Region where the clot perturbation is expected (footprint +/- a width, low y band).
    clot_infl = (np.abs(x - cx) <= 1.5 * cw)
    clot_core = (np.abs(x - cx) <= 0.5 * cw) & (y <= ch)

    def _rms(a: np.ndarray, m: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a[m] ** 2))) if m.any() else float("nan")

    finite_ok = bool(np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(du).all())

    far_du_rms = _rms(du, far_mask) / u_ref
    far_dv_rms = _rms(dv, far_mask) / u_ref
    top_du_rms = _rms(du, top_mask) / u_ref
    top_dv_rms = _rms(dv, top_mask) / u_ref
    wall_u_max = float(np.max(np.abs(u[wall_mask]))) / u_ref if wall_mask.any() else float("nan")
    wall_v_max = float(np.max(np.abs(v[wall_mask]))) / u_ref if wall_mask.any() else float("nan")
    inlet_du_max = float(np.max(np.abs(du[inlet_mask]))) / u_ref if inlet_mask.any() else float("nan")

    clot_du_peak = float(np.max(np.abs(du[clot_infl]))) / u_ref if clot_infl.any() else float("nan")
    clot_du_mean = float(np.mean(du[clot_core])) / u_ref if clot_core.any() else float("nan")
    far_du_abs = _rms(du, far_mask)
    snr = (clot_du_peak * u_ref) / far_du_abs if (far_du_abs > 1e-30) else float("inf")

    u_top = float(np.max(u[top_mask])) if top_mask.any() else float(np.max(u))
    mag_rel_err = abs(u_top - gdot * H) / max(gdot * H, 1e-12)

    flux_in = _flux(u, y, inlet_mask)
    flux_out = _flux(u, y, outlet_mask)
    mass_rel_err = abs(flux_out - flux_in) / max(abs(flux_in), 1e-30)

    mu_max = float(np.max(mu)) if mu is not None else float("nan")
    mu_min = float(np.min(mu)) if mu is not None else float("nan")
    clot_dmu_ratio = (mu_max / float(d["clot_mu_peak"])) if (mu is not None and d.get("clot_mu_peak", 0) > 0) else float("nan")

    flags: List[str] = []
    if not finite_ok:
        flags.append("nonfinite")
    if math.isfinite(far_du_rms) and far_du_rms > thr.baseline_rms:
        flags.append("baseline_impure")
    if math.isfinite(top_du_rms) and top_du_rms > thr.top_rms:
        flags.append("top_not_freestream")
    if math.isfinite(wall_u_max) and wall_u_max > thr.wall_noslip:
        flags.append("wall_noslip_violation")
    if math.isfinite(inlet_du_max) and inlet_du_max > thr.inlet_du:
        flags.append("inlet_baseline_off")
    if math.isfinite(mass_rel_err) and mass_rel_err > thr.mass_rel_err:
        flags.append("mass_imbalance")
    if math.isfinite(mag_rel_err) and mag_rel_err > thr.magnitude_rel_err:
        flags.append("magnitude_off")
    if math.isfinite(snr) and snr < thr.min_snr and not bool(d.get("is_dry_run", False)):
        flags.append("low_snr")
    if math.isfinite(clot_du_mean) and clot_du_mean >= 0 and not bool(d.get("is_dry_run", False)):
        flags.append("clot_not_slowing")
    if mu is not None and math.isfinite(clot_dmu_ratio) and clot_dmu_ratio < thr.min_clot_dmu:
        flags.append("clot_mu_not_applied")

    return {
        "idx": int(d["idx"]),
        "finite_ok": finite_ok,
        "u_ref": u_ref,
        "far_du_rms": far_du_rms,
        "far_dv_rms": far_dv_rms,
        "top_du_rms": top_du_rms,
        "top_dv_rms": top_dv_rms,
        "wall_u_max": wall_u_max,
        "wall_v_max": wall_v_max,
        "inlet_du_max": inlet_du_max,
        "clot_du_peak": clot_du_peak,
        "clot_du_mean": clot_du_mean,
        "snr": snr,
        "mag_rel_err": mag_rel_err,
        "mass_rel_err": mass_rel_err,
        "mu_max": mu_max,
        "mu_min": mu_min,
        "clot_dmu_ratio": clot_dmu_ratio,
        "is_dry_run": bool(d.get("is_dry_run", False)),
        "flags": flags,
    }


def _percentiles(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def qc_dataset(
    directory: Path,
    *,
    thr: Optional[QCThresholds] = None,
    max_samples: Optional[int] = None,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """Run QC over all patches in ``directory``; print summary; return aggregate dict."""
    directory = Path(directory)
    paths = sorted(
        glob.glob(str(directory / "patch_*.npz")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    if max_samples is not None:
        paths = paths[: int(max_samples)]
    if not paths:
        logger.warning("No patch_*.npz found in %s", directory)
        return {"n": 0}

    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(compute_sample_qc(load_patch(Path(p)), thr))
        except Exception as exc:
            logger.warning("QC failed on %s: %s", os.path.basename(p), exc)
            rows.append({"idx": int(os.path.basename(p).split("_")[1].split(".")[0]), "flags": ["load_error"]})

    n = len(rows)
    flag_counts: Dict[str, int] = {}
    for r in rows:
        for fl in r.get("flags", []):
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    clean = sum(1 for r in rows if not r.get("flags"))

    metric_keys = [
        "far_du_rms", "top_du_rms", "wall_u_max", "inlet_du_max",
        "snr", "mag_rel_err", "mass_rel_err", "clot_du_mean", "clot_dmu_ratio",
    ]
    stats = {k: _percentiles([r.get(k) for r in rows if k in r]) for k in metric_keys}

    print("\n================ Patch Factory QC ================")
    print(f"  dir: {directory}")
    print(f"  samples: {n}   clean (no flags): {clean}   flagged: {n - clean}")
    if rows and rows[0].get("is_dry_run"):
        print("  [i] dry-run data: perturbation/SNR checks are skipped (du==0 by construction)")
    print("\n  Metric (normalized by U_ref unless noted)   min      median   p95      max")
    label = {
        "far_du_rms": "baseline far-field |du| RMS (want ~0)",
        "top_du_rms": "lid far-field |du| RMS    (want ~0)",
        "wall_u_max": "bottom-wall |u|           (want ~0)",
        "inlet_du_max": "inlet |du|                (want ~0)",
        "snr": "clot-peak / far-field SNR  (want high)",
        "mag_rel_err": "max u vs shear*H rel err  (want ~0)",
        "mass_rel_err": "mass in/out rel err       (want ~0)",
        "clot_du_mean": "mean du over clot (want <0)",
        "clot_dmu_ratio": "max(mu)/clot_mu (want ~1)",
    }
    for k in metric_keys:
        s = stats[k]
        print(f"  {label[k]:<42} {s['min']:<8.3g} {s['median']:<8.3g} {s['p95']:<8.3g} {s['max']:<8.3g}")

    if flag_counts:
        print("\n  Flags raised:")
        for fl, c in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {fl:<26} {c}/{n}")
    else:
        print("\n  No flags raised.")
    print("==================================================\n")

    csv_path = None
    if write_csv:
        import csv
        out_dir = reports_dir() / "patch_factory"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "patch_factory_qc.csv"
        cols = ["idx", "finite_ok"] + metric_keys + ["flags"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) if c != "flags" else ";".join(r.get("flags", [])) for c in cols])
        logger.info("Wrote per-sample QC to %s", csv_path)

    return {
        "n": n,
        "clean": clean,
        "flagged": n - clean,
        "flag_counts": flag_counts,
        "stats": stats,
        "rows": rows,
        "csv": str(csv_path) if csv_path else None,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_samples(
    directory: Path, indices: List[int], out_png: Optional[Path] = None
) -> Optional[Path]:
    """Field panels (u, du, dv, mu) per sample with the clot footprint overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    directory = Path(directory)
    fields = ["u", "du", "dv", "mu"]
    rows = []
    for i in indices:
        p = directory / f"patch_{i}.npz"
        if p.exists():
            rows.append(load_patch(p))
    if not rows:
        logger.warning("No requested patches found for visualization.")
        return None

    um = 1e6
    fig, axes = plt.subplots(len(rows), len(fields), figsize=(4.2 * len(fields), 2.6 * len(rows)), squeeze=False)
    for r_i, d in enumerate(rows):
        nx, ny = d["grid_nx"], d["grid_ny"]
        x2d = _reshape(d["x"], ny, nx) * um
        y2d = _reshape(d["y"], ny, nx) * um
        cx, cw, ch = d["clot_x_center"] * um, d["clot_width"] * um, d["clot_height"] * um
        for c_i, fld in enumerate(fields):
            ax = axes[r_i][c_i]
            if fld not in d:
                ax.set_visible(False)
                continue
            f2d = _reshape(d[fld], ny, nx)
            cmap = "RdBu_r" if fld in ("du", "dv") else "viridis"
            if fld in ("du", "dv"):
                vmax = np.percentile(np.abs(f2d), 99) + 1e-30
                pc = ax.pcolormesh(x2d, y2d, f2d, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
            else:
                pc = ax.pcolormesh(x2d, y2d, f2d, cmap=cmap, shading="auto")
            fig.colorbar(pc, ax=ax, shrink=0.85)
            ax.add_patch(Rectangle((cx - cw / 2, 0), cw, ch, fill=False, edgecolor="k", lw=1.0))
            ax.set_aspect("equal")
            ax.set_title(f"patch {d['idx']} | {fld}", fontsize=8)
            ax.set_xlabel("x [um]", fontsize=7)
            ax.set_ylabel("y [um]", fontsize=7)
    fig.tight_layout()
    if out_png is None:
        out_png = reports_dir() / "patch_factory" / "patch_factory_qc_fields.png"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    logger.info("Wrote QC field panels to %s", out_png)
    return Path(out_png)


def visualize_distributions(result: Dict[str, Any], out_png: Optional[Path] = None) -> Optional[Path]:
    """Histograms of the key QC metrics across the dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result.get("rows", [])
    if not rows:
        return None
    keys = ["far_du_rms", "snr", "mass_rel_err", "wall_u_max", "mag_rel_err", "clot_du_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for ax, k in zip(axes, keys):
        vals = np.asarray([r.get(k) for r in rows if math.isfinite(r.get(k, float("nan")))], dtype=np.float64)
        if vals.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=30, color="#4878a8", edgecolor="white")
        ax.set_title(k, fontsize=10)
        ax.axvline(float(np.median(vals)), color="crimson", ls="--", lw=1, label=f"median {np.median(vals):.2g}")
        ax.legend(fontsize=7)
    fig.suptitle("Patch Factory QC metric distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if out_png is None:
        out_png = reports_dir() / "patch_factory" / "patch_factory_qc_distributions.png"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    logger.info("Wrote QC distributions to %s", out_png)
    return Path(out_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch Factory dataset QC.")
    p.add_argument(
        "--dir", type=str,
        default=str(data_root() / "processed" / "cfd_results_patch_factory"),
        help="Directory of patch_*.npz files.",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples scanned.")
    p.add_argument("--plot", action="store_true", help="Write field panels + distribution plots.")
    p.add_argument("--plot-indices", type=str, default=None, help="Comma-separated patch indices for field panels.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = qc_dataset(Path(args.dir), max_samples=args.max_samples)
    if args.plot and result.get("n", 0) > 0:
        if args.plot_indices:
            idxs = [int(s) for s in args.plot_indices.split(",")]
        else:
            idxs = [r["idx"] for r in result["rows"][:4]]
        visualize_samples(Path(args.dir), idxs)
        visualize_distributions(result)
