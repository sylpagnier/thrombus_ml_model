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
    with np.load(npz_path) as z:  # context-manager closes the handle (Windows file lock)
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


def purge_dry_run(directory: Path) -> List[int]:
    """Delete dry-run placeholder ``patch_*.npz`` (+ ``.json``) so a resumed solve refills them.

    Returns the list of purged indices. A real (solved) batch can then be re-run without
    ``--overwrite``: resume fills only the now-missing indices.
    """
    directory = Path(directory)
    purged: List[int] = []
    for p in sorted(glob.glob(str(directory / "patch_*.npz"))):
        try:
            with np.load(p) as z:  # context-manager closes the handle (Windows file lock)
                is_dry = bool(np.asarray(z["dry_run"]).ravel()[0]) if "dry_run" in z.files else False
        except Exception:
            is_dry = False
        if not is_dry:
            continue
        idx = int(os.path.basename(p).split("_")[1].split(".")[0])
        os.remove(p)
        sidecar = directory / f"patch_{idx}.json"
        if sidecar.exists():
            sidecar.unlink()
        purged.append(idx)
    logger.info("Purged %d dry-run placeholder file(s) from %s", len(purged), directory)
    return purged


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


def run_or_load_convergence(
    directory: Path,
    *,
    n_samples: int = 3,
    refine_factor: float = 2.0,
    tol: float = 0.02,
    template: Optional[Path] = None,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return a mesh-convergence report for the dataset.

    Loads ``convergence_report.json`` if present (cheap); otherwise re-solves a few real
    dataset samples at base + refined mapped mesh via COMSOL. Returns ``None`` (graceful)
    when no report exists and COMSOL/template is unreachable.
    """
    directory = Path(directory)
    report_path = directory / "convergence_report.json"
    if report_path.exists() and not force:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", report_path.name, exc)

    # Need a fresh check: pick real (solved, non-dry-run) samples evenly across the dataset.
    npzs = sorted(
        glob.glob(str(directory / "patch_*.npz")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    candidates: List[Dict[str, Any]] = []
    for p in npzs:
        idx = int(os.path.basename(p).split("_")[1].split(".")[0])
        sidecar = directory / f"patch_{idx}.json"
        if not sidecar.exists():
            continue
        try:
            with np.load(p) as z:
                if "dry_run" in z.files and bool(np.asarray(z["dry_run"]).ravel()[0]):
                    continue
            with open(sidecar, "r", encoding="utf-8") as f:
                candidates.append(json.load(f))
        except Exception:
            continue
    if not candidates:
        logger.warning("Convergence: no solved samples with sidecars found.")
        return None

    step = max(1, len(candidates) // max(1, n_samples))
    chosen_meta = candidates[::step][:n_samples]

    try:
        from src.data_gen.lib.patch_factory_comsol import (
            PatchFactoryComsolGenerator,
            PatchFactoryConfig,
            PatchSample,
        )
    except Exception as exc:
        logger.warning("Convergence: cannot import COMSOL generator: %s", exc)
        return None

    cfg = PatchFactoryConfig(output_dir=directory)
    if template is not None:
        cfg.template_path = Path(template)
    samples = [PatchSample.from_meta(m) for m in chosen_meta]
    try:
        with PatchFactoryComsolGenerator(cfg) as gen:
            return gen.convergence_check(
                samples, refine_factor=refine_factor, tol=tol, write_report=True
            )
    except Exception as exc:
        logger.warning("Convergence check skipped (COMSOL/template not reachable): %s", exc)
        return None


def qc_dataset(
    directory: Path,
    *,
    thr: Optional[QCThresholds] = None,
    max_samples: Optional[int] = None,
    write_csv: bool = True,
    run_convergence: bool = True,
    convergence_samples: int = 3,
    convergence_refine: float = 2.0,
    convergence_tol: float = 0.02,
    force_convergence: bool = False,
    template: Optional[Path] = None,
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

    # Dry-run detection. A dry-run placeholder (du==0, no COMSOL solve) is fine if the WHOLE
    # dataset is a dry run, but is contamination if mixed into a real (solved) dataset.
    dry_idxs = [r["idx"] for r in rows if r.get("is_dry_run")]
    n_dry = len(dry_idxs)
    all_dry = (n_dry == n)
    if 0 < n_dry < n:
        for r in rows:
            if r.get("is_dry_run") and "dry_run_placeholder" not in r.get("flags", []):
                r.setdefault("flags", []).append("dry_run_placeholder")

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
    if all_dry:
        print("  [i] dry-run dataset: perturbation/SNR checks skipped (du==0 by construction)")
    elif n_dry > 0:
        preview = ", ".join(str(i) for i in sorted(dry_idxs)[:15])
        more = "" if n_dry <= 15 else f", ... (+{n_dry - 15} more)"
        print(f"  [WARN] CONTAMINATION: {n_dry}/{n} dry-run placeholder files (du==0, no clot")
        print(f"         signal) are mixed into a solved dataset. These will poison training.")
        print(f"         Regenerate them with --overwrite (or delete). Indices: {preview}{more}")
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

    # Mesh-convergence (default on). Skipped for dry-run datasets (du==0 -> meaningless).
    convergence: Optional[Dict[str, Any]] = None
    if run_convergence and not all_dry:
        convergence = run_or_load_convergence(
            directory,
            n_samples=convergence_samples,
            refine_factor=convergence_refine,
            tol=convergence_tol,
            template=template,
            force=force_convergence,
        )
        print("\n  Mesh convergence (du, refined vs base)")
        if convergence is None:
            print("    [i] not available: no report and COMSOL/template not reachable.")
            print("        Run with --template <master.mph>, or generate with --convergence.")
        else:
            status = "PASS" if convergence.get("passed") else "FAIL"
            print(
                f"    {status}: median rel L2 {convergence.get('median_rel_l2_du', float('nan')):.3g}"
                f"  max {convergence.get('max_rel_l2_du', float('nan')):.3g}"
                f"  (tol {convergence.get('tol')}, refine x{convergence.get('refine_factor')},"
                f" n={convergence.get('n_evaluated', convergence.get('n_samples'))})"
            )
            if not convergence.get("mesh_refined", True):
                print("    [WARN] FE mesh could not be refined (no mapped numelem distributions)"
                      " -> result inconclusive.")
    elif run_convergence and all_dry:
        print("\n  Mesh convergence: skipped (dry-run dataset).")
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
        "convergence": convergence,
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
    p.add_argument(
        "--purge-dry-run",
        action="store_true",
        help="Delete dry-run placeholder files (du==0) so a resumed solve refills only those indices.",
    )
    # Mesh-convergence check (on by default; cached in convergence_report.json).
    p.add_argument(
        "--no-convergence",
        action="store_true",
        help="Skip the mesh-convergence check (otherwise run/load by default).",
    )
    p.add_argument(
        "--force-convergence",
        action="store_true",
        help="Re-run the convergence check even if convergence_report.json exists.",
    )
    p.add_argument("--convergence-samples", type=int, default=3, help="Patches for the convergence check.")
    p.add_argument("--convergence-refine", type=float, default=2.0, help="Mesh element-count scale factor.")
    p.add_argument("--convergence-tol", type=float, default=0.02, help="Max rel L2 du to PASS.")
    p.add_argument("--template", type=str, default=None, help="Master .mph path for the convergence re-solve.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.purge_dry_run:
        purged = purge_dry_run(Path(args.dir))
        print(f"Purged {len(purged)} dry-run placeholder file(s): {purged[:20]}{' ...' if len(purged) > 20 else ''}")
        print("Now re-run the solver to refill them:")
        print("  python -m src.data_gen.lib.patch_factory_comsol -n 1000 --seed 0")
        raise SystemExit(0)
    result = qc_dataset(
        Path(args.dir),
        max_samples=args.max_samples,
        run_convergence=not args.no_convergence,
        convergence_samples=args.convergence_samples,
        convergence_refine=args.convergence_refine,
        convergence_tol=args.convergence_tol,
        force_convergence=args.force_convergence,
        template=Path(args.template) if args.template else None,
    )
    if args.plot and result.get("n", 0) > 0:
        if args.plot_indices:
            idxs = [int(s) for s in args.plot_indices.split(",")]
        else:
            idxs = [r["idx"] for r in result["rows"][:4]]
        visualize_samples(Path(args.dir), idxs)
        visualize_distributions(result)
