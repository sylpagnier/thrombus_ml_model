"""Interactive anchor CFD (.npz) inspector and batch health scan (Tier 1 / Tier 2 raw COMSOL exports).

Run::

    python -m src.tools.inspect_anchor_cfd --tier tier2 --scan-only

``--scan-only`` writes ``outputs/reports/<tier>_anchor_health.csv`` and exits without matplotlib windows.
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from src.config import VesselConfig
from src.utils.paths import get_project_root, reports_dir


def _resolve_data_dir(active_tier: str) -> Path:
    root = get_project_root()
    cfg = VesselConfig(tier=active_tier)
    if Path(cfg.output_dir).is_absolute():
        return Path(cfg.output_dir)
    return root / cfg.output_dir


def _load_anchor_npz(sample_idx: int, active_tier: str):
    data_dir = _resolve_data_dir(active_tier)
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


def _compute_metrics(data) -> dict:
    keys = list(data.keys())
    if "x" not in keys or "y" not in keys:
        return {"ok": False, "reason": "missing_xy"}
    if "u" not in keys or "v" not in keys or "p" not in keys:
        return {"ok": False, "reason": "missing_uvp"}

    x = data["x"].flatten()
    u = data["u"].flatten()
    v = data["v"].flatten()
    p = data["p"].flatten()
    vel_mag = np.sqrt(u**2 + v**2)

    has_mu = "mu" in keys
    mu = data["mu"].flatten() if has_mu else None

    total_nodes = len(x)
    nan_count = int(np.isnan(u).sum() + np.isnan(v).sum() + np.isnan(p).sum())
    if has_mu:
        nan_count += int(np.isnan(mu).sum())
    denom = max(total_nodes * (4 if has_mu else 3), 1)
    nan_ratio = nan_count / denom

    p_std = float(np.nanstd(p)) if p.size else 0.0
    u_abs_max = float(np.nanmax(np.abs(u))) if u.size else 0.0

    flags = []
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


def inspect_all_anchors(active_tier="tier1", export_csv=True):
    data_dir = _resolve_data_dir(active_tier)
    files = sorted(Path(data_dir).glob("vessel_*.npz"))
    if len(files) == 0:
        print(f"No vessel_*.npz files found in {data_dir}")
        return []

    print(f"\nScanning {len(files)} anchors in {data_dir} ...")
    rows = []
    for f in files:
        try:
            sample_idx = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            d = np.load(f)
            m = _compute_metrics(d)
            d.close()
        except Exception as e:
            m = {"ok": False, "reason": f"load_error:{e}"}
        m["sample_idx"] = sample_idx
        rows.append(m)

    valid = [r for r in rows if r.get("ok")]
    invalid = [r for r in rows if not r.get("ok")]
    flagged = [r for r in valid if len(r.get("quality_flags", [])) > 0]

    print("\n--- Anchor Health Summary ---")
    print(f"Total files: {len(rows)}")
    print(f"Valid files: {len(valid)}")
    print(f"Invalid files: {len(invalid)}")
    print(f"Flagged quality files: {len(flagged)}")

    if len(valid) > 0:
        pstd = np.array([r["p_std"] for r in valid], dtype=float)
        uabs = np.array([r["u_abs_max"] for r in valid], dtype=float)
        nanr = np.array([r["nan_ratio"] for r in valid], dtype=float)
        print(
            f"p_std median={np.median(pstd):.3e} | "
            f"u_abs_max median={np.median(uabs):.3e} | "
            f"nan_ratio max={np.max(nanr):.3e}"
        )

    if len(flagged) > 0:
        print("\nTop flagged anchors:")
        ranked = sorted(flagged, key=lambda x: (len(x["quality_flags"]), x["nan_ratio"]), reverse=True)
        for r in ranked[:20]:
            print(
                f"  vessel_{r['sample_idx']}: flags={r['quality_flags']} "
                f"p_std={r['p_std']:.2e} u_abs_max={r['u_abs_max']:.2e} nan_ratio={r['nan_ratio']:.2e}"
            )
    if export_csv and len(rows) > 0:
        out_path = reports_dir() / f"{active_tier}_anchor_health.csv"
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
        print(f"📄 Wrote anchor health CSV: {out_path}")
    return rows


def _list_sample_indices(active_tier: str):
    data_dir = _resolve_data_dir(active_tier)
    out = []
    for f in sorted(Path(data_dir).glob("vessel_*.npz")):
        try:
            out.append(int(f.stem.split("_")[-1]))
        except ValueError:
            continue
    return out


def inspect_data(sample_idx=0, active_tier="tier1", enable_regenerate=True):
    current_idx = int(sample_idx)
    sample_indices = _list_sample_indices(active_tier) if enable_regenerate else []

    while True:
        data, file_path = _load_anchor_npz(sample_idx=current_idx, active_tier=active_tier)
        if data is None:
            return

        next_idx_holder = {"value": None}

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

            print(f"--- Data Summary (Sample {current_idx}) ---")
            if "d_bar" in keys:
                print(f"Mean Diameter (d_bar): {data['d_bar']:.4f} m")
            print(f"Nodes: {len(x)}")
            print(f"Velocity Range: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s")
            print(f"Pressure Range: {p.min():.4f} - {p.max():.4f} Pa")
            if has_mu:
                print(f"Viscosity Range: {mu.min():.6f} - {mu.max():.6f} Pa*s")

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
            if enable_regenerate:

                def _pick_next_random():
                    candidates = [i for i in sample_indices if i != current_idx]
                    if len(candidates) == 0:
                        print("Only one sample available; cannot regenerate another random sample.")
                        return None
                    return random.choice(candidates)

                def _regenerate():
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
                fig._regen_btn = regen_btn
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


def _prompt_tier_choice(default="tier1"):
    while True:
        raw = input("Select tier [1/2] [1]: ").strip().lower()
        if raw == "":
            return "tier1"
        if raw in ("1", "tier1"):
            return "tier1"
        if raw in ("2", "tier2"):
            return "tier2"
        print("Invalid tier. Enter 1 or 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect anchor CFD outputs")
    parser.add_argument("--tier", type=str, default=None, help="tier1 or tier2")
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index to plot")
    parser.add_argument("--scan-only", action="store_true", help="Health scan + CSV only, no GUI")
    parser.add_argument("--no-export-csv", action="store_true", help="Disable CSV export during scan")
    args = parser.parse_args()

    tier = args.tier if args.tier in ("tier1", "tier2") else _prompt_tier_choice(default="tier1")
    data_dir = _resolve_data_dir(tier)

    inspect_all_anchors(active_tier=tier, export_csv=(not args.no_export_csv))

    if args.scan_only:
        raise SystemExit(0)

    if args.sample_idx is not None:
        sample_idx = args.sample_idx
    else:
        files = sorted(Path(data_dir).glob("vessel_*.npz"))
        if len(files) == 0:
            print(f"No vessel_*.npz files found in {data_dir}")
            sample_idx = None
        else:
            sample_idx = int(random.choice(files).stem.split("_")[-1])
            print(f"\nRandom sample selected for plotting: vessel_{sample_idx}.npz")

    if sample_idx is not None:
        inspect_data(sample_idx=sample_idx, active_tier=tier)
