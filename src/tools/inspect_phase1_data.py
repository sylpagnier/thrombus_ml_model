"""
Phase 1 data inspector (restored + modernized).

Focuses on tier1/tier2 COMSOL anchor exports (`vessel_*.npz`) and optional live template tag check.

Examples:
    python -m src.tools.inspect_phase1_data --tier tier1 --summary
    python -m src.tools.inspect_phase1_data --tier tier2 --sample-idx 10 --plot
    python -m src.tools.inspect_phase1_data --inspect-template-tags
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import VesselConfig
from src.utils.paths import get_project_root


def _resolve_anchor_dir(tier: str) -> Path:
    return Path(VesselConfig(tier=tier).output_dir)


def _iter_anchor_files(anchor_dir: Path):
    return sorted(anchor_dir.glob("vessel_*.npz"))


def _extract_idx(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def _compute_metrics(npz: np.lib.npyio.NpzFile) -> dict:
    keys = set(npz.keys())
    required = {"x", "y", "u", "v", "p"}
    missing = sorted(required - keys)
    if missing:
        return {"ok": False, "reason": f"missing:{','.join(missing)}"}

    u = np.asarray(npz["u"]).reshape(-1)
    v = np.asarray(npz["v"]).reshape(-1)
    p = np.asarray(npz["p"]).reshape(-1)
    vel = np.sqrt(u**2 + v**2)

    bad = (~np.isfinite(u)).sum() + (~np.isfinite(v)).sum() + (~np.isfinite(p)).sum()
    denom = max(u.size + v.size + p.size, 1)
    nan_ratio = float(bad) / float(denom)

    out = {
        "ok": True,
        "nodes": int(u.size),
        "vel_min": float(np.nanmin(vel)),
        "vel_max": float(np.nanmax(vel)),
        "p_std": float(np.nanstd(p)),
        "nan_ratio": nan_ratio,
        "has_mu": "mu" in keys,
    }
    if "mu" in keys:
        mu = np.asarray(npz["mu"]).reshape(-1)
        out["mu_min"] = float(np.nanmin(mu))
        out["mu_max"] = float(np.nanmax(mu))
    return out


def summary(tier: str) -> None:
    anchor_dir = _resolve_anchor_dir(tier)
    files = list(_iter_anchor_files(anchor_dir))
    if not files:
        print(f"No vessel_*.npz files found in {anchor_dir}")
        return

    rows = []
    for f in files:
        idx = _extract_idx(f)
        try:
            with np.load(f) as npz:
                m = _compute_metrics(npz)
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


def plot_sample(tier: str, sample_idx: int) -> None:
    anchor_dir = _resolve_anchor_dir(tier)
    file_path = anchor_dir / f"vessel_{sample_idx}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Sample not found: {file_path}")
    with np.load(file_path) as npz:
        x = np.asarray(npz["x"]).reshape(-1)
        y = np.asarray(npz["y"]).reshape(-1)
        u = np.asarray(npz["u"]).reshape(-1)
        v = np.asarray(npz["v"]).reshape(-1)
        p = np.asarray(npz["p"]).reshape(-1)
        vel = np.sqrt(u**2 + v**2)
        mu = np.asarray(npz["mu"]).reshape(-1) if "mu" in npz else None

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
    parser = argparse.ArgumentParser(description="Inspect Phase1 anchor data and template tags.")
    parser.add_argument("--tier", type=str, default="tier1", choices=["tier1", "tier2"])
    parser.add_argument("--summary", action="store_true", help="Print summary statistics for vessel_*.npz.")
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index (vessel_<idx>.npz).")
    parser.add_argument("--plot", action="store_true", help="Plot sample fields (requires --sample-idx).")
    parser.add_argument(
        "--inspect-template-tags",
        action="store_true",
        help="Run live COMSOL tag inspection for phase1 template (.mph + mph package required).",
    )
    args = parser.parse_args()

    if args.summary or (not args.plot and not args.inspect_template_tags):
        summary(args.tier)

    if args.plot:
        if args.sample_idx is None:
            raise ValueError("--plot requires --sample-idx")
        plot_sample(args.tier, args.sample_idx)

    if args.inspect_template_tags:
        inspect_template_tags()


if __name__ == "__main__":
    main()
