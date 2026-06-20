"""Held-out evaluation + visualization for the local kinematic corrector.

Unlike ``verify_local_corrector_live`` (which injects a *dummy* clot onto a patient
graph to eyeball plausibility), this scores the corrector against the **real COMSOL
ground-truth** residuals it was trained on. It rebuilds the *same* validation split as
training, computes a global relative-L2 error (and per-sample distribution), and renders
true-vs-predicted ``dU`` maps for the best / median / worst patches.

Relative L2 (global over the split) is the tangible accuracy number:

    relL2 = sqrt( sum||pred - truth||^2 / sum||truth||^2 )

A per-sample mean would divide by tiny far-field norms; the global ratio is robust.

CLI:
    python -m src.tools.eval_local_corrector \
        --patch-dir data/processed/cfd_results_patch_factory \
        --corrector outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth \
        --stride 2 --val-frac 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from src.config import PhysicsConfig
from src.core_physics.coupled_shear_gnn import load_local_corrector
from src.training.train_local_kinematic_corrector import (
    DEFAULT_OUT_DIR,
    DEFAULT_PATCH_DIR,
    PatchFactoryDataset,
    PatchNdConfig,
)
from src.utils.paths import reports_dir

DEFAULT_CORRECTOR = DEFAULT_OUT_DIR / "local_kinematic_corrector_best.pth"


def _read_meta(path: Path) -> dict[str, Any]:
    """Best-effort extraction of the training meta dict from a corrector checkpoint."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if isinstance(blob, dict):
        if isinstance(blob.get("meta"), dict):
            return blob["meta"]
        if "nd_cfg" in blob:
            return blob
    return {}


def _nd_cfg_from_meta(meta: dict[str, Any], args: argparse.Namespace) -> PatchNdConfig:
    """Prefer the checkpoint's nd_cfg so the eval representation matches training exactly."""
    nd = meta.get("nd_cfg") if isinstance(meta, dict) else None
    if isinstance(nd, dict):
        return PatchNdConfig(
            crop_x_factor=float(nd.get("crop_x_factor", args.crop_x_factor)),
            crop_y_frac=float(nd.get("crop_y_frac", args.crop_y_frac)),
            crop_y_min_abs_factor=float(nd.get("crop_y_min_abs_factor", 8.0)),
            stride=int(nd.get("stride", args.stride)),
            mu_thresh_si=float(nd.get("mu_thresh_si", 1e-4)),
        )
    return PatchNdConfig(
        crop_x_factor=args.crop_x_factor, crop_y_frac=args.crop_y_frac, stride=args.stride
    )


def _print_buckets(per_sample: List[dict[str, Any]]) -> None:
    """Stratified relL2 by terciles of each difficulty axis (Stage-A per-level lesson).

    Energy-weighted relL2 per bucket = sqrt(sum sq_err / sum sq_tgt) over the bucket, so it
    matches the global metric and reveals which clot regime drives the failure tail.
    """
    axes = [
        ("difficulty", "difficulty (composite)", 1.0, ""),
        ("clot_mu", "clot_mu", 1.0, " Pa.s"),
        ("clot_w", "clot_w", 1e6, " um"),
        ("clot_h", "clot_h", 1e6, " um"),
        ("occlusion", "clot_h/H", 100.0, " %"),
        ("shear", "shear_rate", 1.0, " 1/s"),
    ]
    print("\n  Stratified relL2 (energy-weighted) by terciles -- where the error lives:")
    print(f"  {'axis':<22} {'low tercile':<22} {'mid tercile':<22} {'high tercile':<22}")
    for key, label, scale, unit in axes:
        rows = [r for r in per_sample if math.isfinite(r.get(key, float('nan')))]
        if len(rows) < 3:
            continue
        rows.sort(key=lambda r: r[key])
        k = len(rows) // 3
        thirds = [rows[:k], rows[k:2 * k], rows[2 * k:]]
        cells: List[str] = []
        for grp in thirds:
            if not grp:
                cells.append("-")
                continue
            se = sum(r["sq_err"] for r in grp)
            st = sum(r["sq_tgt"] for r in grp)
            rel = (se / st) ** 0.5 if st > 0 else float("nan")
            lo = grp[0][key] * scale
            hi = grp[-1][key] * scale
            cells.append(f"{rel * 100:5.1f}% [{lo:.0f}-{hi:.0f}{unit}]")
        print(f"  {label:<22} {cells[0]:<22} {cells[1]:<22} {cells[2]:<22}")


def _rel_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    sst = float((truth ** 2).sum().item())
    if sst <= 0.0:
        return float("nan")
    return float((((pred - truth) ** 2).sum().item() / sst) ** 0.5)


def evaluate(
    patch_dir: Path | str = DEFAULT_PATCH_DIR,
    corrector_path: Path | str = DEFAULT_CORRECTOR,
    *,
    val_frac: float = 0.1,
    seed: int = 0,
    nd_cfg: PatchNdConfig | None = None,
    device: torch.device | str | None = None,
    n_viz: int = 3,
    out_png: Path | str | None = None,
) -> dict[str, Any]:
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    phys = PhysicsConfig(phase="kinematics")
    cfg = nd_cfg or PatchNdConfig()

    dataset = PatchFactoryDataset(Path(patch_dir), phys, cfg)
    n = len(dataset)
    if n == 0:
        raise RuntimeError(f"No usable patches in {patch_dir}.")

    # Reproduce the training val split bit-for-bit (same seed + val_frac + ordering).
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    val_set = [dataset[i] for i in val_idx]
    print(f"[i] patches: {n} usable | evaluating on {len(val_set)} val (seed={seed}, frac={val_frac})")

    corrector = load_local_corrector(Path(corrector_path), dev)
    corrector.eval()

    sq_err = 0.0
    sq_tgt = 0.0
    per_sample: List[dict[str, Any]] = []
    with torch.no_grad():
        for local_i, data in enumerate(val_set):
            d = data.to(dev)
            pred = corrector(d.x, d.edge_index)
            truth = d.y
            se = float(((pred - truth) ** 2).sum().item())
            st = float((truth ** 2).sum().item())
            sq_err += se
            sq_tgt += st
            per_sample.append({
                "split_pos": local_i,
                "rel_l2": _rel_l2(pred, truth),
                "n_nodes": int(d.x.shape[0]),
                "sq_err": se,
                "sq_tgt": st,
                # stratification axes (attached in patch_to_data)
                "difficulty": float(getattr(data, "difficulty", float("nan"))),
                "clot_mu": float(getattr(data, "clot_mu", float("nan"))),
                "clot_w": float(getattr(data, "clot_w", float("nan"))),
                "clot_h": float(getattr(data, "clot_h", float("nan"))),
                "occlusion": float(getattr(data, "occlusion", float("nan"))),
                "shear": float(getattr(data, "shear", float("nan"))),
            })

    global_rel = float((sq_err / sq_tgt) ** 0.5) if sq_tgt > 0 else float("nan")
    rels = np.array([p["rel_l2"] for p in per_sample if np.isfinite(p["rel_l2"])], dtype=np.float64)
    pct = lambda q: float(np.percentile(rels, q)) if rels.size else float("nan")

    print("\n========== Local Corrector Eval (held-out COMSOL truth) ==========")
    print(f"  global relL2 (energy-weighted) : {global_rel * 100:.2f}%")
    print(f"  per-sample relL2  median       : {pct(50) * 100:.2f}%")
    print(f"                    p90 / p95    : {pct(90) * 100:.2f}% / {pct(95) * 100:.2f}%")
    print(f"                    min / max    : {pct(0) * 100:.2f}% / {pct(100) * 100:.2f}%")
    print("=================================================================")
    _print_buckets(per_sample)
    print()

    # Pick best / median / worst (by per-sample relL2) for the truth-vs-pred maps.
    order = sorted([p for p in per_sample if np.isfinite(p["rel_l2"])], key=lambda p: p["rel_l2"])
    picks: List[dict[str, Any]] = []
    if order:
        picks = [order[0], order[len(order) // 2], order[-1]][:max(1, n_viz)]
    out = _plot_truth_vs_pred(val_set, picks, corrector, dev, out_png=out_png)
    if out is not None:
        print(f"[save] {out}")

    return {
        "global_rel_l2": global_rel,
        "median_rel_l2": pct(50),
        "p90_rel_l2": pct(90),
        "n_val": len(val_set),
        "per_sample": per_sample,
        "viz": str(out) if out else None,
    }


def _plot_truth_vs_pred(val_set, picks, corrector, dev, *, out_png) -> Optional[Path]:
    if not picks:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (["best", "median", "worst"] + [f"#{i}" for i in range(len(picks))])[: len(picks)]
    cols = ["truth dU", "pred dU", "|err| dU"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(4.0 * len(cols), 3.0 * len(picks)), squeeze=False)
    for r_i, p in enumerate(picks):
        data = val_set[p["split_pos"]]
        d = data.to(dev)
        with torch.no_grad():
            pred = corrector(d.x, d.edge_index).cpu().numpy()
        truth = d.y.cpu().numpy()
        # feature layout: [dx, dy, dist_to_wall, u0, v0, delta_mu]
        dx = d.x[:, 0].cpu().numpy()
        dy = d.x[:, 1].cpu().numpy()
        du_t, du_p = truth[:, 0], pred[:, 0]
        err = np.abs(du_p - du_t)
        vmax = float(np.percentile(np.abs(du_t), 99) + 1e-12)
        panels = [
            (du_t, "RdBu_r", -vmax, vmax),
            (du_p, "RdBu_r", -vmax, vmax),
            (err, "magma", 0.0, vmax),
        ]
        for c_i, (vals, cmap, vmn, vmx) in enumerate(panels):
            ax = axes[r_i][c_i]
            sc = ax.scatter(dx, dy, c=vals, cmap=cmap, vmin=vmn, vmax=vmx, s=6)
            fig.colorbar(sc, ax=ax, shrink=0.85)
            ax.set_aspect("equal")
            ttl = cols[c_i] if r_i == 0 else ""
            ax.set_title(f"{labels[r_i]} (relL2 {p['rel_l2']*100:.1f}%) | {ttl}".strip(" |"), fontsize=8)
            ax.set_xlabel("dx (clot-centered, ND)", fontsize=7)
            ax.set_ylabel("dy (ND)", fontsize=7)
    fig.tight_layout()
    if out_png is None:
        out_png = reports_dir() / "figures" / "kinematics" / "local_corrector_eval.png"
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Held-out eval + viz of the local kinematic corrector vs COMSOL truth.")
    p.add_argument("--patch-dir", type=str, default=str(DEFAULT_PATCH_DIR))
    p.add_argument("--corrector", type=str, default=str(DEFAULT_CORRECTOR))
    p.add_argument("--val-frac", type=float, default=0.1, help="Must match training to reproduce the split.")
    p.add_argument("--seed", type=int, default=0, help="Must match training to reproduce the split.")
    p.add_argument("--stride", type=int, default=2, help="Used only if the checkpoint has no nd_cfg.")
    p.add_argument("--crop-x-factor", type=float, default=4.0)
    p.add_argument("--crop-y-frac", type=float, default=0.5)
    p.add_argument("--n-viz", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    meta = _read_meta(Path(args.corrector))
    cfg = _nd_cfg_from_meta(meta, args)
    evaluate(
        patch_dir=args.patch_dir,
        corrector_path=args.corrector,
        val_frac=args.val_frac,
        seed=args.seed,
        nd_cfg=cfg,
        device=args.device,
        n_viz=args.n_viz,
        out_png=args.out,
    )
