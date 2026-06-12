"""T0 baseline viz: GT clot label vs predictors (GT flow+species only).

Row 0 = GT clot from spf.mu (evaluation label only).
Row 1 = best mu-growth predictor (no spf.sr sidecar).
Row 2 = species-hard (Mat/fi COMSOL steps).
Row 3 = optional nucleation on row-1.

Usage::

    python scripts/viz_t0_baseline_gt.py --anchor patient007
    python scripts/viz_t0_baseline_gt.py --anchor patient007 --sweep-json outputs/.../t0_clot_predictor_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.t0_clot_predictor import (  # noqa: E402
    predict_clot_phi_species_hard,
    t0_gt_baseline_env,
)
from src.core_physics.t0_mu_physics import (  # noqa: E402
    gt_clot_phi_at_time,
    predict_clot_phi_at_time,
    rollout_t0_clot_phi,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def _load_best_mu_config(sweep_path: Path | None) -> dict:
    default = {
        "gamma_mode": "kinematic",
        "gamma_scale": 1.0,
        "poiseuille_scale": 0.85,
        "predictor": "mu_growth",
    }
    if sweep_path is None or not sweep_path.is_file():
        return default
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    best = (
        payload.get("best", {}).get("mu_growth_nucleation")
        or payload.get("best", {}).get("mu_growth")
        or payload.get("best", {}).get("overall", {})
    )
    if not best or best.get("predictor") not in (None, "mu_growth"):
        return default
    return {
        "gamma_mode": str(best.get("gamma_mode", default["gamma_mode"])),
        "gamma_scale": float(best.get("gamma_scale", default["gamma_scale"])),
        "poiseuille_scale": best.get("poiseuille_scale"),
        "predictor": "mu_growth",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 GT baseline clot timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--sweep-json", default="outputs/biochem/clot_trigger/t0_clot_predictor_sweep.json")
    ap.add_argument("--gamma-mode", default="")
    ap.add_argument("--gamma-scale", type=float, default=-1.0)
    ap.add_argument("--poiseuille-scale", type=float, default=-1.0)
    ap.add_argument("--nucleation-row", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_nodes = int(data.num_nodes)
    n_steps = int(data.y.shape[0])
    times = _pick_times(n_steps, int(args.max_frames))
    full_region = np.ones(n_nodes, dtype=bool)

    sweep_path = root / args.sweep_json if args.sweep_json.strip() else None
    cfg = _load_best_mu_config(sweep_path)
    gamma_mode = args.gamma_mode.strip() or cfg["gamma_mode"]
    gamma_scale = float(args.gamma_scale) if args.gamma_scale >= 0 else float(cfg["gamma_scale"])
    poi = cfg.get("poiseuille_scale")
    if args.poiseuille_scale >= 0:
        poi = float(args.poiseuille_scale)

    use_nuc = bool(args.nucleation_row)
    frames: list[dict] = []

    with t0_gt_baseline_env(
        gamma_mode=gamma_mode,
        gamma_scale=gamma_scale,
        poiseuille_scale=poi,
    ) as env:
        traj_nuc = (
            rollout_t0_clot_phi(
                data,
                phys,
                bio,
                device,
                gamma_mode=gamma_mode,
                nucleation=True,
                nucleation_hops=1,
            )
            if use_nuc
            else None
        )
        mu_label = f"mu growth ({gamma_mode} g={gamma_scale:g}"
        if poi is not None:
            mu_label += f" poi={float(poi):g}"
        mu_label += ")"

        for t in times:
            phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
            phi_mu, _step = predict_clot_phi_at_time(
                data, int(t), phys, bio, device, gamma_mode=gamma_mode
            )
            phi_sp = predict_clot_phi_species_hard(data, int(t), bio, device)
            phi_nuc = traj_nuc[int(t)]["phi"] if traj_nuc is not None else None
            mask = torch.ones(n_nodes, device=device, dtype=torch.bool)
            m_mu = clot_trigger_viz_f1(phi_mu, phi_gt, mask)
            m_sp = clot_trigger_viz_f1(phi_sp, phi_gt, mask)
            m_nuc = (
                clot_trigger_viz_f1(phi_nuc, phi_gt, mask) if phi_nuc is not None else None
            )
            frames.append(
                {
                    "t": int(t),
                    "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                    "phi_gt": phi_gt.detach().cpu().numpy(),
                    "phi_mu": phi_mu.detach().cpu().numpy(),
                    "phi_sp": phi_sp.detach().cpu().numpy(),
                    "phi_nuc": phi_nuc.detach().cpu().numpy() if phi_nuc is not None else None,
                    "f1_mu": float(m_mu["clot_f1"]),
                    "f1_sp": float(m_sp["clot_f1"]),
                    "f1_nuc": float(m_nuc["clot_f1"]) if m_nuc else float("nan"),
                }
            )

        nrows = 4 if use_nuc else 3
        ncols = len(frames)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(max(2.5 * ncols, 10), 2.6 * nrows), squeeze=False
        )
        fig.suptitle(
            f"T0 GT baseline -- {args.anchor} | no spf.mu/sr input | {mu_label}",
            fontsize=10,
        )
        row_labels = ["GT (spf.mu)", mu_label, "species Mat|fi", "nucleation"]
        for j, fr in enumerate(frames):
            title = (
                f"t={fr['t']} tau={fr['tau']:.2f}\n"
                f"F1_mu={fr['f1_mu']:.2f} F1_sp={fr['f1_sp']:.2f}"
            )
            if use_nuc and np.isfinite(fr["f1_nuc"]):
                title += f" F1_nuc={fr['f1_nuc']:.2f}"
            rows_data = [
                fr["phi_gt"],
                fr["phi_mu"],
                fr["phi_sp"],
            ]
            if use_nuc and fr["phi_nuc"] is not None:
                rows_data.append(fr["phi_nuc"])
            for i, vals in enumerate(rows_data):
                _scatter_fullmesh_region(
                    axes[i, j],
                    pos,
                    vals,
                    full_region,
                    row_labels[i] if j == 0 else "",
                    cmap="bwr",
                    vmin=0,
                    vmax=1,
                    s=float(args.scatter_size),
                    layer_positive_on_top=True,
                )
            axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_baseline_gt_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary = {
        "anchor": args.anchor,
        "env": env,
        "frames": [{k: v for k, v in fr.items() if not k.startswith("phi_")} for fr in frames],
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
