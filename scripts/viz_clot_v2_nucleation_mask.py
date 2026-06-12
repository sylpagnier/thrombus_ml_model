"""V0 viz: GT new commits vs nucleation eligibility vs legacy ceiling."""

from __future__ import annotations

import argparse
import json
import os
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
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_nucleation_mask import (  # noqa: E402
    gt_new_commit_mask,
    resolve_catalytic_hood,
    resolve_commits_at_time,
    resolve_nucleation_eligibility,
)
from src.evaluation.viz_clot_phi_simple import _scatter_fullmesh_region  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def render_nucleation_timeline(
    *,
    anchor: str,
    frames: list[dict],
    pos: np.ndarray,
    out_path: Path,
    scatter_size: float,
) -> None:
    if not frames:
        raise RuntimeError("no frames")
    ncols = len(frames)
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.6 * ncols, 10), 2.3 * nrows + 1.2), squeeze=False)
    fig.suptitle(f"V0 nucleation audit -- {anchor} (GT seed)", fontsize=11)

    row_titles = ["GT new", "E_seed", "ceiling", "catalytic H"]
    for j, fr in enumerate(frames):
        title = f"t={fr['t']} tau={fr['tau']:.2f}\nnew={fr['n_new_gt']} rec={fr['recall']:.2f}"
        panels = [
            (fr["new_gt"], "Reds", 0.0, 1.0),
            (fr["elig"], "Greens", 0.0, 1.0),
            (fr["ceiling"], "Purples", 0.0, 1.0),
            (fr["catalytic"], "Oranges", 0.0, 1.0),
        ]
        for row, (vals, cmap, vmin, vmax) in enumerate(panels):
            _scatter_fullmesh_region(
                axes[row, j],
                pos,
                vals,
                np.ones(len(vals), dtype=bool),
                row_titles[row] if j == 0 else "",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=scatter_size,
            )
            if row == 0:
                axes[row, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="V0 nucleation mask timeline viz")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument("--growth-seed", default="gt", choices=("gt", "pred"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ.setdefault("CLOT_ML_USE_MACRO_TAU", "1")
    os.environ.setdefault("CLOT_PHI_GROWTH_SEED", args.growth_seed)

    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio).detach().cpu().numpy().astype(float)

    phi_pred_by_time = None
    if args.growth_seed == "pred":
        from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
        from src.training.clot_ml_step1_residual import (
            apply_step1_eval_env,
            load_step1_checkpoint,
            resolve_step1_rule_cfg,
            rollout_step1_phi,
        )

        apply_step1_eval_env()
        rule_cfg = resolve_step1_rule_cfg(
            root / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
        )
        ckpt = root / "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
        model, _ = load_step1_checkpoint(ckpt, device=device)
        reset_temporal_kinematics_cache()
        phi_pred_by_time = rollout_step1_phi(
            data,
            rule_cfg,
            model,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=0.35,
        )

    frames: list[dict] = []
    for t in _pick_times(n_steps, int(args.max_frames)):
        elig_t = resolve_nucleation_eligibility(
            data,
            t,
            device,
            phys,
            bio,
            growth_seed=args.growth_seed,
            phi_pred_by_time=phi_pred_by_time,
        )
        new_gt = gt_new_commit_mask(data, t, phys, device)
        commits_prev = resolve_commits_at_time(
            data,
            max(t - 1, 0),
            device=device,
            phys_cfg=phys,
            growth_seed=args.growth_seed,
            phi_pred_by_time=phi_pred_by_time,
        )
        hood = resolve_catalytic_hood(commits_prev, data.edge_index.to(device))
        n_new = int(new_gt.sum().item())
        rec = float((new_gt & elig_t).sum().item()) / max(n_new, 1) if n_new else float("nan")
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "n_new_gt": n_new,
                "recall": rec,
                "new_gt": new_gt.detach().cpu().numpy().astype(float),
                "elig": elig_t.detach().cpu().numpy().astype(float),
                "ceiling": ceiling,
                "catalytic": hood.detach().cpu().numpy().astype(float),
            }
        )

    out_default = root / f"outputs/biochem/viz/clot_v2/v0_nucleation_{args.anchor}_{args.growth_seed}.png"
    out_path = Path(args.out) if args.out.strip() else out_default
    if not out_path.is_absolute():
        out_path = root / out_path

    render_nucleation_timeline(
        anchor=args.anchor,
        frames=frames,
        pos=pos,
        out_path=out_path,
        scatter_size=float(args.scatter_size),
    )
    print(f"[save] {out_path}")

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "growth_seed": args.growth_seed,
                "frames": [{k: v for k, v in fr.items() if k not in ("new_gt", "elig", "ceiling", "catalytic")} for fr in frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
