"""T0 timeline viz: GT vs deploy physics (nucleation projected, pred seed)."""

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

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_phi_simple import build_clot_phi_step
from src.core_physics.clot_trigger_rollout import (
    forward_path_uses_gt_commits,
    lumen_false_positive_frac,
    rollout_clot_trigger_physics,
    snapshot_clot_trigger_rollout_config,
)
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1, scatter_clot_vessel
from src.training.clot_trigger_stack import (
    apply_clot_trigger_honest_env,
    apply_clot_trigger_oracle_forward_env,
    apply_oracle_neighbor_mask_env,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def _clot_phi_soft_labels_enabled() -> bool:
    import os

    return (os.environ.get("CLOT_PHI_SOFT_LABELS") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="T0 physics trigger viz (deploy nucleation projection)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument(
        "--band-mask",
        action="store_true",
        help="Grey out nodes outside loss support B_t (debug)",
    )
    ap.add_argument(
        "--show-raw",
        action="store_true",
        help="Add third row: unprojected raw gelation (debug failure mode)",
    )
    ap.add_argument(
        "--oracle-band",
        action="store_true",
        help="Legacy loss env: GT-mu seeds + dgamma slice (debug only)",
    )
    ap.add_argument(
        "--oracle-forward",
        action="store_true",
        help="Forward envelope from GT commits (not deploy)",
    )
    ap.add_argument("--prior-gate", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    apply_clot_trigger_honest_env()
    apply_physics_trigger_baseline_env()
    if bool(args.oracle_band):
        apply_oracle_neighbor_mask_env()
    if bool(args.oracle_forward):
        apply_clot_trigger_oracle_forward_env()
    if args.prior_gate:
        import os

        os.environ["CLOT_PHI_PHYSICS_GELATION_GATE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])
    use_soft = _clot_phi_soft_labels_enabled()

    traj = rollout_clot_trigger_physics(
        data, phys_cfg=phys, bio_cfg=bio, device=device, time_stride=1
    )
    full_ones = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)

    frames: list[dict] = []
    for t in _pick_times(n_steps, int(args.max_frames)):
        step = build_clot_phi_step(data, t, phys, bio, device)
        support = step.loss_mask.reshape(-1).bool()
        phi_gt = step.phi_gt.reshape(-1)
        bundle = traj[int(t)]
        phi_deploy = bundle["phi"]
        phi_raw = bundle["phi_raw"]
        m_sup = clot_trigger_viz_f1(phi_deploy, phi_gt, support)
        m_full = clot_trigger_viz_f1(phi_deploy, phi_gt, full_ones)
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_gt": phi_gt.detach().cpu().numpy(),
                "phi_deploy": phi_deploy.detach().cpu().numpy(),
                "phi_raw": phi_raw.detach().cpu().numpy(),
                "region": support.detach().cpu().numpy(),
                "f1_support": float(m_sup["clot_f1"]),
                "f1_full": float(m_full["clot_f1"]),
                "lumen_fp": lumen_false_positive_frac(
                    phi_deploy, phi_gt, data=data, device=device
                ),
            }
        )

    nrows = 3 if bool(args.show_raw) else 2
    ncols = len(frames)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.5 * ncols, 10), 2.8 * nrows), squeeze=False)
    deploy_label = (
        "physics deploy (pred nucleation)"
        if not forward_path_uses_gt_commits()
        else "physics (GT-seed nucleation)"
    )
    fig.suptitle(
        f"T0 physics trigger -- {args.anchor} | row0=GT | row1={deploy_label}",
        fontsize=11,
    )
    for j, fr in enumerate(frames):
        title = (
            f"t={fr['t']} tau={fr['tau']:.2f}\n"
            f"F1 sup={fr['f1_support']:.2f} full={fr['f1_full']:.2f} lumen_fp={fr['lumen_fp']:.2f}"
        )
        scatter_clot_vessel(
            axes[0, j],
            pos,
            fr["phi_gt"],
            "GT" if j == 0 else "",
            scatter_size=float(args.scatter_size),
            mask_outside_region=bool(args.band_mask),
            region=fr["region"],
        )
        scatter_clot_vessel(
            axes[1, j],
            pos,
            fr["phi_deploy"],
            deploy_label if j == 0 else "",
            scatter_size=float(args.scatter_size),
            mask_outside_region=bool(args.band_mask),
            region=fr["region"],
        )
        axes[0, j].set_title(title, fontsize=7)
        if nrows > 2:
            scatter_clot_vessel(
                axes[2, j],
                pos,
                fr["phi_raw"],
                "raw gelation (debug)" if j == 0 else "",
                scatter_size=float(args.scatter_size),
            )

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t0_{args.anchor}.png"
    )
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "step": "t0_oracle",
                "rollout_config": snapshot_clot_trigger_rollout_config(),
                "deploy_faithful_forward": not forward_path_uses_gt_commits(),
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_gt", "phi_deploy", "phi_raw", "region")}
                    for fr in frames
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
