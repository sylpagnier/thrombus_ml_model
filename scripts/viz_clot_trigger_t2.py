"""T2 clot trigger timeline viz: GT | physics (pred kine) | hybrid (pred kine)."""

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
from src.core_physics.clot_phi_simple import build_clot_phi_model, build_clot_phi_step, clot_phi_feature_dim
from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_config_from_checkpoint, apply_clot_phi_eval_defaults
from src.evaluation.viz_clot_trigger import clot_trigger_viz_f1, clot_trigger_viz_phis, scatter_clot_vessel
from src.training.clot_trigger_stack import (
    apply_star2_eval_env,
    default_t1_checkpoint_path,
    reset_star2_kinematics_cache,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


def _pick_times(n_steps: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n_steps <= max_frames:
        return list(range(n_steps))
    idx = np.linspace(0, n_steps - 1, num=max_frames, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


def main() -> int:
    ap = argparse.ArgumentParser(description="T2 clot trigger timeline viz (pred kine)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--kine-ckpt", default="outputs/kinematics/kinematics_best.pth")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--scatter-size", type=float, default=4.0)
    ap.add_argument(
        "--band-mask",
        action="store_true",
        help="Grey out nodes outside the supervision band (legacy debug)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    apply_star2_eval_env(kine_ckpt=args.kine_ckpt)
    apply_clot_phi_eval_defaults()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    ckpt_path = Path(args.checkpoint) if args.checkpoint.strip() else default_t1_checkpoint_path()
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(raw.get("config") or {})
    apply_clot_phi_config_from_checkpoint(cfg)
    hidden = int(cfg.get("hidden", 32))
    in_dim = int(cfg.get("in_dim", clot_phi_feature_dim()))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()

    reset_star2_kinematics_cache()
    graph_path = root / args.anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    pos = data.x[:, :2].detach().cpu().numpy()
    n_steps = int(data.y.shape[0])
    edge_index = data.edge_index.to(device)

    frames: list[dict] = []
    for t in _pick_times(n_steps, int(args.max_frames)):
        step = build_clot_phi_step(data, t, phys, bio, device)
        mask = step.loss_mask.reshape(-1).bool()
        display = clot_trigger_viz_phis(
            step,
            data,
            phys_cfg=phys,
            bio_cfg=bio,
            device=device,
            model=model,
            edge_index=edge_index,
        )
        m_phys = clot_trigger_viz_f1(display["phi_phys"], display["phi_gt"], mask)
        m_hyb = clot_trigger_viz_f1(display["phi_hybrid"], display["phi_gt"], mask)
        frames.append(
            {
                "t": int(t),
                "tau": float(macro_tau_at_index(data, int(t), bio_cfg=bio)),
                "phi_gt": display["phi_gt"].detach().cpu().numpy(),
                "phi_phys": display["phi_phys"].detach().cpu().numpy(),
                "phi_hybrid": display["phi_hybrid"].detach().cpu().numpy(),
                "region": mask.detach().cpu().numpy(),
                "f1_phys": float(m_phys["clot_f1"]),
                "f1_hybrid": float(m_hyb["clot_f1"]),
            }
        )

    ncols = len(frames)
    fig, axes = plt.subplots(3, ncols, figsize=(max(2.5 * ncols, 10), 7.5), squeeze=False)
    fig.suptitle(
        f"T2 clot trigger -- {args.anchor} | full vessel (F1 in support) | row0=GT | row1=physics | row2=hybrid",
        fontsize=11,
    )
    row_labels = ("GT", "physics (pred kine)", "hybrid (pred kine)")
    keys = ("phi_gt", "phi_phys", "phi_hybrid")
    for j, fr in enumerate(frames):
        title = (
            f"t={fr['t']} tau={fr['tau']:.2f}\n"
            f"phys F1={fr['f1_phys']:.2f} hyb F1={fr['f1_hybrid']:.2f}"
        )
        for ri, (label, key) in enumerate(zip(row_labels, keys)):
            scatter_clot_vessel(
                axes[ri, j],
                pos,
                fr[key],
                label if j == 0 else "",
                scatter_size=float(args.scatter_size),
                mask_outside_region=bool(args.band_mask),
                region=fr["region"],
            )
        axes[0, j].set_title(title, fontsize=7)

    fig.tight_layout()
    out_path = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/viz/clot_trigger/t2_{args.anchor}.png"
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
                "step": "t2_pred_flow",
                "checkpoint": str(
                    ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path
                ),
                "kine_ckpt": args.kine_ckpt,
                "frames": [
                    {k: v for k, v in fr.items() if k not in ("phi_gt", "phi_phys", "phi_hybrid", "region")}
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
