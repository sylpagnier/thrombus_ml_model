"""Star 2 (T2): frozen T1 hybrid trigger with pred GINO-DEQ flow + GT species.

Usage:
  python scripts/eval_clot_trigger_t2_pred_flow.py
  python scripts/eval_clot_trigger_t2_pred_flow.py --checkpoint outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    clot_phi_feature_dim,
    clot_phi_model_uses_mpnn,
)
from src.evaluation.clot_phi_checkpoint_env import apply_clot_phi_config_from_checkpoint, apply_clot_phi_eval_defaults
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor
from src.training.clot_trigger_stack import (
    apply_star2_eval_env,
    default_t1_checkpoint_path,
    forward_clot_trigger_hybrid,
    forward_physics_trigger_phi,
    reset_star2_kinematics_cache,
)
from src.training.train_clot_phi_simple import _clot_metrics


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def _load_trigger_model(ckpt_path: Path, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(raw.get("config") or {})
    apply_clot_phi_config_from_checkpoint(cfg)
    hidden = int(cfg.get("hidden", 32))
    in_dim = int(cfg.get("in_dim", clot_phi_feature_dim()))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def eval_anchor_t2(
    graph_path: Path,
    model: torch.nn.Module,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict:
    reset_star2_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    edge_index = data.edge_index.to(device) if clot_phi_model_uses_mpnn(model) else None
    n_steps = int(data.y.shape[0])
    phi_hyb_by_t: dict[int, torch.Tensor] = {}
    per_step: list[dict] = []

    for t in range(n_steps):
        step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1)
        phi_phys, _ = forward_physics_trigger_phi(
            step, data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, apply_region=True
        )
        bundle = forward_clot_trigger_hybrid(
            model, step, data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, edge_index=edge_index
        )
        phi_hyb = bundle["phi_hybrid"]
        phi_hyb_by_t[int(t)] = phi_hyb.reshape(-1)
        m_phys = _clot_metrics(phi_phys, phi_gt, step.loss_mask.reshape(-1).bool())
        m_hyb = _clot_metrics(phi_hyb, phi_gt, step.loss_mask.reshape(-1).bool())
        per_step.append(
            {
                "t": t,
                "f1_phys": float(m_phys["clot_f1"]),
                "f1_hybrid": float(m_hyb["clot_f1"]),
                "mesh_f1_phys": float(m_phys["clot_f1"]),
                "mesh_f1_hybrid": float(m_hyb["clot_f1"]),
                "pred_pos_frac": float(m_hyb["pred_pos_frac"]),
                "gt_pos_frac": float(m_hyb["gt_pos_frac"]),
            }
        )

    traj = eval_phi_trajectory_on_anchor(
        phi_hyb_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="t2_pred_kine",
    )
    return {
        "anchor": graph_path.stem,
        "inputs": "pred_kine + GT_species",
        "n_steps": n_steps,
        "mean_f1_phys": _mean(per_step, "f1_phys"),
        "mean_f1_hybrid": _mean(per_step, "f1_hybrid"),
        "final_f1_phys": per_step[-1]["f1_phys"] if per_step else float("nan"),
        "final_f1_hybrid": per_step[-1]["f1_hybrid"] if per_step else float("nan"),
        "trajectory_score": float(traj.get("trajectory_score", float("nan"))),
        "tfinal_wall_ring_frac": float(traj.get("tfinal_wall_ring_frac", float("nan"))),
        "per_step": per_step,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T2 clot trigger eval (pred kine + GT species)")
    ap.add_argument("--anchor-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument(
        "--checkpoint",
        default="",
        help="Frozen T1 trigger ckpt (default: outputs/biochem/clot_trigger/t1/clot_trigger_t1_best.pth)",
    )
    ap.add_argument("--kine-ckpt", default="outputs/kinematics/kinematics_best.pth")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t2_pred_flow.json")
    args = ap.parse_args()

    apply_star2_eval_env(kine_ckpt=args.kine_ckpt)
    apply_clot_phi_eval_defaults()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    ckpt_path = Path(args.checkpoint) if args.checkpoint.strip() else default_t1_checkpoint_path()
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO / ckpt_path
    if not ckpt_path.is_file():
        print(f"[ERR] missing T1 checkpoint: {ckpt_path}", file=sys.stderr)
        return 2

    model, cfg = _load_trigger_model(ckpt_path, device)
    print(
        f"[i] T2 eval ckpt={ckpt_path.name} star={cfg.get('clot_trigger_star', '?')} "
        f"kine={args.kine_ckpt}",
        flush=True,
    )

    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else (
        _REPO / VesselConfig(phase="biochem_anchors").graph_output_dir
    )
    paths = sorted(anchor_dir.glob("*.pt"))
    if not paths:
        print(f"[ERR] no graphs in {anchor_dir}", file=sys.stderr)
        return 2

    rows = [
        eval_anchor_t2(p, model, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
        for p in paths
    ]
    for r in rows:
        print(
            f"[OK] {r['anchor']}: mean_hyb_F1={r['mean_f1_hybrid']:.3f} "
            f"final_hyb_F1={r['final_f1_hybrid']:.3f} "
            f"mean_phys_F1={r['mean_f1_phys']:.3f} traj={r['trajectory_score']:.3f}",
            flush=True,
        )

    val_row = next((r for r in rows if r["anchor"] == args.val), None)
    summary = {
        "step": "t2_pred_flow",
        "inputs": "pred_GINO_DEQ_flow + GT_species (frozen T1 trigger)",
        "checkpoint": str(ckpt_path.relative_to(_REPO) if ckpt_path.is_relative_to(_REPO) else ckpt_path),
        "kine_ckpt": args.kine_ckpt,
        "mean_f1_hybrid": _mean(rows, "mean_f1_hybrid"),
        "mean_f1_phys": _mean(rows, "mean_f1_phys"),
        "mean_trajectory_score": _mean(rows, "trajectory_score"),
        "val_anchor": args.val,
        "val_mean_f1_hybrid": float(val_row["mean_f1_hybrid"]) if val_row else float("nan"),
        "val_final_f1_hybrid": float(val_row["final_f1_hybrid"]) if val_row else float("nan"),
        "val_trajectory_score": float(val_row["trajectory_score"]) if val_row else float("nan"),
        "per_anchor": [{k: v for k, v in r.items() if k != "per_step"} for r in rows],
    }
    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    print(
        f"[summary] mean_hyb_F1={summary['mean_f1_hybrid']:.3f} "
        f"mean_traj={summary['mean_trajectory_score']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
