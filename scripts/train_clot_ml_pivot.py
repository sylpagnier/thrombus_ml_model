"""Train clot ML side pivots: soft_commit | rule_mixture | data_driven.

Usage:
  python scripts/train_clot_ml_pivot.py --pivot soft_commit
  python scripts/train_clot_ml_pivot.py --pivot rule_mixture --epochs 30
  python scripts/train_clot_ml_pivot.py --pivot data_driven
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_training_device  # noqa: E402
from src.training.clot_ml_pivot_common import load_pivot_checkpoint, save_pivot_checkpoint  # noqa: E402
from src.training.clot_ml_pivot_data_driven import (  # noqa: E402
    ClotDataDrivenPhiGNN,
    PivotDataTrainConfig,
    build_data_driven_model,
    default_data_driven_out_dir,
    eval_data_driven_on_anchor,
    train_one_graph as train_data_one,
)
from src.training.clot_ml_pivot_rule_mixture import (  # noqa: E402
    ClotRuleMixtureModel,
    PivotMixtureTrainConfig,
    build_rule_mixture_model,
    default_mixture_out_dir,
    eval_rule_mixture_on_anchor,
    resolve_mixture_rule_cfg,
    train_one_graph as train_mix_one,
)
from src.training.clot_ml_pivot_soft_commit import (  # noqa: E402
    ClotSoftCommitModel,
    PivotSoftTrainConfig,
    build_soft_commit_model,
    default_soft_out_dir,
    eval_soft_commit_on_anchor,
    resolve_soft_rule_cfg,
    train_one_graph as train_soft_one,
)
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402


def _apply_deploy_env() -> None:
    os.environ["BIOCHEM_PRIOR_COMSOL_ALIGNED"] = "1"
    os.environ["BIOCHEM_PRIOR_NORM_MASK"] = "adjacent"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"
    os.environ["CLOT_PHI_CEILING_HOPS"] = "2"
    os.environ["CLOT_FORECAST_MASK"] = "ceiling_growth"
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    os.environ.setdefault("CLOT_PHI_KINE_CKPT", "outputs/kinematics/kinematics_best.pth")


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Clot ML side pivot trainer")
    ap.add_argument(
        "--pivot",
        required=True,
        choices=("soft_commit", "rule_mixture", "data_driven"),
    )
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    _apply_deploy_env()
    pivot = args.pivot
    anchor_dir = REPO / args.anchor_dir
    paths = [str(p) for p in discover_anchor_paths(anchor_dir)]
    train_paths, _ = _split_train_val(paths, args.val)
    device = resolve_clot_ml_training_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    if pivot == "soft_commit":
        cfg = PivotSoftTrainConfig(
            step0_json=args.step0_json,
            hidden=int(args.hidden),
            lr=float(args.lr),
            epochs=int(args.epochs),
        )
        rule_cfg = resolve_soft_rule_cfg(REPO / cfg.step0_json)
        model = ClotSoftCommitModel(hidden=cfg.hidden).to(device)
        out_dir = Path(args.out_dir) if args.out_dir else default_soft_out_dir()
        ckpt_name = "clot_ml_pivot_soft_commit_best.pth"
        train_fn = lambda m, d: train_soft_one(
            m,
            d,
            rule_cfg,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            early_paint_weight=cfg.early_paint_weight,
            final_bce_weight=cfg.final_bce_weight,
        )
        eval_fn = lambda m, p: eval_soft_commit_on_anchor(
            m, rule_cfg, graph_path=Path(p), device=device, phys_cfg=phys, bio_cfg=bio
        )
        model_ctor = build_soft_commit_model
        meta_extra = {"step0_json": cfg.step0_json, "hidden": cfg.hidden}
    elif pivot == "rule_mixture":
        cfg = PivotMixtureTrainConfig(
            step0_json=args.step0_json,
            hidden=int(args.hidden),
            lr=float(args.lr),
            epochs=int(args.epochs),
        )
        rule_cfg = resolve_mixture_rule_cfg(REPO / cfg.step0_json)
        model = ClotRuleMixtureModel(hidden=cfg.hidden).to(device)
        out_dir = Path(args.out_dir) if args.out_dir else default_mixture_out_dir()
        ckpt_name = "clot_ml_pivot_rule_mixture_best.pth"
        train_fn = lambda m, d: train_mix_one(
            m,
            d,
            rule_cfg,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            early_paint_weight=cfg.early_paint_weight,
            final_rank_weight=cfg.final_rank_weight,
        )
        eval_fn = lambda m, p: eval_rule_mixture_on_anchor(
            m, rule_cfg, graph_path=Path(p), device=device, phys_cfg=phys, bio_cfg=bio
        )
        model_ctor = build_rule_mixture_model
        meta_extra = {"step0_json": cfg.step0_json, "hidden": cfg.hidden}
    else:
        cfg = PivotDataTrainConfig(hidden=int(args.hidden), lr=float(args.lr), epochs=int(args.epochs))
        model = ClotDataDrivenPhiGNN(hidden=cfg.hidden).to(device)
        out_dir = Path(args.out_dir) if args.out_dir else default_data_driven_out_dir()
        ckpt_name = "clot_ml_pivot_data_driven_best.pth"
        train_fn = lambda m, d: train_data_one(
            m,
            d,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            early_paint_weight=cfg.early_paint_weight,
            final_bce_weight=cfg.final_bce_weight,
        )
        eval_fn = lambda m, p: eval_data_driven_on_anchor(
            m, graph_path=Path(p), device=device, phys_cfg=phys, bio_cfg=bio
        )
        model_ctor = build_data_driven_model
        meta_extra = {"hidden": cfg.hidden}

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / ckpt_name
    log_path = out_dir / "train_log.jsonl"
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    best_mean_deploy = -1.0

    print(
        f"[i] pivot={pivot} epochs={args.epochs} hidden={args.hidden} "
        f"val_holdout={args.val} ckpt=mean_deploy_all_anchors device={device}",
        flush=True,
    )

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            opt.zero_grad(set_to_none=True)
            loss = train_fn(model, data)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_paths), 1)

        model.eval()
        all_rows = [eval_fn(model, p) for p in paths]
        mean_deploy = sum(r["deploy_score"] for r in all_rows) / len(all_rows)
        row_log = {"epoch": ep, "train_loss": train_loss, "mean_deploy_all": mean_deploy, "per_anchor": all_rows}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(row_log)) + "\n")
        print(f"[i] ep={ep} loss={train_loss:.4f} mean_deploy={mean_deploy:.3f}", flush=True)

        if mean_deploy > best_mean_deploy:
            best_mean_deploy = mean_deploy
            meta = {"pivot": pivot, "val_holdout": args.val, "best_mean_deploy": best_mean_deploy, **meta_extra}
            save_pivot_checkpoint(ckpt_path, model=model, meta=meta)

    model, meta = load_pivot_checkpoint(ckpt_path, device=device, model_ctor=model_ctor)
    per_anchor = [eval_fn(model, p) for p in paths]
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    summary = {
        "pivot": pivot,
        "checkpoint": str(ckpt_path),
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "meta": meta,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    print(f"[OK] pivot={pivot} mean_deploy={mean_deploy:.3f} best_mean={best_mean_deploy:.3f}", flush=True)
    print(f"[save] {ckpt_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
