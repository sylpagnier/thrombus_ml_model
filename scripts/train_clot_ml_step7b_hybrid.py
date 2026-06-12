"""Step 7b ladder: train residual MLP on frozen rule_mixture shell.

Usage:
  python scripts/train_clot_ml_step7b_hybrid.py
  python scripts/train_clot_ml_step7b_hybrid.py --epochs 40
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
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_step7b_hybrid import (  # noqa: E402
    ClotRuleResidualMLP,
    Step7bTrainConfig,
    default_step7b_out_dir,
    eval_step7b_on_anchor,
    load_frozen_mixture,
    load_step7b_checkpoint,
    resolve_step7b_rule_cfg,
    save_step7b_checkpoint,
    step7b_feature_dim,
    train_one_graph,
)
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
    ap = argparse.ArgumentParser(description="Step 7b: hybrid rule_mixture + residual MLP")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument(
        "--mixture-ckpt",
        default="outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth",
    )
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    _apply_deploy_env()
    cfg_train = Step7bTrainConfig(
        step0_json=args.step0_json,
        mixture_ckpt=args.mixture_ckpt,
        alpha=float(args.alpha),
        hidden=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
    )

    anchor_dir = REPO / args.anchor_dir
    paths = [str(p) for p in discover_anchor_paths(anchor_dir)]
    train_paths, _ = _split_train_val(paths, args.val)
    rule_cfg = resolve_step7b_rule_cfg(REPO / cfg_train.step0_json)

    device = resolve_clot_ml_training_device()
    mixture_path = REPO / cfg_train.mixture_ckpt
    if not mixture_path.is_file():
        raise FileNotFoundError(f"mixture checkpoint missing: {mixture_path}")
    mixture, mixture_meta = load_frozen_mixture(mixture_path, device=device)
    for p in mixture.parameters():
        p.requires_grad = False
    mixture.eval()

    residual = ClotRuleResidualMLP(in_dim=step7b_feature_dim(), hidden=cfg_train.hidden).to(device)
    opt = torch.optim.Adam(residual.parameters(), lr=cfg_train.lr)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    out_dir = Path(args.out_dir) if args.out_dir else default_step7b_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_ml_step7b_best.pth"
    log_path = out_dir / "train_log.jsonl"

    best_mean_deploy = -1.0
    print(
        f"[i] Step7b train epochs={cfg_train.epochs} alpha={cfg_train.alpha} "
        f"mixture={cfg_train.mixture_ckpt} early_paint_w={cfg_train.early_paint_weight} "
        f"val_holdout={args.val} ckpt=mean_deploy_all_anchors device={device}",
        flush=True,
    )

    for ep in range(1, cfg_train.epochs + 1):
        residual.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            opt.zero_grad(set_to_none=True)
            loss = train_one_graph(
                residual,
                mixture,
                data,
                rule_cfg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=cfg_train.alpha,
                early_paint_weight=cfg_train.early_paint_weight,
                final_bce_weight=cfg_train.final_bce_weight,
            )
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_paths), 1)

        residual.eval()
        all_rows = [
            eval_step7b_on_anchor(
                residual,
                mixture,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                alpha=cfg_train.alpha,
            )
            for p in paths
        ]
        mean_deploy = sum(r["deploy_score"] for r in all_rows) / len(all_rows)
        row_log = {
            "epoch": ep,
            "train_loss": train_loss,
            "mean_deploy_all": mean_deploy,
            "per_anchor": all_rows,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(row_log)) + "\n")
        print(f"[i] ep={ep} loss={train_loss:.4f} mean_deploy={mean_deploy:.3f}", flush=True)

        if mean_deploy > best_mean_deploy:
            best_mean_deploy = mean_deploy
            meta = {
                "step": "7b",
                "step0_json": str(cfg_train.step0_json),
                "mixture_ckpt": str(cfg_train.mixture_ckpt),
                "mixture_meta": mixture_meta,
                "alpha": cfg_train.alpha,
                "hidden": cfg_train.hidden,
                "early_paint_weight": cfg_train.early_paint_weight,
                "val_holdout": args.val,
                "best_mean_deploy": best_mean_deploy,
            }
            save_step7b_checkpoint(ckpt_path, residual=residual, meta=meta)

    mixture, residual, meta = load_step7b_checkpoint(ckpt_path, device=device)
    per_anchor = [
        eval_step7b_on_anchor(
            residual,
            mixture,
            rule_cfg,
            graph_path=Path(p),
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            alpha=float(meta.get("alpha", cfg_train.alpha)),
        )
        for p in paths
    ]
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    summary = {
        "step": "7b",
        "checkpoint": str(ckpt_path),
        "mixture_ckpt": cfg_train.mixture_ckpt,
        "alpha": cfg_train.alpha,
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "meta": meta,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    print(f"[OK] mean_deploy={mean_deploy:.3f} best_mean={best_mean_deploy:.3f}", flush=True)
    print(f"[save] {ckpt_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
