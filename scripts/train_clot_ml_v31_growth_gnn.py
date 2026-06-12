"""V3.1: band GNN with spatial growth loss + soft commits."""

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
from src.training.clot_ml_v2_growth_gnn import (  # noqa: E402
    ClotGrowthRateGNN,
    V31TrainConfig,
    apply_step3_v3_env,
    default_v31_out_dir,
    eval_v3_on_anchor,
    growth_gnn_feature_dim,
    load_v3_checkpoint,
    resolve_v3_rule_cfg,
    save_v3_checkpoint,
    train_one_graph_v31,
    val_score_from_eval_row,
    teacher_phi_by_t_from_step1,
)
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="V3.1 growth GNN train")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--init-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--no-init", action="store_true")
    ap.add_argument("--no-teacher", action="store_true")
    args = ap.parse_args()

    apply_step3_v3_env(v31=True)
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    cfg = V31TrainConfig(
        step0_json=args.step0_json,
        hidden=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        init_ckpt=str(args.init_ckpt),
        teacher_weight=0.0 if args.no_teacher else 0.08,
    )

    device = resolve_clot_ml_training_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_v3_rule_cfg(REPO / cfg.step0_json)

    paths = [str(p) for p in discover_anchor_paths(REPO / args.anchor_dir)]
    train_paths, val_paths = _split_train_val(paths, args.val)

    model = ClotGrowthRateGNN(in_dim=growth_gnn_feature_dim(), hidden=cfg.hidden).to(device)
    if not args.no_init and cfg.init_ckpt and Path(REPO / cfg.init_ckpt).exists():
        init_model, _ = load_v3_checkpoint(REPO / cfg.init_ckpt, device=device, v31=False)
        model.load_state_dict(init_model.state_dict(), strict=True)
        print(f"[i] init from {cfg.init_ckpt}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out_dir = Path(args.out_dir) if args.out_dir else default_v31_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_ml_v31_growth_gnn_best.pth"
    log_path = out_dir / "train_log.jsonl"

    best_val = -1.0
    print(f"[i] V3.1 train epochs={cfg.epochs} val={args.val} device={device}", flush=True)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            teacher_phi = None
            if cfg.teacher_weight > 0.0 and Path(REPO / cfg.teacher_ckpt).exists():
                teacher_phi = teacher_phi_by_t_from_step1(
                    data,
                    rule_cfg,
                    device=device,
                    phys_cfg=phys,
                    bio_cfg=bio,
                    teacher_ckpt=REPO / cfg.teacher_ckpt,
                    alpha=0.35,
                )
            opt.zero_grad(set_to_none=True)
            loss = train_one_graph_v31(
                model,
                data,
                rule_cfg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                teacher_phi_by_t=teacher_phi,
                teacher_weight=cfg.teacher_weight,
                final_frame_boost=cfg.final_frame_boost,
            )
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_paths), 1)

        model.eval()
        val_rows = [
            eval_v3_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
            )
            for p in val_paths
        ]
        for r in val_rows:
            r["val_score"] = val_score_from_eval_row(r)
        val_score = sum(r["val_score"] for r in val_rows) / max(len(val_rows), 1)
        row_log = {"epoch": ep, "train_loss": train_loss, "val_score": val_score, "val_rows": val_rows}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(row_log)) + "\n")
        print(
            f"[i] ep={ep} loss={train_loss:.4f} val_score={val_score:.3f} "
            f"shape={val_rows[0].get('tfinal_clot_shape', float('nan')):.3f}",
            flush=True,
        )
        if val_score > best_val:
            best_val = val_score
            meta = {
                "step": "v31",
                "recipe": "v31",
                "step0_json": str(cfg.step0_json),
                "hidden": cfg.hidden,
                "in_dim": growth_gnn_feature_dim(),
                "val_stem": args.val,
                "best_val_score": best_val,
                "init_ckpt": cfg.init_ckpt,
            }
            save_v3_checkpoint(ckpt_path, model=model, meta=meta)

    model, meta = load_v3_checkpoint(ckpt_path, device=device, v31=True)
    per_anchor = [
        eval_v3_on_anchor(
            model,
            rule_cfg,
            graph_path=Path(p),
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        for p in paths
    ]
    for r in per_anchor:
        r["val_score"] = val_score_from_eval_row(r)
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    mean_shape = sum(float(r.get("tfinal_clot_shape", 0.0)) for r in per_anchor) / len(per_anchor)
    summary = {
        "step": "v31_growth_gnn",
        "checkpoint": str(ckpt_path),
        "mean_deploy": mean_deploy,
        "mean_clot_shape": mean_shape,
        "per_anchor": per_anchor,
        "meta": meta,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    print(f"[OK] mean_deploy={mean_deploy:.3f} mean_shape={mean_shape:.3f} best_val={best_val:.3f}", flush=True)
    print(f"[save] {ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
