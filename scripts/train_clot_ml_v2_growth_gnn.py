"""V3 ladder: train band GNN growth rate + nucleation (LOAO val)."""

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
from src.training.clot_ml_step1_residual import load_step1_checkpoint  # noqa: E402
from src.training.clot_ml_v2_growth_gnn import (  # noqa: E402
    ClotGrowthRateGNN,
    V3TrainConfig,
    apply_step3_v3_env,
    default_v3_out_dir,
    eval_v3_on_anchor,
    growth_gnn_feature_dim,
    load_v3_checkpoint,
    resolve_v3_rule_cfg,
    save_v3_checkpoint,
    train_one_graph,
)
from src.training.clot_ml_v2_step1_nucleation import rollout_step1_v1_nucleation  # noqa: E402
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


@torch.no_grad()
def _teacher_phi_by_t(
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    teacher_ckpt: str,
    alpha: float,
) -> dict[int, torch.Tensor]:
    model, meta = load_step1_checkpoint(REPO / teacher_ckpt, device=device)
    a = float(meta.get("alpha", alpha))
    return rollout_step1_v1_nucleation(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=a,
        sim_end_scale=1.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="V3 band GNN growth rate train")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--teacher-weight", type=float, default=0.15)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--teacher-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    args = ap.parse_args()

    apply_step3_v3_env()
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    cfg = V3TrainConfig(
        step0_json=args.step0_json,
        hidden=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        teacher_weight=0.0 if args.no_teacher else float(args.teacher_weight),
        teacher_ckpt=str(args.teacher_ckpt),
    )

    device = resolve_clot_ml_training_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_v3_rule_cfg(REPO / cfg.step0_json)

    paths = [str(p) for p in discover_anchor_paths(REPO / args.anchor_dir)]
    train_paths, val_paths = _split_train_val(paths, args.val)

    model = ClotGrowthRateGNN(in_dim=growth_gnn_feature_dim(), hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(args.out_dir) if args.out_dir else default_v3_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_ml_v3_growth_gnn_best.pth"
    log_path = out_dir / "train_log.jsonl"

    best_val_deploy = -1.0
    print(
        f"[i] V3 train epochs={cfg.epochs} val={args.val} teacher_w={cfg.teacher_weight} device={device}",
        flush=True,
    )

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            teacher_phi = None
            if cfg.teacher_weight > 0.0 and Path(REPO / cfg.teacher_ckpt).exists():
                teacher_phi = _teacher_phi_by_t(
                    data,
                    rule_cfg,
                    device=device,
                    phys_cfg=phys,
                    bio_cfg=bio,
                    teacher_ckpt=cfg.teacher_ckpt,
                    alpha=0.35,
                )
            opt.zero_grad(set_to_none=True)
            loss = train_one_graph(
                model,
                data,
                rule_cfg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                early_paint_weight=cfg.early_paint_weight,
                final_bce_weight=cfg.final_bce_weight,
                teacher_phi_by_t=teacher_phi,
                teacher_weight=cfg.teacher_weight,
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
        val_deploy = sum(r["deploy_score"] for r in val_rows) / max(len(val_rows), 1)
        row_log = {"epoch": ep, "train_loss": train_loss, "val_deploy": val_deploy, "val_rows": val_rows}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_sanitize(row_log)) + "\n")
        print(f"[i] ep={ep} loss={train_loss:.4f} val_deploy={val_deploy:.3f}", flush=True)

        if val_deploy > best_val_deploy:
            best_val_deploy = val_deploy
            meta = {
                "step": "v3",
                "step0_json": str(cfg.step0_json),
                "hidden": cfg.hidden,
                "in_dim": growth_gnn_feature_dim(),
                "val_stem": args.val,
                "best_val_deploy": best_val_deploy,
                "teacher_weight": cfg.teacher_weight,
            }
            save_v3_checkpoint(ckpt_path, model=model, meta=meta)

    model, meta = load_v3_checkpoint(ckpt_path, device=device)
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
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    summary = {
        "step": "v3_growth_gnn",
        "checkpoint": str(ckpt_path),
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "meta": meta,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    print(f"[OK] mean_deploy={mean_deploy:.3f} best_val={best_val_deploy:.3f}", flush=True)
    print(f"[save] {ckpt_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
